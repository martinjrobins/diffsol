"""diffsol solves as JAX operations, with the ODE equations given as JAX functions.

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from diffsol_jax import Model

    model = Model(
        rhs=lambda x, p, t: -p[0] * x,
        init=lambda p, t: jnp.ones(2),
        nstates=2, nparams=1,
    )
    ys = model.solve(params, t_eval)       # params [..., nparams] -> ys [..., nt, nout]
    jax.grad(lambda p: model.solve(p, t_eval).sum())(params)

The functions act on one lane. diffsol calls the callbacks (compiled with the PJRT C API) for
all the lanes of a solve together. `jax.vmap` over `params` gives more lanes in one solve.

Reverse mode (`ad="adjoint"`, the default) uses the diffsol VJP: the forward solve keeps a
checkpoint in C++, and gives an int64 handle to the backward solve. The model keeps at most
`max_checkpoints` checkpoints. A backward solve whose checkpoint was freed runs the forward
solve again (from the `params` and `t_eval` that it also gets), so it gives the same gradient;
`max_checkpoints` controls only the memory.
Forward mode (`ad="forward"`) uses diffsol forward sensitivities; `jax.grad` also works with it.

There is no derivative with respect to `t_eval`.
"""

import weakref

import jax
import jax.numpy as jnp

from . import _lib, _lower, _primitives
from ._lib import DiffsolJaxError
from ._lower import CallbackId, Equations

__all__ = ["Model", "DiffsolJaxError", "CallbackId"]

_ODE_SOLVERS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}
_LINEAR_SOLVERS = {"default": 0, "lu": 1, "klu": 2}
_MATRIX_TYPES = {"nalgebra_dense": 0, "faer_dense": 1, "faer_sparse": 2}


class Model:
    def __init__(
        self,
        rhs,
        init,
        nstates,
        nparams,
        out=None,
        nout=0,
        mass=None,
        ode_solver="bdf",
        linear_solver="default",
        matrix_type="nalgebra_dense",
        rtol=1e-6,
        atol=1e-6,
        ad="adjoint",
        max_checkpoints=8,
    ):
        if not jax.config.read("jax_enable_x64"):
            raise DiffsolJaxError("diffsol_jax needs jax_enable_x64 (the solver uses f64)")
        if (out is None) != (nout == 0):
            raise DiffsolJaxError("give out and nout > 0 together")
        if ad not in ("adjoint", "forward"):
            raise DiffsolJaxError("ad must be 'adjoint' or 'forward'")
        self.eq = Equations(rhs, init, nstates, nparams, out, nout, mass)
        self.nstates, self.nparams, self.nout = nstates, nparams, nout
        self.nsolution = nout if nout > 0 else nstates
        self.rtol, self.atol, self.ad = float(rtol), float(atol), ad
        so = _lib.lib()
        self.id = _lib.call(
            so.diffsol_jax_model_new,
            nstates,
            nparams,
            nout,
            _MATRIX_TYPES[matrix_type],
            _LINEAR_SOLVERS[linear_solver],
            _ODE_SOLVERS[ode_solver],
            max_checkpoints,
        )
        self._finalizer = weakref.finalize(self, so.diffsol_jax_model_free, self.id)
        self._compile_options = None
        _primitives.MODELS[self.id] = self
        self.solve = self._make_solve()

    # ---- callbacks ----------------------------------------------------------------------------

    def _options(self):
        if self._compile_options is None:
            from jax._src.lib import xla_client

            self._compile_options = xla_client.CompileOptions().SerializeAsString()
        return self._compile_options

    def _add(self, cb, nbatch, p):
        so = _lib.lib()
        if so.diffsol_jax_model_has_variant(self.id, nbatch, int(cb), p):
            return
        code = _lower.lower(self.eq, cb, nbatch, p)
        opts = self._options()
        _lib.call(
            so.diffsol_jax_model_add_variant,
            self.id,
            nbatch,
            int(cb),
            p,
            code,
            len(code),
            opts,
            len(opts),
        )

    def ensure_callbacks(self, kind, nbatch, ndirections=1):
        """Compile the callbacks that a solve of `kind` with `nbatch` lanes needs (called by the
        lowering rules). All the callbacks of the model get a `P = 1` executable, so the set of
        callbacks of the diffsol ODE does not change between solve kinds."""
        available = self.eq.available()
        for cb in available:
            self._add(cb, nbatch, 1)
        if kind == "fwd_sens" and self.nparams > 1:
            for cb in _lower.FWD_SENS_MULTI_P:
                if cb in available:
                    self._add(cb, nbatch, self.nparams)
        if kind == "vjp_bkwd" and ndirections > 1:
            for cb in _lower.VJP_BKWD_MULTI_P:
                if cb in available:
                    self._add(cb, nbatch, ndirections)

    def clear_checkpoints(self):
        _lib.lib().diffsol_jax_model_clear_checkpoints(self.id)

    def num_checkpoints(self):
        return _lib.lib().diffsol_jax_model_num_checkpoints(self.id)

    def callback_stats(self, nbatch, reset=False):
        """`(ncalls, nstaged, nviews)` for the solver with `nbatch` lanes: the number of callback
        calls, of inputs copied because they were not 64-byte aligned, and of PJRT views made
        (the misses of the view cache)."""
        import ctypes

        ncalls, nstaged, nviews = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        ok = _lib.lib().diffsol_jax_model_stats(
            self.id,
            nbatch,
            ctypes.byref(ncalls),
            ctypes.byref(nstaged),
            ctypes.byref(nviews),
            int(reset),
        )
        return (ncalls.value, nstaged.value, nviews.value) if ok else (0, 0, 0)

    def num_recomputes(self):
        """The number of backward solves that ran the forward solve again, because the
        registry had freed their checkpoint."""
        return _lib.lib().diffsol_jax_model_num_recomputes(self.id)

    # ---- solve --------------------------------------------------------------------------------

    def _make_solve(self):
        kw = dict(model_id=self.id, rtol=self.rtol, atol=self.atol)

        def check(params, t_eval):
            params = jnp.asarray(params, jnp.float64)
            t_eval = jnp.asarray(t_eval, jnp.float64)
            if params.ndim < 1 or params.shape[-1] != self.nparams:
                raise DiffsolJaxError(f"params must have the shape [..., {self.nparams}]")
            if t_eval.ndim != 1:
                raise DiffsolJaxError("t_eval must have one axis")
            return params, t_eval

        if self.ad == "adjoint":

            @jax.custom_vjp
            def solve(params, t_eval):
                return _primitives.solve_dense_p.bind(params, t_eval, **kw)

            def fwd(params, t_eval):
                ys, handle = _primitives.solve_vjp_fwd_p.bind(params, t_eval, **kw)
                # params and t_eval let the backward solve run the forward solve again if the
                # registry has freed the checkpoint
                return ys, (handle, params, t_eval)

            def bwd(res, ct):
                handle, params, t_eval = res
                grad = _primitives.solve_vjp_bkwd_p.bind(handle, params, t_eval, ct, **kw)
                return grad, jnp.zeros_like(t_eval)

            solve.defvjp(fwd, bwd)
        else:

            @jax.custom_jvp
            def solve(params, t_eval):
                return _primitives.solve_dense_p.bind(params, t_eval, **kw)

            @solve.defjvp
            def _(primals, tangents):
                params, t_eval = primals
                dparams, _ = tangents
                ys, sens = _primitives.solve_fwd_sens_p.bind(params, t_eval, **kw)
                return ys, jnp.einsum("...jtk,...j->...tk", sens, dparams)

        def run(params, t_eval):
            """Solve for each lane of `params [..., nparams]` at the times `t_eval [nt]`.
            Return `ys [..., nt, nout]` (`nout = nstates` if the model has no `out`)."""
            return solve(*check(params, t_eval))

        return run
