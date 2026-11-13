"""The JAX primitives of the diffsol solves: abstract evaluation, batching and lowering.

We use our own primitives (not only `jax.ffi.ffi_call`) because the callback executables depend
on the number of lanes `B`, and `B` is known only at lowering, after `vmap`. The lowering rule
compiles the callbacks that are missing for this `B`, and then emits the XLA FFI custom call.
"""

import math
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import batching, mlir

# model id -> Model, for the lowering rules (primitive params must be hashable)
MODELS: "weakref.WeakValueDictionary[int, object]" = weakref.WeakValueDictionary()


def _model(model_id):
    model = MODELS.get(model_id)
    if model is None:
        raise RuntimeError(f"the diffsol model {model_id} was freed")
    return model


def _f64(shape):
    return jax.core.ShapedArray(shape, jnp.float64)


# ---- primitives -------------------------------------------------------------------------------

solve_dense_p = Primitive("diffsol_solve_dense")
solve_fwd_sens_p = Primitive("diffsol_solve_fwd_sens")
solve_fwd_sens_p.multiple_results = True
solve_vjp_fwd_p = Primitive("diffsol_solve_vjp_fwd")
solve_vjp_fwd_p.multiple_results = True
solve_vjp_bkwd_p = Primitive("diffsol_solve_vjp_bkwd")


def _ys_shape(params, t_eval, model_id):
    return params.shape[:-1] + (t_eval.shape[0], _model(model_id).nsolution)


@solve_dense_p.def_abstract_eval
def _(params, t_eval, *, model_id, rtol, atol):
    return _f64(_ys_shape(params, t_eval, model_id))


@solve_fwd_sens_p.def_abstract_eval
def _(params, t_eval, *, model_id, rtol, atol):
    ys = _ys_shape(params, t_eval, model_id)
    nparams = params.shape[-1]
    sens = params.shape[:-1] + (nparams,) + ys[-2:]
    return [_f64(ys), _f64(sens)]


@solve_vjp_fwd_p.def_abstract_eval
def _(params, t_eval, *, model_id, rtol, atol):
    handle = jax.core.ShapedArray(params.shape[:-1], jnp.int64)
    return [_f64(_ys_shape(params, t_eval, model_id)), handle]


@solve_vjp_bkwd_p.def_abstract_eval
def _(handle, params, t_eval, dgdu, *, model_id, rtol, atol):
    return _f64(dgdu.shape[:-2] + (_model(model_id).nparams,))


for _p in (solve_dense_p, solve_fwd_sens_p, solve_vjp_fwd_p, solve_vjp_bkwd_p):
    _p.def_impl(lambda *args, _p=_p, **kw: jax.jit(lambda *a: _p.bind(*a, **kw))(*args))


# ---- batching ---------------------------------------------------------------------------------
# The leading axes of `params` are the diffsol lanes, so a vmapped axis of `params` moves to the
# front and becomes more lanes: one solve for all of them.


def _batch_solve(prim):
    def rule(args, dims, **kw):
        params, t_eval = args
        pdim, tdim = dims
        if tdim is not None:
            raise NotImplementedError("diffsol_jax: vmap over t_eval is not supported")
        params = jnp.moveaxis(params, pdim, 0)
        out = prim.bind(params, t_eval, **kw)
        return out, ([0] * len(out) if prim.multiple_results else 0)

    batching.primitive_batchers[prim] = rule


for _p in (solve_dense_p, solve_fwd_sens_p, solve_vjp_fwd_p):
    _batch_solve(_p)


def _batch_bkwd(args, dims, **kw):
    """`handle [*B]` and `params [*B, np]` have the lane axes. A vmapped axis of `handle` or
    `params` is a lane axis. A vmapped axis of only `dgdu` is a direction axis: it goes after the
    lane axes, and the backward solve does all the directions together."""
    handle, params, t_eval, dgdu = args
    hdim, pdim, tdim, gdim = dims
    if tdim is not None:
        raise NotImplementedError("diffsol_jax: vmap over t_eval is not supported")
    if hdim is not None or pdim is not None:
        size = handle.shape[hdim] if hdim is not None else params.shape[pdim]

        def front(x, d):
            if d is None:
                return jnp.broadcast_to(x, (size,) + x.shape)
            return jnp.moveaxis(x, d, 0)

        handle, params, dgdu = front(handle, hdim), front(params, pdim), front(dgdu, gdim)
        return solve_vjp_bkwd_p.bind(handle, params, t_eval, dgdu, **kw), 0
    pos = handle.ndim
    dgdu = jnp.moveaxis(dgdu, gdim, pos)
    return solve_vjp_bkwd_p.bind(handle, params, t_eval, dgdu, **kw), pos


batching.primitive_batchers[solve_vjp_bkwd_p] = _batch_bkwd


# ---- lowering ---------------------------------------------------------------------------------


def _lower_solve(prim, kind):
    target = prim.name

    def rule(ctx, params, t_eval, *, model_id, rtol, atol):
        aval = ctx.avals_in[0]
        nbatch = math.prod(aval.shape[:-1])
        _model(model_id).ensure_callbacks(kind, nbatch)
        return jax.ffi.ffi_lowering(target)(
            ctx,
            params,
            t_eval,
            model_id=np.int64(model_id),
            rtol=np.float64(rtol),
            atol=np.float64(atol),
        )

    mlir.register_lowering(prim, rule, platform="cpu")


_lower_solve(solve_dense_p, "dense")
_lower_solve(solve_fwd_sens_p, "fwd_sens")
_lower_solve(solve_vjp_fwd_p, "vjp_fwd")


def _lower_bkwd(ctx, handle, params, t_eval, dgdu, *, model_id, rtol, atol):
    haval, gaval = ctx.avals_in[0], ctx.avals_in[3]
    nbatch = math.prod(haval.shape)
    ndirections = math.prod(gaval.shape[haval.ndim : -2])
    _model(model_id).ensure_callbacks("vjp_bkwd", nbatch, ndirections)
    return jax.ffi.ffi_lowering(solve_vjp_bkwd_p.name)(
        ctx,
        handle,
        params,
        t_eval,
        dgdu,
        model_id=np.int64(model_id),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
    )


mlir.register_lowering(solve_vjp_bkwd_p, _lower_bkwd, platform="cpu")
