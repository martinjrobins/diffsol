"""Make the diffsol callbacks from the user functions with JAX AD, and lower them to StableHLO.

The user functions act on one lane: `rhs(x[nstates], p[nparams], t)`, `init(p, t)`,
`out(x, p, t)` and `mass(v, p, t)` (which must be linear in `v`).

Each executable acts on all the lanes, with the row-major layouts that the C++ trampolines give
(see `src/callbacks.h`):
    x: [B, nstates]   p: [B, nparams]   v: [B, P, nv]   y: [B, P, ny]   t, beta: []
The `P` axis is only on `v` and `y`: the other inputs are the same for each `P`.
"""

import enum
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp


class CallbackId(enum.IntEnum):
    """The same values as `CallbackId` in `src/callbacks.h`."""

    RHS = 0
    RHS_JAC = 1
    RHS_SENS = 2
    RHS_ADJOINT = 3
    RHS_SENS_ADJOINT = 4
    INIT = 5
    INIT_SENS = 6
    INIT_SENS_ADJOINT = 7
    MASS = 8
    MASS_ADJOINT = 9
    OUT = 10
    OUT_JAC = 11
    OUT_SENS = 12
    OUT_ADJOINT = 13
    OUT_SENS_ADJOINT = 14


# The callbacks that diffsol calls with more than one lane on `P`: in a forward sensitivity solve
# (`P = nparams`) and in a VJP backward solve (`P = ndirections`). See the diffsol-c test
# `lane_layout_rule`.
FWD_SENS_MULTI_P = (CallbackId.RHS_JAC, CallbackId.OUT_JAC, CallbackId.MASS)
VJP_BKWD_MULTI_P = (
    CallbackId.RHS_ADJOINT,
    CallbackId.RHS_SENS_ADJOINT,
    CallbackId.OUT_ADJOINT,
    CallbackId.OUT_SENS_ADJOINT,
    CallbackId.INIT_SENS_ADJOINT,
    CallbackId.MASS_ADJOINT,
)


@dataclass(frozen=True)
class Equations:
    rhs: Callable
    init: Callable
    nstates: int
    nparams: int
    out: Optional[Callable] = None
    nout: int = 0
    mass: Optional[Callable] = None

    def available(self):
        """The callbacks of this model."""
        ids = [
            CallbackId.RHS,
            CallbackId.RHS_JAC,
            CallbackId.RHS_SENS,
            CallbackId.RHS_ADJOINT,
            CallbackId.RHS_SENS_ADJOINT,
            CallbackId.INIT,
            CallbackId.INIT_SENS,
            CallbackId.INIT_SENS_ADJOINT,
        ]
        if self.mass is not None:
            ids += [CallbackId.MASS, CallbackId.MASS_ADJOINT]
        if self.out is not None:
            ids += [
                CallbackId.OUT,
                CallbackId.OUT_JAC,
                CallbackId.OUT_SENS,
                CallbackId.OUT_ADJOINT,
                CallbackId.OUT_SENS_ADJOINT,
            ]
        return ids


def _jvp(f, primal, tangent):
    return jax.jvp(f, (primal,), (tangent,))[1]


def _neg_vjp(f, primal, cotangent):
    return -jax.vjp(f, primal)[1](cotangent)[0]


def _lane_fn(eq: Equations, cb: CallbackId):
    """The callback for one lane, its kind, and the lengths `(nv, ny)`."""
    n, npar = eq.nstates, eq.nparams
    f, g, h, m = eq.rhs, eq.init, eq.out, eq.mass
    no = eq.nout
    C = CallbackId
    table = {
        C.RHS: ("fn", lambda x, p, t: f(x, p, t), 0, n),
        C.RHS_JAC: ("mul", lambda x, p, t, v: _jvp(lambda x: f(x, p, t), x, v), n, n),
        C.RHS_SENS: ("mul", lambda x, p, t, v: _jvp(lambda p: f(x, p, t), p, v), npar, n),
        C.RHS_ADJOINT: ("mul", lambda x, p, t, v: _neg_vjp(lambda x: f(x, p, t), x, v), n, n),
        C.RHS_SENS_ADJOINT: (
            "mul",
            lambda x, p, t, v: _neg_vjp(lambda p: f(x, p, t), p, v),
            n,
            npar,
        ),
        C.INIT: ("init", lambda p, t: g(p, t), 0, n),
        C.INIT_SENS: ("initmul", lambda p, t, v: _jvp(lambda p: g(p, t), p, v), npar, n),
        C.INIT_SENS_ADJOINT: (
            "initmul",
            lambda p, t, v: _neg_vjp(lambda p: g(p, t), p, v),
            n,
            npar,
        ),
    }
    if m is not None:
        table[C.MASS] = ("mass", lambda v, p, t, beta, y: m(v, p, t) + beta * y, n, n)
        table[C.MASS_ADJOINT] = (
            "mass",
            # M^T v (not negative), from the VJP of the linear map v -> M v
            lambda v, p, t, beta, y: jax.vjp(lambda u: m(u, p, t), v)[1](v)[0] + beta * y,
            n,
            n,
        )
    if h is not None:
        table[C.OUT] = ("fn", lambda x, p, t: h(x, p, t), 0, no)
        table[C.OUT_JAC] = ("mul", lambda x, p, t, v: _jvp(lambda x: h(x, p, t), x, v), n, no)
        table[C.OUT_SENS] = (
            "mul",
            lambda x, p, t, v: _jvp(lambda p: h(x, p, t), p, v),
            npar,
            no,
        )
        table[C.OUT_ADJOINT] = (
            "mul",
            lambda x, p, t, v: _neg_vjp(lambda x: h(x, p, t), x, v),
            no,
            n,
        )
        table[C.OUT_SENS_ADJOINT] = (
            "mul",
            lambda x, p, t, v: _neg_vjp(lambda p: h(x, p, t), p, v),
            no,
            npar,
        )
    return table[cb]


def batched(eq: Equations, cb: CallbackId):
    """The callback for all the lanes, with the executable layouts, and its argument shapes for
    `B` lanes and `P` lanes on `P`."""
    kind, fn, nv, ny = _lane_fn(eq, cb)
    vmap = jax.vmap
    f64 = jnp.float64

    def spec(*shape):
        return jax.ShapeDtypeStruct(shape, f64)

    def shapes(B, P):
        x, p, t = spec(B, eq.nstates), spec(B, eq.nparams), spec()
        v = spec(B, P, nv)
        y = spec(B, P, ny)
        return {
            "fn": (x, p, t),
            "mul": (x, p, t, v),
            "init": (p, t),
            "initmul": (p, t, v),
            "mass": (v, p, t, spec(), y),
        }[kind]

    if kind == "fn":
        # [B, ny] -> [B, 1, ny]
        def all_lanes(x, p, t):
            return vmap(fn, (0, 0, None))(x, p, t)[:, None, :]

    elif kind == "init":

        def all_lanes(p, t):
            return vmap(fn, (0, None))(p, t)[:, None, :]

    elif kind == "mul":
        over_p = vmap(fn, (None, None, None, 0))

        def all_lanes(x, p, t, v):
            return vmap(over_p, (0, 0, None, 0))(x, p, t, v)

    elif kind == "initmul":
        over_p = vmap(fn, (None, None, 0))

        def all_lanes(p, t, v):
            return vmap(over_p, (0, None, 0))(p, t, v)

    else:  # mass
        over_p = vmap(fn, (0, None, None, None, 0))

        def all_lanes(v, p, t, beta, y):
            return vmap(over_p, (0, 0, None, None, 0))(v, p, t, beta, y)

    return all_lanes, shapes


def _serialize(module) -> bytes:
    """StableHLO as a portable artifact (VHLO), so that a PJRT plugin of a different version can
    read it."""
    from jax._src.lib.mlir.dialects import hlo

    return hlo.serialize_portable_artifact_str(str(module), hlo.get_minimum_version())


def lower(eq: Equations, cb: CallbackId, B: int, P: int) -> bytes:
    """The StableHLO of callback `cb` for `B` lanes and `P` lanes on `P`."""
    all_lanes, shapes = batched(eq, cb)
    # the trampolines always give all the arguments, also those that the function does not use
    lowered = jax.jit(all_lanes, keep_unused=True).lower(*shapes(B, P))
    return _serialize(lowered.compiler_ir("stablehlo"))
