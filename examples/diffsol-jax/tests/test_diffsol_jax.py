"""Tests of diffsol_jax. Needs `DIFFSOL_JAX_PJRT_PLUGIN` (the CPU PJRT plugin), and
`libdiffsol_jax` in `examples/diffsol-jax/build` (or `DIFFSOL_JAX_LIB`)."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from diffsol_jax import Model  # noqa: E402

T_EVAL = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
X0 = 0.1


def logistic_exact(r, t):
    return X0 * jnp.exp(r * t) / (1.0 - X0 + X0 * jnp.exp(r * t))


def logistic(ad="adjoint", **kw):
    return Model(
        rhs=lambda x, p, t: p[0] * x * (1.0 - x / p[1]),
        init=lambda p, t: jnp.array([X0]),
        nstates=1,
        nparams=2,
        rtol=1e-10,
        atol=1e-10,
        ad=ad,
        **kw,
    )


def robertson(**kw):
    def rhs(x, p, t):
        k1, k2, k3 = p
        return jnp.array(
            [
                -k1 * x[0] + k3 * x[1] * x[2],
                k1 * x[0] - k2 * x[1] ** 2 - k3 * x[1] * x[2],
                k2 * x[1] ** 2,
            ]
        )

    return Model(
        rhs=rhs,
        init=lambda p, t: jnp.array([1.0, 0.0, 0.0]),
        nstates=3,
        nparams=3,
        rtol=1e-8,
        atol=1e-10,
        **kw,
    )


ROBERTSON_P = jnp.array([0.04, 3e7, 1e4])
# the classic reference solution at t = 40
ROBERTSON_40 = np.array([0.7158270687193, 9.185534764529e-6, 0.2841637457471])


def test_logistic_solve():
    model = logistic()
    ys = model.solve(jnp.array([1.0, 1.0]), T_EVAL)
    assert ys.shape == (5, 1)
    np.testing.assert_allclose(ys[:, 0], logistic_exact(1.0, T_EVAL), rtol=1e-7)


def test_logistic_out():
    model = Model(
        rhs=lambda x, p, t: p[0] * x * (1.0 - x),
        init=lambda p, t: jnp.array([X0]),
        out=lambda x, p, t: jnp.array([2.0 * x[0], x[0] ** 2]),
        nout=2,
        nstates=1,
        nparams=1,
        rtol=1e-10,
        atol=1e-10,
    )
    ys = model.solve(jnp.array([1.0]), T_EVAL)
    x = logistic_exact(1.0, T_EVAL)
    np.testing.assert_allclose(ys, jnp.stack([2 * x, x**2], axis=-1), rtol=1e-7)


@pytest.mark.parametrize("solver", ["bdf", "esdirk34", "tr_bdf2"])
def test_robertson(solver):
    model = robertson(ode_solver=solver)
    ys = jax.jit(model.solve)(ROBERTSON_P, jnp.array([0.0, 40.0]))
    np.testing.assert_allclose(ys[-1], ROBERTSON_40, rtol=1e-4)


def test_vmap_is_one_solve_and_matches_a_loop():
    model = logistic()
    params = jnp.array([[0.5, 1.0], [1.0, 1.0], [2.0, 1.5]])
    batched = jax.jit(jax.vmap(model.solve, in_axes=(0, None)))
    ys = batched(params, T_EVAL)
    loop = jnp.stack([model.solve(p, T_EVAL) for p in params])
    np.testing.assert_allclose(ys, loop, rtol=1e-8)
    hlo = batched.lower(params, T_EVAL).as_text()
    assert hlo.count("diffsol_solve_dense") == 1


def loss(model, p):
    return jnp.sum(model.solve(p, T_EVAL) ** 2)


def finite_difference(f, p, h=1e-6):
    return jnp.array(
        [(f(p.at[i].add(h)) - f(p.at[i].add(-h))) / (2 * h) for i in range(p.shape[0])]
    )


@pytest.mark.parametrize("ad", ["adjoint", "forward"])
def test_grad(ad):
    model = logistic(ad)
    p = jnp.array([1.2, 1.5])
    grad = jax.jit(jax.grad(lambda p: loss(model, p)))(p)
    np.testing.assert_allclose(grad, finite_difference(lambda p: loss(model, p), p), rtol=1e-5)


def test_forward_and_adjoint_jacobians_agree():
    p = jnp.array([1.2, 1.5])
    fwd = jax.jacfwd(lambda p: logistic("forward").solve(p, T_EVAL))(p)
    # jacrev maps over the cotangent: one backward solve with nout * nt directions
    model = logistic("adjoint")
    rev = jax.jit(jax.jacrev(lambda p: model.solve(p, T_EVAL)))(p)
    np.testing.assert_allclose(rev, fwd, rtol=1e-6, atol=1e-10)


def test_vmap_of_grad():
    model = logistic()
    params = jnp.array([[0.5, 1.0], [1.0, 1.0], [2.0, 1.5]])
    grads = jax.jit(jax.vmap(jax.grad(lambda p: loss(model, p))))(params)
    loop = jnp.stack([jax.grad(lambda p: loss(model, p))(p) for p in params])
    # a batched solve takes one step sequence for all the lanes, and the backward pass uses the
    # default adjoint tolerances: the lanes agree to the solver accuracy, not to round-off
    np.testing.assert_allclose(grads, loop, rtol=1e-6)


def test_robertson_grad():
    model = robertson()
    t = jnp.array([0.0, 1.0, 10.0, 40.0])
    f = lambda logp: jnp.sum(model.solve(jnp.exp(logp), t)[:, 0])  # noqa: E731
    logp = jnp.log(ROBERTSON_P)
    grad = jax.jit(jax.grad(f))(logp)
    np.testing.assert_allclose(grad, finite_difference(f, logp, 1e-5), rtol=1e-3)


def test_vjp_function_can_run_two_times():
    model = logistic()
    ys, vjp = jax.vjp(lambda p: model.solve(p, T_EVAL), jnp.array([1.0, 1.0]))
    (g1,) = vjp(jnp.ones_like(ys))
    (g2,) = vjp(jnp.ones_like(ys))
    np.testing.assert_allclose(g1, g2)


def test_evicted_checkpoint_is_recomputed():
    model = logistic(max_checkpoints=1)
    p = jnp.array([1.0, 1.0])
    ys, vjp = jax.vjp(lambda p: model.solve(p, T_EVAL), p)
    (expected,) = vjp(jnp.ones_like(ys))
    assert model.num_recomputes() == 0
    # a second forward solve frees the first checkpoint
    jax.vjp(lambda p: model.solve(p, T_EVAL), jnp.array([2.0, 1.0]))
    (grad,) = vjp(jnp.ones_like(ys))
    assert model.num_recomputes() == 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_grad_of_scan_with_more_solves_than_checkpoints():
    """`jax.grad` of a scan runs all the forward solves before the backward solves, so most of
    the checkpoints are freed; the backward solves run the forward solves again."""
    model = logistic(max_checkpoints=2)

    def f(p):
        def body(acc, r):
            return acc + jnp.sum(model.solve(p * r, T_EVAL) ** 2), None

        return jax.lax.scan(body, 0.0, jnp.linspace(0.8, 1.2, 6))[0]

    p = jnp.array([1.0, 1.5])
    grad = jax.jit(jax.grad(f))(p)
    np.testing.assert_allclose(grad, finite_difference(f, p), rtol=1e-5)
    assert model.num_recomputes() > 0


def _bkwd(model, handle, params, ct):
    """Call the backward primitive directly, as a wrong residual would reach it."""
    from diffsol_jax import _primitives

    kw = dict(model_id=model.id, rtol=model.rtol, atol=model.atol)
    return _primitives.solve_vjp_bkwd_p.bind(handle, params, T_EVAL, ct, **kw)


def _fwd(model, params):
    from diffsol_jax import _primitives

    kw = dict(model_id=model.id, rtol=model.rtol, atol=model.atol)
    return _primitives.solve_vjp_fwd_p.bind(params, T_EVAL, **kw)


def test_handle_with_other_params_is_not_used():
    model = logistic()
    p1, p2 = jnp.array([[1.0, 1.0]]), jnp.array([[2.0, 1.5]])
    ys, h1 = _fwd(model, p1)
    ct = jnp.ones_like(ys)
    expected = _bkwd(model, _fwd(model, p2)[1], p2, ct)
    before = model.num_recomputes()
    grad = _bkwd(model, h1, p2, ct)  # the handle of p1 with the params p2
    assert model.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_mixed_handles_are_not_used():
    model = logistic()
    p = jnp.array([[1.0, 1.0], [2.0, 1.5]])
    ys, h = _fwd(model, p)
    ct = jnp.ones_like(ys)
    expected = _bkwd(model, h, p, ct)
    # lane 1 has the id of a different forward solve
    _, other = _fwd(model, p[1:])
    before = model.num_recomputes()
    grad = _bkwd(model, h.at[1].set(other[0]), p, ct)
    assert model.num_recomputes() == before + 1
    np.testing.assert_allclose(grad, expected, rtol=1e-12)
