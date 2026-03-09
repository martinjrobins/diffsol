use crate::{
    ode_solver::problem::OdeSolverSolution, scalar::scale, ConstantOp, Matrix, MatrixHost,
    OdeBuilder, OdeEquations, OdeEquationsImplicit, OdeEquationsImplicitAdjoint,
    OdeEquationsImplicitSens, OdeSolverProblem, Op, Vector,
};
use nalgebra::ComplexField;
use num_traits::{FromPrimitive, One, Zero};
use std::ops::MulAssign;

// exponential decay problem
// dy/dt = -ay (p = [a, y0])
fn exponential_decay<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-p.get_index(0)));
}

// mass matrix is identity
fn exponential_decay_mass<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, beta: M::T, y: &mut M::V) {
    y.axpy(M::T::one(), x, beta);
}

// df/dp v = -yv (p = [a, y0])
// df/dp = | -y  0 |
//         | -y  0 |
// df/dp v = | -y  0 | |v_1| = |-yv_1|
//           | -y  0 | |v_2|   |-yv_1 |
fn exponential_decay_sens<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-v[0]));
}

// df/dp^T v = | -y  -y | |v_1| = |-yv_1 - yv_2|
//             |  0   0 | |v_2|   |  0  |
fn exponential_decay_sens_transpose<M: MatrixHost>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = x[0] * v[0] + x[1] * v[1];
    y[1] = M::T::zero();
}

// J = | -a  0 |
//     | 0  -a |
// Jv = -av
fn exponential_decay_jacobian<M: Matrix>(_x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.mul_assign(scale(-p.get_index(0)));
}

// -J^Tv = av
fn exponential_decay_jacobian_adjoint<M: MatrixHost>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.copy_from(v);
    y.mul_assign(scale(p[0]));
}

fn exponential_decay_init<M: Matrix>(p: &M::V, _t: M::T, y: &mut M::V) {
    y.fill(p.get_index(1));
}

// dy0/dp = | 0 1 |
//          | 0 1 |
// dy0/dp v = | 0 1 | |v_1| = |v_2|
//            | 0 1 | |v_2|   |v_2|
fn exponential_decay_init_sens<M: MatrixHost>(_p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y[0] = v[1];
    y[1] = v[1];
}

// dy0/dp^T v = | 0 0 | |v_1| = |0 |
//              | 1 1 | |v_2|   |v_1 + v_2|
fn exponential_decay_init_sens_adjoint<M: MatrixHost>(_p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y[0] = M::T::zero();
    y[1] = -v[0] - v[1];
}

fn exponential_decay_root<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = x[0] - M::T::from_f64(0.6).unwrap();
}

/// g_1 = 1 * x_1  +  2 * x_2
/// g_2 = 3 * x_1  +  4 * x_2
fn exponential_decay_out<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = M::T::one() * x[0] + M::T::from_f64(2.0).unwrap() * x[1];
    y[1] = M::T::from_f64(3.0).unwrap() * x[0] + M::T::from_f64(4.0).unwrap() * x[1];
}

/// J = |1 2|
///    |3 4|
/// J v = |1 2| |v_1| = |v_1 + 2v_2|
///       |3 4| |v_2|   |3v_1 + 4v_2|
fn exponential_decay_out_jac_mul<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = v[0] + M::T::from_f64(2.0).unwrap() * v[1];
    y[1] = M::T::from_f64(3.0).unwrap() * v[0] + M::T::from_f64(4.0).unwrap() * v[1];
}

/// J = |1 2|
///    |3 4|
/// -J^T v = |-1 -3| |v_1| = |-v_1 - 3v_2|
///         |-2 -4| |v_2|   |-2v_1 - 4v_2|
fn exponential_decay_out_adj_mul<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -v[0] - M::T::from_f64(3.0).unwrap() * v[1];
    y[1] = -M::T::from_f64(2.0).unwrap() * v[0] - M::T::from_f64(4.0).unwrap() * v[1];
}

/// J = |0 0|
///     |0 0|
fn exponential_decay_out_sens<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

/// J = |0 0|
///     |0 0|
fn exponential_decay_out_sens_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

#[allow(clippy::type_complexity)]
pub fn negative_exponential_decay_problem<M: MatrixHost + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let h = -1.0;
    let k = 0.1;
    let y0 = (-10.0 * k).exp();
    let problem = OdeBuilder::<M>::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>, 2)
        .build()
        .unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    let mut soln = OdeSolverSolution {
        negative_time: true,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from_f64(-i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[cfg(feature = "diffsl")]
pub fn exponential_decay_problem_diffsl<
    M: MatrixHost<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
>(
    prep_adjoint: bool,
) -> (
    OdeSolverProblem<crate::DiffSl<M, CG>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let out = if prep_adjoint {
        "1 * x  +  2 * y, 3 * x  +  4 * y,"
    } else {
        "u_i"
    };
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .integrate_out(prep_adjoint)
        .build_from_diffsl(
            format!(
                "
            in_i {{ k = 0.1, y0 = 1.0 }}
            u_i {{ x = y0, y = y0 }}
            F_i {{ -k * u_i }}
            out_i {{
                {out}
            }}"
            )
            .as_str(),
        )
        .unwrap();
    let p = [k, y0];
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    for i in 0..10 {
        let t = i as f64;
        let y0 = problem.eqn.init().call(0.0);
        let y = y0.clone() * scale((-p[0] * t).exp());
        let ypk = y0 * scale(-t * M::T::exp(-p[0] * t));
        let ypy0 = M::V::from_vec(
            vec![(-p[0] * t).exp(), (-p[0] * t).exp()],
            y.context().clone(),
        );
        soln.push_sens(y, t, &[ypk, ypy0]);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let h = 1.0;
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>, 2)
        .build()
        .unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0.clone() * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_with_mass<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let h = 1.0;
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .mass(exponential_decay_mass::<M>)
        .init(exponential_decay_init::<M>, 2)
        .build()
        .unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_with_root<M: MatrixHost + 'static>(
    use_coloring: bool,
    integrate_out: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let out_fn = |_x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V| {
        y.copy_from(_x);
    };
    let out_jac_fn = |_x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
        y.copy_from(v);
    };
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>, 2)
        .root(exponential_decay_root::<M>, 1)
        .out_implicit(out_fn, out_jac_fn, 2)
        .integrate_out(integrate_out)
        .build()
        .unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_adjoint<M: MatrixHost>(
    integrate_out: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .integrate_out(integrate_out)
        .rhs_adjoint_implicit(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_jacobian_adjoint::<M>,
            exponential_decay_sens_transpose::<M>,
        )
        .init_adjoint(
            exponential_decay_init::<M>,
            exponential_decay_init_sens_adjoint::<M>,
            2,
        )
        .out_adjoint_implicit(
            exponential_decay_out::<M>,
            exponential_decay_out_jac_mul::<M>,
            exponential_decay_out_adj_mul::<M>,
            exponential_decay_out_sens_adj::<M>,
            2,
        )
        .build()
        .unwrap();
    let ctx = problem.eqn.context();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    let t0 = M::T::zero();
    let t1 = M::T::from_f64(9.0).unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let g = y0.clone() * scale((M::T::exp(-p[0] * t0) - M::T::exp(-p[0] * t)) / p[0]);
        let g = M::V::from_vec(
            vec![
                g[0] + M::T::from_f64(2.0).unwrap() * g[1],
                M::T::from_f64(3.0).unwrap() * g[0] + M::T::from_f64(4.0).unwrap() * g[1],
            ],
            ctx.clone(),
        );
        let dydk = y0.clone()
            * scale(
                M::T::exp(-p[0] * (t1 + t0))
                    * (M::T::exp(t0 * p[0]) * (p[0] * t1 + M::T::one())
                        - M::T::exp(t1 * p[0]) * (p[0] * t0 + M::T::one()))
                    / (p[0] * p[0]),
            );
        let dydy0 = (M::T::exp(-p[0] * t0) - M::T::exp(-p[0] * t1)) / p[0];
        let dg1dk = dydk[0] + M::T::from_f64(2.0).unwrap() * dydk[1];
        let dg2dk = M::T::from_f64(3.0).unwrap() * dydk[0] + M::T::from_f64(4.0).unwrap() * dydk[1];
        let dg1dy0 = dydy0 + M::T::from_f64(2.0).unwrap() * dydy0;
        let dg2dy0 = M::T::from_f64(3.0).unwrap() * dydy0 + M::T::from_f64(4.0).unwrap() * dydy0;
        let dg1 = M::V::from_vec(vec![dg1dk, dg1dy0], ctx.clone());
        let dg2 = M::V::from_vec(vec![dg2dk, dg2dy0], ctx.clone());
        soln.push_sens(g, t, &[dg1, dg2]);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_sens<M: MatrixHost + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6])
        .use_coloring(use_coloring)
        .rhs_sens_implicit(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_sens::<M>,
        )
        .init_sens(
            exponential_decay_init::<M>,
            exponential_decay_init_sens::<M>,
            2,
        )
        .build()
        .unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0.clone() * scale(M::T::exp(-p[0] * t));
        let ypk = y0 * scale(-t * M::T::exp(-p[0] * t));
        let ypy0 = M::V::from_vec(
            vec![(-p[0] * t).exp(), (-p[0] * t).exp()],
            y.context().clone(),
        );
        soln.push_sens(y, t, &[ypk, ypy0]);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_sens_with_out<M: MatrixHost + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6])
        .use_coloring(use_coloring)
        .rhs_sens_implicit(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_sens::<M>,
        )
        .init_sens(
            exponential_decay_init::<M>,
            exponential_decay_init_sens::<M>,
            2,
        )
        // g_1 = 1 * x_1  +  2 * x_2
        // g_2 = 3 * x_1  +  4 * x_2
        .out_sens_implicit(
            exponential_decay_out::<M>,
            exponential_decay_out_jac_mul::<M>,
            exponential_decay_out_sens::<M>,
            2,
        )
        .build()
        .unwrap();
    let p = [M::T::from_f64(k).unwrap(), M::T::from_f64(y0).unwrap()];
    let mut soln = OdeSolverSolution::default();

    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0.clone() * scale(M::T::exp(-p[0] * t));
        let y_out = M::V::from_vec(
            vec![
                M::T::one() * y[0] + M::T::from_f64(2.0).unwrap() * y[1],
                M::T::from_f64(3.0).unwrap() * y[0] + M::T::from_f64(4.0).unwrap() * y[1],
            ],
            y.context().clone(),
        );
        let ypk = y0 * scale(-t * M::T::exp(-p[0] * t));
        let ypk_out = M::V::from_vec(
            vec![
                M::T::one() * ypk[0] + M::T::from_f64(2.0).unwrap() * ypk[1],
                M::T::from_f64(3.0).unwrap() * ypk[0] + M::T::from_f64(4.0).unwrap() * ypk[1],
            ],
            y.context().clone(),
        );
        let ypy0 = M::V::from_vec(
            vec![(-p[0] * t).exp(), (-p[0] * t).exp()],
            y.context().clone(),
        );
        let ypy0_out = M::V::from_vec(
            vec![
                M::T::one() * ypy0[0] + M::T::from_f64(2.0).unwrap() * ypy0[1],
                M::T::from_f64(3.0).unwrap() * ypy0[0] + M::T::from_f64(4.0).unwrap() * ypy0[1],
            ],
            y.context().clone(),
        );
        soln.push_sens(y_out, t, &[ypk_out, ypy0_out]);
    }
    (problem, soln)
}

// ------------------------------------------------------------------
// Exponential-decay problem extended with a Reset feature
//
// Reuses the standard exponential_decay / exponential_decay_jacobian /
// exponential_decay_init functions (p = [k, y0], nstates = 2).
//
// Root 0: y[0] - 0.6  (reset trigger, fires at t ≈ 5.108)
// Root 1: y[0] - 0.3  (stop condition, fires after reset at t ≈ 17.148)
// Reset:  y → [y0, y0] = [1.0, 1.0]
// ------------------------------------------------------------------

fn exponential_decay_two_root<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    // index 0: reset trigger (y[0] = 0.6)
    y.set_index(0, x.get_index(0) - M::T::from_f64(0.6).unwrap());
    // index 1: stopping condition (y[0] = 0.3)
    y.set_index(1, x.get_index(0) - M::T::from_f64(0.3).unwrap());
}

fn exponential_decay_reset<M: Matrix>(_x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    // Reset state to 0.4 — below root-0 threshold (0.6) so root 0 won't fire
    // again, but above root-1 threshold (0.3) so root 1 can still fire.
    y.fill(M::T::from_f64(0.4).unwrap());
}

/// Exponential decay problem with a Root function (2 outputs) **and** a Reset
/// (triggered by root index 0).
///
/// Uses the standard `dy/dt = -k*y` equations with `k=0.1`, `y(0)=[1,1]`.
/// Root 0 fires at t ≈ 5.108 and triggers a reset (y → [0.4, 0.4]).
/// After the reset, y[0] = 0.4 < 0.6 so root 0 cannot fire again.
/// Root 1 fires at t ≈ 5.108 + ln(4/3)/0.1 ≈ 7.985 and causes `solve()` to stop.
///
/// Returns the problem alongside a three-point `OdeSolverSolution`:
///   1. pre-reset  state `y = [0.6, 0.6]` at t ≈ 5.108
///   2. post-reset state `y = [0.4, 0.4]` at t ≈ 5.108
///   3. stop state `y = [0.3, 0.3]` at t ≈ 7.985
#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_reset_problem<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1, 1.0])
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>, 2)
        .root(exponential_decay_two_root::<M>, 2)
        .reset(exponential_decay_reset::<M>)
        .build()
        .unwrap();
    // Root 0 fires when y[0] = 0.6: t_root_0 = -ln(0.6)/0.1
    let t_root_0 = M::T::from_f64(0.6_f64.ln().abs() / 0.1).unwrap();
    // After reset to y=[0.4,0.4], root 1 fires when y[0]=0.3:
    //   t_from_reset = -ln(0.3/0.4)/0.1 = ln(4/3)/0.1
    let t_from_reset = M::T::from_f64((4.0_f64 / 3.0_f64).ln() / 0.1).unwrap();
    let t_stop = t_root_0 + t_from_reset;
    let y0: M::V = problem.eqn.init().call(M::T::zero());
    let reset_val = M::T::from_f64(0.4).unwrap();
    let ctx = y0.context().clone();
    let nstates = y0.len();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    // Point 1: state just before reset (y[0] decayed to 0.6)
    soln.push(y0.clone() * scale(M::T::from_f64(0.6).unwrap()), t_root_0);
    // Point 2: state just after reset (y → [0.4, 0.4])
    let mut y_reset = M::V::zeros(nstates, ctx);
    y_reset.fill(reset_val);
    soln.push(y_reset, t_root_0);
    // Point 3: state at stop (y[0] decayed from 0.4 to 0.3 after reset)
    // y_stop = 0.4 * exp(-0.1 * t_from_reset) = 0.4 * (3/4) = 0.3
    soln.push(
        M::V::from_element(nstates, M::T::from_f64(0.3).unwrap(), y0.context().clone()),
        t_stop,
    );
    (problem, soln)
}

/// Same equations but **without** Reset — used to test that `step()` returns
/// `RootFound(t, index)` with the correct root index.
/// Root 0 fires first at t ≈ 5.108.
///
/// Returns the problem alongside a one-point `OdeSolverSolution` whose single
/// entry is the expected root-0 state `y = [0.6, 0.6]` at t ≈ 5.108.
#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_two_roots_problem<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1, 1.0])
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>, 2)
        .root(exponential_decay_two_root::<M>, 2)
        .build()
        .unwrap();
    // Root 0 fires when y[0] = 0.6: t = -ln(0.6)/0.1
    let t_root_0 = M::T::from_f64(0.6_f64.ln().abs() / 0.1).unwrap();
    let y0: M::V = problem.eqn.init().call(M::T::zero());
    let y_root_0 = y0 * scale(M::T::from_f64(0.6).unwrap());
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    soln.push(y_root_0, t_root_0);
    (problem, soln)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_exponential_decay_diffsl_llvm() {
        use super::*;
        use crate::{
            matrix::dense_nalgebra_serial::NalgebraMat, ConstantOpSens, ConstantOpSensAdjoint,
            NalgebraVec, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
            NonLinearOpSensAdjoint,
        };
        let (problem, _soln) =
            exponential_decay_problem_diffsl::<NalgebraMat<f64>, crate::LlvmModule>(true);
        let ctx = problem.eqn.context();
        let x = NalgebraVec::from_vec(vec![1.0, 2.0], *ctx);
        let t = 0.0;
        let v = NalgebraVec::from_vec(vec![2.0, 3.0], *ctx);
        let p = NalgebraVec::from_vec(vec![0.1, 1.0], *ctx);

        // check the adjoint jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_jacobian_adjoint::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .rhs()
                .jac_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the sens jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_sens::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem.eqn().rhs().sens_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the sens adjoint jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_sens_transpose::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .rhs()
                .sens_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the set_u0 sens adjoint jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_init_sens_adjoint::<NalgebraMat<f64>>(&p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .init()
                .sens_transpose_mul_inplace(t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the set_u0 sens jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_init_sens::<NalgebraMat<f64>>(&p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem.eqn().init().sens_mul_inplace(t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_out_jac_mul::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .jac_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out adjoint jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_out_adj_mul::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .jac_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out sens adjoint jacobian
        let mut y_check = NalgebraVec::zeros(2, *ctx);
        exponential_decay_out_sens_adj::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(2, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .sens_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }
    }
}
