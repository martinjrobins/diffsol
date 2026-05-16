use crate::{
    ode_solver::problem::OdeSolverSolution,
    scalar::{scale, Scalar},
    ConstantOp, Matrix, MatrixHost, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
    NonLinearOpSensAdjoint, NonLinearOpTimePartial, OdeBuilder, OdeEquations, OdeEquationsImplicit,
    OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverProblem, Op, Vector,
};
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
        let y = y0 * scale((-p[0] * t).exp());
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
        let ypk = y0 * scale(-t * (-p[0] * t).exp());
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
        let y = y0.clone() * scale((-p[0] * t).exp());
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
        let y = y0 * scale((-p[0] * t).exp());
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
        let y = y0 * scale((-p[0] * t).exp());
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
        let g = y0.clone() * scale(((-p[0] * t0).exp() - (-p[0] * t).exp()) / p[0]);
        let g = M::V::from_vec(
            vec![
                g[0] + M::T::from_f64(2.0).unwrap() * g[1],
                M::T::from_f64(3.0).unwrap() * g[0] + M::T::from_f64(4.0).unwrap() * g[1],
            ],
            ctx.clone(),
        );
        let dydk = y0.clone()
            * scale(
                (-(p[0] * (t1 + t0))).exp()
                    * ((t0 * p[0]).exp() * (p[0] * t1 + M::T::one())
                        - (t1 * p[0]).exp() * (p[0] * t0 + M::T::one()))
                    / (p[0] * p[0]),
            );
        let dydy0 = ((-p[0] * t0).exp() - (-p[0] * t1).exp()) / p[0];
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

fn exponential_decay_fixed_init<M: Matrix>(_p: &M::V, _t: M::T, y: &mut M::V) {
    y.fill(M::T::one());
}

fn exponential_decay_fixed_init_sens_adjoint<M: MatrixHost>(
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

fn exponential_decay_root_single_half<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = x[0] - M::T::from_f64(0.5).unwrap();
}

fn exponential_decay_root_single_half_jac<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = v[0];
}

fn exponential_decay_root_single_half_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -v[0];
    y[1] = M::T::zero();
}

fn exponential_decay_root_single_half_sens_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

fn exponential_decay_reset_y_plus_2r<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    let increment = M::T::from_f64(2.0).unwrap() * p.get_index(1);
    for i in 0..x.len() {
        y.set_index(i, x.get_index(i) + increment);
    }
}

fn exponential_decay_reset_y_plus_2r_jac<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.copy_from(v);
}

fn exponential_decay_reset_y_plus_2r_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.copy_from(v);
    y.mul_assign(scale(-M::T::one()));
}

fn exponential_decay_reset_y_plus_2r_sens_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = M::T::zero();
    y[1] = -(M::T::from_f64(2.0).unwrap() * v[0] + M::T::from_f64(2.0).unwrap() * v[1]);
}

fn exponential_decay_out_first_state<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = x[0];
}

fn exponential_decay_out_first_state_jac<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = v[0];
}

fn exponential_decay_out_first_state_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -v[0];
    y[1] = M::T::zero();
}

fn exponential_decay_out_first_state_sens_adj<M: MatrixHost>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_single_reset_root_problem_adjoint<M: MatrixHost + 'static>(
    integrate_out: bool,
) -> (
    OdeSolverProblem<
        impl OdeEquationsImplicitAdjoint<
            M = M,
            V = M::V,
            T = M::T,
            C = M::C,
            Reset: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpAdjoint<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpSensAdjoint<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpTimePartial<M = M, V = M::V, T = M::T, C = M::C>,
            Root: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpAdjoint<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpSensAdjoint<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpTimePartial<M = M, V = M::V, T = M::T, C = M::C>,
        >,
    >,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let r = 0.25;
    let problem = OdeBuilder::<M>::new()
        .p([k, r])
        .integrate_out(integrate_out)
        .rhs_adjoint_implicit(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_jacobian_adjoint::<M>,
            exponential_decay_sens_transpose::<M>,
        )
        .init_adjoint(
            exponential_decay_fixed_init::<M>,
            exponential_decay_fixed_init_sens_adjoint::<M>,
            2,
        )
        .out_adjoint_implicit(
            exponential_decay_out_first_state::<M>,
            exponential_decay_out_first_state_jac::<M>,
            exponential_decay_out_first_state_adj::<M>,
            exponential_decay_out_first_state_sens_adj::<M>,
            1,
        )
        .root_adjoint_implicit(
            exponential_decay_root_single_half::<M>,
            exponential_decay_root_single_half_jac::<M>,
            exponential_decay_root_single_half_adj::<M>,
            exponential_decay_root_single_half_sens_adj::<M>,
            1,
        )
        .reset_adjoint_implicit(
            exponential_decay_reset_y_plus_2r::<M>,
            exponential_decay_reset_y_plus_2r_jac::<M>,
            exponential_decay_reset_y_plus_2r_adj::<M>,
            exponential_decay_reset_y_plus_2r_sens_adj::<M>,
        )
        .build()
        .unwrap();

    let ctx = problem.eqn.context();
    let t_root = M::T::from_f64(f64::ln(2.0) / k).unwrap();
    let t_stop = t_root + t_root;
    let g = M::T::from_f64((0.5 + 2.0 * r) / k).unwrap();
    let dgdk = M::T::from_f64(-(0.5 + 2.0 * r) / (k * k)).unwrap();
    let dgdr = M::T::from_f64(2.0 / k).unwrap();

    let mut soln = OdeSolverSolution {
        atol: problem
            .out_atol
            .clone()
            .unwrap_or_else(|| problem.atol.clone()),
        rtol: problem.out_rtol.unwrap_or(problem.rtol),
        ..Default::default()
    };
    let g_state = M::V::from_vec(vec![g], ctx.clone());
    let dgdk_state = M::V::from_vec(vec![dgdk], ctx.clone());
    let dgdr_state = M::V::from_vec(vec![dgdr], ctx.clone());
    soln.push_sens(g_state, t_stop, &[dgdk_state, dgdr_state]);
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
        let y = y0.clone() * scale((-p[0] * t).exp());
        let ypk = y0 * scale(-t * (-p[0] * t).exp());
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
        let y = y0.clone() * scale((-p[0] * t).exp());
        let y_out = M::V::from_vec(
            vec![
                M::T::one() * y[0] + M::T::from_f64(2.0).unwrap() * y[1],
                M::T::from_f64(3.0).unwrap() * y[0] + M::T::from_f64(4.0).unwrap() * y[1],
            ],
            y.context().clone(),
        );
        let ypk = y0 * scale(-t * (-p[0] * t).exp());
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
// Exponential-decay problem with a Reset feature present
//
// Reuses the standard exponential_decay / exponential_decay_jacobian /
// exponential_decay_init functions (p = [k, y0], nstates = 2).
//
// Root 0: y[0] - 0.6  (first root, fires at t ≈ 5.108)
// Root 1: y[0] - 0.3  (later root, reached at t = 7.985 if solve is continued after reset)
// Reset:  y → [0.4, 0.4]
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

/// Exponential decay problem with a Root function (2 outputs) and a Reset.
///
/// Uses the standard `dy/dt = -k*y` equations with `k=0.1`, `y(0)=[1,1]`.
/// Root 0 fires first at t ≈ 5.108.
/// If reset is applied manually (`y -> [0.4, 0.4]`), root 1 then fires at
/// t ≈ 5.108 + ln(4/3)/0.1 ≈ 7.985.
///
/// Returns the problem alongside a one-point `OdeSolverSolution`:
///   1. second-root state `y = [0.3, 0.3]` at t ≈ 7.985 (for manual-reset continuation tests)
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
    let t_root_0 = M::T::from_f64(-(0.6_f64.ln()) / 0.1).unwrap();
    // After manual reset to y=[0.4, 0.4], root 1 fires when y[0]=0.3:
    //   t_from_reset = -ln(0.3/0.4)/0.1 = ln(4/3)/0.1
    let t_from_reset = M::T::from_f64((4.0_f64 / 3.0_f64).ln() / 0.1).unwrap();
    let t_stop = t_root_0 + t_from_reset;
    let nstates = problem.eqn.nstates();
    let ctx = problem.context().clone();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    // Only point: second-root stop state after manual reset.
    soln.push(
        M::V::from_element(nstates, M::T::from_f64(0.3).unwrap(), ctx),
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

// ------------------------------------------------------------------
// Exponential-decay problem with Reset (R(y) = y + 2) and forward
// sensitivity equations. Public dense sensitivity solves apply the reset
// automatically; staged sensitivity solves still expose root events.
//
// Root 0: y[0] - 0.6  (first root, fires at t_root = 10·ln(5/3) ≈ 5.108)
// Root 1: y[0] - 2.0  (later root, not reached because solve halts at root 0)
// Reset:  y → y + 2   (component-wise; y_new = [2.6, 2.6])
// ------------------------------------------------------------------

pub(crate) fn exponential_decay_reset_y_plus_2<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    y: &mut M::V,
) {
    // R(y) = y + 2  (component-wise)
    for i in 0..x.len() {
        y.set_index(i, x.get_index(i) + M::T::from_f64(2.0).unwrap());
    }
}

pub(crate) fn exponential_decay_reset_y_plus_2_jac<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    // J_R = I  →  J_R · v = v
    y.copy_from(v);
}

pub(crate) fn exponential_decay_root_0_6_and_2_0<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    y: &mut M::V,
) {
    // Root 0: y[0] = 0.6  (reset trigger)
    y.set_index(0, x.get_index(0) - M::T::from_f64(0.6).unwrap());
    // Root 1: y[0] = 2.0  (stop; fires after reset when y decays from 2.6 to 2.0)
    y.set_index(1, x.get_index(0) - M::T::from_f64(2.0).unwrap());
}

pub(crate) fn exponential_decay_root_0_6_and_2_0_jac<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.set_index(0, v.get_index(0));
    y.set_index(1, v.get_index(0));
}

pub(crate) fn exponential_decay_root_0_6_and_2_0_sens<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

pub(crate) fn exponential_decay_reset_y_plus_2_sens<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

/// Exponential decay with a root function, reset map, and forward sensitivity equations.
///
/// `dy/dt = -k·y`, `y(0) = [y0, y0]`, `p = [k, y0] = [0.1, 1.0]`, 2 states, 2 params.
///
/// At t_root ≈ 5.108 root 0 fires. If reset is applied (`y -> y + 2`), root 1
/// then fires at t_stop ≈ 7.732.
///
/// Returns the problem with a single solution point at t_stop containing the exact y and
/// sensitivity vectors for comparison in reset-continuation tests.
#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_reset_problem_sens<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<
        impl OdeEquationsImplicitSens<
            M = M,
            V = M::V,
            T = M::T,
            C = M::C,
            Reset: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpSens<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpTimePartial<M = M, V = M::V, T = M::T, C = M::C>,
            Root: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpSens<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpTimePartial<M = M, V = M::V, T = M::T, C = M::C>,
        >,
    >,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1, 1.0])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6])
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
        .root_sens_implicit(
            exponential_decay_root_0_6_and_2_0::<M>,
            exponential_decay_root_0_6_and_2_0_jac::<M>,
            exponential_decay_root_0_6_and_2_0_sens::<M>,
            2,
        )
        .reset_sens_implicit(
            exponential_decay_reset_y_plus_2::<M>,
            exponential_decay_reset_y_plus_2_jac::<M>,
            exponential_decay_reset_y_plus_2_sens::<M>,
        )
        .build()
        .unwrap();

    // t_root: y[0] = exp(-0.1·t) = 0.6  →  t_root = 10·ln(5/3)
    let t_root = M::T::from_f64(10.0 * (5.0_f64 / 3.0_f64).ln()).unwrap();
    // dt to stop after reset: 2.6·exp(-0.1·dt) = 2.0  →  dt = 10·ln(1.3)
    let dt = M::T::from_f64(10.0 * 1.3_f64.ln()).unwrap();
    let t_stop = t_root + dt;

    // y at t_stop: both components = 2.0 (by construction of root 1)
    let y0 = problem.eqn.init().call(M::T::zero());
    let ctx = y0.context().clone();
    let y_stop = M::V::from_element(2, M::T::from_f64(2.0).unwrap(), ctx.clone());

    // Exact fixed-time sensitivities at t_stop after a root-aware reset:
    // for t >= t_root, x(t) = 2.6 * exp(-k * (t - t_root(k, y0))).
    // Evaluated at the nominal fixed time t_stop:
    //   s_k(t_stop) = -2 * t_stop
    //   s_y0(t_stop) = 2 / y0 = 2
    let s_k_val = -M::T::from_f64(2.0).unwrap() * t_stop;
    let s_y0_val = M::T::from_f64(2.0).unwrap();

    let s_k = M::V::from_element(2, s_k_val, ctx.clone());
    let s_y0 = M::V::from_element(2, s_y0_val, ctx.clone());

    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    soln.push_sens(y_stop, t_stop, &[s_k, s_y0]);

    (problem, soln)
}

fn exponential_decay_constant_reset_jac<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

fn exponential_decay_constant_reset_sens<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

fn exponential_decay_two_root_jac<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.set_index(0, v.get_index(0));
    y.set_index(1, v.get_index(0));
}

fn exponential_decay_two_root_sens<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

/// Exponential decay with two roots (0.6 trigger, 0.3 stop), constant reset to 0.4,
/// and forward sensitivity equations.
#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_constant_reset_problem_sens<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<
        impl OdeEquationsImplicitSens<
            M = M,
            V = M::V,
            T = M::T,
            C = M::C,
            Reset: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpSens<M = M, V = M::V, T = M::T, C = M::C>
                       + NonLinearOpTimePartial<M = M, V = M::V, T = M::T, C = M::C>,
            Root: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpSens<M = M, V = M::V, T = M::T, C = M::C>
                      + NonLinearOpTimePartial<M = M, V = M::V, T = M::T, C = M::C>,
        >,
    >,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1, 1.0])
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
        .root_sens_implicit(
            exponential_decay_two_root::<M>,
            exponential_decay_two_root_jac::<M>,
            exponential_decay_two_root_sens::<M>,
            2,
        )
        .reset_sens_implicit(
            exponential_decay_reset::<M>,
            exponential_decay_constant_reset_jac::<M>,
            exponential_decay_constant_reset_sens::<M>,
        )
        .build()
        .unwrap();

    let nstates = problem.eqn.rhs().nstates();
    let ctx = problem.context().clone();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    soln.push(M::V::from_element(nstates, M::T::zero(), ctx), M::T::zero());

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
