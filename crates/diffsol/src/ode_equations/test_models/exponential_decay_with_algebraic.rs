use crate::{
    matrix::Matrix,
    ode_solver::problem::OdeSolverSolution,
    scalar::{scale, Scalar},
    ConstantOp, Context, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpTimePartial, OdeBuilder,
    OdeEquations, OdeEquationsImplicit, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens,
    OdeSolverProblem, Op, Vector,
};
use num_traits::{FromPrimitive, One, Zero};

// exponential decay problem with algebraic constraint
// dy/dt = -ay
// 0 = z - y
// remove warning about unused mut
// J = | -y.get_index(0) |
//     | -y.get_index(1) |
//     | 0    |
// Jv = | -y.get_index(0)v[0] |
//      | -y.get_index(1)v[1] |
//      | 0        |
fn exponential_decay_with_algebraic_sens<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([x, v], |y, [x, v], _| {
        for (y, x) in y.iter_mut().zip(x.iter()) {
            *y = -v[0] * *x;
        }
        *y.last_mut().unwrap() = M::T::zero();
    });
}

// -J^Tv = | y.get_index(0)v[0] + y.get_index(1)v[1] + 0 |
fn exponential_decay_with_algebraic_sens_adjoint<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([x, v], |y, [x, v], _| y[0] = x[0] * v[0] + x[1] * v[1]);
}

// J = | -a, 0, 0 |
//     | 0, -a, 0 |
//     | 0,  -1, 1 |
// Jv = | -av[0] |
//      | -av[1] |
//      | v[2] - v[1] |
// -J^T v = | av[0] |
//          | av[1] + v[2] |
//          |  -v[2]    |
fn exponential_decay_with_algebraic_adjoint<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([p, v], |y, [p, v], _| {
        y[0] = p[0] * v[0];
        y[1] = p[0] * v[1] + v[2];
        y[2] = -v[2];
    });
}

// y = Mx + beta * y = | 1 0 | | x[0] | + beta | y.get_index(0) |
//                     | 0 0 | | x[1] |         | y.get_index(1) |
// y = M^T x + beta * y = | 1 0 | | x[0] | + beta | y.get_index(0) |
//                        | 0 0 | | x[1] |         | y.get_index(1) |
fn exponential_decay_with_algebraic_mass_transpose<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    y.for_each_batch([x], |y, [x], _| {
        let n = y.len();
        for (y, x) in y.iter_mut().zip(x.iter()).take(n - 1) {
            *y = *x + beta * *y;
        }
        y[n - 1] *= beta;
    });
}

fn exponential_decay_with_algebraic_init<M: Matrix>(_p: &M::V, _t: M::T, y: &mut M::V) {
    y.for_each_batch([], |y, _, _| {
        y[0] = M::T::one();
        y[1] = M::T::one();
        y[2] = M::T::zero();
    });
}

fn exponential_decay_with_algebraic_init_sens<M: Matrix>(
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

fn exponential_decay_with_algebraic_init_sens_adjoint<M: Matrix>(
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

// out(x) = | a * x[2] |
fn exponential_decay_with_algebraic_out<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y.for_each_batch([x, p], |y, [x, p], _| y[0] = p[0] * x[2]);
}

// J = | 0 0 a |
// Jv = | a * v[2] |
fn exponential_decay_with_algebraic_out_jac_mul<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([p, v], |y, [p, v], _| y[0] = p[0] * v[2]);
}

// J = | 0 0 a |
// -J^T v = | 0 |
//          | 0 |
//          | -a * v[0] |
fn exponential_decay_with_algebraic_out_jac_adj_mul<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([p, v], |y, [p, v], _| {
        y.fill(M::T::zero());
        y[2] = -p[0] * v[0];
    });
}

// J = | x[2] |
// Jv = | x[2]v[0] |
//fn exponential_decay_with_algebraic_out_sens<M: Matrix>(
//    x: &M::V,
//    _p: &M::V,
//    _t: M::T,
//    v: &M::V,
//    y: &mut M::V,
//) {
//    y.get_index(0) = x[1] * v[1];
//}

// -J^T v = | -x[2]v[2] |
fn exponential_decay_with_algebraic_out_sens_adj<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([x, v], |y, [x, v], _| y[0] = -x[2] * v[0]);
}

pub fn exponential_decay_with_algebraic_batched<M: Matrix>(
    x: &M::V,
    p: &M::V,
    _t: M::T,
    y: &mut M::V,
) {
    y.for_each_batch([x, p], |y, [x, p], _| {
        let n = y.len();
        for (y, x) in y.iter_mut().zip(x.iter()) {
            *y = -p[0] * *x;
        }
        y[n - 1] = x[n - 1] - x[n - 2];
    });
}

pub fn exponential_decay_with_algebraic_jacobian_batched<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([p, v], |y, [p, v], _| {
        let n = y.len();
        for (y, v) in y.iter_mut().zip(v.iter()) {
            *y = -p[0] * *v;
        }
        y[n - 1] = v[n - 1] - v[n - 2];
    });
}

#[allow(dead_code)]
pub fn exponential_decay_with_algebraic_mass_batched<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    y.for_each_batch([x], |y, [x], _| {
        let n = y.len();
        for (y, x) in y.iter_mut().zip(x.iter()).take(n - 1) {
            *y = *x + beta * *y;
        }
        y[n - 1] *= beta;
    });
}

pub fn exponential_decay_with_algebraic_init_batched<M: Matrix>(_p: &M::V, _t: M::T, y: &mut M::V) {
    y.for_each_batch([], |y, _, _| y.fill(M::T::one()));
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1])
        .use_coloring(use_coloring)
        .rhs_implicit(
            exponential_decay_with_algebraic_batched::<M>,
            exponential_decay_with_algebraic_jacobian_batched::<M>,
        )
        .mass(exponential_decay_with_algebraic_mass_batched::<M>)
        .init(exponential_decay_with_algebraic_init::<M>, 3)
        .build()
        .unwrap();

    let ctx = problem.eqn.context();
    let p = M::V::from_vec(vec![M::T::from_f64(0.1).unwrap()], ctx.clone());
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 10.0).unwrap();
        let y0 = M::V::from_vec(vec![M::T::one(), M::T::one(), M::T::one()], ctx.clone());
        let y: M::V = y0 * scale((-p.get_index(0) * t).exp());
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_problem_batched<M: Matrix + 'static>(
    nbatch: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let ctx = M::C::default().clone_with_nbatch(nbatch).unwrap();
    let mut p_f64 = Vec::with_capacity(nbatch);
    for b in 0..nbatch {
        p_f64.push(0.1 * (b + 1) as f64);
    }
    let problem = OdeBuilder::<M>::new()
        .context(ctx.clone())
        .p(p_f64.clone())
        .rhs_implicit(
            exponential_decay_with_algebraic_batched::<M>,
            exponential_decay_with_algebraic_jacobian_batched::<M>,
        )
        .mass(exponential_decay_with_algebraic_mass_batched::<M>)
        .init(exponential_decay_with_algebraic_init_batched::<M>, 3)
        .build()
        .unwrap();
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 10.0).unwrap();
        let mut y_data = Vec::with_capacity(3 * nbatch);
        for &k in p_f64.iter().take(nbatch) {
            let k = M::T::from_f64(k).unwrap();
            let val = (-k * t).exp();
            y_data.push(val);
            y_data.push(val);
            y_data.push(val);
        }
        let y = M::V::from_vec(y_data, ctx.clone());
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_adjoint_problem<M: Matrix + 'static>(
    integrate_out: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let a = 0.1;
    let nout = 1;
    let problem = {
        let mut builder = OdeBuilder::<M>::new()
            .p([a])
            .integrate_out(integrate_out)
            .sens_rtol(1e-6)
            .sens_atol([1e-6, 1e-6, 1e-6])
            .param_rtol(1e-6)
            .param_atol([1e-6])
            .rhs_adjoint_implicit(
                exponential_decay_with_algebraic_batched::<M>,
                exponential_decay_with_algebraic_jacobian_batched::<M>,
                exponential_decay_with_algebraic_adjoint::<M>,
                exponential_decay_with_algebraic_sens_adjoint::<M>,
            )
            .init_adjoint(
                exponential_decay_with_algebraic_init::<M>,
                exponential_decay_with_algebraic_init_sens_adjoint::<M>,
                3,
            )
            .mass_adjoint(
                exponential_decay_with_algebraic_mass_batched::<M>,
                exponential_decay_with_algebraic_mass_transpose::<M>,
            )
            .out_adjoint_implicit(
                exponential_decay_with_algebraic_out::<M>,
                exponential_decay_with_algebraic_out_jac_mul::<M>,
                exponential_decay_with_algebraic_out_jac_adj_mul::<M>,
                exponential_decay_with_algebraic_out_sens_adj::<M>,
                nout,
            );
        if integrate_out {
            let val = M::T::from_f64(1e-6).unwrap();
            builder = builder.out_rtol(1e-6).out_atol([val]);
        }
        builder.build().unwrap()
    };

    let ctx = problem.eqn.context();
    let a = M::T::from_f64(a).unwrap();
    let p = M::V::from_vec(vec![a], ctx.clone());
    let atol_out = M::V::from_element(nout, M::T::from_f64(1e-6).unwrap(), ctx.clone());
    let mut soln = OdeSolverSolution {
        atol: atol_out,
        rtol: problem.rtol,
        ..Default::default()
    };
    let t0 = M::T::zero();
    let t1 = M::T::from_f64(9.0).unwrap();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y0 = M::V::from_vec(vec![M::T::one(), M::T::one(), M::T::one()], ctx.clone());
        let g = y0.clone()
            * scale(((-p.get_index(0) * t0).exp() - (-p.get_index(0) * t).exp()) / p.get_index(0));
        let g = M::V::from_vec(vec![p.get_index(0) * g.get_index(2)], ctx.clone());
        let dgdk = t1 * (-p.get_index(0) * t1).exp();
        let dg = M::V::from_vec(vec![dgdk], ctx.clone());
        soln.push_sens(g, t, &[dg]);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_problem_sens<M: Matrix + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let problem = OdeBuilder::<M>::new()
        .p([k])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6, 1e-6])
        .rhs_sens_implicit(
            exponential_decay_with_algebraic_batched::<M>,
            exponential_decay_with_algebraic_jacobian_batched::<M>,
            exponential_decay_with_algebraic_sens::<M>,
        )
        .init_sens(
            exponential_decay_with_algebraic_init::<M>,
            exponential_decay_with_algebraic_init_sens::<M>,
            3,
        )
        .mass(exponential_decay_with_algebraic_mass_batched::<M>)
        .build()
        .unwrap();

    let ctx = problem.eqn.context();
    let k = M::T::from_f64(k).unwrap();
    let p = M::V::from_vec(vec![k], ctx.clone());
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 10.0).unwrap();
        let y0 = M::V::from_vec(vec![M::T::one(), M::T::one(), M::T::one()], ctx.clone());
        let y: M::V = y0.clone() * scale((-p.get_index(0) * t).exp());
        let yp = y0 * scale(-t * (-p.get_index(0) * t).exp());
        soln.push_sens(y, t, &[yp]);
    }
    (problem, soln)
}

#[cfg(feature = "diffsl")]
pub fn exponential_decay_with_algebraic_problem_diffsl<
    M: Matrix<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
>(
    prep_adjoint: bool,
) -> (
    OdeSolverProblem<crate::DiffSl<M, CG>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let out = if prep_adjoint { "k * z" } else { "u_i" };
    let problem = OdeBuilder::<M>::new()
        .p([k])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6, 1e-6])
        .param_rtol(1e-6)
        .param_atol([1e-6])
        .integrate_out(prep_adjoint)
        .out_rtol(1e-6)
        .out_atol([1e-6])
        .build_from_diffsl(
            format!(
                "
        in  {{ k = 0.1 }}
        u_i {{ x = 1, y = 1, z = 0 }}
        dudt_i {{ dxdt = 0, dydt = 0, dzdt = 0 }}
        M_i {{ dxdt, dydt, 0 }}
        F_i {{ -k * x, -k * y, z - y }}
        out_i {{ {out} }}
    "
            )
            .as_str(),
        )
        .unwrap();
    let p = [k];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = i as f64 / 10.0;
        let y0 = M::V::from_vec(vec![1.0, 1.0, 1.0], problem.eqn.context().clone());
        let y: M::V = y0 * scale((-p[0] * t).exp());
        soln.push(y, t);
    }
    (problem, soln)
}

fn exponential_decay_with_algebraic_reset_init<M: Matrix>(p: &M::V, _t: M::T, y: &mut M::V) {
    y.fill(p.get_index(1));
}

fn exponential_decay_with_algebraic_reset_init_sens<M: Matrix>(
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.for_each_batch([v], |y, [v], _| y.fill(v[1]));
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_with_reset_problem_sens<M: Matrix + 'static>() -> (
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
    use super::exponential_decay::{
        exponential_decay_reset_y_plus_2, exponential_decay_reset_y_plus_2_jac,
        exponential_decay_reset_y_plus_2_sens, exponential_decay_root_0_6_and_2_0,
        exponential_decay_root_0_6_and_2_0_jac, exponential_decay_root_0_6_and_2_0_sens,
    };

    let problem = OdeBuilder::<M>::new()
        .p([0.1, 1.0])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6, 1e-6])
        .rhs_sens_implicit(
            exponential_decay_with_algebraic_batched::<M>,
            exponential_decay_with_algebraic_jacobian_batched::<M>,
            exponential_decay_with_algebraic_sens::<M>,
        )
        .init_sens(
            exponential_decay_with_algebraic_reset_init::<M>,
            exponential_decay_with_algebraic_reset_init_sens::<M>,
            3,
        )
        .mass(exponential_decay_with_algebraic_mass_batched::<M>)
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

    let t_root = M::T::from_f64(10.0 * (5.0_f64 / 3.0_f64).ln()).unwrap();
    let dt = M::T::from_f64(10.0 * 1.3_f64.ln()).unwrap();
    let t_stop = t_root + dt;

    let y0 = problem.eqn.init().call(M::T::zero());
    let ctx = y0.context().clone();
    let nstates = problem.eqn.rhs().nstates();
    let y_stop = M::V::from_element(nstates, M::T::from_f64(2.0).unwrap(), ctx.clone());

    let s_k_val = -M::T::from_f64(2.0).unwrap() * t_stop;
    let s_y0_val = M::T::from_f64(2.0).unwrap();

    let s_k = M::V::from_element(nstates, s_k_val, ctx.clone());
    let s_y0 = M::V::from_element(nstates, s_y0_val, ctx.clone());

    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    soln.push_sens(y_stop, t_stop, &[s_k, s_y0]);

    (problem, soln)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_exponential_decay_with_algebraic_diffsl_llvm() {
        use super::*;
        use crate::{
            matrix::dense_nalgebra_serial::NalgebraMat, ConstantOpSens, ConstantOpSensAdjoint,
            NalgebraVec, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
            NonLinearOpSensAdjoint, OdeEquations,
        };
        let (problem, _soln) = exponential_decay_with_algebraic_problem_diffsl::<
            NalgebraMat<f64>,
            crate::LlvmModule,
        >(true);
        let ctx = problem.eqn.context();
        let x = NalgebraVec::from_vec(vec![1.0, 2.0, 3.0], *ctx);
        let t = 0.0;
        let v = NalgebraVec::from_vec(vec![2.0, 3.0, 4.0], *ctx);
        let v_in = NalgebraVec::from_vec(vec![5.0], *ctx);
        let p = NalgebraVec::from_vec(vec![0.1], *ctx);

        // check the adjoint jacobian
        let mut y_check = NalgebraVec::zeros(3, *ctx);
        exponential_decay_with_algebraic_adjoint::<NalgebraMat<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = NalgebraVec::zeros(3, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .rhs()
                .jac_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the sens jacobian
        let mut y_check = NalgebraVec::zeros(3, *ctx);
        exponential_decay_with_algebraic_sens::<NalgebraMat<f64>>(&x, &p, t, &v_in, &mut y_check);
        let mut y = NalgebraVec::zeros(3, *ctx);
        for _i in 0..2 {
            problem.eqn().rhs().sens_mul_inplace(&x, t, &v_in, &mut y);
            assert_eq!(y, y_check);
        }

        // check the sens adjoint jacobian
        let mut y_check = NalgebraVec::zeros(1, *ctx);
        exponential_decay_with_algebraic_sens_adjoint::<NalgebraMat<f64>>(
            &x,
            &p,
            t,
            &v,
            &mut y_check,
        );
        let mut y = NalgebraVec::zeros(1, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .rhs()
                .sens_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the set_u0 sens adjoint jacobian
        let mut y_check = NalgebraVec::zeros(1, *ctx);
        exponential_decay_with_algebraic_init_sens_adjoint::<NalgebraMat<f64>>(
            &p,
            t,
            &v,
            &mut y_check,
        );
        let mut y = NalgebraVec::zeros(1, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .init()
                .sens_transpose_mul_inplace(t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the set_u0 sens jacobian
        let mut y_check = NalgebraVec::zeros(3, *ctx);
        exponential_decay_with_algebraic_init_sens::<NalgebraMat<f64>>(&p, t, &v_in, &mut y_check);
        let mut y = NalgebraVec::zeros(3, *ctx);
        for _i in 0..2 {
            problem.eqn().init().sens_mul_inplace(t, &v_in, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out jacobian
        let mut y_check = NalgebraVec::zeros(1, *ctx);
        exponential_decay_with_algebraic_out_jac_mul::<NalgebraMat<f64>>(
            &x,
            &p,
            t,
            &v,
            &mut y_check,
        );
        let mut y = NalgebraVec::zeros(1, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .jac_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out adjoint jacobian
        let mut y_check = NalgebraVec::zeros(3, *ctx);
        exponential_decay_with_algebraic_out_jac_adj_mul::<NalgebraMat<f64>>(
            &x,
            &p,
            t,
            &v_in,
            &mut y_check,
        );
        let mut y = NalgebraVec::zeros(3, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .jac_transpose_mul_inplace(&x, t, &v_in, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out sens adjoint jacobian
        let mut y_check = NalgebraVec::zeros(1, *ctx);
        exponential_decay_with_algebraic_out_sens_adj::<NalgebraMat<f64>>(
            &x,
            &p,
            t,
            &v_in,
            &mut y_check,
        );
        let mut y = NalgebraVec::zeros(1, *ctx);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .sens_transpose_mul_inplace(&x, t, &v_in, &mut y);
            assert_eq!(y, y_check);
        }
    }
}
