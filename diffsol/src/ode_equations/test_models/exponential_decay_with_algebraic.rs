use crate::{
    matrix::Matrix, ode_solver::problem::OdeSolverSolution, scalar::scale, MatrixHost, OdeBuilder,
    OdeEquationsImplicit, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverProblem,
    Op, Vector,
};
use nalgebra::ComplexField;
use num_traits::{FromPrimitive, One, Zero};
use std::ops::MulAssign;

// exponential decay problem with algebraic constraint
// dy/dt = -ay
// 0 = z - y
// remove warning about unused mut
#[allow(unused_mut)]
fn exponential_decay_with_algebraic<M: MatrixHost>(x: &M::V, p: &M::V, _t: M::T, mut y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-p[0]));
    let nstates = y.len();
    y[nstates - 1] = x[nstates - 1] - x[nstates - 2];
}

// J = | -y[0] |
//     | -y[1] |
//     | 0    |
// Jv = | -y[0]v[0] |
//      | -y[1]v[1] |
//      | 0        |
#[allow(unused_mut)]
fn exponential_decay_with_algebraic_sens<M: MatrixHost>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    mut y: &mut M::V,
) {
    y.copy_from(x);
    y.mul_assign(scale(-v[0]));
    let nstates = y.len();
    y[nstates - 1] = M::T::zero();
}

// -J^Tv = | y[0]v[0] + y[1]v[1] + 0 |
#[allow(unused_mut)]
fn exponential_decay_with_algebraic_sens_adjoint<M: MatrixHost>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    mut y: &mut M::V,
) {
    y[0] = x[0] * v[0] + x[1] * v[1];
}

// J = | -a, 0, 0 |
//     | 0, -a, 0 |
//     | 0,  -1, 1 |
// Jv = | -av[0] |
//      | -av[1] |
//      | v[2] - v[1] |
#[allow(unused_mut)]
fn exponential_decay_with_algebraic_jacobian<M: MatrixHost>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    mut y: &mut M::V,
) {
    y.copy_from(v);
    y.mul_assign(scale(-p[0]));
    let nstates = y.len();
    y[nstates - 1] = v[nstates - 1] - v[nstates - 2];
}

// -J^T v = | av[0] |
//          | av[1] + v[2] |
//          |  -v[2]    |
fn exponential_decay_with_algebraic_adjoint<M: MatrixHost>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = p[0] * v[0];
    y[1] = p[0] * v[1] + v[2];
    y[2] = -v[2];
}

// y = Mx + beta * y = | 1 0 | | x[0] | + beta | y[0] |
//                     | 0 0 | | x[1] |         | y[1] |
fn exponential_decay_with_algebraic_mass<M: MatrixHost>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    let nstates = y.len();
    let yn = beta * y[nstates - 1];
    y.axpy(M::T::one(), x, beta);
    y[nstates - 1] = yn;
}

// y = M^T x + beta * y = | 1 0 | | x[0] | + beta | y[0] |
//                        | 0 0 | | x[1] |         | y[1] |
fn exponential_decay_with_algebraic_mass_transpose<M: MatrixHost>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    let nstates = y.len();
    let yn = beta * y[nstates - 1];
    y.axpy(M::T::one(), x, beta);
    y[nstates - 1] = yn;
}

fn exponential_decay_with_algebraic_init<M: MatrixHost>(_p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = M::T::one();
    y[1] = M::T::one();
    y[2] = M::T::zero();
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
fn exponential_decay_with_algebraic_out<M: MatrixHost>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = p[0] * x[2];
}

// J = | 0 0 a |
// Jv = | a * v[2] |
fn exponential_decay_with_algebraic_out_jac_mul<M: MatrixHost>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = p[0] * v[2];
}

// J = | 0 0 a |
// -J^T v = | 0 |
//          | 0 |
//          | -a * v[0] |
fn exponential_decay_with_algebraic_out_jac_adj_mul<M: MatrixHost>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
    y[2] = -p[0] * v[0];
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
//    y[0] = x[1] * v[1];
//}

// -J^T v = | -x[2]v[2] |
fn exponential_decay_with_algebraic_out_sens_adj<M: MatrixHost>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -x[2] * v[0];
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_problem<M: MatrixHost + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1])
        .use_coloring(use_coloring)
        .rhs_implicit(
            exponential_decay_with_algebraic::<M>,
            exponential_decay_with_algebraic_jacobian::<M>,
        )
        .mass(exponential_decay_with_algebraic_mass::<M>)
        .init(exponential_decay_with_algebraic_init::<M>, 3)
        .build()
        .unwrap();

    let ctx = problem.eqn.context();
    let p = M::V::from_vec(vec![M::T::from_f64(0.1).unwrap()], ctx.clone());
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 10.0).unwrap();
        let y0 = M::V::from_vec(vec![M::T::one(), M::T::one(), M::T::one()], ctx.clone());
        let y: M::V = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_adjoint_problem<M: MatrixHost + 'static>(
    integrate_out: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let a = 0.1;
    let nout = 1;
    let problem = OdeBuilder::<M>::new()
        .p([a])
        .integrate_out(integrate_out)
        .rhs_adjoint_implicit(
            exponential_decay_with_algebraic::<M>,
            exponential_decay_with_algebraic_jacobian::<M>,
            exponential_decay_with_algebraic_adjoint::<M>,
            exponential_decay_with_algebraic_sens_adjoint::<M>,
        )
        .init_adjoint(
            exponential_decay_with_algebraic_init::<M>,
            exponential_decay_with_algebraic_init_sens_adjoint::<M>,
            3,
        )
        .mass_adjoint(
            exponential_decay_with_algebraic_mass::<M>,
            exponential_decay_with_algebraic_mass_transpose::<M>,
        )
        .out_adjoint_implicit(
            exponential_decay_with_algebraic_out::<M>,
            exponential_decay_with_algebraic_out_jac_mul::<M>,
            exponential_decay_with_algebraic_out_jac_adj_mul::<M>,
            exponential_decay_with_algebraic_out_sens_adj::<M>,
            nout,
        )
        .build()
        .unwrap();

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
        let g = y0.clone() * scale((M::T::exp(-p[0] * t0) - M::T::exp(-p[0] * t)) / p[0]);
        let g = M::V::from_vec(vec![p[0] * g[2]], ctx.clone());
        let dgdk = t1 * M::T::exp(-p[0] * t1);
        let dg = M::V::from_vec(vec![dgdk], ctx.clone());
        soln.push_sens(g, t, &[dg]);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_with_algebraic_problem_sens<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let problem = OdeBuilder::<M>::new()
        .p([k])
        .rhs_sens_implicit(
            exponential_decay_with_algebraic::<M>,
            exponential_decay_with_algebraic_jacobian::<M>,
            exponential_decay_with_algebraic_sens::<M>,
        )
        .init_sens(
            exponential_decay_with_algebraic_init::<M>,
            exponential_decay_with_algebraic_init_sens::<M>,
            3,
        )
        .mass(exponential_decay_with_algebraic_mass::<M>)
        .build()
        .unwrap();

    let ctx = problem.eqn.context();
    let k = M::T::from_f64(k).unwrap();
    let p = M::V::from_vec(vec![k], ctx.clone());
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 10.0).unwrap();
        let y0 = M::V::from_vec(vec![M::T::one(), M::T::one(), M::T::one()], ctx.clone());
        let y: M::V = y0.clone() * scale(M::T::exp(-p[0] * t));
        let yp = y0 * scale(-t * M::T::exp(-p[0] * t));
        soln.push_sens(y, t, &[yp]);
    }
    (problem, soln)
}

#[cfg(feature = "diffsl")]
pub fn exponential_decay_with_algebraic_problem_diffsl<
    M: MatrixHost<T = f64>,
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
        .integrate_out(prep_adjoint)
        .build_from_diffsl(
            format!(
                "
        in = [k]
        k {{ 0.1 }}
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
        let y: M::V = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
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
