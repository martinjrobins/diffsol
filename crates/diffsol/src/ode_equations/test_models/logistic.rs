use crate::{
    ode_solver::problem::OdeSolverSolution, Context, Matrix, OdeBuilder,
    OdeEquationsImplicitAdjoint, OdeSolverProblem, Op, Vector,
};
use num_traits::{FromPrimitive, One, Zero};

fn logistic_state<T: crate::Scalar>(r: T, k: T, y0: T, t: T) -> T {
    let exp_rt = (r * t).exp();
    let numerator = y0 * exp_rt;
    let denominator = T::one() - y0 / k + (y0 / k) * exp_rt;
    numerator / denominator
}

/// The equations behind [`logistic_problem_adjoint_no_out`], with the context left open so tests
/// can build the same operators at `nbatch > 1`.
#[allow(clippy::type_complexity)]
fn logistic_adjoint_problem<M: Matrix + 'static>(
    ctx: M::C,
) -> OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>> {
    let nbatch = ctx.nbatch();
    let (r, k, y0) = (1.0, 1.0, 0.1);
    OdeBuilder::<M>::new()
        .context(ctx)
        .p([r, k, y0].repeat(nbatch))
        .sens_rtol(1e-6)
        .sens_atol([1e-6])
        .param_rtol(1e-6)
        .param_atol([1e-6])
        .rhs_adjoint_implicit(
            |x: &M::V, p: &M::V, _t: M::T, y: &mut M::V| {
                y.for_each_batch([x, p], |y, [x, p], _| {
                    let (r, k, u) = (p[0], p[1], x[0]);
                    y[0] = r * u * (M::T::one() - u / k);
                });
            },
            |x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                y.for_each_batch([x, p, v], |out, [x, p, v], _| {
                    let (r, k, u) = (p[0], p[1], x[0]);
                    out[0] = r * (M::T::one() - M::T::from_f64(2.0).unwrap() * u / k) * v[0]
                });
            },
            |x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                y.for_each_batch([x, p, v], |out, [x, p, v], _| {
                    let (r, k, u) = (p[0], p[1], x[0]);
                    out[0] = -r * (M::T::one() - M::T::from_f64(2.0).unwrap() * u / k) * v[0]
                });
            },
            |x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                y.for_each_batch([x, p, v], |out, [x, p, v], _| {
                    let (r, k, u) = (p[0], p[1], x[0]);
                    out[0] = -u * (M::T::one() - u / k) * v[0];
                    out[1] = -(r * u * u / (k * k) * v[0]);
                    out[2] = M::T::zero();
                });
            },
        )
        .init_adjoint(
            |p: &M::V, _t: M::T, y: &mut M::V| y.for_each_batch([p], |y, [p], _| y[0] = p[2]),
            |_p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                y.for_each_batch([v], |out, [v], _| {
                    out[0] = M::T::zero();
                    out[1] = M::T::zero();
                    out[2] = -v[0];
                });
            },
            1,
        )
        .build()
        .unwrap()
}

#[allow(clippy::type_complexity)]
pub fn logistic_problem_adjoint_no_out<M: Matrix + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let (r, k, y0) = (1.0, 1.0, 0.1);
    let problem = logistic_adjoint_problem::<M>(M::C::default());

    let r = M::T::from_f64(r).unwrap();
    let k = M::T::from_f64(k).unwrap();
    let y0 = M::T::from_f64(y0).unwrap();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from_f64(i as f64).unwrap();
        let y = M::V::from_vec(
            vec![logistic_state(r, k, y0, t)],
            problem.eqn.context().clone(),
        );
        soln.push(y, t);
    }
    (problem, soln)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NalgebraContext, NalgebraMat, NalgebraVec, NonLinearOp, NonLinearOpAdjoint,
        NonLinearOpJacobian, NonLinearOpSensAdjoint, OdeEquations,
    };

    type M = NalgebraMat<f64>;

    /// Every operator must run at `nbatch > 1`: reading a parameter or state with `get_index`
    /// outside the per-batch closure panics there, and used to.
    #[test]
    fn operators_run_batched() {
        let nbatch = 2;
        let ctx = NalgebraContext::default()
            .clone_with_nbatch(nbatch)
            .unwrap();
        let problem = logistic_adjoint_problem::<M>(ctx);
        let eqn = problem.eqn.rhs();

        let x = NalgebraVec::<f64>::from_element(1, 0.5, ctx);
        let v = NalgebraVec::<f64>::from_element(1, 1.0, ctx);
        let mut y = NalgebraVec::<f64>::zeros(1, ctx);
        let mut yp = NalgebraVec::<f64>::zeros(3, ctx);
        let t = 0.0;

        // r = k = 1, u = 0.5:  r·u·(1 - u/k) = 0.25, and J = r(1 - 2u/k) = 0
        eqn.call_inplace(&x, t, &mut y);
        assert_eq!(y.clone_as_vec(), vec![0.25; nbatch]);
        eqn.jac_mul_inplace(&x, t, &v, &mut y);
        assert_eq!(y.clone_as_vec(), vec![0.0; nbatch]);
        eqn.jac_transpose_mul_inplace(&x, t, &v, &mut y);
        assert_eq!(y.clone_as_vec(), vec![0.0; nbatch]);
        // -u(1 - u/k) = -0.25, -(r·u²/k²) = -0.25, 0
        eqn.sens_transpose_mul_inplace(&x, t, &v, &mut yp);
        assert_eq!(
            yp.clone_as_vec(),
            vec![-0.25, -0.25, 0.0, -0.25, -0.25, 0.0]
        );
    }
}
