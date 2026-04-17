use crate::{
    ode_solver::problem::OdeSolverSolution, MatrixHost, OdeBuilder, OdeEquationsImplicitAdjoint,
    OdeSolverProblem, Op, Vector,
};
use num_traits::{FromPrimitive, One, Zero};

fn logistic_state<T: crate::Scalar>(r: T, k: T, y0: T, t: T) -> T {
    let exp_rt = (r * t).exp();
    let numerator = y0 * exp_rt;
    let denominator = T::one() - y0 / k + (y0 / k) * exp_rt;
    numerator / denominator
}

#[allow(clippy::type_complexity)]
pub fn logistic_problem_adjoint_no_out<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let r = 1.0;
    let k = 1.0;
    let y0 = 0.1;
    let problem = OdeBuilder::<M>::new()
        .p([r, k, y0])
        .rhs_adjoint_implicit(
            |x: &M::V, p: &M::V, _t: M::T, y: &mut M::V| {
                let r = p.get_index(0);
                let k = p.get_index(1);
                let u = x.get_index(0);
                y[0] = r * u * (M::T::one() - u / k);
            },
            |x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                let r = p.get_index(0);
                let k = p.get_index(1);
                let u = x.get_index(0);
                y[0] = r * (M::T::one() - M::T::from_f64(2.0).unwrap() * u / k) * v[0];
            },
            |x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                let r = p.get_index(0);
                let k = p.get_index(1);
                let u = x.get_index(0);
                y[0] = -r * (M::T::one() - M::T::from_f64(2.0).unwrap() * u / k) * v[0];
            },
            |x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                let r = p.get_index(0);
                let k = p.get_index(1);
                let u = x.get_index(0);
                y[0] = -u * (M::T::one() - u / k) * v[0];
                y[1] = -(r * u * u / (k * k) * v[0]);
                y[2] = M::T::zero();
            },
        )
        .init_adjoint(
            |p: &M::V, _t: M::T, y: &mut M::V| y[0] = p.get_index(2),
            |_p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
                y[0] = M::T::zero();
                y[1] = M::T::zero();
                y[2] = -v[0];
            },
            1,
        )
        .build()
        .unwrap();

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
