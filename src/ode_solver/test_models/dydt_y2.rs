use crate::{
    ode_solver::problem::OdeSolverSolution, scalar::scale, DenseMatrix, OdeBuilder, OdeEquations,
    OdeSolverProblem, Vector,
};
use num_traits::One;
use std::ops::MulAssign;

// dy/dt = y^2
fn rhs<M: DenseMatrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.component_mul_assign(x);
}

// Jv = 2yv
fn rhs_jac<M: DenseMatrix>(x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.component_mul_assign(x);
    y.mul_assign(scale(M::T::from(2.)));
}

pub fn dydt_y2_problem<M: DenseMatrix + 'static>(
    use_coloring: bool,
    size: usize,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let size2 = size;
    let y0 = -200.;
    let tlast = 20.0;
    let problem = OdeBuilder::new()
        .use_coloring(use_coloring)
        .rtol(1e-4)
        .build_ode(rhs::<M>, rhs_jac::<M>, move |_p, _t| {
            M::V::from_vec([y0.into()].repeat(size2))
        })
        .unwrap();
    let mut soln = OdeSolverSolution::default();
    let y0 = M::V::from_vec([y0.into()].repeat(size));
    let n = 10;
    let dt = tlast / n as f64;
    for i in 0..=n {
        let t = M::T::from(i as f64 * dt);
        // y = y0 / (1 - y0 * t)
        let mut denom = y0.clone() * (scale(-t));
        denom.add_scalar_mut(M::T::one());
        let mut y = y0.clone();
        y.component_div_assign(&denom);
        soln.push(y, t);
    }
    (problem, soln)
}
