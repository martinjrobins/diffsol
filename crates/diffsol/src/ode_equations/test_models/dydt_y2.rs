use crate::{
    ode_solver::problem::OdeSolverSolution, Context, DenseMatrix, OdeBuilder, OdeEquationsImplicit,
    OdeSolverProblem,
};
use num_traits::{FromPrimitive, One};

// dy/dt = y^2
fn rhs<M: DenseMatrix>(x: &[M::T], _p: &[M::T], _t: M::T, y: &mut [M::T]) {
    for (y, x) in y.iter_mut().zip(x.iter()) {
        *y = *x * *x;
    }
}

// Jv = 2yv
fn rhs_jac<M: DenseMatrix>(x: &[M::T], _p: &[M::T], _t: M::T, v: &[M::T], y: &mut [M::T]) {
    let two = M::T::from_f64(2.).unwrap();
    for ((y, x), v) in y.iter_mut().zip(x.iter()).zip(v.iter()) {
        *y = two * *x * *v;
    }
}

#[allow(clippy::type_complexity)]
pub fn dydt_y2_problem<M: DenseMatrix + 'static>(
    use_coloring: bool,
    size: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let y0 = -200.;
    let tlast = 20.0;
    let problem = OdeBuilder::<M>::new()
        .use_coloring(use_coloring)
        .rtol(1e-4)
        .rhs_implicit(rhs::<M>, rhs_jac::<M>)
        .init(move |_p, _t, y| y.fill(M::T::from_f64(y0).unwrap()), size)
        .build()
        .unwrap();
    let mut soln = OdeSolverSolution::default();
    let y0: Vec<M::T> = [M::T::from_f64(y0).unwrap()].repeat(size);
    let n = 10;
    let dt = tlast / n as f64;
    for i in 0..=n {
        let t = M::T::from_f64(i as f64 * dt).unwrap();
        // y = y0 / (1 - y0 * t)
        let y = y0
            .iter()
            .map(|&y| y / (M::T::one() - y * t))
            .collect::<Vec<_>>();
        soln.push(problem.context().vector_from_vec(y), t);
    }
    (problem, soln)
}
