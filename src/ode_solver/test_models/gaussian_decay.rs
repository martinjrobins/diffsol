use crate::ode_solver::problem::OdeSolverSolution;
use crate::OdeSolverProblem;
use crate::{scalar::scale, DenseMatrix, OdeBuilder, OdeEquations, Vector};
use num_traits::Pow;
use num_traits::Zero;
use std::ops::MulAssign;

// dy/dt = -aty (p = [a])
fn gaussian_decay<M: DenseMatrix>(x: &M::V, p: &M::V, t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.component_mul_assign(p);
    y.mul_assign(scale(-t));
}

// Jv = -atv
fn gaussian_decay_jacobian<M: DenseMatrix>(_x: &M::V, p: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.component_mul_assign(p);
    y.mul_assign(scale(-t));
}

pub fn gaussian_decay_problem<M: DenseMatrix + 'static>(
    use_coloring: bool,
    size: usize,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let size2 = size;
    let problem = OdeBuilder::new()
        .p([0.1].repeat(size))
        .use_coloring(use_coloring)
        .build_ode(
            gaussian_decay::<M>,
            gaussian_decay_jacobian::<M>,
            move |_p, _t| M::V::from_vec([1.0.into()].repeat(size2)),
        )
        .unwrap();
    let p = [M::T::from(0.1)].repeat(size);
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64 / 1.0);
        let px = M::V::from_vec(p.clone()) * scale(t.pow(2)) / scale(M::T::from(-2.0));
        let mut y: M::V = problem.eqn.init(M::T::zero());
        y.component_mul_assign(&px.exp());
        soln.push(y, t);
    }
    (problem, soln)
}
