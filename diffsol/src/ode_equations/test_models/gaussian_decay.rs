use crate::ode_solver::problem::OdeSolverSolution;
use crate::OdeSolverProblem;
use crate::{
    scalar::scale, ConstantOp, DenseMatrix, OdeBuilder, OdeEquations, OdeEquationsImplicit, Vector,
};
use nalgebra::ComplexField;
use num_traits::{FromPrimitive, Pow};
use num_traits::{One, Zero};
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

#[allow(clippy::type_complexity)]
pub fn gaussian_decay_problem<M: DenseMatrix + 'static>(
    use_coloring: bool,
    size: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1].repeat(size))
        .use_coloring(use_coloring)
        .rhs_implicit(gaussian_decay::<M>, gaussian_decay_jacobian::<M>)
        .init(move |_p, _t, y| y.fill(M::T::one()), size)
        .build()
        .unwrap();
    let p = [M::T::from_f64(0.1).unwrap()].repeat(size);
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 1.0).unwrap();
        let mut y: M::V = problem.eqn.init().call(M::T::zero());
        let px = M::V::from_vec(
            p.iter()
                .map(|&x| (x * t.pow(2) / M::T::from_f64(-2.0).unwrap()).exp())
                .collect::<Vec<_>>(),
            problem.context().clone(),
        );
        y.component_mul_assign(&px);
        soln.push(y, t);
    }
    (problem, soln)
}
