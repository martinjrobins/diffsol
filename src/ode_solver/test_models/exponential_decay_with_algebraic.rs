use crate::{
    ode_solver::problem::OdeSolverSolution, scalar::scale, DenseMatrix, OdeBuilder, OdeEquations,
    OdeSolverProblem, Vector,
};
use nalgebra::ComplexField;
use num_traits::One;
use std::ops::MulAssign;

// exponential decay problem with algebraic constraint
// dy/dt = -ay
// 0 = z - y
// remove warning about unused mut
#[allow(unused_mut)]
fn exponential_decay_with_algebraic<M: DenseMatrix>(
    x: &M::V,
    p: &M::V,
    _t: M::T,
    mut y: &mut M::V,
) {
    y.copy_from(x);
    y.mul_assign(scale(-p[0]));
    let nstates = y.len();
    y[nstates - 1] = x[nstates - 1] - x[nstates - 2];
}

// Jv = [[-av, 0], [-1, 1]]v = [-av, -v[0] + v[1]]
#[allow(unused_mut)]
fn exponential_decay_with_algebraic_jacobian<M: DenseMatrix>(
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

fn exponential_decay_with_algebraic_mass<M: DenseMatrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    y.axpy(M::T::one(), x, beta);
    let nstates = y.len();
    y[nstates - 1] = beta * x[nstates - 1];
}

fn exponential_decay_with_algebraic_init<M: DenseMatrix>(_p: &M::V, _t: M::T) -> M::V {
    M::V::from_vec(vec![1.0.into(), 1.0.into(), 0.0.into()])
}

pub fn exponential_decay_with_algebraic_problem<M: DenseMatrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let p = M::V::from_vec(vec![0.1.into()]);
    let problem = OdeBuilder::new()
        .p([0.1])
        .use_coloring(use_coloring)
        .build_ode_with_mass(
            exponential_decay_with_algebraic::<M>,
            exponential_decay_with_algebraic_jacobian::<M>,
            exponential_decay_with_algebraic_mass::<M>,
            exponential_decay_with_algebraic_init::<M>,
        )
        .unwrap();

    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64 / 10.0);
        let y0 = M::V::from_vec(vec![1.0.into(), 1.0.into(), 1.0.into()]);
        let y: M::V = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}
