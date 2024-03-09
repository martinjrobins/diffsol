use crate::{op::{ConstantOp, LinearOp, NonLinearOp}, ode_solver::{OdeSolverProblem, OdeSolverSolution}, DenseMatrix, Vector};
use std::ops::MulAssign;
use num_traits::Zero;
use nalgebra::ComplexField;


// exponential decay problem
// dy/dt = -ay (p = [a])
fn exponential_decay<M: DenseMatrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(-p[0]);
}

// Jv = -av
fn exponential_decay_jacobian<M: DenseMatrix>(_x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.mul_assign(-p[0]);
}

fn exponential_decay_init<M: DenseMatrix>(_p: &M::V, _t: M::T) -> M::V {
    M::V::from_vec(vec![1.0.into(), 1.0.into()])
}

pub fn exponential_decay_problem<M: DenseMatrix + 'static>() -> (OdeSolverProblem<impl NonLinearOp<M = M, V = M::V, T = M::T>, impl LinearOp<M = M, V = M::V, T = M::T> , impl ConstantOp<M = M, V = M::V, T = M::T>>, OdeSolverSolution<M::V>) {
    let p = M::V::from_vec(vec![0.1.into()]);
    let rhs = {
        let p = p.clone();
        move |x: &M::V, t: M::T, y: &mut M::V| exponential_decay::<M>(x, &p, t, y)
    };
    let jac = {
        let p = p.clone();
        move |x: &M::V, t: M::T, v: &M::V, y: &mut M::V| exponential_decay_jacobian::<M>(x, &p, t, v, y)
    };
    let init = {
        let p = p.clone();
        move |t: M::T| exponential_decay_init::<M>(&p, t)
    };
    let problem = OdeSolverProblem::new_ode(rhs, jac, init);
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64 / 10.0);
        let y0: M::V = problem.init().call(M::T::zero());
        let y = y0 * M::T::exp(-p[0] * t);
        soln.push(y, t);
    }
    (problem, soln)
}