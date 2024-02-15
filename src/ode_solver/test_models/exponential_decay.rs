use crate::{callable::{closure::Closure, constant_closure::ConstantClosure, linear_closure::LinearClosure, ConstantOp, LinearOp, NonLinearOp}, ode_solver::{OdeSolverProblem, OdeSolverSolution}, Matrix, Vector};
use std::ops::MulAssign;
use num_traits::Zero;
use nalgebra::ComplexField;


// exponential decay problem
// dy/dt = -ay (p = [a])
fn exponential_decay<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(-p[0]);
}

// Jv = -av
fn exponential_decay_jacobian<M: Matrix>(_x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.mul_assign(-p[0]);
}

fn exponential_decay_mass<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
}

fn exponential_decay_init<M: Matrix>(_p: &M::V, _t: M::T, y: &mut M::V) {
    let y0 = M::V::from_vec(vec![1.0.into(), 1.0.into()]);
    y.copy_from(&y0);
}

pub fn exponential_decay_problem<M: Matrix>() -> (OdeSolverProblem<impl NonLinearOp<M = M, V = M::V, T = M::T>, impl LinearOp<M = M, V = M::V, T = M::T> , impl ConstantOp<M = M, V = M::V, T = M::T>>, OdeSolverSolution<M::V>) {
    let nstates = 2;
    let rhs = Closure::new(
        exponential_decay::<M>,
        exponential_decay_jacobian::<M>,
        nstates, 
        nstates,
        1,
    );
    let mass = LinearClosure::new(
        exponential_decay_mass::<M>,
        nstates, 
        nstates,
        1,
    );
    let init: ConstantClosure<M, fn(&M::V, M::T, &mut M::V)> = ConstantClosure::new(
        exponential_decay_init::<M>,
        nstates,
        nstates,
        1,
    );
    let p = M::V::from_vec(vec![0.1.into()]);

    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64 / 10.0);
        let y0 = init.call(&p, M::T::zero());
        let y = y0 * M::T::exp(-p[0] * t);
        soln.push(y, t);
    }
    let problem = OdeSolverProblem::new(rhs, mass, init, p);
    (problem, soln)
}