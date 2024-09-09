use crate::{
    matrix::Matrix, ode_solver::problem::OdeSolverSolution, scalar::scale, ConstantOp, OdeBuilder,
    OdeEquations, OdeSolverProblem, Vector,
};
use nalgebra::ComplexField;
use num_traits::Zero;
use std::ops::MulAssign;

// exponential decay problem
// dy/dt = -ay (p = [a])
fn exponential_decay<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-p[0]));
}

// df/dp v = -yv (p = [a])
fn exponential_decay_sens<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-v[0]));
}

// Jv = -av
fn exponential_decay_jacobian<M: Matrix>(_x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.mul_assign(scale(-p[0]));
}

fn exponential_decay_init<M: Matrix>(p: &M::V, _t: M::T) -> M::V {
    M::V::from_vec(vec![p[1], p[1]])
}

fn exponential_decay_init_sens<M: Matrix>(_p: &M::V, _t: M::T, _v: &M::V, y: &mut M::V) {
    y.fill(M::T::zero());
}

fn exponential_decay_root<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = x[0] - M::T::from(0.6);
}

pub fn negative_exponential_decay_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let h = -1.0;
    let k = 0.1;
    let y0 = (-10.0 * k).exp();
    let problem = OdeBuilder::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .build_ode(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_init::<M>,
        )
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution {
        negative_time: true,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from(-i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

pub fn exponential_decay_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let h = 1.0;
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .build_ode(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_init::<M>,
        )
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

pub fn exponential_decay_problem_with_root<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::new()
        .p([k, y0])
        .use_coloring(use_coloring)
        .build_ode_with_root(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_init::<M>,
            exponential_decay_root::<M>,
            1,
        )
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

pub fn exponential_decay_problem_sens<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::new()
        .p([k, y0])
        .use_coloring(use_coloring)
        .sensitivities_error_control(true)
        .build_ode_with_sens(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_sens::<M>,
            exponential_decay_init::<M>,
            exponential_decay_init_sens::<M>,
        )
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0.clone() * scale(M::T::exp(-p[0] * t));
        let yp = y0 * scale(-t * M::T::exp(-p[0] * t));
        soln.push_sens(y, t, &[yp]);
    }
    (problem, soln)
}
