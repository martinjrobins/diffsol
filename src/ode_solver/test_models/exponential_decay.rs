use crate::{
    matrix::Matrix, ode_solver::problem::OdeSolverSolution,
    op::closure_with_adjoint::ClosureWithAdjoint, scalar::scale, ConstantClosure, ConstantOp,
    OdeBuilder, OdeEquations, OdeSolverEquations, OdeSolverProblem, UnitCallable, Vector,
};
use nalgebra::ComplexField;
use num_traits::Zero;
use std::{ops::MulAssign, rc::Rc};

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

// df/dp^T v = -yv (p = [a])
fn exponential_decay_sens_transpose<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.copy_from(x);
    y.mul_assign(scale(-v[0]));
}

// Jv = -av
fn exponential_decay_jacobian<M: Matrix>(_x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.mul_assign(scale(-p[0]));
}

// -J^Tv = av
fn exponential_decay_jacobian_adjoint<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.copy_from(v);
    y.mul_assign(scale(p[0]));
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

/// g_1 = 1 * x_1  +  2 * x_2
/// g_2 = 3 * x_1  +  4 * x_2
fn exponential_decay_out<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = M::T::from(1.0) * x[0] + M::T::from(2.0) * x[1];
    y[1] = M::T::from(3.0) * x[0] + M::T::from(4.0) * x[1];
}

/// J = |1 2|
///     |3 4|
/// Jv = |1 2| |v_1| = |v_1 + 2v_2|
///     |3 4| |v_2|   |3v_1 + 4v_2|
fn exponential_decay_out_jac_mul<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = v[0] + M::T::from(2.0) * v[1];
    y[1] = M::T::from(3.0) * v[0] + M::T::from(4.0) * v[1];
}

/// J = |1 2|
///    |3 4|
/// -J^T v = |1 3| |v_1| = |-v_1 - 3v_2|
///         |2 4| |v_2|   |-2v_1 - 4v_2|
fn exponential_decay_out_adj_mul<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -v[0] - M::T::from(3.0) * v[1];
    y[1] = -M::T::from(2.0) * v[0] - M::T::from(4.0) * v[1];
}

/// J = |0|
///     |0|
fn exponential_decay_out_sens<M: Matrix>(_x: &M::V, _p: &M::V, _t: M::T, _v: &M::V, y: &mut M::V) {
    y.fill(M::T::zero());
}

/// J = |0|
///     |0|
fn exponential_decay_out_sens_adj<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
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

pub fn exponential_decay_problem_adjoint<M: Matrix>() -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let k = M::T::from(0.1);
    let y0 = M::T::from(1.0);
    let t0 = M::T::from(0.0);
    let h0 = M::T::from(1.0);
    let p = Rc::new(M::V::from_vec(vec![k, y0]));
    let init = exponential_decay_init::<M>;
    let y0 = init(&p, t0);
    let nstates = y0.len();
    let rhs = exponential_decay::<M>;
    let rhs_jac = exponential_decay_jacobian::<M>;
    let rhs_adj_jac = exponential_decay_jacobian_adjoint::<M>;
    let rhs_sens_adj = exponential_decay_sens_transpose::<M>;
    let mut rhs = ClosureWithAdjoint::new(
        rhs,
        rhs_jac,
        rhs_adj_jac,
        rhs_sens_adj,
        nstates,
        nstates,
        p.clone(),
    );
    let nout = 2;
    let out = exponential_decay_out::<M>;
    let out_jac = exponential_decay_out_sens::<M>;
    let out_jac_adj = exponential_decay_out_adj_mul::<M>;
    let out_sens_adj = exponential_decay_out_sens_adj::<M>;
    let mut out = ClosureWithAdjoint::new(
        out,
        out_jac,
        out_jac_adj,
        out_sens_adj,
        nstates,
        nout,
        p.clone(),
    );
    let init = exponential_decay_init::<M>;
    let init = ConstantClosure::new(init, p.clone());
    if M::is_sparse() {
        rhs.calculate_jacobian_sparsity(&y0, t0);
        rhs.calculate_adjoint_sparsity(&y0, t0);
        rhs.calculate_sens_adjoint_sparsity(&y0, t0);
        out.calculate_sens_adjoint_sparsity(&y0, t0);
    }
    let rhs = Rc::new(rhs);
    let init = Rc::new(init);
    let out = Some(Rc::new(out));
    let mass: Option<Rc<UnitCallable<M>>> = None;
    let root: Option<Rc<UnitCallable<M>>> = None;
    let eqn = OdeSolverEquations::new(rhs, mass, root, init, out, p.clone());
    let rtol = M::T::from(1e-6);
    let atol = M::V::from_element(nstates, M::T::from(1e-6));
    let problem = OdeSolverProblem::new(eqn, rtol, atol, t0, h0).unwrap();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.as_ref().clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    let t0 = M::T::from(0.0);
    let t1 = M::T::from(10.0);
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let g = y0.clone() * scale((M::T::exp(-p[0] * t) - M::T::exp(-p[0] * t0)) / p[0]);
        let g = M::V::from_vec(vec![
            -g[0] - M::T::from(2.0) * g[1],
            -M::T::from(3.0) * g[0] - M::T::from(4.0) * g[1],
        ]);
        let gp = y0
            * scale(
                (M::T::exp(-p[0] * t1) - M::T::exp(-p[0] * t)) / (p[0] * p[0])
                    - (t * M::T::exp(-p[0] * t) - t1 * M::T::exp(-p[0] * t1)) / p[0],
            );
        let gp = M::V::from_vec(vec![
            gp[0] + M::T::from(2.0) * gp[1],
            M::T::from(3.0) * gp[0] + M::T::from(4.0) * gp[1],
        ]);
        soln.push_sens(g, t, &[gp]);
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
