use crate::{
    matrix::Matrix,
    ode_solver::problem::OdeSolverSolution,
    op::{
        closure_with_sens::ClosureWithSens, constant_closure_with_sens::ConstantClosureWithSens,
        linear_closure_with_sens::LinearClosureWithSens,
    },
    scalar::scale,
    ConstantOp, OdeBuilder, OdeEquationsImplicit, OdeEquationsSens, OdeSolverEquations,
    OdeSolverProblem, UnitCallable, Vector,
};
use nalgebra::ComplexField;
use num_traits::{One, Zero};
use std::{ops::MulAssign, rc::Rc};

// exponential decay problem with algebraic constraint
// dy/dt = -ay
// 0 = z - y
// remove warning about unused mut
#[allow(unused_mut)]
fn exponential_decay_with_algebraic<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, mut y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-p[0]));
    let nstates = y.len();
    y[nstates - 1] = x[nstates - 1] - x[nstates - 2];
}

#[allow(unused_mut)]
fn exponential_decay_with_algebraic_sens<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    mut y: &mut M::V,
) {
    y.copy_from(x);
    y.mul_assign(scale(-v[0]));
    let nstates = y.len();
    y[nstates - 1] = M::T::zero();
}

// Jv = [[-av, 0], [-1, 1]]v = [-av, -v[0] + v[1]]
#[allow(unused_mut)]
fn exponential_decay_with_algebraic_jacobian<M: Matrix>(
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

// y = Mx + beta * y = | 1 0 | | x[0] | + beta | y[0] |
//                     | 0 0 | | x[1] |         | y[1] |
fn exponential_decay_with_algebraic_mass<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    let nstates = y.len();
    let yn = beta * y[nstates - 1];
    y.axpy(M::T::one(), x, beta);
    y[nstates - 1] = yn;
}

fn exponential_decay_with_algebraic_mass_sens<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

fn exponential_decay_with_algebraic_init<M: Matrix>(_p: &M::V, _t: M::T) -> M::V {
    M::V::from_vec(vec![1.0.into(), 1.0.into(), 0.0.into()])
}

fn exponential_decay_with_algebraic_init_sens<M: Matrix>(
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

pub fn exponential_decay_with_algebraic_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T>>,
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

pub fn exponential_decay_with_algebraic_problem_sens<M: Matrix + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsSens<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let p = Rc::new(M::V::from_vec(vec![0.1.into()]));
    let mut rhs = ClosureWithSens::new(
        exponential_decay_with_algebraic::<M>,
        exponential_decay_with_algebraic_jacobian::<M>,
        exponential_decay_with_algebraic_sens::<M>,
        3,
        3,
        p.clone(),
    );
    let mut mass = LinearClosureWithSens::new(
        exponential_decay_with_algebraic_mass::<M>,
        exponential_decay_with_algebraic_mass_sens::<M>,
        3,
        3,
        p.clone(),
    );
    let init = ConstantClosureWithSens::new(
        exponential_decay_with_algebraic_init::<M>,
        exponential_decay_with_algebraic_init_sens::<M>,
        3,
        3,
        p.clone(),
    );
    let t0 = M::T::zero();

    if M::is_sparse() {
        let y0 = init.call(t0);
        rhs.calculate_jacobian_sparsity(&y0, t0);
        rhs.calculate_sens_sparsity(&y0, t0);
        mass.calculate_sparsity(t0);
    }

    let out: Option<Rc<UnitCallable<M>>> = None;
    let root: Option<Rc<UnitCallable<M>>> = None;
    let eqn = OdeSolverEquations::new(
        Rc::new(rhs),
        Some(Rc::new(mass)),
        root,
        Rc::new(init),
        out,
        p.clone(),
    );
    let problem = OdeSolverProblem::new(
        eqn,
        M::T::from(1e-6),
        M::V::from_element(3, M::T::from(1e-6)),
        t0,
        M::T::from(1.0),
        false,
    )
    .unwrap();

    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64 / 10.0);
        let y0 = M::V::from_vec(vec![1.0.into(), 1.0.into(), 1.0.into()]);
        let y: M::V = y0.clone() * scale(M::T::exp(-p[0] * t));
        let yp = y0 * scale(-t * M::T::exp(-p[0] * t));
        soln.push_sens(y, t, &[yp]);
    }
    (problem, soln)
}
