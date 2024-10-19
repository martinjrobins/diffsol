use crate::{
    matrix::Matrix,
    ode_solver::problem::OdeSolverSolution,
    op::{
        closure_with_sens::ClosureWithSens, constant_closure_with_sens::ConstantClosureWithSens,
        linear_closure_with_adjoint::LinearClosureWithAdjoint,
    },
    scalar::scale,
    ClosureWithAdjoint, ConstantClosureWithAdjoint, ConstantOp, LinearClosure, OdeBuilder,
    OdeEquationsAdjoint, OdeEquationsImplicit, OdeEquationsSens, OdeSolverEquations,
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

// J = | -y[0] |
//     | -y[1] |
//     | 0    |
// Jv = | -y[0]v[0] |
//      | -y[1]v[1] |
//      | 0        |
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

// -J^Tv = | y[0]v[0] + y[1]v[1] + 0 |
#[allow(unused_mut)]
fn exponential_decay_with_algebraic_sens_adjoint<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    mut y: &mut M::V,
) {
    y[0] = x[0] * v[0] + x[1] * v[1];
}

// J = | -a, 0, 0 |
//     | 0, -a, 0 |
//     | 0,  -1, 1 |
// Jv = | -av[0] |
//      | -av[1] |
//      | v[2] - v[1] |
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

// -J^T v = | av[0] |
//          | av[1] + v[2] |
//          |  -v[2]    |
fn exponential_decay_with_algebraic_adjoint<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = p[0] * v[0];
    y[1] = p[0] * v[1] + v[2];
    y[2] = -v[2];
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

// y = M^T x + beta * y = | 1 0 | | x[0] | + beta | y[0] |
//                        | 0 0 | | x[1] |         | y[1] |
fn exponential_decay_with_algebraic_mass_transpose<M: Matrix>(
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

fn exponential_decay_with_algebraic_init_sens_adjoint<M: Matrix>(
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

// out(x) = | a * x[2] |
fn exponential_decay_with_algebraic_out<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = p[0] * x[2];
}

// J = | 0 0 a |
// Jv = | a * v[2] |
fn exponential_decay_with_algebraic_out_jac_mul<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = p[0] * v[2];
}

// J = | 0 0 a |
// -J^T v = | 0 |
//          | 0 |
//          | -a * v[0] |
fn exponential_decay_with_algebraic_out_jac_adj_mul<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
    y[2] = -p[0] * v[0];
}

// J = | x[2] |
// Jv = | x[2]v[0] |
//fn exponential_decay_with_algebraic_out_sens<M: Matrix>(
//    x: &M::V,
//    _p: &M::V,
//    _t: M::T,
//    v: &M::V,
//    y: &mut M::V,
//) {
//    y[0] = x[1] * v[1];
//}

// -J^T v = | -x[2]v[2] |
fn exponential_decay_with_algebraic_out_sens_adj<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -x[2] * v[0];
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

pub fn exponential_decay_with_algebraic_adjoint_problem<M: Matrix + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsAdjoint<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let a = M::T::from(0.1);
    let t0 = M::T::from(0.0);
    let h0 = M::T::from(1.0);
    let p = Rc::new(M::V::from_vec(vec![a]));
    let init = exponential_decay_with_algebraic_init::<M>;
    let y0 = init(&p, t0);
    let nstates = y0.len();
    let rhs = exponential_decay_with_algebraic::<M>;
    let rhs_jac = exponential_decay_with_algebraic_jacobian::<M>;
    let rhs_adj_jac = exponential_decay_with_algebraic_adjoint::<M>;
    let rhs_sens_adj = exponential_decay_with_algebraic_sens_adjoint::<M>;
    let mut rhs = ClosureWithAdjoint::new(
        rhs,
        rhs_jac,
        rhs_adj_jac,
        rhs_sens_adj,
        nstates,
        nstates,
        p.clone(),
    );
    let nout = 1;
    let out = exponential_decay_with_algebraic_out::<M>;
    let out_jac = exponential_decay_with_algebraic_out_jac_mul::<M>;
    let out_jac_adj = exponential_decay_with_algebraic_out_jac_adj_mul::<M>;
    let out_sens_adj = exponential_decay_with_algebraic_out_sens_adj::<M>;
    let out = ClosureWithAdjoint::new(
        out,
        out_jac,
        out_jac_adj,
        out_sens_adj,
        nstates,
        nout,
        p.clone(),
    );
    let init = ConstantClosureWithAdjoint::new(
        exponential_decay_with_algebraic_init::<M>,
        exponential_decay_with_algebraic_init_sens_adjoint::<M>,
        p.clone(),
    );
    let mut mass = LinearClosureWithAdjoint::new(
        exponential_decay_with_algebraic_mass::<M>,
        exponential_decay_with_algebraic_mass_transpose::<M>,
        nstates,
        nstates,
        p.clone(),
    );
    if M::is_sparse() {
        rhs.calculate_jacobian_sparsity(&y0, t0);
        rhs.calculate_adjoint_sparsity(&y0, t0);
        mass.calculate_sparsity(t0);
        mass.calculate_adjoint_sparsity(t0);
    }
    let rhs = Rc::new(rhs);
    let init = Rc::new(init);
    let out = Some(Rc::new(out));

    let root: Option<Rc<UnitCallable<M>>> = None;
    let mass = Some(Rc::new(mass));
    let eqn = OdeSolverEquations::new(rhs, mass, root, init, out, p.clone());
    let rtol = M::T::from(1e-6);
    let atol = M::V::from_element(nstates, M::T::from(1e-6));
    let out_rtol = Some(M::T::from(1e-6));
    let out_atol = Some(M::V::from_element(nout, M::T::from(1e-6)));
    let param_rtol = Some(M::T::from(1e-6));
    let param_atol = Some(M::V::from_element(1, M::T::from(1e-6)));
    let sens_atol = Some(M::V::from_element(nstates, M::T::from(1e-6)));
    let sens_rtol = Some(M::T::from(1e-6));
    let integrate_out = true;
    let problem = OdeSolverProblem::new(
        eqn,
        rtol,
        atol,
        sens_rtol,
        sens_atol,
        out_rtol,
        out_atol,
        param_rtol,
        param_atol,
        t0,
        h0,
        integrate_out,
    )
    .unwrap();
    let atol_out = M::V::from_element(nout, M::T::from(1e-6));
    let mut soln = OdeSolverSolution {
        atol: atol_out,
        rtol: problem.rtol,
        ..Default::default()
    };
    let t0 = M::T::from(0.0);
    let t1 = M::T::from(9.0);
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0 = M::V::from_vec(vec![1.0.into(), 1.0.into(), 1.0.into()]);
        let g = y0.clone() * scale((M::T::exp(-p[0] * t0) - M::T::exp(-p[0] * t)) / p[0]);
        let g = M::V::from_vec(vec![p[0] * g[2]]);
        let dgdk = t1 * M::T::exp(-p[0] * t1);
        let dg = M::V::from_vec(vec![dgdk]);
        soln.push_sens(g, t, &[dg]);
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
    let mut mass = LinearClosure::new(exponential_decay_with_algebraic_mass::<M>, 3, 3, p.clone());
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
    let sens_rtol = Some(M::T::from(1e-6));
    let sens_atol = Some(M::V::from_element(3, M::T::from(1e-6)));
    let problem = OdeSolverProblem::new(
        eqn,
        M::T::from(1e-6),
        M::V::from_element(3, M::T::from(1e-6)),
        sens_rtol,
        sens_atol,
        None,
        None,
        None,
        None,
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
