use crate::{Matrix, OdeBuilder, OdeEquationsImplicit, OdeSolverProblem};
use num_traits::{FromPrimitive, One};

// du/dt = -u
// 0 = u - v - v^3
//
// From u = 2 the consistent state is v = 1, du = -2. The initial guess v = 5 is where
// dg/dv = -76 against -4 at the solution, so Newton on a jacobian frozen at the guess
// contracts by about 0.95 a step and cannot finish in the initial-condition budget.
fn nonlinear_algebraic_rhs<M: Matrix>(x: &[M::T], _p: &[M::T], _t: M::T, y: &mut [M::T]) {
    y[0] = -x[0];
    y[1] = x[0] - x[1] - x[1] * x[1] * x[1];
}

fn nonlinear_algebraic_jac_mul<M: Matrix>(
    x: &[M::T],
    _p: &[M::T],
    _t: M::T,
    v: &[M::T],
    y: &mut [M::T],
) {
    let three = M::T::from_f64(3.0).unwrap();
    y[0] = -v[0];
    y[1] = v[0] - (M::T::one() + three * x[1] * x[1]) * v[1];
}

fn nonlinear_algebraic_mass<M: Matrix>(
    x: &[M::T],
    _p: &[M::T],
    _t: M::T,
    beta: M::T,
    y: &mut [M::T],
) {
    y[0] = x[0] + beta * y[0];
    y[1] *= beta;
}

fn nonlinear_algebraic_init<M: Matrix>(_p: &[M::T], _t: M::T, y: &mut [M::T]) {
    y[0] = M::T::from_f64(2.0).unwrap();
    y[1] = M::T::from_f64(5.0).unwrap();
}

pub fn nonlinear_algebraic_problem<M: Matrix + 'static>(
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>> {
    OdeBuilder::<M>::new()
        .rhs_implicit(
            nonlinear_algebraic_rhs::<M>,
            nonlinear_algebraic_jac_mul::<M>,
        )
        .mass(nonlinear_algebraic_mass::<M>)
        .init(nonlinear_algebraic_init::<M>, 2)
        .build()
        .unwrap()
}
