use crate::{Matrix, OdeBuilder, OdeEquationsImplicit, OdeSolverProblem};
use num_traits::{FromPrimitive, One, Pow};

// du/dt = -u + v - 1
// 0 = u - v - v^3
//
// From u = 2 the consistent state is v = 1, du = -2. The initial guess v = 5 is where
// dg/dv = -76 against -4 at the solution, so Newton on a jacobian frozen at the guess
// contracts by about 0.95 a step and cannot finish in the initial-condition budget.
fn nonlinear_algebraic_rhs<M: Matrix>(x: &[M::T], _p: &[M::T], _t: M::T, y: &mut [M::T]) {
    y[0] = -x[0] + x[1] - M::T::one();
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
    y[0] = -v[0] + v[1];
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

// as above, with mass M(t) = diag(t, 0)
//
// u only becomes differential once t > 0, so restarting at t = 2 from u = 2 gives the
// consistent state v = 1, du = -u / t = -1 only if the algebraic partition and the
// mass are taken at the state's time rather than t0 = 0.
fn nonlinear_algebraic_time_dependent_mass<M: Matrix>(
    x: &[M::T],
    _p: &[M::T],
    t: M::T,
    beta: M::T,
    y: &mut [M::T],
) {
    y[0] = t * x[0] + beta * y[0];
    y[1] *= beta;
}

pub fn nonlinear_algebraic_time_dependent_mass_problem<M: Matrix + 'static>(
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>> {
    OdeBuilder::<M>::new()
        .rhs_implicit(
            nonlinear_algebraic_rhs::<M>,
            nonlinear_algebraic_jac_mul::<M>,
        )
        .mass(nonlinear_algebraic_time_dependent_mass::<M>)
        .init(nonlinear_algebraic_init::<M>, 2)
        .build()
        .unwrap()
}

// du/dt = -u
// 0 = u - v^10
//
// From u = 1 the consistent state is v = 1. dg/dv grows from -0.0076 at the guess v = 0.45 to
// -10, so a jacobian frozen at the guess overshoots more than the line search can backtrack.
fn power_algebraic_rhs<M: Matrix>(x: &[M::T], _p: &[M::T], _t: M::T, y: &mut [M::T]) {
    y[0] = -x[0];
    y[1] = x[0] - x[1].pow(10i32);
}

fn power_algebraic_jac_mul<M: Matrix>(
    x: &[M::T],
    _p: &[M::T],
    _t: M::T,
    v: &[M::T],
    y: &mut [M::T],
) {
    let ten = M::T::from_f64(10.0).unwrap();
    y[0] = -v[0];
    y[1] = v[0] - ten * x[1].pow(9i32) * v[1];
}

fn power_algebraic_init<M: Matrix>(_p: &[M::T], _t: M::T, y: &mut [M::T]) {
    y[0] = M::T::one();
    y[1] = M::T::from_f64(0.45).unwrap();
}

pub fn power_algebraic_problem<M: Matrix + 'static>(
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>> {
    OdeBuilder::<M>::new()
        .rhs_implicit(power_algebraic_rhs::<M>, power_algebraic_jac_mul::<M>)
        .mass(nonlinear_algebraic_mass::<M>)
        .init(power_algebraic_init::<M>, 2)
        .build()
        .unwrap()
}
