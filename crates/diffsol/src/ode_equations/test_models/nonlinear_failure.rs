use crate::{Matrix, OdeBuilder, OdeEquationsImplicit, OdeSolverProblem, Scalar};
use num_traits::One;

/// A NaN right-hand side forces every nonlinear solve to fail.
pub fn nonlinear_failure_problem<M: Matrix + 'static>(
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>> {
    OdeBuilder::<M>::new()
        .rhs_implicit(
            |_y, _p, _t, dy| dy[0] = M::T::NAN,
            |_y, _p, _t, v, jv| jv[0] = -v[0],
        )
        .init(|_p, _t, y| y[0] = M::T::one(), 1)
        .build()
        .unwrap()
}
