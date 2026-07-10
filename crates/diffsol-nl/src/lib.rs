//! # diffsol-nl
//!
//! Time-unaware non-linear solver traits and implementations for
//! [diffsol](https://github.com/martinjrobins/diffsol).
//!
//! This crate provides the non-linear operator and solver abstractions used by
//! diffsol, together with a Newton implementation built on top of the linear
//! solvers provided by [`diffsol_la`].

/// Convergence testing for iterative non-linear solvers.
///
/// This module defines the [Convergence] struct and [ConvergenceStatus] enum used
/// to test for convergence (and divergence) of the Newton iteration.
pub mod convergence;

/// Error types and handling.
///
/// This module defines the [NlError] enum for the non-linear solver layer, which
/// wraps solver-specific errors ([NonLinearSolverError]) as well as linear-algebra
/// errors from [`diffsol_la`].
pub mod error;

/// Line search implementations for globalising the Newton iteration.
///
/// This module defines the [LineSearch] trait and provides the [NoLineSearch] and
/// [BacktrackingLineSearch] implementations.
pub mod line_search;

/// The Newton non-linear solver.
///
/// This module provides [NewtonNonlinearSolver], an implementation of the
/// [NonLinearSolver] trait using Newton's method, as well as the standalone
/// [newton_iteration] function.
pub mod newton;

/// The [NonLinearOp] and [NonLinearOpJacobian] traits describing a time-unaware
/// non-linear operator `F(x)` and its Jacobian `J(x)`.
pub mod nonlinear_op;

/// Non-linear solver traits.
///
/// This module defines the [NonLinearSolver] trait for solving `F(x) = 0`.
pub mod nonlinear_solver;

pub use convergence::{Convergence, ConvergenceStatus};
pub use error::{NlError, NonLinearSolverError};
pub use line_search::{BacktrackingLineSearch, LineSearch, NoLineSearch};
pub use newton::{newton_iteration, NewtonNonlinearSolver};
pub use nonlinear_op::{NonLinearOp, NonLinearOpJacobian};
pub use nonlinear_solver::{NonLinearSolveSolution, NonLinearSolver};
