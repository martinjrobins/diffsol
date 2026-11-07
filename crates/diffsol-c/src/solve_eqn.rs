//! Hooks for each equation type that [`GenericSolve`](crate::solve::GenericSolve) uses.
//!
//! With these hooks, the same runtime API can solve DiffSL equations and other equation types,
//! for example [`OdeCallbacks`](crate::ode_callbacks::OdeCallbacks).

use crate::{error::DiffsolRtError, ode_solver_type::OdeSolverType};
use diffsol::error::DiffsolError;

/// What a solve needs from the equations. See [`SolveEqn::check_requirement`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EqnRequirement {
    /// A plain forward solve with the given method.
    Solve(OdeSolverType),
    /// A forward sensitivity solve.
    FwdSens,
    /// A forward pass followed by an adjoint backward pass, both with the given method.
    Adjoint(OdeSolverType),
}

/// The hooks for one equation type.
pub(crate) trait SolveEqn {
    /// Serialize the equations, so that the solver can be made again later.
    fn serialized(&self) -> Result<Vec<u8>, DiffsolRtError> {
        unsupported("ODE serialization is only supported for JIT-backed solvers")
    }

    /// Before a solve, check that the equations give all that `requirement` needs.
    fn check_requirement(&self, _requirement: EqnRequirement) -> Result<(), DiffsolRtError> {
        Ok(())
    }

    /// Take and clear the message of the error that the equations recorded during evaluation,
    /// if any.
    fn take_error(&self) -> Option<String> {
        None
    }
}

pub(crate) fn unsupported(message: &str) -> Result<Vec<u8>, DiffsolRtError> {
    Err(DiffsolError::Other(message.to_string()).into())
}
