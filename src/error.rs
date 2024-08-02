use faer::sparse::CreationError;
use thiserror::Error;

/// Custom error type for Diffsol
///
/// This error type is used to wrap all possible errors that can occur when using Diffsol
#[derive(thiserror::Error, Debug)]
pub enum DiffsolError {
    #[error("Linear solver error: {0}")]
    LinearSolverError(#[from] LinearSolverError),
    #[error("Non-linear solver error: {0}")]
    NonLinearSolverError(#[from] NonLinearSolverError),
    #[error("ODE solver error: {0}")]
    OdeSolverError(#[from] OdeSolverError),
    #[error("Matrix error: {0}")]
    MatrixError(#[from] MatrixError),
    #[error("Other error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving a linear problem
#[derive(Error, Debug)]
pub enum LinearSolverError {
    #[error("LU not initialized")]
    LuNotInitialized,
    #[error("LU solve failed")]
    LuSolveFailed,
    #[error("Other error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving a non-linear problem
#[derive(Error, Debug)]
pub enum NonLinearSolverError {
    #[error("Newton did not converge")]
    NewtonDidNotConverge,
    #[error("LU solve failed")]
    LuSolveFailed,
    #[error("Other error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving an ODE
#[derive(Debug, Error)]
pub enum OdeSolverError {
    #[error(
        "Stop time = {} is less than current state time = {}",
        stop_time,
        state_time
    )]
    StopTimeBeforeCurrentTime { stop_time: f64, state_time: f64 },
    #[error("Interpolation time is after current time")]
    InterpolationTimeAfterCurrentTime,
    #[error("Interpolation time is not within the current step. Step size is zero after calling state_mut()")]
    InterpolationTimeOutsideCurrentStep,
    #[error("State not set")]
    StateNotSet,
    #[error("Sensitivity solve failed")]
    SensitivitySolveFailed,
    #[error("Step size is too small at time = {time}")]
    StepSizeTooSmall { time: f64 },
    #[error("Sensitivity requested but equations do not support it")]
    SensitivityNotSupported,
    #[error("Failed to get mutable reference to equations, is there a solver created with this problem?")]
    FailedToGetMutableReference,
    #[error("atol must have length 1 or equal to the number of states")]
    AtolLengthMismatch,
    #[error("t_eval must be increasing and all values must be greater than or equal to the current time")]
    InvalidTEval,
    #[error("Other error: {0}")]
    Other(String),
}

/// Possible errors for matrix operations
#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("Failed to create matrix from triplets: {0}")]
    FailedToCreateMatrixFromTriplets(#[from] CreationError),
    #[error("Cannot union matrices with different shapes")]
    UnionIncompatibleShapes,
    #[error("Cannot create a matrix with zero rows or columns")]
    MatrixShapeError,
    #[error("Other error: {0}")]
    Other(String),
}

#[macro_export]
macro_rules! linear_solver_error {
    ($variant:ident) => {
        DiffsolError::from(LinearSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        DiffsolError::from(LinearSolverError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! non_linear_solver_error {
    ($variant:ident) => {
        DiffsolError::from(NonLinearSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        DiffsolError::from(NonLinearSolverError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! ode_solver_error {
    ($variant:ident) => {
        DiffsolError::from(OdeSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        DiffsolError::from(OdeSolverError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! matrix_error {
    ($variant:ident) => {
        DiffsolError::from(MatrixError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        DiffsolError::from(MatrixError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! other_error {
    ($msg:expr) => {
        DiffsolError::Other($msg.to_string())
    };
}
