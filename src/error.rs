use thiserror::Error;
use faer::sparse::CreationError as CreationError;

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
    #[error("template")]
    Template,
    #[error("LU not initialized")]
    LuNotInitialized,
    #[error("LU solve failed")]
    LuSolveFailed,
}

/// Possible errors that can occur when solving a non-linear problem
#[derive(Error, Debug)]
pub enum NonLinearSolverError {
    #[error("template")]
    Template,
    #[error("Newton did not converge")]
    NewtonDidNotConverge,
    #[error("LU solve failed")]
    LuSolveFailed,
}

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
}


/// Possible errors that can occur when solving a non-linear problem
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