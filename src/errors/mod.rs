use thiserror::Error;

#[derive(Error, Debug)]
pub enum PSError {
    #[error("Sensitivity solve failed")]
    SensitivityError,
    #[error("Tstop ({}) is less than current time t {}", tstop, t)]
    StopBeforeCurrentTime { tstop: f64, t: f64 },
    #[error("Interpolation time is after current time")]
    InterpolationBeforeCurrentTime,
    #[error("Interpolation time is not within the current step. Step size is zero after calling state_mut()")]
    InterpolationOutsideCurrentStep,
    #[error("State not set")]
    StateNotSet,
    #[error("Absolute tolerance must be of length 1 or the same length as the state vector")]
    AbsoluteToleranceLengthMismatch,
    #[error("Step size too small at t = {}", t)]
    StepSizeTooSmall { t: f64 },
    #[error("LU not initialized")]
    LuNotInitialized,
    #[error("LU solve failed")]
    LuFailed,
    #[error("Error: {}", e)]
    Other { e: String },
    #[error("Maximum number of iterations reached, solver did not converge.")]
    MaxIterReached,
    #[error("Maximum number of iterations reached, solver did not converge.")]
    LinearPSError,
    #[error("Failed to get mutable reference to equations, is there a solver created with this problem?")]
    MutableReferenceError,
    #[error("Sensitivity requested but equations do not support it")]
    SensitivityNotSupported,
    #[error("LU not initialized")]
    UnionShapeMismatch,
    #[error("Failed to create sparsity pattern: {}", e)]
    SparsityPatternError { e: String },
    #[error("Unknown error: {}", e)]
    Unknown { e: String },
    #[error("Cannot create a matrix with zero rows or columns")]
    ZeroColRow,
}
