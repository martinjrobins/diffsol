use diffsol_la::error::LaError;
use thiserror::Error;

/// Custom error type for Diffsol
///
/// This error type is used to wrap all possible errors that can occur when using Diffsol
#[derive(Error, Debug, Clone)]
pub enum DiffsolError {
    #[error("Linear algebra error: {0}")]
    LaError(#[from] LaError),
    #[error("Non-linear solver error: {0}")]
    NonLinearSolverError(#[from] NonLinearSolverError),
    #[error("ODE solver error: {0}")]
    OdeSolverError(#[from] OdeSolverError),
    #[error("DiffSL Parser error: {0}")]
    DiffslParserError(String),
    #[error("DiffSL Compiler error: {0}")]
    DiffslCompilerError(String),
    #[error("Error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving a non-linear problem
#[derive(Error, Debug, Clone)]
pub enum NonLinearSolverError {
    #[error("Initial condition solver did not converge")]
    InitialConditionDidNotConverge,
    #[error("Newton iterations did not converge, maximum iterations reached")]
    NewtonMaxIterations,
    #[error("Newton iteration diverged")]
    NewtonDiverged,
    #[error("Newton linesearch failed to find a suitable step in max iterations")]
    LinesearchFailedMaxIterations,
    #[error("Newton linesearch failed, minimum step size reached")]
    LinesearchFailedMinStep,
    #[error("LU solve failed")]
    LuSolveFailed,
    #[error("Jacobian not reset before calling solve")]
    JacobianNotReset,
    #[error("State has wrong length: expected {expected}, got {found}")]
    WrongStateLength { expected: usize, found: usize },
    #[error("Error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving an ODE
#[derive(Debug, Error, Clone)]
pub enum OdeSolverError {
    #[error(
        "Stop time = {} is less than current state time = {}",
        stop_time,
        state_time
    )]
    StopTimeBeforeCurrentTime { stop_time: f64, state_time: f64 },
    #[error("Mass matrix not supported for this solver")]
    MassMatrixNotSupported,
    #[error("Stop time is at the current state time")]
    StopTimeAtCurrentTime,
    #[error("Interpolation vector is not the correct length, expected {expected}, got {found}")]
    InterpolationVectorWrongSize { expected: usize, found: usize },
    #[error("Number of sensitivities does not match number of parameters")]
    SensitivityCountMismatch { expected: usize, found: usize },
    #[error("Interpolation time is after current time")]
    InterpolationTimeAfterCurrentTime,
    #[error("Interpolation time is not within the current step. Step size is zero after calling state_mut()")]
    InterpolationTimeOutsideCurrentStep,
    #[error("Interpolation time is greater than current time")]
    InterpolationTimeGreaterThanCurrentTime,
    #[error("State not set")]
    StateNotSet,
    #[error("Sensitivity solve failed")]
    SensitivitySolveFailed,
    #[error(
        "Exceeded maximum number of nonlinear solver failures ({num_failures}) at time = {time}"
    )]
    TooManyNonlinearSolverFailures { time: f64, num_failures: usize },
    #[error("Exceeded maximum number of error test failures ({num_failures}) at time = {time}")]
    TooManyErrorTestFailures { time: f64, num_failures: usize },
    #[error("Step size is too small at time = {time}")]
    StepSizeTooSmall { time: f64 },
    #[error("Sensitivity requested but equations do not support it")]
    SensitivityNotSupported,
    #[error("A reset operator requires a root operator to propagate sensitivities")]
    ResetRequiresRootOperator,
    #[error("Failed to get mutable reference to equations. If there is a solver created with this problem, call solver.take_state() to release the problem")]
    FailedToGetMutableReference,
    #[error("Builder error: {0}")]
    BuilderError(String),
    #[error("State is not consistent with the problem equations")]
    StateProblemMismatch,
    #[error("t_eval must be increasing and all values must be greater than or equal to the current time")]
    InvalidTEval,
    #[error("Sundials error: {0}")]
    SundialsError(String),
    #[error("Problem not set")]
    ProblemNotSet,
    #[error("Jacobian not available")]
    JacobianNotAvailable,
    #[error("Invalid Tableau: {0}")]
    InvalidTableau(String),
    #[error("Error: {0}")]
    Other(String),
}

#[macro_export]
macro_rules! non_linear_solver_error {
    ($variant:ident) => {
        $crate::DiffsolError::from($crate::error::NonLinearSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::DiffsolError::from($crate::error::NonLinearSolverError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! ode_solver_error {
    ($variant:ident) => {
        $crate::DiffsolError::from($crate::error::OdeSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::DiffsolError::from($crate::error::OdeSolverError::$variant($($arg)*.to_string()))
    };
}

#[macro_export]
macro_rules! other_error {
    ($msg:expr) => {
        $crate::DiffsolError::Other($msg.to_string())
    };
}
