use faer::sparse::CreationError;
use thiserror::Error;

/// Error type for the diffsol linear algebra crate (`diffsol-la`).
///
/// This wraps the errors that can occur in the linear algebra layer: matrix
/// operations, linear solvers, and (optionally) CUDA. It is re-exported by the
/// `diffsol` crate and can be converted into `diffsol`'s top-level error type.
#[derive(Error, Debug, Clone)]
pub enum LaError {
    #[error("Linear solver error: {0}")]
    LinearSolverError(#[from] LinearSolverError),
    #[error("Matrix error: {0}")]
    MatrixError(#[from] MatrixError),
    #[cfg(feature = "cuda")]
    #[error("Cuda error: {0}")]
    CudaError(#[from] CudaError),
    #[error("Error: {0}")]
    Other(String),
}

/// Possible errors that can occur when solving a linear problem
#[derive(Error, Debug, Clone)]
pub enum LinearSolverError {
    #[error("LU not initialized")]
    LuNotInitialized,
    #[error("LU solve failed")]
    LuSolveFailed,
    #[error("Linear solver not setup")]
    LinearSolverNotSetup,
    #[error("Linear solver matrix not square")]
    LinearSolverMatrixNotSquare,
    #[error("Linear solver matrix not compatible with vector")]
    LinearSolverMatrixVectorNotCompatible,
    #[error("KLU failed to analyze")]
    KluFailedToAnalyze,
    #[error("KLU failed to factorize")]
    KluFailedToFactorize,
    #[error("Error: {0}")]
    Other(String),
}

/// Possible errors for matrix operations
#[derive(Error, Debug, Clone)]
pub enum MatrixError {
    #[error("Failed to create matrix from triplets: {0}")]
    FailedToCreateMatrixFromTriplets(#[from] CreationError),
    #[error("Cannot union matrices with different shapes")]
    UnionIncompatibleShapes,
    #[error("Cannot create a matrix with zero rows or columns")]
    MatrixShapeError,
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Error: {0}")]
    Other(String),
}

#[cfg(feature = "cuda")]
#[derive(Error, Debug, Clone)]
pub enum CudaError {
    #[error("Failed to allocate memory on GPU")]
    CudaMemoryAllocationError,
    #[error("Failed to initialize CUDA: {0}")]
    CudaInitializationError(String),
    #[error("Cuda error: {0}")]
    Other(String),
}

#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! cuda_error {
    ($variant:ident) => {
        $crate::error::LaError::from($crate::error::CudaError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::LaError::from($crate::error::CudaError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! linear_solver_error {
    ($variant:ident) => {
        $crate::error::LaError::from($crate::error::LinearSolverError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::LaError::from($crate::error::LinearSolverError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! matrix_error {
    ($variant:ident) => {
        $crate::error::LaError::from($crate::error::MatrixError::$variant)
    };
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::LaError::from($crate::error::MatrixError::$variant($($arg)*))
    };
}

#[macro_export]
macro_rules! la_other_error {
    ($msg:expr) => {
        $crate::error::LaError::Other($msg.to_string())
    };
}
