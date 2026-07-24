//! # diffsol-la
//!
//! Linear algebra traits and backends for [diffsol](https://github.com/martinjrobins/diffsol).
//!
//! This crate provides the vector, matrix, and linear-solver abstractions used by
//! diffsol, together with concrete implementations backed by
//! [nalgebra](https://nalgebra.org), [faer](https://github.com/sarah-ek/faer-rs),
//! suitesparse (KLU), and CUDA.

/// Context objects for managing device resources for vectors and matrices (e.g. device streams, threading pools, etc.).
///
/// This module provides context types that encapsulate information about where data is stored and computed
/// (CPU, GPU, etc.). Different backends like nalgebra and faer may require different context implementations.
/// The [Context] trait defines the interface that must be implemented.
pub mod context;

/// Error types and handling.
///
/// This module defines the [LaError] enum and specialized error variants for
/// the linear algebra layer, including matrix, linear-solver, and (optional)
/// CUDA errors.
pub mod error;

/// The [LinearOp] trait describing a linear operator `A` for use with [LinearSolver].
pub mod linear_op;

/// Linear solver implementations and traits.
///
/// This module defines the [LinearSolver] trait for solving linear systems and provides implementations:
/// - Direct solvers: [NalgebraLU], [FaerLU], [FaerSparseLU]
/// - Optional sparse solvers: `KLU` (requires `suitesparse` feature)
/// - GPU solvers: `CudaLU` (requires `cuda` feature)
///
/// The linear solver is a critical component used internally by nonlinear solvers to solve Newton systems.
pub mod linear_solver;

/// Matrix types and operations.
///
/// This module defines the [Matrix] trait and related traits for matrix operations:
/// - [DenseMatrix] for dense column-major matrices
/// - [MatrixView] and [MatrixViewMut] for borrowed views
/// - Sparsity detection and handling
///
/// Implementations are provided for:
/// - Dense matrices: [NalgebraMat], [FaerMat]
/// - Sparse matrices: [FaerSparseMat]
/// - GPU matrices: `CudaMat` (requires `cuda` feature)
pub mod matrix;

/// Scalar types and traits.
///
/// This module defines the [Scalar] trait that all floating-point types used in DiffSol must implement.
/// It aggregates requirements from nalgebra, faer, and num_traits to ensure compatibility with linear algebra operations.
///
/// Implementations are provided for `f32` and `f64`.
/// GPU scalar types are available via `ScalarCuda` (requires `cuda` feature).
pub mod scalar;

/// Vector types and traits.
///
/// This module defines the [Vector] trait and related traits for vector operations:
/// - [VectorView] and [VectorViewMut] for borrowed views
/// - [VectorIndex] for index collections
/// - [VectorHost] for CPU-resident vectors with direct access
///
/// Implementations are provided for:
/// - [NalgebraVec] using nalgebra vectors
/// - [FaerVec] using faer vectors
/// - `CudaVec` for GPU computation (requires `cuda` feature)
pub mod vector;

pub use error::LaError;

pub use context::{faer::FaerContext, nalgebra::NalgebraContext, Context};

pub use linear_op::LinearOp;
pub use linear_solver::LinearSolver;
pub use linear_solver::{faer::sparse_lu::FaerSparseLU, FaerLU, NalgebraLU};

#[cfg(feature = "suitesparse")]
pub use linear_solver::suitesparse::klu::KLU;

pub use matrix::{
    default_solver::DefaultSolver, dense_faer_serial::FaerMat, dense_nalgebra_serial::NalgebraMat,
    sparse_faer::FaerSparseMat, DenseMatrix, Matrix, MatrixCommon,
};

pub(crate) use matrix::extract_block::ColMajBlock;
pub use matrix::{
    sparsity::Dense, sparsity::DenseRef, sparsity::MatrixSparsity, sparsity::MatrixSparsityRef,
    MatrixHost, MatrixRef, MatrixView, MatrixViewMut,
};

pub use scalar::{scale, FaerScalar, IndexType, NalgebraScalar, Scalar, Scale};

pub use vector::DefaultDenseMatrix;
pub use vector::{
    faer_serial::{FaerVec, FaerVecIndex, FaerVecMut, FaerVecRef},
    nalgebra_serial::{NalgebraVec, NalgebraVecMut, NalgebraVecRef},
    Vector, VectorCommon, VectorHost, VectorIndex, VectorRef, VectorView, VectorViewMut,
};

#[cfg(feature = "cuda")]
pub use context::cuda::CudaContext;
#[cfg(feature = "cuda")]
pub use linear_solver::cuda::lu::CudaLU;
#[cfg(feature = "cuda")]
pub use matrix::cuda::{CudaMat, CudaMatMut, CudaMatRef};
#[cfg(feature = "cuda")]
pub use scalar::cuda::{CudaType, ScalarCuda};
#[cfg(feature = "cuda")]
pub use vector::cuda::{CudaIndex, CudaVec, CudaVecMut, CudaVecRef};
