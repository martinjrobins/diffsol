//! Matrix types and operations.
//!
//! This module defines the [`Matrix`] trait and related traits for matrix operations:
//! - [`DenseMatrix`] for dense column-major matrices
//! - [`MatrixView`] and [`MatrixViewMut`] for borrowed views
//! - Sparsity detection and handling
//!
//! Implementations are provided for:
//! - Dense matrices: [`NalgebraMat`](crate::NalgebraMat), [`FaerMat`](crate::FaerMat)
//! - Sparse matrices: [`FaerSparseMat`](crate::FaerSparseMat)
//! - GPU matrices: `CudaMat` (requires the `cuda` feature)

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::error::DiffsolError;
use crate::scalar::Scale;
use crate::vector::VectorHost;
use crate::{Context, IndexType, Scalar, Vector, VectorIndex};

use extract_block::combine;
use num_traits::{One, Zero};
use sparsity::{Dense, MatrixSparsity, MatrixSparsityRef};

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "nalgebra")]
pub mod dense_nalgebra_serial;

#[cfg(feature = "faer")]
pub mod dense_faer_serial;

#[cfg(feature = "faer")]
pub mod sparse_faer;

pub mod default_solver;
pub mod extract_block;
pub mod sparsity;

#[macro_use]
mod utils;

/// Common interface for matrix types, providing access to scalar type, context, and dimensions.
pub trait MatrixCommon: Sized + Debug {
    type V: Vector<T = Self::T, C = Self::C, Index: VectorIndex<C = Self::C>>;
    type T: Scalar;
    type C: Context;
    type Inner;

    /// Get the number of rows in this matrix.
    fn nrows(&self) -> IndexType;
    /// Get the number of columns in this matrix.
    fn ncols(&self) -> IndexType;
    /// Get a reference to the inner representation of the matrix.
    fn inner(&self) -> &Self::Inner;
}

impl<M> MatrixCommon for &M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;
    type C = M::C;
    type Inner = M::Inner;

    fn nrows(&self) -> IndexType {
        M::nrows(*self)
    }
    fn ncols(&self) -> IndexType {
        M::ncols(*self)
    }
    fn inner(&self) -> &Self::Inner {
        M::inner(*self)
    }
}

impl<M> MatrixCommon for &mut M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;
    type C = M::C;
    type Inner = M::Inner;

    fn ncols(&self) -> IndexType {
        M::ncols(*self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(*self)
    }
    fn inner(&self) -> &Self::Inner {
        M::inner(*self)
    }
}

/// Operations on matrices by value (addition and subtraction).
///
/// This trait defines matrix addition and subtraction when both operands are owned or references.
pub trait MatrixOpsByValue<Rhs = Self, Output = Self>:
    MatrixCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<M, Rhs, Output> MatrixOpsByValue<Rhs, Output> for M where
    M: MatrixCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

/// In-place operations on matrices (addition and subtraction).
///
/// This trait defines in-place matrix addition and subtraction (self += rhs, self -= rhs).
pub trait MatrixMutOpsByValue<Rhs = Self>: MatrixCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

impl<M, Rhs> MatrixMutOpsByValue<Rhs> for M where M: MatrixCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

/// A trait allowing for references to implement matrix operations
pub trait MatrixRef<M: MatrixCommon>: Mul<Scale<M::T>, Output = M> {}
impl<RefT, M: MatrixCommon> MatrixRef<M> for RefT where RefT: Mul<Scale<M::T>, Output = M> {}

/// A mutable view of a dense matrix, supporting in-place operations and modifications.
///
/// This trait represents a temporary mutable reference to a matrix's data, allowing in-place
/// arithmetic operations (+=, -=, *=) and matrix-matrix multiplication. Mutable views can be
/// created via the `columns_mut()` or `column_mut()` methods on a `DenseMatrix`.
pub trait MatrixViewMut<'a>:
    for<'b> MatrixMutOpsByValue<&'b Self>
    + for<'b> MatrixMutOpsByValue<&'b Self::View>
    + MulAssign<Scale<Self::T>>
{
    type Owned;
    type View;
    /// Convert this mutable view into an owned matrix, cloning the data if necessary.
    fn into_owned(self) -> Self::Owned;
    /// Perform matrix-matrix multiplication with owned matrices: self = alpha * a * b + beta * self
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T);
    /// Perform matrix-matrix multiplication with a view and owned matrix: self = alpha * a * b + beta * self
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T);
}

/// A borrowed immutable view of a dense matrix, supporting read-only arithmetic operations.
///
/// This trait represents a temporary immutable reference to a matrix's data, allowing read-only
/// operations like addition, subtraction, scalar multiplication, and matrix-vector multiplication.
/// Matrix views can be created via the `columns()` methods on a `DenseMatrix`.
pub trait MatrixView<'a>:
    for<'b> MatrixOpsByValue<&'b Self::Owned, Self::Owned> + Mul<Scale<Self::T>, Output = Self::Owned>
{
    type Owned;

    /// Convert this view into an owned matrix, cloning the data if necessary.
    fn into_owned(self) -> Self::Owned;

    /// Perform a matrix-vector multiplication with a vector view: y = alpha * self * x + beta * y
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    );

    /// Perform a matrix-vector multiplication with an owned vector: y = alpha * self * x + beta * y
    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);
}

/// A base matrix trait supporting both sparse and dense matrices.
///
/// This trait provides a complete interface for matrix operations including:
/// - Matrix creation and memory management
/// - Matrix-vector and matrix-matrix multiplication
/// - Element access and modification
/// - Sparsity information and handling
/// - Matrix decomposition and combination operations
/// - Triplet-based construction for sparse matrices
///
/// Implementing matrices can be dense or sparse, and may be hosted on CPU or GPU.
/// Users typically do not need to implement this trait; use provided implementations.
pub trait Matrix: MatrixCommon + Mul<Scale<Self::T>, Output = Self> + Clone + 'static {
    type Sparsity: MatrixSparsity<Self>;
    type SparsityRef<'a>: MatrixSparsityRef<'a, Self>
    where
        Self: 'a;

    /// Return sparsity information, or `None` if the matrix is dense.
    fn sparsity(&self) -> Option<Self::SparsityRef<'_>>;

    /// Get the context associated with this matrix (for device placement, memory management, etc.).
    fn context(&self) -> &Self::C;

    /// Returns true if this matrix is stored in a sparse format
    fn is_sparse() -> bool {
        Self::zeros(1, 1, Default::default()).sparsity().is_some()
    }

    /// Partition the diagonal indices into two groups: those with zero diagonal elements and those with non-zero diagonal elements.
    ///
    /// This is useful for identifying algebraic constraints, which typically have zero diagonal elements in the mass matrix.
    /// Returns a tuple of (zero_diagonal_indices, non_zero_diagonal_indices).
    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index);

    /// Perform a matrix-vector multiplication: y = alpha * self * x + beta * y
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);

    /// Copy the contents of `other` into this matrix.
    fn copy_from(&mut self, other: &Self);

    /// Create a new matrix of shape `nrows` x `ncols` filled with zeros.
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self;

    /// Create a new matrix from a sparsity pattern. Non-zero elements are not initialized.
    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self;

    /// Create a new diagonal matrix from a vector holding the diagonal elements.
    fn from_diagonal(v: &Self::V) -> Self;

    /// Set the values of column `j` to be equal to the values in `v`.
    ///
    /// For sparse matrices, only the existing non-zero elements are updated.
    fn set_column(&mut self, j: IndexType, v: &Self::V);

    /// Add a column of this matrix to a vector: v += self[:, j]
    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V);

    /// Assign the values in the `data` vector to this matrix at the indices in `dst_indices`
    /// from the indices in `src_indices`.
    ///
    /// For dense matrices, the index is the data index in column-major order.
    /// For sparse matrices, the index is the index into the data array.
    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    );

    /// Gather values from another matrix at specified indices into this matrix.
    ///
    /// For sparse matrices: the index `idx_i` in `indices` is an index into the data array for `other`,
    /// and is copied to the index `idx_i` in the data array for this matrix.
    /// For dense matrices: the index is the data index in column-major order.
    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index);

    /// Split this matrix into four submatrices based on algebraic constraint indices.
    ///
    /// Partitions the matrix into blocks:
    /// ```text
    /// M = [UL, UR]
    ///     [LL, LR]
    /// ```
    /// where:
    /// - UL contains rows and columns NOT in `algebraic_indices`
    /// - UR contains rows NOT in `algebraic_indices` and columns in `algebraic_indices`
    /// - LL contains rows in `algebraic_indices` and columns NOT in `algebraic_indices`
    /// - LR contains rows and columns in `algebraic_indices`
    ///
    /// Returns an array of tuples, where each tuple contains a submatrix and the indices that were used to create it.
    /// These indices can be used with `gather()` to update the submatrix.
    fn split(
        &self,
        algebraic_indices: &<Self::V as Vector>::Index,
    ) -> [(Self, <Self::V as Vector>::Index); 4] {
        match self.sparsity() {
            Some(sp) => sp.split(algebraic_indices).map(|(sp, src_indices)| {
                let mut m = Self::new_from_sparsity(
                    sp.nrows(),
                    sp.ncols(),
                    Some(sp),
                    self.context().clone(),
                );
                m.gather(self, &src_indices);
                (m, src_indices)
            }),
            None => Dense::<Self>::new(self.nrows(), self.ncols())
                .split(algebraic_indices)
                .map(|(sp, src_indices)| {
                    let mut m = Self::new_from_sparsity(
                        sp.nrows(),
                        sp.ncols(),
                        None,
                        self.context().clone(),
                    );
                    m.gather(self, &src_indices);
                    (m, src_indices)
                }),
        }
    }

    /// Combine four submatrices back into a single matrix based on algebraic constraint indices.
    ///
    /// Inverse operation of `split()`. Takes submatrices `ul`, `ur`, `ll`, `lr` and combines them
    /// back into the original matrix structure.
    fn combine(
        ul: &Self,
        ur: &Self,
        ll: &Self,
        lr: &Self,
        algebraic_indices: &<Self::V as Vector>::Index,
    ) -> Self {
        combine(ul, ur, ll, lr, algebraic_indices)
    }

    /// Perform the assignment: self = x + beta * y where x and y are matrices and beta is a scalar.
    ///
    /// Note: Panics if the sparsity patterns of self, x, and y do not match.
    /// The sparsity of self must be the union of the sparsity of x and y.
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self);

    /// Iterate over all non-zero elements in triplet format (row, column, value).
    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, Self::T)>;

    /// Create a new matrix from a vector of triplets (row, column, value).
    ///
    /// This is useful for sparse matrix construction. The sparsity pattern is inferred from the triplets.
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, Self::T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError>;
}

/// A host matrix is a matrix type whose vector type is hosted on the CPU.
///
/// This trait extends `Matrix` to ensure the associated vector type implements `VectorHost`,
/// enabling direct CPU-side access to data. GPU matrices typically do not implement this trait.
pub trait MatrixHost: Matrix<V: VectorHost> {}

impl<T: Matrix<V: VectorHost>> MatrixHost for T {}

/// A dense column-major matrix with efficient column access operations.
///
/// This trait represents matrices stored in column-major order, where accessing matrix columns
/// is efficient. It supports:
/// - Matrix views and mutable views
/// - Matrix-matrix multiplication (GEMM)
/// - Column operations (axpy, access, modification)
/// - Element access and modification
/// - Matrix resizing
///
/// The column-major layout makes operations on individual or ranges of columns very efficient.
pub trait DenseMatrix:
    Matrix
    + for<'b> MatrixOpsByValue<&'b Self, Self>
    + for<'b> MatrixMutOpsByValue<&'b Self>
    + for<'a, 'b> MatrixOpsByValue<&'b Self::View<'a>, Self>
    + for<'a, 'b> MatrixMutOpsByValue<&'b Self::View<'a>>
{
    /// A view of the dense matrix type
    type View<'a>: MatrixView<'a, Owned = Self, T = Self::T, V = Self::V>
    where
        Self: 'a;

    /// A mutable view of the dense matrix type
    type ViewMut<'a>: MatrixViewMut<
        'a,
        Owned = Self,
        T = Self::T,
        V = Self::V,
        View = Self::View<'a>,
    >
    where
        Self: 'a;

    /// Perform a matrix-matrix multiplication: self = alpha * a * b + beta * self
    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T);

    /// Perform a column AXPY operation: column i = alpha * column j + column i
    ///
    /// This is equivalent to: self[:, i] += alpha * self[:, j]
    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, i: IndexType);

    /// Get an immutable view of columns from `start` (inclusive) to `end` (exclusive).
    fn columns(&self, start: IndexType, end: IndexType) -> Self::View<'_>;

    /// Get an immutable vector view of column `i`.
    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_>;

    /// Get a mutable view of columns from `start` (inclusive) to `end` (exclusive).
    fn columns_mut(&mut self, start: IndexType, end: IndexType) -> Self::ViewMut<'_>;

    /// Get a mutable vector view of column `i`.
    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_>;

    /// Set the value at the given row and column indices.
    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T);

    /// Get the value at the given row and column indices.
    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T;

    /// Perform matrix-matrix multiplication using GEMM, allocating a new matrix for the result.
    fn mat_mul(&self, b: &Self) -> Self {
        let nrows = self.nrows();
        let ncols = b.ncols();
        let mut ret = Self::zeros(nrows, ncols, self.context().clone());
        ret.gemm(Self::T::one(), self, b, Self::T::zero());
        ret
    }

    /// Resize the number of columns in the matrix, preserving existing data.
    ///
    /// New elements (if added) are uninitialized. If the number of columns decreases, trailing columns are discarded.
    fn resize_cols(&mut self, ncols: IndexType);

    /// Create a new matrix from a vector of values in column-major order.
    ///
    /// The values are assumed to be stored in column-major order (first column, then second column, etc.).
    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self;
}

#[cfg(test)]
mod tests {
    use super::{DenseMatrix, Matrix};
    use crate::{scalar::IndexType, VectorIndex};
    use num_traits::{FromPrimitive, One, Zero};

    pub fn test_partition_indices_by_zero_diagonal<M: Matrix>() {
        let triplets = vec![
            (0, 0, M::T::one()),
            (1, 1, M::T::from_f64(2.0).unwrap()),
            (3, 3, M::T::one()),
        ];
        let m = M::try_from_triplets(4, 4, triplets, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(zero_diagonal_indices.clone_as_vec(), vec![2]);
        assert_eq!(non_zero_diagonal_indices.clone_as_vec(), vec![0, 1, 3]);

        let triplets = vec![
            (0, 0, M::T::one()),
            (1, 1, M::T::from_f64(2.0).unwrap()),
            (2, 2, M::T::zero()),
            (3, 3, M::T::one()),
        ];
        let m = M::try_from_triplets(4, 4, triplets, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(zero_diagonal_indices.clone_as_vec(), vec![2]);
        assert_eq!(non_zero_diagonal_indices.clone_as_vec(), vec![0, 1, 3]);

        let triplets = vec![
            (0, 0, M::T::one()),
            (1, 1, M::T::from_f64(2.0).unwrap()),
            (2, 2, M::T::from_f64(3.0).unwrap()),
            (3, 3, M::T::one()),
        ];
        let m = M::try_from_triplets(4, 4, triplets, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(
            zero_diagonal_indices.clone_as_vec(),
            Vec::<IndexType>::new()
        );
        assert_eq!(non_zero_diagonal_indices.clone_as_vec(), vec![0, 1, 2, 3]);
    }

    pub fn test_column_axpy<M: DenseMatrix>() {
        // M = [1 2]
        //     [3 4]
        let mut a = M::zeros(2, 2, Default::default());
        a.set_index(0, 0, M::T::one());
        a.set_index(0, 1, M::T::from_f64(2.0).unwrap());
        a.set_index(1, 0, M::T::from_f64(3.0).unwrap());
        a.set_index(1, 1, M::T::from_f64(4.0).unwrap());

        // op is M(:, 1) = 2 * M(:, 0) + M(:, 1)
        a.column_axpy(M::T::from_f64(2.0).unwrap(), 0, 1);
        // M = [1 4]
        //     [3 10]
        assert_eq!(a.get_index(0, 0), M::T::one());
        assert_eq!(a.get_index(0, 1), M::T::from_f64(4.0).unwrap());
        assert_eq!(a.get_index(1, 0), M::T::from_f64(3.0).unwrap());
        assert_eq!(a.get_index(1, 1), M::T::from_f64(10.0).unwrap());
    }

    pub fn test_resize_cols<M: DenseMatrix>() {
        let mut a = M::zeros(2, 2, Default::default());
        a.set_index(0, 0, M::T::one());
        a.set_index(0, 1, M::T::from_f64(2.0).unwrap());
        a.set_index(1, 0, M::T::from_f64(3.0).unwrap());
        a.set_index(1, 1, M::T::from_f64(4.0).unwrap());

        a.resize_cols(3);
        assert_eq!(a.ncols(), 3);
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.get_index(0, 0), M::T::one());
        assert_eq!(a.get_index(0, 1), M::T::from_f64(2.0).unwrap());
        assert_eq!(a.get_index(1, 0), M::T::from_f64(3.0).unwrap());
        assert_eq!(a.get_index(1, 1), M::T::from_f64(4.0).unwrap());

        a.set_index(0, 2, M::T::from_f64(5.0).unwrap());
        a.set_index(1, 2, M::T::from_f64(6.0).unwrap());
        assert_eq!(a.get_index(0, 2), M::T::from_f64(5.0).unwrap());
        assert_eq!(a.get_index(1, 2), M::T::from_f64(6.0).unwrap());

        a.resize_cols(2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.get_index(0, 0), M::T::one());
        assert_eq!(a.get_index(0, 1), M::T::from_f64(2.0).unwrap());
        assert_eq!(a.get_index(1, 0), M::T::from_f64(3.0).unwrap());
        assert_eq!(a.get_index(1, 1), M::T::from_f64(4.0).unwrap());
    }
}
