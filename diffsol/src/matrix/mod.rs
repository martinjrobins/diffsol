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
pub trait Matrix:
    MatrixCommon + Mul<Scale<Self::T>, Output = Self> + Clone + Send + 'static
{
    type Sparsity: MatrixSparsity<Self>;
    type SparsityRef<'a>: MatrixSparsityRef<'a, Self>
    where
        Self: 'a;

    /// Return sparsity information, or `None` if the matrix is dense.
    fn sparsity(&self) -> Option<Self::SparsityRef<'_>>;

    /// Get the context associated with this matrix (for device placement, memory management, etc.).
    fn context(&self) -> &Self::C;

    /// Get a mutable reference to the inner representation of the matrix.
    fn inner_mut(&mut self) -> &mut Self::Inner;

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

    /// Iterate over structural positions and values of the matrix.
    ///
    /// Returns a tuple:
    /// - First iterator: `(row, col)` pairs for each non-zero element (length `nnz`)
    /// - Second iterator: values (length `nnz * nbatch`), laid out batch-contiguously:
    ///   `[batch0_val0..batch0_valN, batch1_val0..batch1_valN, ...]`
    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    );

    /// Create a new matrix from structural indices and values.
    ///
    /// - `indices`: `(row, col)` pairs for each non-zero element (length `nnz`)
    /// - `values`: values laid out batch-contiguously (length `nnz * ctx.nbatch()`):
    ///   `[batch0_val0..batch0_valN, batch1_val0..batch1_valN, ...]`
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
        values: Vec<Self::T>,
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
pub(crate) mod tests {
    use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
    use crate::scalar::Scale;
    use crate::{scalar::IndexType, Context, Vector, VectorIndex};
    use num_traits::{FromPrimitive, One, Zero};

    fn f<M: Matrix>(x: f64) -> M::T {
        M::T::from_f64(x).unwrap()
    }

    fn triplet_values<M: Matrix>(m: &M) -> Vec<M::T> {
        let (_, vals) = m.triplet_iter();
        vals.collect()
    }

    fn triplet_indices<M: Matrix>(m: &M) -> Vec<(IndexType, IndexType)> {
        let (idx, _) = m.triplet_iter();
        idx.collect()
    }

    pub fn test_partition_indices_by_zero_diagonal<M: Matrix>() {
        let indices = vec![(0, 0), (1, 1), (3, 3)];
        let values = vec![M::T::one(), M::T::from_f64(2.0).unwrap(), M::T::one()];
        let m = M::try_from_triplets(4, 4, indices, values, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(zero_diagonal_indices.clone_as_vec(), vec![2]);
        assert_eq!(non_zero_diagonal_indices.clone_as_vec(), vec![0, 1, 3]);

        let indices = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let values = vec![
            M::T::one(),
            M::T::from_f64(2.0).unwrap(),
            M::T::zero(),
            M::T::one(),
        ];
        let m = M::try_from_triplets(4, 4, indices, values, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(zero_diagonal_indices.clone_as_vec(), vec![2]);
        assert_eq!(non_zero_diagonal_indices.clone_as_vec(), vec![0, 1, 3]);

        let indices = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let values = vec![
            M::T::one(),
            M::T::from_f64(2.0).unwrap(),
            M::T::from_f64(3.0).unwrap(),
            M::T::one(),
        ];
        let m = M::try_from_triplets(4, 4, indices, values, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(
            zero_diagonal_indices.clone_as_vec(),
            Vec::<IndexType>::new()
        );
        assert_eq!(non_zero_diagonal_indices.clone_as_vec(), vec![0, 1, 2, 3]);
    }

    // --- Matrix-generic tests (work with both dense and sparse) ---

    pub fn test_zeros<M: Matrix>() {
        let a = M::zeros(2, 3, Default::default());
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 3);
        let vals = triplet_values(&a);
        assert!(vals.is_empty() || vals.iter().all(|v| v.is_zero()));
    }

    pub fn test_from_diagonal<M: Matrix>() {
        let v = M::V::from_vec(
            vec![f::<M>(2.0), f::<M>(3.0), f::<M>(5.0)],
            Default::default(),
        );
        let a = M::from_diagonal(&v);
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);
        let idx = triplet_indices(&a);
        let vals = triplet_values(&a);
        // diagonal matrix triplet_iter returns only the diagonal nnz entries
        for &(i, j) in &idx {
            let pos = idx.iter().position(|&x| x == (i, j)).unwrap();
            if i == j {
                assert!(
                    vals[pos] != M::T::zero(),
                    "diagonal entry should be non-zero"
                );
            } else {
                assert!(vals[pos].is_zero(), "off-diagonal entry should be zero");
            }
        }
    }

    pub fn test_from_diagonal_dense<M: DenseMatrix>() {
        let v = M::V::from_vec(
            vec![f::<M>(2.0), f::<M>(3.0), f::<M>(5.0)],
            Default::default(),
        );
        let a = M::from_diagonal(&v);
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);
        assert_eq!(a.get_index(0, 0), f::<M>(2.0));
        assert_eq!(a.get_index(1, 1), f::<M>(3.0));
        assert_eq!(a.get_index(2, 2), f::<M>(5.0));
        assert_eq!(a.get_index(0, 1), f::<M>(0.0));
        assert_eq!(a.get_index(1, 0), f::<M>(0.0));
    }

    pub fn test_gemv<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)];
        let a = M::try_from_triplets(2, 2, indices, values, Default::default()).unwrap();
        let x = M::V::from_vec(vec![f::<M>(1.0), f::<M>(2.0)], Default::default());
        let mut y = M::V::zeros(2, Default::default());
        a.gemv(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(5.0), f::<M>(11.0)]);
    }

    pub fn test_set_column<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(0.0), f::<M>(0.0), f::<M>(0.0), f::<M>(0.0)];
        let mut a = M::try_from_triplets(2, 2, indices, values, Default::default()).unwrap();
        let v = M::V::from_vec(vec![f::<M>(7.0), f::<M>(8.0)], Default::default());
        a.set_column(1, &v);
        let idx = triplet_indices(&a);
        let vals = triplet_values(&a);
        assert_eq!(idx, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        assert_eq!(
            vals,
            vec![f::<M>(0.0), f::<M>(0.0), f::<M>(7.0), f::<M>(8.0)]
        );
    }

    pub fn test_copy_from<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let a = M::try_from_triplets(2, 2, indices, values, Default::default()).unwrap();
        let mut b = M::zeros(2, 2, Default::default());
        b.copy_from(&a);
        let vals = triplet_values(&b);
        assert_eq!(
            vals,
            vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)]
        );
    }

    pub fn test_scale_add_and_assign<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let x_vals = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let y_vals = vec![f::<M>(10.0), f::<M>(20.0), f::<M>(30.0), f::<M>(40.0)];
        let x = M::try_from_triplets(2, 2, indices.clone(), x_vals, Default::default()).unwrap();
        let y = M::try_from_triplets(2, 2, indices, y_vals, Default::default()).unwrap();
        let mut result = M::zeros(2, 2, Default::default());
        result.copy_from(&x);
        result.scale_add_and_assign(&x, f::<M>(2.0), &y);
        let vals = triplet_values(&result);
        assert_eq!(
            vals,
            vec![f::<M>(21.0), f::<M>(42.0), f::<M>(63.0), f::<M>(84.0)]
        );
    }

    // --- DenseMatrix-specific tests ---

    pub fn test_column_axpy<M: DenseMatrix>() {
        let mut a = M::zeros(2, 2, Default::default());
        a.set_index(0, 0, M::T::one());
        a.set_index(0, 1, M::T::from_f64(2.0).unwrap());
        a.set_index(1, 0, M::T::from_f64(3.0).unwrap());
        a.set_index(1, 1, M::T::from_f64(4.0).unwrap());

        a.column_axpy(M::T::from_f64(2.0).unwrap(), 0, 1);
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

    pub fn test_from_vec<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.get_index(0, 0), f::<M>(1.0));
        assert_eq!(a.get_index(1, 0), f::<M>(3.0));
        assert_eq!(a.get_index(0, 1), f::<M>(2.0));
        assert_eq!(a.get_index(1, 1), f::<M>(4.0));
    }

    pub fn test_gemm<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(2.0), f::<M>(1.0), f::<M>(0.0), f::<M>(3.0)],
            Default::default(),
        );
        let mut c = M::zeros(2, 2, Default::default());
        c.gemm(f::<M>(1.0), &a, &b, f::<M>(0.0));
        assert_eq!(c.get_index(0, 0), f::<M>(4.0));
        assert_eq!(c.get_index(1, 0), f::<M>(10.0));
        assert_eq!(c.get_index(0, 1), f::<M>(6.0));
        assert_eq!(c.get_index(1, 1), f::<M>(12.0));
    }

    pub fn test_mat_mul<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(2.0), f::<M>(1.0), f::<M>(0.0), f::<M>(3.0)],
            Default::default(),
        );
        let c = a.mat_mul(&b);
        assert_eq!(c.get_index(0, 0), f::<M>(4.0));
        assert_eq!(c.get_index(1, 0), f::<M>(10.0));
        assert_eq!(c.get_index(0, 1), f::<M>(6.0));
        assert_eq!(c.get_index(1, 1), f::<M>(12.0));
    }

    pub fn test_columns_view<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
            ],
            Default::default(),
        );
        let view = a.columns(0, 2);
        assert_eq!(view.ncols(), 2);
        assert_eq!(view.nrows(), 2);
        let owned = view.into_owned();
        assert_eq!(owned.get_index(0, 0), f::<M>(1.0));
        assert_eq!(owned.get_index(1, 0), f::<M>(4.0));
        assert_eq!(owned.get_index(0, 1), f::<M>(2.0));
        assert_eq!(owned.get_index(1, 1), f::<M>(5.0));
    }

    pub fn test_column_view<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let col = a.column(1);
        use crate::VectorView;
        assert_eq!(col.get_index(0), f::<M>(2.0));
        assert_eq!(col.get_index(1), f::<M>(4.0));
    }

    // --- Batched Matrix-generic tests ---

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_zeros_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let a = M::zeros(2, 3, ctx);
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 3);
        let vals = triplet_values(&a);
        assert!(vals.is_empty() || vals.iter().all(|v| v.is_zero()));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![
            f::<M>(1.0),
            f::<M>(3.0),
            f::<M>(2.0),
            f::<M>(4.0), // batch 0
            f::<M>(5.0),
            f::<M>(7.0),
            f::<M>(6.0),
            f::<M>(8.0), // batch 1
        ];
        let a = M::try_from_triplets(2, 2, indices, values, ctx.clone()).unwrap();
        let x = M::V::from_vec(
            vec![f::<M>(1.0), f::<M>(2.0), f::<M>(1.0), f::<M>(1.0)],
            ctx.clone(),
        );
        let mut y = M::V::zeros(2, ctx);
        a.gemv(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(5.0), f::<M>(11.0), f::<M>(11.0), f::<M>(15.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_broadcast_x_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![
            f::<M>(1.0),
            f::<M>(3.0),
            f::<M>(2.0),
            f::<M>(4.0),
            f::<M>(5.0),
            f::<M>(7.0),
            f::<M>(6.0),
            f::<M>(8.0),
        ];
        let a = M::try_from_triplets(2, 2, indices, values, ctx.clone()).unwrap();
        let x = M::V::from_vec(vec![f::<M>(1.0), f::<M>(2.0)], Default::default());
        let mut y = M::V::zeros(2, ctx);
        a.gemv(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(5.0), f::<M>(11.0), f::<M>(17.0), f::<M>(23.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_broadcast_mat_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)];
        let a =
            M::try_from_triplets(2, 2, indices, values, ctx.clone_with_nbatch(1).unwrap()).unwrap();
        let x = M::V::from_vec(
            vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)],
            ctx.clone(),
        );
        let mut y = M::V::zeros(2, ctx);
        a.gemv(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(5.0), f::<M>(11.0), f::<M>(11.0), f::<M>(25.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_from_diagonal_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = M::V::from_vec(
            vec![f::<M>(2.0), f::<M>(3.0), f::<M>(4.0), f::<M>(5.0)],
            ctx,
        );
        let a = M::from_diagonal(&v);
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        let idx = triplet_indices(&a);
        let vals = triplet_values(&a);
        for &(i, j) in &idx {
            let pos = idx.iter().position(|&x| x == (i, j)).unwrap();
            if i == j {
                assert!(
                    vals[pos] != M::T::zero(),
                    "diagonal entry should be non-zero"
                );
            } else {
                assert!(vals[pos].is_zero(), "off-diagonal entry should be zero");
            }
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_copy_from_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![
            f::<M>(1.0),
            f::<M>(2.0),
            f::<M>(3.0),
            f::<M>(4.0),
            f::<M>(5.0),
            f::<M>(6.0),
            f::<M>(7.0),
            f::<M>(8.0),
        ];
        let a = M::try_from_triplets(2, 2, indices, values, ctx.clone()).unwrap();
        let mut b = M::zeros(2, 2, ctx);
        b.copy_from(&a);
        let vals = triplet_values(&b);
        assert_eq!(
            vals,
            vec![
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(3.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(8.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_set_column_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
        ];
        let mut a = M::try_from_triplets(2, 2, indices, values, ctx.clone()).unwrap();
        let v = M::V::from_vec(
            vec![f::<M>(5.0), f::<M>(6.0), f::<M>(7.0), f::<M>(8.0)],
            ctx,
        );
        a.set_column(0, &v);
        let vals = triplet_values(&a);
        assert_eq!(
            vals,
            vec![
                f::<M>(5.0),
                f::<M>(6.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(7.0),
                f::<M>(8.0),
                f::<M>(0.0),
                f::<M>(0.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_scale_add_and_assign_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let x_vals = vec![
            f::<M>(1.0),
            f::<M>(2.0),
            f::<M>(3.0),
            f::<M>(4.0),
            f::<M>(5.0),
            f::<M>(6.0),
            f::<M>(7.0),
            f::<M>(8.0),
        ];
        let y_vals = vec![
            f::<M>(10.0),
            f::<M>(20.0),
            f::<M>(30.0),
            f::<M>(40.0),
            f::<M>(50.0),
            f::<M>(60.0),
            f::<M>(70.0),
            f::<M>(80.0),
        ];
        let x = M::try_from_triplets(2, 2, indices.clone(), x_vals, ctx.clone()).unwrap();
        let y = M::try_from_triplets(2, 2, indices, y_vals, ctx.clone()).unwrap();
        let mut result = M::zeros(2, 2, ctx);
        result.copy_from(&x);
        result.scale_add_and_assign(&x, f::<M>(2.0), &y);
        let vals = triplet_values(&result);
        assert_eq!(
            vals,
            vec![
                f::<M>(21.0),
                f::<M>(42.0),
                f::<M>(63.0),
                f::<M>(84.0),
                f::<M>(105.0),
                f::<M>(126.0),
                f::<M>(147.0),
                f::<M>(168.0),
            ]
        );
    }

    // --- Batched DenseMatrix-specific tests ---

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_from_vec<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x2 matrix, nbatch=2: physical is 2x4
        // batch0: col0=[1,3], col1=[2,4]; batch1: col0=[5,7], col1=[6,8]
        let a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(7.0),
                f::<M>(6.0),
                f::<M>(8.0),
            ],
            ctx,
        );
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.get_index(0, 0), f::<M>(1.0));
        assert_eq!(a.get_index(1, 0), f::<M>(3.0));
        assert_eq!(a.get_index(0, 1), f::<M>(2.0));
        assert_eq!(a.get_index(1, 1), f::<M>(4.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // batch0: A=[[1,0],[0,1]](identity), batch1: A=[[2,0],[0,2]]
        let a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(2.0),
            ],
            ctx.clone(),
        );
        // batch0: B=[[3,4],[5,6]], batch1: B=[[1,1],[1,1]]
        let b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(3.0),
                f::<M>(5.0),
                f::<M>(4.0),
                f::<M>(6.0),
                f::<M>(1.0),
                f::<M>(1.0),
                f::<M>(1.0),
                f::<M>(1.0),
            ],
            ctx.clone(),
        );
        let mut c = M::zeros(2, 2, ctx);
        c.gemm(f::<M>(1.0), &a, &b, f::<M>(0.0));
        // batch0: I*B=B=[[3,4],[5,6]], batch1: 2I*B=[[2,2],[2,2]]
        assert_eq!(c.get_index(0, 0), f::<M>(3.0));
        assert_eq!(c.get_index(1, 0), f::<M>(5.0));
        assert_eq!(c.get_index(0, 1), f::<M>(4.0));
        assert_eq!(c.get_index(1, 1), f::<M>(6.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_columns<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x3 matrix, nbatch=2
        // batch0: [[1,3,5],[2,4,6]], batch1: [[7,9,11],[8,10,12]]
        let a = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(3.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(8.0),
                f::<M>(9.0),
                f::<M>(10.0),
                f::<M>(11.0),
                f::<M>(12.0),
            ],
            ctx.clone(),
        );
        let view = a.columns(0, 2);
        assert_eq!(view.ncols(), 2);
        assert_eq!(view.nrows(), 2);
        let owned = view.into_owned();
        assert_eq!(owned.nrows(), 2);
        assert_eq!(owned.ncols(), 2);
        // Verify via gemv_o: multiply columns(0,2) by [1,1] for each batch
        let view2 = a.columns(0, 2);
        let x = M::V::from_vec(
            vec![f::<M>(1.0), f::<M>(1.0), f::<M>(1.0), f::<M>(1.0)],
            ctx.clone(),
        );
        let mut y = M::V::zeros(2, ctx);
        view2.gemv_o(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // batch0: [1,2]*1 + [3,4]*1 = [4,6], batch1: [7,8]*1 + [9,10]*1 = [16,18]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(4.0), f::<M>(6.0), f::<M>(16.0), f::<M>(18.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_o_on_columns<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x3 diff matrix, nbatch=2
        // batch0: [[1,2,3],[4,5,6]], batch1: [[7,8,9],[10,11,12]]
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(10.0),
                f::<M>(8.0),
                f::<M>(11.0),
                f::<M>(9.0),
                f::<M>(12.0),
            ],
            ctx.clone(),
        );
        // take columns 0..2 from each batch
        let view = diff.columns(0, 2);
        // x has nbatch=2, length=2 (matches ncols of view)
        // batch0: x=[1,1], batch1: x=[2,2]
        let x = M::V::from_vec(
            vec![f::<M>(1.0), f::<M>(1.0), f::<M>(2.0), f::<M>(2.0)],
            ctx.clone(),
        );
        let mut y = M::V::zeros(2, ctx);
        view.gemv_o(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // batch0: [[1,2],[4,5]] * [1,1] = [1+2, 4+5] = [3, 9]
        // batch1: [[7,8],[10,11]] * [2,2] = [14+16, 20+22] = [30, 42]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(9.0), f::<M>(30.0), f::<M>(42.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_v_broadcast_mat<M: DenseMatrix>(ctx3: M::C) {
        assert_eq!(ctx3.nbatch(), 2);
        // matrix view with nbatch=1 broadcasts to x/y with nbatch=2
        let ctx1 = M::C::default();
        // 2x3 matrix, nbatch=1: [[1,2,3],[4,5,6]]
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
            ],
            ctx1,
        );
        let view = diff.columns(0, 2);
        // x with nbatch=2, length=2
        // batch0: [1,1], batch1: [2,2]
        let x = M::V::from_vec(
            vec![f::<M>(1.0), f::<M>(1.0), f::<M>(2.0), f::<M>(2.0)],
            ctx3.clone(),
        );
        let mut y = M::V::zeros(2, ctx3);
        view.gemv_v(f::<M>(1.0), &x.as_view(), f::<M>(0.0), &mut y);
        // batch0: [[1,2],[4,5]] * [1,1] = [3, 9]
        // batch1: [[1,2],[4,5]] * [2,2] = [6, 18]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(9.0), f::<M>(6.0), f::<M>(18.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_o_broadcast_mat<M: DenseMatrix>(ctx3: M::C) {
        assert_eq!(ctx3.nbatch(), 2);
        // matrix view with nbatch=1 broadcasts to x/y with nbatch=2
        let ctx1 = M::C::default();
        // 2x3 matrix, nbatch=1: [[1,2,3],[4,5,6]]
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
            ],
            ctx1,
        );
        let view = diff.columns(0, 2);
        // x with nbatch=2, length=2
        // batch0: [1,1], batch1: [2,2]
        let x = M::V::from_vec(
            vec![f::<M>(1.0), f::<M>(1.0), f::<M>(2.0), f::<M>(2.0)],
            ctx3.clone(),
        );
        let mut y = M::V::zeros(2, ctx3);
        view.gemv_o(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // batch0: [[1,2],[4,5]] * [1,1] = [3, 9]
        // batch1: [[1,2],[4,5]] * [2,2] = [6, 18]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(9.0), f::<M>(6.0), f::<M>(18.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_vo_on_columns<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x3 diff matrix, nbatch=2
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(10.0),
                f::<M>(8.0),
                f::<M>(11.0),
                f::<M>(9.0),
                f::<M>(12.0),
            ],
            ctx.clone(),
        );
        // R is 2x2 (nbatch=2): batch0=identity, batch1=2*identity
        let r = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(2.0),
            ],
            ctx.clone(),
        );
        let mut result = M::zeros(2, 3, ctx);
        {
            let d_view = diff.columns(0, 2);
            let mut r_view = result.columns_mut(0, 2);
            r_view.gemm_vo(f::<M>(1.0), &d_view, &r, f::<M>(0.0));
        }
        // batch0: [[1,2],[4,5]] * I = [[1,2],[4,5]]
        // batch1: [[7,8],[10,11]] * 2I = [[14,16],[20,22]]
        assert_eq!(result.get_index(0, 0), f::<M>(1.0));
        assert_eq!(result.get_index(1, 0), f::<M>(4.0));
        assert_eq!(result.get_index(0, 1), f::<M>(2.0));
        assert_eq!(result.get_index(1, 1), f::<M>(5.0));
    }

    // --- Broadcasting tests ---

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_broadcast_b<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // batch0: A=[[1,0],[0,1]], batch1: A=[[2,0],[0,3]]
        let a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(3.0),
            ],
            ctx.clone(),
        );
        // B with nbatch=1: [[1,2],[3,4]]
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let mut c = M::zeros(2, 2, ctx);
        c.gemm(f::<M>(1.0), &a, &b, f::<M>(0.0));
        // batch0: I*B=[[1,2],[3,4]], batch1: diag(2,3)*B=[[2,4],[9,12]]
        assert_eq!(c.get_index(0, 0), f::<M>(1.0));
        assert_eq!(c.get_index(1, 0), f::<M>(3.0));
        assert_eq!(c.get_index(0, 1), f::<M>(2.0));
        assert_eq!(c.get_index(1, 1), f::<M>(4.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_broadcast_a<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // A with nbatch=1: [[1,0],[0,2]]
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(0.0), f::<M>(0.0), f::<M>(2.0)],
            Default::default(),
        );
        // batch0: B=[[3,4],[5,6]], batch1: B=[[1,1],[1,1]]
        let b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(3.0),
                f::<M>(5.0),
                f::<M>(4.0),
                f::<M>(6.0),
                f::<M>(1.0),
                f::<M>(1.0),
                f::<M>(1.0),
                f::<M>(1.0),
            ],
            ctx.clone(),
        );
        let mut c = M::zeros(2, 2, ctx);
        c.gemm(f::<M>(1.0), &a, &b, f::<M>(0.0));
        // batch0: [[1,0],[0,2]]*[[3,4],[5,6]]=[[3,4],[10,12]]
        assert_eq!(c.get_index(0, 0), f::<M>(3.0));
        assert_eq!(c.get_index(1, 0), f::<M>(10.0));
        assert_eq!(c.get_index(0, 1), f::<M>(4.0));
        assert_eq!(c.get_index(1, 1), f::<M>(12.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_o_broadcast_x<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x3 diff matrix, nbatch=2
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(10.0),
                f::<M>(8.0),
                f::<M>(11.0),
                f::<M>(9.0),
                f::<M>(12.0),
            ],
            ctx.clone(),
        );
        let view = diff.columns(0, 2);
        // x with nbatch=1, length=2 (broadcast)
        let x = M::V::from_vec(vec![f::<M>(1.0), f::<M>(1.0)], Default::default());
        let mut y = M::V::zeros(2, ctx);
        view.gemv_o(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // batch0: [[1,2],[4,5]] * [1,1] = [3, 9]
        // batch1: [[7,8],[10,11]] * [1,1] = [15, 21]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(9.0), f::<M>(15.0), f::<M>(21.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_vo_broadcast_b<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x3 diff matrix, nbatch=2
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(10.0),
                f::<M>(8.0),
                f::<M>(11.0),
                f::<M>(9.0),
                f::<M>(12.0),
            ],
            ctx.clone(),
        );
        // R with nbatch=1: 2x2 identity
        let r = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(0.0), f::<M>(0.0), f::<M>(1.0)],
            Default::default(),
        );
        let mut result = M::zeros(2, 3, ctx);
        {
            let d_view = diff.columns(0, 2);
            let mut r_view = result.columns_mut(0, 2);
            r_view.gemm_vo(f::<M>(1.0), &d_view, &r, f::<M>(0.0));
        }
        // Both batches: sub-matrix * I = sub-matrix (unchanged)
        // batch0: [[1,2],[4,5]], batch1: [[7,8],[10,11]]
        assert_eq!(result.get_index(0, 0), f::<M>(1.0));
        assert_eq!(result.get_index(1, 0), f::<M>(4.0));
        assert_eq!(result.get_index(0, 1), f::<M>(2.0));
        assert_eq!(result.get_index(1, 1), f::<M>(5.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_vo_broadcast_a<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // diff with nbatch=1: 2x3 matrix [[1,2,3],[4,5,6]]
        let diff = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(5.0),
                f::<M>(3.0),
                f::<M>(6.0),
            ],
            Default::default(),
        );
        // b with nbatch=2: batch0=[[1,0],[0,1]], batch1=[[2,0],[0,3]]
        let b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(3.0),
            ],
            ctx.clone(),
        );
        let mut result = M::zeros(2, 3, ctx);
        {
            let d_view = diff.columns(0, 2);
            let mut r_view = result.columns_mut(0, 2);
            r_view.gemm_vo(f::<M>(1.0), &d_view, &b, f::<M>(0.0));
        }
        // batch0: [[1,2],[4,5]]*I=[[1,2],[4,5]], batch1: [[1,2],[4,5]]*[[2,0],[0,3]]=[[2,6],[8,15]]
        assert_eq!(result.get_index(0, 0), f::<M>(1.0));
        assert_eq!(result.get_index(1, 0), f::<M>(4.0));
        assert_eq!(result.get_index(0, 1), f::<M>(2.0));
        assert_eq!(result.get_index(1, 1), f::<M>(5.0));
    }

    // --- Incompatible batch tests ---

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_incompatible_a<M: DenseMatrix>(ctx2: M::C, ctx3: M::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let a = M::zeros(2, 2, ctx3);
        let b = M::zeros(2, 2, ctx2.clone());
        let mut c = M::zeros(2, 2, ctx2);
        c.gemm(f::<M>(1.0), &a, &b, f::<M>(0.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_incompatible<M: DenseMatrix>(ctx2: M::C, ctx3: M::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let a = M::zeros(2, 2, ctx2.clone());
        let x = M::V::zeros(2, ctx3);
        let mut y = M::V::zeros(2, ctx2);
        a.gemv(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_incompatible<M: DenseMatrix>(ctx2: M::C, ctx3: M::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let a = M::zeros(2, 2, ctx2.clone());
        let b = M::zeros(2, 2, ctx3);
        let mut c = M::zeros(2, 2, ctx2);
        c.gemm(f::<M>(1.0), &a, &b, f::<M>(0.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_resize_cols<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // 2x2, nbatch=2: batch0=[[1,2],[3,4]], batch1=[[5,6],[7,8]]
        let mut a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(7.0),
                f::<M>(6.0),
                f::<M>(8.0),
            ],
            ctx.clone(),
        );
        // grow to 3 columns
        a.resize_cols(3);
        assert_eq!(a.ncols(), 3);
        assert_eq!(a.nrows(), 2);
        // existing data preserved per batch
        assert_eq!(a.get_index(0, 0), f::<M>(1.0));
        assert_eq!(a.get_index(1, 0), f::<M>(3.0));
        assert_eq!(a.get_index(0, 1), f::<M>(2.0));
        assert_eq!(a.get_index(1, 1), f::<M>(4.0));
        // new column is zero
        assert_eq!(a.get_index(0, 2), f::<M>(0.0));
        assert_eq!(a.get_index(1, 2), f::<M>(0.0));
        // verify via gemv that batch 1 data is intact
        let x = M::V::from_vec(
            vec![
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
            ],
            ctx.clone(),
        );
        let mut y = M::V::zeros(2, ctx.clone());
        a.gemv(f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // batch0: col0=[1,3], x=[1,0,0] → [1,3]
        // batch1: col0=[5,7], x=[1,0,0] → [5,7]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(5.0), f::<M>(7.0)]
        );

        // shrink to 1 column
        a.resize_cols(1);
        assert_eq!(a.ncols(), 1);
        assert_eq!(a.get_index(0, 0), f::<M>(1.0));
        assert_eq!(a.get_index(1, 0), f::<M>(3.0));
        // verify batch1 col0 via gemv
        let x2 = M::V::from_vec(vec![f::<M>(1.0), f::<M>(1.0)], ctx.clone());
        let mut y2 = M::V::zeros(2, ctx);
        a.gemv(f::<M>(1.0), &x2, f::<M>(0.0), &mut y2);
        // batch0: [[1],[3]] * [1] = [1,3], batch1: [[5],[7]] * [1] = [5,7]
        assert_eq!(
            y2.clone_as_vec(),
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(5.0), f::<M>(7.0)]
        );
    }

    // --- New unbatched Matrix-generic tests ---

    pub fn test_mul_scalar<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)];
        let a = M::try_from_triplets(2, 2, indices, values, Default::default()).unwrap();
        let result = a * Scale(f::<M>(2.0));
        let (_, vals) = result.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![f::<M>(2.0), f::<M>(6.0), f::<M>(4.0), f::<M>(8.0)]
        );
    }

    pub fn test_add_column_to_vector<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let mat = M::try_from_triplets(2, 2, indices, values, Default::default()).unwrap();
        let mut v = M::V::zeros(2, Default::default());
        mat.add_column_to_vector(1, &mut v);
        assert_eq!(v.clone_as_vec(), vec![f::<M>(3.0), f::<M>(4.0)]);
    }

    // --- New unbatched DenseMatrix-specific tests ---

    pub fn test_add<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(5.0), f::<M>(7.0), f::<M>(6.0), f::<M>(8.0)],
            Default::default(),
        );
        let result = a + &b;
        assert_eq!(result.get_index(0, 0), f::<M>(6.0));
        assert_eq!(result.get_index(1, 1), f::<M>(12.0));
    }

    pub fn test_sub<M: DenseMatrix>() {
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(5.0), f::<M>(7.0), f::<M>(6.0), f::<M>(8.0)],
            Default::default(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let result = a - &b;
        assert_eq!(result.get_index(0, 0), f::<M>(4.0));
        assert_eq!(result.get_index(1, 1), f::<M>(4.0));
    }

    pub fn test_add_assign<M: DenseMatrix>() {
        let mut a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(5.0), f::<M>(7.0), f::<M>(6.0), f::<M>(8.0)],
            Default::default(),
        );
        a += &b;
        assert_eq!(a.get_index(0, 0), f::<M>(6.0));
        assert_eq!(a.get_index(1, 1), f::<M>(12.0));
    }

    pub fn test_sub_assign<M: DenseMatrix>() {
        let mut a = M::from_vec(
            2,
            2,
            vec![f::<M>(5.0), f::<M>(7.0), f::<M>(6.0), f::<M>(8.0)],
            Default::default(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        a -= &b;
        assert_eq!(a.get_index(0, 0), f::<M>(4.0));
        assert_eq!(a.get_index(1, 1), f::<M>(4.0));
    }

    pub fn test_gather<M: DenseMatrix>() {
        let mat1 = M::from_vec(
            3,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(3.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(6.0),
                f::<M>(7.0),
                f::<M>(8.0),
                f::<M>(9.0),
            ],
            Default::default(),
        );
        let mut mat2 = M::zeros(2, 2, Default::default());
        let indices = <M::V as Vector>::Index::from_vec(vec![0, 1, 3, 4], Default::default());
        mat2.gather(&mat1, &indices);
        assert_eq!(mat2.get_index(0, 0), f::<M>(1.0));
        assert_eq!(mat2.get_index(1, 0), f::<M>(2.0));
        assert_eq!(mat2.get_index(0, 1), f::<M>(4.0));
        assert_eq!(mat2.get_index(1, 1), f::<M>(5.0));
    }

    pub fn test_set_data_with_indices<M: DenseMatrix>() {
        let mut mat = M::zeros(2, 2, Default::default());
        let dst_indices = <M::V as Vector>::Index::from_vec(vec![0, 3], Default::default());
        let src_indices = <M::V as Vector>::Index::from_vec(vec![0, 1], Default::default());
        let data = M::V::from_vec(vec![f::<M>(5.0), f::<M>(6.0)], Default::default());
        mat.set_data_with_indices(&dst_indices, &src_indices, &data);
        assert_eq!(mat.get_index(0, 0), f::<M>(5.0));
        assert_eq!(mat.get_index(1, 1), f::<M>(6.0));
    }

    pub fn test_mul_assign_scalar<M: DenseMatrix>() {
        let mut mat = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        {
            let mut view = mat.columns_mut(0, 2);
            view *= Scale(f::<M>(2.0));
        }
        assert_eq!(mat.get_index(0, 0), f::<M>(2.0));
        assert_eq!(mat.get_index(1, 1), f::<M>(8.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_combine<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        #[rustfmt::skip]
        let data: Vec<M::T> = vec![
            // batch 0: 4x4 column-major (cols 0-3)
            f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0),
            f::<M>(5.0), f::<M>(6.0), f::<M>(7.0), f::<M>(8.0),
            f::<M>(9.0), f::<M>(10.0), f::<M>(11.0), f::<M>(12.0),
            f::<M>(13.0), f::<M>(14.0), f::<M>(15.0), f::<M>(16.0),
            // batch 1: 4x4 column-major (cols 0-3)
            f::<M>(101.0), f::<M>(102.0), f::<M>(103.0), f::<M>(104.0),
            f::<M>(105.0), f::<M>(106.0), f::<M>(107.0), f::<M>(108.0),
            f::<M>(109.0), f::<M>(110.0), f::<M>(111.0), f::<M>(112.0),
            f::<M>(113.0), f::<M>(114.0), f::<M>(115.0), f::<M>(116.0),
        ];
        let m = M::from_vec(4, 4, data, ctx.clone());

        let alg_indices = <M::V as Vector>::Index::from_vec(vec![1, 3], Default::default());
        let [(ul, _), (ur, _), (ll, _), (lr, _)] = m.split(&alg_indices);

        let recombined = M::combine(&ul, &ur, &ll, &lr, &alg_indices);

        let (_orig_idx, orig_vals) = m.triplet_iter();
        let (_recom_idx, recom_vals) = recombined.triplet_iter();
        let orig_vals: Vec<_> = orig_vals.collect();
        let recom_vals: Vec<_> = recom_vals.collect();
        assert_eq!(orig_vals, recom_vals);
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_add_column_to_vector_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![
            f::<M>(1.0),
            f::<M>(2.0),
            f::<M>(3.0),
            f::<M>(4.0),
            f::<M>(5.0),
            f::<M>(6.0),
            f::<M>(7.0),
            f::<M>(8.0),
        ];
        let mat = M::try_from_triplets(2, 2, indices, values, ctx.clone()).unwrap();
        let mut v = M::V::zeros(2, ctx);
        mat.add_column_to_vector(1, &mut v);
        assert_eq!(
            v.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(4.0), f::<M>(7.0), f::<M>(8.0)]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_set_data_with_indices_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let zero_values = vec![
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
        ];
        let mut mat = M::try_from_triplets(2, 2, indices, zero_values, ctx.clone()).unwrap();
        let dst_indices = <M::V as Vector>::Index::from_vec(vec![0, 3], Default::default());
        let src_indices = <M::V as Vector>::Index::from_vec(vec![0, 1], Default::default());
        let data = M::V::from_vec(
            vec![f::<M>(5.0), f::<M>(6.0), f::<M>(50.0), f::<M>(60.0)],
            ctx,
        );
        mat.set_data_with_indices(&dst_indices, &src_indices, &data);
        let (_, vals) = mat.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(5.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(6.0),
                f::<M>(50.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(60.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gather_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices: Vec<(IndexType, IndexType)> =
            (0..3).flat_map(|j| (0..3).map(move |i| (i, j))).collect();
        let values = vec![
            f::<M>(1.0),
            f::<M>(2.0),
            f::<M>(3.0),
            f::<M>(4.0),
            f::<M>(5.0),
            f::<M>(6.0),
            f::<M>(7.0),
            f::<M>(8.0),
            f::<M>(9.0),
            f::<M>(10.0),
            f::<M>(20.0),
            f::<M>(30.0),
            f::<M>(40.0),
            f::<M>(50.0),
            f::<M>(60.0),
            f::<M>(70.0),
            f::<M>(80.0),
            f::<M>(90.0),
        ];
        let mat1 = M::try_from_triplets(3, 3, indices, values, ctx.clone()).unwrap();
        let dest_indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let zero_values = vec![
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
            f::<M>(0.0),
        ];
        let mut mat2 = M::try_from_triplets(2, 2, dest_indices, zero_values, ctx).unwrap();
        let gather_indices =
            <M::V as Vector>::Index::from_vec(vec![0, 1, 3, 4], Default::default());
        mat2.gather(&mat1, &gather_indices);
        let (_, vals) = mat2.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(10.0),
                f::<M>(20.0),
                f::<M>(40.0),
                f::<M>(50.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_mul_scalar_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![
            f::<M>(1.0),
            f::<M>(3.0),
            f::<M>(2.0),
            f::<M>(4.0),
            f::<M>(5.0),
            f::<M>(7.0),
            f::<M>(6.0),
            f::<M>(8.0),
        ];
        let a = M::try_from_triplets(2, 2, indices, values, ctx.clone()).unwrap();
        let result = a * Scale(f::<M>(2.0));
        let (_, vals) = result.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(2.0),
                f::<M>(6.0),
                f::<M>(4.0),
                f::<M>(8.0),
                f::<M>(10.0),
                f::<M>(14.0),
                f::<M>(12.0),
                f::<M>(16.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_partition_indices<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let zero_val = M::T::zero();
        let one_val = f::<M>(1.0);
        let two_val = f::<M>(2.0);
        let indices = vec![(0, 0), (1, 1), (2, 2)];
        let values = vec![one_val, zero_val, one_val, two_val, zero_val, two_val];
        let m = M::try_from_triplets(3, 3, indices, values, ctx).unwrap();
        let (zero_idx, nonzero_idx) = m.partition_indices_by_zero_diagonal();
        assert_eq!(zero_idx.clone_as_vec(), vec![1]);
        assert_eq!(nonzero_idx.clone_as_vec(), vec![0, 2]);
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_column_axpy<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(7.0),
                f::<M>(6.0),
                f::<M>(8.0),
            ],
            ctx,
        );
        a.column_axpy(f::<M>(2.0), 0, 1);
        assert_eq!(a.get_index(0, 0), f::<M>(1.0));
        assert_eq!(a.get_index(0, 1), f::<M>(4.0));
        assert_eq!(a.get_index(1, 0), f::<M>(3.0));
        assert_eq!(a.get_index(1, 1), f::<M>(10.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_mat_mul<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(2.0),
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(3.0),
            ],
            ctx.clone(),
        );
        let b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(2.0),
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(3.0),
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(2.0),
                f::<M>(1.0),
            ],
            ctx.clone(),
        );
        let c = a.mat_mul(&b);
        assert_eq!(c.get_index(0, 0), f::<M>(4.0));
        assert_eq!(c.get_index(1, 0), f::<M>(10.0));
        assert_eq!(c.get_index(0, 1), f::<M>(6.0));
        assert_eq!(c.get_index(1, 1), f::<M>(12.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_from_diagonal_dense<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = M::V::from_vec(
            vec![f::<M>(2.0), f::<M>(3.0), f::<M>(4.0), f::<M>(5.0)],
            ctx,
        );
        let a = M::from_diagonal(&v);
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.get_index(0, 0), f::<M>(2.0));
        assert_eq!(a.get_index(1, 1), f::<M>(3.0));
        assert_eq!(a.get_index(0, 1), f::<M>(0.0));
        assert_eq!(a.get_index(1, 0), f::<M>(0.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn make_strided_matrix<M: DenseMatrix>(nbatch: usize) -> M {
        let ctx = M::C::default().clone_with_nbatch(nbatch).unwrap();
        let nrows = 3;
        let ncols = 4;
        let mut data = Vec::with_capacity(nrows * ncols * nbatch);
        for b in 0..nbatch {
            for col in 0..ncols {
                for row in 0..nrows {
                    data.push(f::<M>(row as f64 + col as f64 * 10.0 + b as f64 * 100.0));
                }
            }
        }
        M::from_vec(nrows, ncols, data, ctx)
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_into_owned<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_matrix::<M>(ctx.nbatch());
        let view = matrix.columns(0, 2);
        let owned = view.into_owned();
        assert_eq!(owned.nrows(), 3);
        assert_eq!(owned.ncols(), 2);
        // column 0, batch 0: [0,1,2]
        assert_eq!(owned.get_index(0, 0), f::<M>(0.0));
        assert_eq!(owned.get_index(1, 0), f::<M>(1.0));
        assert_eq!(owned.get_index(2, 0), f::<M>(2.0));
        // column 1, batch 0: [10,11,12]
        assert_eq!(owned.get_index(0, 1), f::<M>(10.0));
        assert_eq!(owned.get_index(1, 1), f::<M>(11.0));
        assert_eq!(owned.get_index(2, 1), f::<M>(12.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_add_owned<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_matrix::<M>(ctx.nbatch());
        let view = matrix.columns(0, 2);
        // owned 3x2 with nbatch=1 (broadcast) — column-major: col0=[1,2,3], col1=[4,5,6]
        let rhs = M::from_vec(
            3,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(3.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(6.0),
            ],
            M::C::default(),
        );
        let result = view + &rhs;
        // batch0: [0,1,2,10,11,12] + [1,2,3,4,5,6] = [1,3,5,14,16,18]
        assert_eq!(result.get_index(0, 0), f::<M>(1.0));
        assert_eq!(result.get_index(0, 1), f::<M>(14.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_sub_owned<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_matrix::<M>(ctx.nbatch());
        let view = matrix.columns(0, 2);
        let rhs = M::from_vec(
            3,
            2,
            vec![
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(10.0),
                f::<M>(11.0),
                f::<M>(12.0),
            ],
            M::C::default(),
        );
        let result = view - &rhs;
        // batch0: [0,1,2,10,11,12] - [0,1,2,10,11,12] = all zeros
        assert_eq!(result.get_index(0, 0), f::<M>(0.0));
        assert_eq!(result.get_index(0, 1), f::<M>(0.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_mul_scalar<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_matrix::<M>(ctx.nbatch());
        let view = matrix.columns(0, 2);
        let result = view * Scale(f::<M>(2.0));
        assert_eq!(result.get_index(0, 0), f::<M>(0.0));
        assert_eq!(result.get_index(1, 0), f::<M>(2.0));
        assert_eq!(result.get_index(0, 1), f::<M>(20.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_mut_add_assign_view<M: DenseMatrix>(ctx: M::C) {
        let mut a = make_strided_matrix::<M>(ctx.nbatch());
        let b = make_strided_matrix::<M>(ctx.nbatch());
        {
            let mut a_view = a.columns_mut(0, 2);
            let b_view = b.columns(2, 4);
            a_view += &b_view;
        }
        // a columns 0-1 now = original a[0..2] + b[2..4]
        // batch0 a[0..2]: [[0,10],[1,11],[2,12]]
        // batch0 b[2..4]: [[20,30],[21,31],[22,32]]
        // sum: [[20,40],[22,42],[24,44]]
        assert_eq!(a.get_index(0, 0), f::<M>(20.0));
        assert_eq!(a.get_index(1, 0), f::<M>(22.0));
        assert_eq!(a.get_index(0, 1), f::<M>(40.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_mut_sub_assign_view<M: DenseMatrix>(ctx: M::C) {
        let mut a = make_strided_matrix::<M>(ctx.nbatch());
        let b = make_strided_matrix::<M>(ctx.nbatch());
        {
            let mut a_view = a.columns_mut(0, 2);
            let b_view = b.columns(0, 2);
            a_view -= &b_view;
        }
        // same columns subtracted = all zero
        assert_eq!(a.get_index(0, 0), f::<M>(0.0));
        assert_eq!(a.get_index(1, 0), f::<M>(0.0));
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_mut_mul_assign_scalar<M: DenseMatrix>(ctx: M::C) {
        let mut a = make_strided_matrix::<M>(ctx.nbatch());
        {
            let mut a_view = a.columns_mut(0, 2);
            a_view *= Scale(f::<M>(2.0));
        }
        assert_eq!(a.get_index(0, 0), f::<M>(0.0));
        assert_eq!(a.get_index(1, 0), f::<M>(2.0));
        assert_eq!(a.get_index(0, 1), f::<M>(20.0));
    }

    // --- View-mut tests (into_owned, gemm_oo, += / -= between two mutable views) ---

    pub fn test_view_mut_into_owned<M: DenseMatrix>() {
        let mut a = M::from_vec(
            2,
            3,
            vec![
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(3.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(6.0),
            ],
            Default::default(),
        );
        let owned = a.columns_mut(0, 2).into_owned();
        assert_eq!(owned.nrows(), 2);
        assert_eq!(owned.ncols(), 2);
        assert_eq!(owned.get_index(0, 0), f::<M>(1.0));
        assert_eq!(owned.get_index(1, 0), f::<M>(2.0));
        assert_eq!(owned.get_index(0, 1), f::<M>(3.0));
        assert_eq!(owned.get_index(1, 1), f::<M>(4.0));
    }

    pub fn test_view_mut_add_assign_view_mut<M: DenseMatrix>() {
        let mut a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        let mut b = M::from_vec(
            2,
            2,
            vec![f::<M>(10.0), f::<M>(30.0), f::<M>(20.0), f::<M>(40.0)],
            Default::default(),
        );
        {
            let mut a_view = a.columns_mut(0, 2);
            let b_view = b.columns_mut(0, 2);
            a_view += &b_view;
        }
        assert_eq!(a.get_index(0, 0), f::<M>(11.0));
        assert_eq!(a.get_index(1, 0), f::<M>(33.0));
        assert_eq!(a.get_index(0, 1), f::<M>(22.0));
        assert_eq!(a.get_index(1, 1), f::<M>(44.0));
    }

    pub fn test_view_mut_sub_assign_view_mut<M: DenseMatrix>() {
        let mut a = M::from_vec(
            2,
            2,
            vec![f::<M>(10.0), f::<M>(30.0), f::<M>(20.0), f::<M>(40.0)],
            Default::default(),
        );
        let mut b = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        {
            let mut a_view = a.columns_mut(0, 2);
            let b_view = b.columns_mut(0, 2);
            a_view -= &b_view;
        }
        assert_eq!(a.get_index(0, 0), f::<M>(9.0));
        assert_eq!(a.get_index(1, 0), f::<M>(27.0));
        assert_eq!(a.get_index(0, 1), f::<M>(18.0));
        assert_eq!(a.get_index(1, 1), f::<M>(36.0));
    }

    pub fn test_gemm_oo_on_columns<M: DenseMatrix>() {
        // a = [[1,2],[3,4]] (col-major [1,3,2,4])
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        // b = identity
        let b = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(0.0), f::<M>(0.0), f::<M>(1.0)],
            Default::default(),
        );
        let mut result = M::zeros(2, 3, Default::default());
        {
            let mut r_view = result.columns_mut(0, 2);
            r_view.gemm_oo(f::<M>(1.0), &a, &b, f::<M>(0.0));
        }
        // result columns 0-1 = a * I = a; column 2 untouched (zero)
        assert_eq!(result.get_index(0, 0), f::<M>(1.0));
        assert_eq!(result.get_index(1, 0), f::<M>(3.0));
        assert_eq!(result.get_index(0, 1), f::<M>(2.0));
        assert_eq!(result.get_index(1, 1), f::<M>(4.0));
        assert_eq!(result.get_index(0, 2), f::<M>(0.0));
        assert_eq!(result.get_index(1, 2), f::<M>(0.0));
    }

    pub fn test_try_from_triplets_wrong_length<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        // one value too few: triggers the length assertion inside try_from_triplets
        let values = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0)];
        let _ = M::try_from_triplets(2, 2, indices, values, Default::default());
    }

    // --- Batched view-mut tests ---

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_strided_matrix_view_mut_into_owned<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_matrix::<M>(ctx.nbatch());
        let owned = matrix.columns_mut(0, 2).into_owned();
        assert_eq!(owned.nrows(), 3);
        assert_eq!(owned.ncols(), 2);
        // batch 0 col0=[0,1,2], col1=[10,11,12]
        assert_eq!(owned.get_index(0, 0), f::<M>(0.0));
        assert_eq!(owned.get_index(1, 0), f::<M>(1.0));
        assert_eq!(owned.get_index(2, 0), f::<M>(2.0));
        assert_eq!(owned.get_index(0, 1), f::<M>(10.0));
        assert_eq!(owned.get_index(1, 1), f::<M>(11.0));
        assert_eq!(owned.get_index(2, 1), f::<M>(12.0));
        // verify both batches via triplet_iter
        let (_, vals) = owned.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(10.0),
                f::<M>(11.0),
                f::<M>(12.0),
                f::<M>(100.0),
                f::<M>(101.0),
                f::<M>(102.0),
                f::<M>(110.0),
                f::<M>(111.0),
                f::<M>(112.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_view_mut_add_assign_view_mut<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // a 2x2 nbatch=2: batch0 [[1,2],[3,4]], batch1 [[5,6],[7,8]]
        let mut a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(7.0),
                f::<M>(6.0),
                f::<M>(8.0),
            ],
            ctx.clone(),
        );
        let mut b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(10.0),
                f::<M>(30.0),
                f::<M>(20.0),
                f::<M>(40.0),
                f::<M>(50.0),
                f::<M>(70.0),
                f::<M>(60.0),
                f::<M>(80.0),
            ],
            ctx,
        );
        {
            let mut a_view = a.columns_mut(0, 2);
            let b_view = b.columns_mut(0, 2);
            a_view += &b_view;
        }
        let (_, vals) = a.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(11.0),
                f::<M>(33.0),
                f::<M>(22.0),
                f::<M>(44.0),
                f::<M>(55.0),
                f::<M>(77.0),
                f::<M>(66.0),
                f::<M>(88.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_view_mut_sub_assign_view_mut<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(10.0),
                f::<M>(30.0),
                f::<M>(20.0),
                f::<M>(40.0),
                f::<M>(50.0),
                f::<M>(70.0),
                f::<M>(60.0),
                f::<M>(80.0),
            ],
            ctx.clone(),
        );
        let mut b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(7.0),
                f::<M>(6.0),
                f::<M>(8.0),
            ],
            ctx,
        );
        {
            let mut a_view = a.columns_mut(0, 2);
            let b_view = b.columns_mut(0, 2);
            a_view -= &b_view;
        }
        let (_, vals) = a.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(9.0),
                f::<M>(27.0),
                f::<M>(18.0),
                f::<M>(36.0),
                f::<M>(45.0),
                f::<M>(63.0),
                f::<M>(54.0),
                f::<M>(72.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemm_oo_on_columns<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // a 2x2 nbatch=2: batch0 [[1,2],[3,4]], batch1 [[5,6],[7,8]]
        let a = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(5.0),
                f::<M>(7.0),
                f::<M>(6.0),
                f::<M>(8.0),
            ],
            ctx.clone(),
        );
        // b 2x2 nbatch=2: batch0 identity, batch1 2*identity
        let b = M::from_vec(
            2,
            2,
            vec![
                f::<M>(1.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(2.0),
            ],
            ctx.clone(),
        );
        let mut result = M::zeros(2, 3, ctx);
        {
            let mut r_view = result.columns_mut(0, 2);
            r_view.gemm_oo(f::<M>(1.0), &a, &b, f::<M>(0.0));
        }
        // batch0: a*I = [[1,2],[3,4]]; batch1: a*2I = [[10,12],[14,16]]; col2 = 0
        assert_eq!(result.get_index(0, 0), f::<M>(1.0));
        assert_eq!(result.get_index(1, 0), f::<M>(3.0));
        assert_eq!(result.get_index(0, 1), f::<M>(2.0));
        assert_eq!(result.get_index(1, 1), f::<M>(4.0));
        let (_, vals) = result.triplet_iter();
        let vals: Vec<_> = vals.collect();
        assert_eq!(
            vals,
            vec![
                f::<M>(1.0),
                f::<M>(3.0),
                f::<M>(2.0),
                f::<M>(4.0),
                f::<M>(0.0),
                f::<M>(0.0),
                f::<M>(10.0),
                f::<M>(14.0),
                f::<M>(12.0),
                f::<M>(16.0),
                f::<M>(0.0),
                f::<M>(0.0),
            ]
        );
    }
}

#[cfg(test)]
macro_rules! generate_matrix_tests_nonbatched {
    ($suffix:ident, $M:ty) => {
        paste::paste! {
            #[test]
            fn [<test_zeros_ $suffix>]() {
                $crate::matrix::tests::test_zeros::<$M>();
            }
            #[test]
            fn [<test_from_diagonal_ $suffix>]() {
                $crate::matrix::tests::test_from_diagonal::<$M>();
            }
            #[test]
            fn [<test_gemv_ $suffix>]() {
                $crate::matrix::tests::test_gemv::<$M>();
            }
            #[test]
            fn [<test_set_column_ $suffix>]() {
                $crate::matrix::tests::test_set_column::<$M>();
            }
            #[test]
            fn [<test_copy_from_ $suffix>]() {
                $crate::matrix::tests::test_copy_from::<$M>();
            }
            #[test]
            fn [<test_scale_add_and_assign_ $suffix>]() {
                $crate::matrix::tests::test_scale_add_and_assign::<$M>();
            }
            #[test]
            fn [<test_partition_indices_ $suffix>]() {
                $crate::matrix::tests::test_partition_indices_by_zero_diagonal::<$M>();
            }
            #[test]
            fn [<test_mul_scalar_ $suffix>]() {
                $crate::matrix::tests::test_mul_scalar::<$M>();
            }
            #[test]
            fn [<test_add_column_to_vector_ $suffix>]() {
                $crate::matrix::tests::test_add_column_to_vector::<$M>();
            }
            #[test]
            #[should_panic]
            fn [<test_try_from_triplets_wrong_length_ $suffix>]() {
                $crate::matrix::tests::test_try_from_triplets_wrong_length::<$M>();
            }
        }
    };
}

#[cfg(test)]
#[cfg_attr(not(feature = "cuda"), allow(unused_macros))]
macro_rules! generate_matrix_tests_batched {
    ($suffix:ident, $M:ty, $ctx1:expr, $ctx2:expr) => {
        paste::paste! {
            #[test]
            fn [<test_batched_add_column_to_vector_ $suffix>]() {
                $crate::matrix::tests::test_batched_add_column_to_vector_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_set_data_with_indices_ $suffix>]() {
                $crate::matrix::tests::test_batched_set_data_with_indices_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gather_ $suffix>]() {
                $crate::matrix::tests::test_batched_gather_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_mul_scalar_ $suffix>]() {
                $crate::matrix::tests::test_batched_mul_scalar_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_partition_indices_ $suffix>]() {
                $crate::matrix::tests::test_batched_partition_indices::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_zeros_ $suffix>]() {
                $crate::matrix::tests::test_batched_zeros_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_broadcast_x_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_broadcast_x_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_broadcast_mat_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_broadcast_mat_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_from_diagonal_ $suffix>]() {
                $crate::matrix::tests::test_batched_from_diagonal_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_copy_from_ $suffix>]() {
                $crate::matrix::tests::test_batched_copy_from_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_set_column_ $suffix>]() {
                $crate::matrix::tests::test_batched_set_column_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_scale_add_ $suffix>]() {
                $crate::matrix::tests::test_batched_scale_add_and_assign_m::<$M>($ctx2);
            }
        }
    };
}

#[cfg(test)]
macro_rules! generate_dense_matrix_tests_nonbatched {
    ($suffix:ident, $M:ty) => {
        paste::paste! {
            #[test]
            fn [<test_from_vec_ $suffix>]() {
                $crate::matrix::tests::test_from_vec::<$M>();
            }
            #[test]
            fn [<test_from_diagonal_dense_ $suffix>]() {
                $crate::matrix::tests::test_from_diagonal_dense::<$M>();
            }
            #[test]
            fn [<test_gemm_ $suffix>]() {
                $crate::matrix::tests::test_gemm::<$M>();
            }
            #[test]
            fn [<test_mat_mul_ $suffix>]() {
                $crate::matrix::tests::test_mat_mul::<$M>();
            }
            #[test]
            fn [<test_columns_view_ $suffix>]() {
                $crate::matrix::tests::test_columns_view::<$M>();
            }
            #[test]
            fn [<test_column_view_ $suffix>]() {
                $crate::matrix::tests::test_column_view::<$M>();
            }
            #[test]
            fn [<test_column_axpy_ $suffix>]() {
                $crate::matrix::tests::test_column_axpy::<$M>();
            }
            #[test]
            fn [<test_resize_cols_ $suffix>]() {
                $crate::matrix::tests::test_resize_cols::<$M>();
            }
            #[test]
            fn [<test_add_ $suffix>]() {
                $crate::matrix::tests::test_add::<$M>();
            }
            #[test]
            fn [<test_sub_ $suffix>]() {
                $crate::matrix::tests::test_sub::<$M>();
            }
            #[test]
            fn [<test_add_assign_ $suffix>]() {
                $crate::matrix::tests::test_add_assign::<$M>();
            }
            #[test]
            fn [<test_sub_assign_ $suffix>]() {
                $crate::matrix::tests::test_sub_assign::<$M>();
            }
            #[test]
            fn [<test_gather_ $suffix>]() {
                $crate::matrix::tests::test_gather::<$M>();
            }
            #[test]
            fn [<test_set_data_with_indices_ $suffix>]() {
                $crate::matrix::tests::test_set_data_with_indices::<$M>();
            }
            #[test]
            fn [<test_mul_assign_scalar_ $suffix>]() {
                $crate::matrix::tests::test_mul_assign_scalar::<$M>();
            }
            #[test]
            fn [<test_view_mut_into_owned_ $suffix>]() {
                $crate::matrix::tests::test_view_mut_into_owned::<$M>();
            }
            #[test]
            fn [<test_view_mut_add_assign_view_mut_ $suffix>]() {
                $crate::matrix::tests::test_view_mut_add_assign_view_mut::<$M>();
            }
            #[test]
            fn [<test_view_mut_sub_assign_view_mut_ $suffix>]() {
                $crate::matrix::tests::test_view_mut_sub_assign_view_mut::<$M>();
            }
            #[test]
            fn [<test_gemm_oo_on_columns_ $suffix>]() {
                $crate::matrix::tests::test_gemm_oo_on_columns::<$M>();
            }
        }
    };
}

#[cfg(test)]
#[cfg_attr(not(feature = "cuda"), allow(unused_macros))]
macro_rules! generate_dense_matrix_tests_batched {
    ($suffix:ident, $M:ty, $ctx1:expr, $ctx2:expr) => {
        paste::paste! {
            #[test]
            fn [<test_batched_column_axpy_ $suffix>]() {
                $crate::matrix::tests::test_batched_column_axpy::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_mat_mul_ $suffix>]() {
                $crate::matrix::tests::test_batched_mat_mul::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_from_diagonal_dense_ $suffix>]() {
                $crate::matrix::tests::test_batched_from_diagonal_dense::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_from_vec_ $suffix>]() {
                $crate::matrix::tests::test_batched_from_vec::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_columns_ $suffix>]() {
                $crate::matrix::tests::test_batched_columns::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_o_on_columns_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_o_on_columns::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_vo_on_columns_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_vo_on_columns::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_broadcast_b_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_broadcast_b::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_o_broadcast_x_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_o_broadcast_x::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_v_broadcast_mat_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_v_broadcast_mat::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_o_broadcast_mat_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_o_broadcast_mat::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_vo_broadcast_b_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_vo_broadcast_b::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_broadcast_a_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_broadcast_a::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_vo_broadcast_a_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_vo_broadcast_a::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_resize_cols_ $suffix>]() {
                $crate::matrix::tests::test_batched_resize_cols::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_combine_ $suffix>]() {
                $crate::matrix::tests::test_batched_combine::<$M>($ctx2);
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_gemv_incompatible_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_incompatible::<$M>($ctx2, $ctx1.clone_with_nbatch(3).unwrap());
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_gemm_incompatible_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_incompatible::<$M>($ctx2, $ctx1.clone_with_nbatch(3).unwrap());
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_gemm_incompatible_a_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_incompatible_a::<$M>($ctx2, $ctx1.clone_with_nbatch(3).unwrap());
            }
            #[test]
            fn [<test_strided_matrix_view_into_owned_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_into_owned::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_add_owned_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_add_owned::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_sub_owned_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_sub_owned::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_mul_scalar_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_mul_scalar::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_mut_add_assign_view_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_mut_add_assign_view::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_mut_sub_assign_view_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_mut_sub_assign_view::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_mut_mul_assign_scalar_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_mut_mul_assign_scalar::<$M>($ctx2);
            }
            #[test]
            fn [<test_strided_matrix_view_mut_into_owned_ $suffix>]() {
                $crate::matrix::tests::test_strided_matrix_view_mut_into_owned::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_view_mut_add_assign_view_mut_ $suffix>]() {
                $crate::matrix::tests::test_batched_view_mut_add_assign_view_mut::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_view_mut_sub_assign_view_mut_ $suffix>]() {
                $crate::matrix::tests::test_batched_view_mut_sub_assign_view_mut::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemm_oo_on_columns_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemm_oo_on_columns::<$M>($ctx2);
            }
        }
    };
}

#[cfg(test)]
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
pub(crate) use generate_dense_matrix_tests_batched;
#[cfg(test)]
pub(crate) use generate_dense_matrix_tests_nonbatched;
#[cfg(test)]
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
pub(crate) use generate_matrix_tests_batched;
#[cfg(test)]
pub(crate) use generate_matrix_tests_nonbatched;
