use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use crate::error::LaError;
use crate::scalar::Scale;
use crate::vector::VectorHost;
use crate::{Context, IndexType, Scalar, Vector, VectorIndex};

use extract_block::combine;
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
    /// For batched matrices the partition is taken from batch 0, so all batches are assumed
    /// to share the same structure.
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
    ) -> Result<Self, LaError>;
}

/// A host matrix is a matrix type whose vector type is hosted on the CPU.
///
/// This trait extends `Matrix` to ensure the associated vector type implements `VectorHost`,
/// enabling direct CPU-side access to data. GPU matrices typically do not implement this trait.
pub trait MatrixHost: Matrix<V: VectorHost> {}

impl<T: Matrix<V: VectorHost>> MatrixHost for T {}

/// Largest column range the small-coefficient kernels accept — [`DenseMatrix::mul_cols_by`] and
/// [`DenseMatrix::gemv_cols`].
///
/// Both take their coefficients from the host: the serial backends stage them in a fixed stack
/// buffer and the CUDA backend passes them to the kernel by value, so neither allocates — which
/// needs a bound. Two things drive it: the BDF difference table is at most `MAX_ORDER + 1 = 6`
/// columns wide, and the widest built-in Runge-Kutta tableau (`tsit45`) has 7 stages. 8 covers
/// both.
///
/// Must match `MAX_SMALL_COLS` in `src/cuda_kernels/cuda_kernels_common.h`, which sizes the
/// kernels' by-value coefficient arrays; nothing checks that across the language boundary.
pub const MAX_SMALL_COLS: usize = 8;

/// A dense column-major matrix with efficient column access operations.
///
/// This trait represents matrices stored in column-major order, where accessing matrix columns
/// is efficient. It supports:
/// - Column operations (access, modification, and the fused column kernels below)
/// - Element access and modification
/// - Matrix resizing
///
/// The column-major layout makes operations on individual or ranges of columns very efficient.
pub trait DenseMatrix:
    Matrix + for<'b> MatrixOpsByValue<&'b Self, Self> + for<'b> MatrixMutOpsByValue<&'b Self>
{
    /// Update a table of backward differences held in columns `0..=order+2` of
    /// this matrix, given the newly computed highest-order correction `d`.
    ///
    /// Equivalent to:
    /// ```text
    /// self[:, order+2] = d - self[:, order+1]
    /// for i in (0..=order+1).rev() { self[:, i] += self[:, i+1] }
    /// ```
    /// where each addition uses the just-updated value of column `i+1` (a
    /// right-to-left cumulative scan).
    fn update_backward_diff(&mut self, order: IndexType, d: &Self::V);

    /// Right-multiply a range of columns by a small matrix, in place:
    ///
    /// ```text
    /// self[:, 0..ncols] = self[:, 0..ncols] * rhs
    /// ```
    ///
    /// `rhs` is an `ncols x ncols` block held column-major in a host slice of exactly
    /// `ncols * ncols` values — it is small (the BDF step-size update uses at most `6 x 6`).
    ///
    /// Panics if `ncols` exceeds [`MAX_SMALL_COLS`], since the serial backends stage the old
    /// column values in a fixed-size buffer rather than allocating.
    fn mul_cols_by(&mut self, ncols: IndexType, rhs: &[Self::T]);

    /// Matrix-vector multiply over a range of columns, with host-side coefficients:
    ///
    /// ```text
    /// y = alpha * self[:, start..end] * x + beta * y
    /// ```
    ///
    /// `x` is a host slice, not a vector, because `x.len()` is the *width of the column range*
    /// rather than a dimension of the problem: every caller in the solvers passes a short
    /// coefficient list computed from host scalars — a Runge-Kutta tableau row, the `b`/`d`
    /// vectors, interpolation weights, BDF's psi weights. Keeping them on the host is what lets
    /// those callers build them in a stack array instead of allocating a vector per call.
    ///
    /// Use [`Matrix::gemv`] instead when `x` is a genuine operand of length `ncols` — a
    /// Jacobian- or mass-times-vector product.
    ///
    /// Only the leading `end - start` columns take part; the rest of the matrix is not read.
    /// `x.len()` must be at least `end - start`; only its leading `end - start` entries are
    /// read, as with a BLAS `gemv` given a longer vector and `incx = 1`. `beta == 0` means `y`
    /// is not read, so it may hold uninitialised values. An empty range contributes nothing,
    /// leaving `y = beta * y`.
    ///
    /// The coefficients are shared by every batch — unlike [`Matrix::gemv`], this cannot apply
    /// a different `x` per batch. Panics if `end - start` exceeds [`MAX_SMALL_COLS`].
    fn gemv_cols(
        &self,
        start: IndexType,
        end: IndexType,
        alpha: Self::T,
        x: &[Self::T],
        beta: Self::T,
        y: &mut Self::V,
    );

    /// Get an immutable vector view of column `i`.
    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_>;

    /// Get a mutable vector view of column `i`.
    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_>;

    /// Set the value at the given row and column indices.
    ///
    /// For batched matrices the value is written to every batch.
    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T);

    /// Get the value at the given row and column indices.
    ///
    /// For batched matrices this reads batch 0, as does `Index`/`IndexMut`.
    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T;

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
#[allow(clippy::items_after_test_module)]
pub(crate) mod tests {
    use std::ops::{Add, Sub};

    use super::{
        DenseMatrix, Matrix, MatrixCommon, MatrixSparsity, MatrixSparsityRef, MAX_SMALL_COLS,
    };
    use crate::scalar::Scale;
    use crate::{scalar::IndexType, Context, Vector, VectorIndex, VectorView, VectorViewMut};
    use num_traits::{FromPrimitive, One, Zero};

    fn f<M: Matrix>(x: f64) -> M::T {
        M::T::from_f64(x).unwrap()
    }

    pub(crate) fn triplet_values<M: Matrix>(m: &M) -> Vec<M::T> {
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

        // column 2 has a single stored entry below the diagonal (row 3) and none at (2, 2):
        // the sparse backend's early-exit ("row index has passed the diagonal") branch fires.
        let indices = vec![(3, 2)];
        let values = vec![M::T::one()];
        let m = M::try_from_triplets(4, 4, indices, values, Default::default()).unwrap();
        let (zero_diagonal_indices, non_zero_diagonal_indices) =
            m.partition_indices_by_zero_diagonal();
        assert_eq!(zero_diagonal_indices.clone_as_vec(), vec![0, 1, 2, 3]);
        assert_eq!(
            non_zero_diagonal_indices.clone_as_vec(),
            Vec::<IndexType>::new()
        );
    }

    // --- Matrix-generic tests (work with both dense and sparse) ---

    pub fn test_zeros<M: Matrix>() {
        let a = M::zeros(2, 3, Default::default());
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 3);
        let vals = triplet_values(&a);
        assert!(vals.is_empty() || vals.iter().all(|v| v.is_zero()));
        assert_eq!(M::is_sparse(), a.sparsity().is_some());
    }

    /// `M::Sparsity`/`M::SparsityRef<'_>` trait methods (`nrows`/`ncols`/`is_sparse`/`indices`/
    /// `union`/`new_diagonal`/`get_index`/`as_ref`/`to_owned`/`split`): only `is_sparse` was ever
    /// called generically before, via `test_zeros`, so none of the sparsity pattern machinery
    /// itself (used by both the `Dense` stand-in for dense backends and the real sparse pattern)
    /// was exercised.
    pub fn test_sparsity<M: Matrix>() {
        let a = M::Sparsity::try_from_indices(2, 2, vec![(0, 0), (1, 1)]).unwrap();
        assert_eq!(MatrixSparsity::<M>::nrows(&a), 2);
        assert_eq!(MatrixSparsity::<M>::ncols(&a), 2);
        assert_eq!(
            M::is_sparse(),
            <M::Sparsity as MatrixSparsity<M>>::is_sparse()
        );
        let _ = a.indices();
        let diag = M::Sparsity::new_diagonal(3);
        assert_eq!(MatrixSparsity::<M>::nrows(&diag), 3);
        assert_eq!(MatrixSparsity::<M>::ncols(&diag), 3);

        let b = M::Sparsity::try_from_indices(2, 2, vec![(0, 1)]).unwrap();
        let unioned = a.clone().union(b.as_ref()).unwrap();
        assert_eq!(MatrixSparsity::<M>::nrows(&unioned), 2);
        assert_eq!(MatrixSparsity::<M>::ncols(&unioned), 2);

        let idx = a.get_index(&[(0, 0), (1, 1)], Default::default());
        assert_eq!(idx.len(), 2);

        let a_ref = a.as_ref();
        assert_eq!(MatrixSparsityRef::<M>::nrows(&a_ref), 2);
        assert_eq!(MatrixSparsityRef::<M>::ncols(&a_ref), 2);
        assert_eq!(
            M::is_sparse(),
            <M::SparsityRef<'_> as MatrixSparsityRef<M>>::is_sparse()
        );
        let _ = a_ref.indices();
        let owned = a_ref.to_owned();
        assert_eq!(MatrixSparsity::<M>::nrows(&owned), 2);

        let algebraic = <M::V as Vector>::Index::from_vec(vec![1], Default::default());
        let [(ul, _), (_, _), (_, _), (lr, _)] = a_ref.split(&algebraic);
        assert_eq!(MatrixSparsity::<M>::nrows(&ul), 1);
        assert_eq!(MatrixSparsity::<M>::nrows(&lr), 1);
    }

    pub fn test_matrix_common_by_ref<M: Matrix>() {
        fn via_common<T: MatrixCommon>(m: T) -> (IndexType, IndexType) {
            let dims = (m.nrows(), m.ncols());
            let _ = m.inner();
            dims
        }
        let mut a = M::zeros(2, 3, Default::default());
        assert_eq!(via_common(&a), (2, 3));
        assert_eq!(via_common(&mut a), (2, 3));
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

        // beta = 1.0: accumulate onto the pre-existing value of y
        a.gemv(f::<M>(1.0), &x, f::<M>(1.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(10.0), f::<M>(22.0)]);
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

    /// `Dense<M>`'s `try_from_indices`/`union` error branches: dense matrices ignore the actual
    /// index pattern (their `Sparsity` is just a bounding box), so only a zero dimension or a
    /// shape mismatch can make these fail — inputs the sparse backend wouldn't reject the same way.
    pub fn test_sparsity_dense_errors<M: DenseMatrix>() {
        assert!(M::Sparsity::try_from_indices(0, 3, vec![]).is_err());
        let a = M::Sparsity::try_from_indices(2, 2, vec![]).unwrap();
        let b = M::Sparsity::try_from_indices(3, 3, vec![]).unwrap();
        assert!(a.union(b.as_ref()).is_err());
    }

    /// Order 1: 4 columns (`0..=order+2`), column 3 starts as garbage (overwritten).
    /// Expected values are the right-to-left cumulative scan replayed by hand:
    ///   col3 = d - col2
    ///   col2 += col3 (updated); col1 += col2 (updated); col0 += col1 (updated)
    pub fn test_update_backward_diff<M: DenseMatrix>() {
        let order = 1;
        let mut a = M::from_vec(
            2,
            4,
            vec![
                f::<M>(1.0),
                f::<M>(10.0),
                f::<M>(2.0),
                f::<M>(20.0),
                f::<M>(3.0),
                f::<M>(30.0),
                f::<M>(0.0),
                f::<M>(0.0),
            ],
            Default::default(),
        );
        let d = M::V::from_vec(vec![f::<M>(100.0), f::<M>(1000.0)], Default::default());

        a.update_backward_diff(order, &d);

        assert_eq!(a.get_index(0, 3), f::<M>(97.0));
        assert_eq!(a.get_index(1, 3), f::<M>(970.0));
        assert_eq!(a.get_index(0, 2), f::<M>(100.0));
        assert_eq!(a.get_index(1, 2), f::<M>(1000.0));
        assert_eq!(a.get_index(0, 1), f::<M>(102.0));
        assert_eq!(a.get_index(1, 1), f::<M>(1020.0));
        assert_eq!(a.get_index(0, 0), f::<M>(103.0));
        assert_eq!(a.get_index(1, 0), f::<M>(1030.0));
    }

    pub fn test_gemv_cols<M: DenseMatrix>() {
        // columns: [1, 10], [2, 20], [3, 30], [4, 40]
        let a = M::from_vec(
            2,
            4,
            (1..=4)
                .flat_map(|j| [f::<M>(j as f64), f::<M>(10.0 * j as f64)])
                .collect(),
            Default::default(),
        );
        let ones = |n: usize| vec![f::<M>(1.0); n];
        let mut y = M::V::from_vec(vec![f::<M>(-1.0), f::<M>(-1.0)], Default::default());

        // unweighted sum of columns 0..3, overwriting y
        a.gemv_cols(0, 3, f::<M>(1.0), &ones(3), f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(6.0), f::<M>(60.0)]);

        // weighted sum of every column
        let w4 = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        a.gemv_cols(0, 4, f::<M>(1.0), &w4, f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(30.0), f::<M>(300.0)]);

        // x lines up with the start of the range, not with column zero
        let w23 = vec![f::<M>(2.0), f::<M>(3.0)];
        a.gemv_cols(2, 4, f::<M>(1.0), &w23, f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(18.0), f::<M>(180.0)]);

        // alpha scales the whole product
        a.gemv_cols(2, 4, f::<M>(2.0), &w23, f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(36.0), f::<M>(360.0)]);

        // beta = 1 accumulates onto y instead of overwriting it
        let half = vec![f::<M>(0.5)];
        a.gemv_cols(1, 2, f::<M>(1.0), &half, f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(1.0), f::<M>(10.0)]);
        a.gemv_cols(1, 2, f::<M>(1.0), &half, f::<M>(1.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(2.0), f::<M>(20.0)]);

        // a general beta scales the old y before accumulating
        a.gemv_cols(1, 2, f::<M>(1.0), &half, f::<M>(3.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(7.0), f::<M>(70.0)]);

        // an empty range contributes nothing, leaving y = beta * y
        a.gemv_cols(1, 1, f::<M>(1.0), &ones(0), f::<M>(2.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(14.0), f::<M>(140.0)]);
        a.gemv_cols(1, 1, f::<M>(1.0), &ones(0), f::<M>(1.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(14.0), f::<M>(140.0)]);
        a.gemv_cols(1, 1, f::<M>(1.0), &ones(0), f::<M>(0.0), &mut y);
        assert_eq!(y.clone_as_vec(), vec![f::<M>(0.0), f::<M>(0.0)]);
    }

    /// A column range wider than the 8-weight chunk the retired `weighted_column_sum` CUDA
    /// kernel could take in one launch — that boundary was never covered by a test.
    /// Entry `(i, j)` of batch `b`. `Matrix::get_index` only ever reads batch 0, and a
    /// column view carries every batch, so this goes through the column.
    fn mat_at<M: DenseMatrix>(m: &M, b: usize, i: usize, j: usize) -> M::T {
        m.column(j).into_owned().clone_as_vec()[b * m.nrows() + i]
    }

    /// Every entry of batch `b`, column-major.
    fn batch_as_vec<M: DenseMatrix>(m: &M, b: usize) -> Vec<M::T> {
        let mut out = Vec::with_capacity(m.nrows() * m.ncols());
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                out.push(mat_at(m, b, i, j));
            }
        }
        out
    }

    pub fn test_mul_cols_by<M: DenseMatrix>() {
        // 2x4, columns [1,10], [2,20], [3,30], [4,40]
        let mk = || {
            M::from_vec(
                2,
                4,
                (1..=4)
                    .flat_map(|j| [f::<M>(j as f64), f::<M>(10.0 * j as f64)])
                    .collect(),
                Default::default(),
            )
        };
        let untouched = batch_as_vec(&mk(), 0);

        // identity must leave the matrix exactly as it was: the sharpest check that the
        // in-place update never reads a value it has already overwritten
        let mut a = mk();
        a.mul_cols_by(2, &[f::<M>(1.0), f::<M>(0.0), f::<M>(0.0), f::<M>(1.0)]);
        assert_eq!(batch_as_vec(&a, 0), untouched);

        // an empty range is a no-op
        let mut a = mk();
        a.mul_cols_by(0, &[]);
        assert_eq!(batch_as_vec(&a, 0), untouched);

        // a single column is just scaled, and the rest are left alone
        let mut a = mk();
        a.mul_cols_by(1, &[f::<M>(3.0)]);
        assert_eq!(
            batch_as_vec(&a, 0),
            vec![
                f::<M>(3.0),
                f::<M>(30.0),
                f::<M>(2.0),
                f::<M>(20.0),
                f::<M>(3.0),
                f::<M>(30.0),
                f::<M>(4.0),
                f::<M>(40.0)
            ]
        );

        // rhs = [[1, 2], [3, 4]] column-major, applied to the first two columns only:
        //   new[:, 0] = 1 * [1,10] + 3 * [2,20] = [7, 70]
        //   new[:, 1] = 2 * [1,10] + 4 * [2,20] = [10, 100]
        // columns 2 and 3 must survive untouched -- the BDF difference table keeps live data
        // above the updated range and reads it back in `update_backward_diff`
        let mut a = mk();
        a.mul_cols_by(2, &[f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)]);
        assert_eq!(
            batch_as_vec(&a, 0),
            vec![
                f::<M>(7.0),
                f::<M>(70.0),
                f::<M>(10.0),
                f::<M>(100.0),
                f::<M>(3.0),
                f::<M>(30.0),
                f::<M>(4.0),
                f::<M>(40.0)
            ]
        );

        // the full width the solvers use (MAX_ORDER + 1), via an identity
        let ncols = 6;
        let mut wide = M::from_vec(
            2,
            ncols,
            (1..=ncols)
                .flat_map(|j| [f::<M>(j as f64), f::<M>(10.0 * j as f64)])
                .collect(),
            Default::default(),
        );
        let wide_before = batch_as_vec(&wide, 0);
        let mut id = vec![f::<M>(0.0); ncols * ncols];
        for j in 0..ncols {
            id[j * ncols + j] = f::<M>(1.0);
        }
        wide.mul_cols_by(ncols, &id);
        assert_eq!(batch_as_vec(&wide, 0), wide_before);

        // more rows than fit one staging tile, so the row-blocked loop runs several times
        let nrows = 700;
        let mut tall = M::from_vec(
            nrows,
            2,
            (0..2 * nrows).map(|i| f::<M>(i as f64)).collect(),
            Default::default(),
        );
        tall.mul_cols_by(2, &[f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)]);
        let got = batch_as_vec(&tall, 0);
        for i in 0..nrows {
            let (c0, c1) = (f::<M>(i as f64), f::<M>((nrows + i) as f64));
            assert_eq!(got[i], c0 + f::<M>(3.0) * c1);
            assert_eq!(got[nrows + i], f::<M>(2.0) * c0 + f::<M>(4.0) * c1);
        }
    }

    /// Same product as [`test_mul_cols_by`] on 2 independent batches: `rhs` is shared by every
    /// batch, the columns are not.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_mul_cols_by<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        // batch0 columns [1,10], [2,20]; batch1 columns [3,30], [4,40]
        let mut a = M::from_vec(
            2,
            2,
            (1..=4)
                .flat_map(|j| [f::<M>(j as f64), f::<M>(10.0 * j as f64)])
                .collect(),
            ctx,
        );
        a.mul_cols_by(2, &[f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)]);
        // batch0: [1,10] + 3*[2,20] = [7,70];   2*[1,10] + 4*[2,20] = [10,100]
        assert_eq!(
            batch_as_vec(&a, 0),
            vec![f::<M>(7.0), f::<M>(70.0), f::<M>(10.0), f::<M>(100.0)]
        );
        // batch1: [3,30] + 3*[4,40] = [15,150]; 2*[3,30] + 4*[4,40] = [22,220]
        assert_eq!(
            batch_as_vec(&a, 1),
            vec![f::<M>(15.0), f::<M>(150.0), f::<M>(22.0), f::<M>(220.0)]
        );
    }

    /// The widest column range `gemv_cols` accepts, which is also the size of the by-value
    /// coefficient array the CUDA kernel takes.
    pub fn test_gemv_cols_many_columns<M: DenseMatrix>() {
        let ncols = MAX_SMALL_COLS;
        // column j is [j + 1, 10 * (j + 1)]
        let a = M::from_vec(
            2,
            ncols,
            (1..=ncols)
                .flat_map(|j| [f::<M>(j as f64), f::<M>(10.0 * j as f64)])
                .collect(),
            Default::default(),
        );
        let x = vec![f::<M>(1.0); ncols];
        let mut y = M::V::from_vec(vec![f::<M>(-1.0), f::<M>(-1.0)], Default::default());
        a.gemv_cols(0, ncols, f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // sum_{j=1}^{ncols} j
        let total = f::<M>((ncols * (ncols + 1) / 2) as f64);
        assert_eq!(y.clone_as_vec(), vec![total, f::<M>(10.0) * total]);
    }

    /// Same products as [`test_gemv_cols`], run on 2 independent batches: `x` is unbatched and
    /// therefore shared by every batch, the columns are not.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_gemv_cols_m<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);

        // batch 0 columns: [1, 10], [2, 20]; batch 1 columns: [3, 30], [4, 40]
        let a = M::from_vec(
            2,
            2,
            (1..=4)
                .flat_map(|j| [f::<M>(j as f64), f::<M>(10.0 * j as f64)])
                .collect(),
            ctx.clone(),
        );
        let mut y = M::V::from_vec(
            vec![f::<M>(-1.0), f::<M>(-1.0), f::<M>(-1.0), f::<M>(-1.0)],
            ctx,
        );

        let ones = vec![f::<M>(1.0), f::<M>(1.0)];
        a.gemv_cols(0, 2, f::<M>(1.0), &ones, f::<M>(0.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(30.0), f::<M>(7.0), f::<M>(70.0)]
        );

        let w23 = vec![f::<M>(2.0), f::<M>(3.0)];
        a.gemv_cols(0, 2, f::<M>(1.0), &w23, f::<M>(0.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(8.0), f::<M>(80.0), f::<M>(18.0), f::<M>(180.0)]
        );

        let half = vec![f::<M>(0.5)];
        a.gemv_cols(1, 2, f::<M>(1.0), &half, f::<M>(0.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(1.0), f::<M>(10.0), f::<M>(2.0), f::<M>(20.0)]
        );

        // beta = 1 accumulates independently in each batch
        a.gemv_cols(1, 2, f::<M>(1.0), &half, f::<M>(1.0), &mut y);
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(2.0), f::<M>(20.0), f::<M>(4.0), f::<M>(40.0)]
        );
    }

    pub fn test_resize_cols<M: DenseMatrix>() {
        let mut a = M::zeros(2, 2, Default::default());
        a.set_index(0, 0, M::T::one());
        a.set_index(0, 1, M::T::from_f64(2.0).unwrap());
        a.set_index(1, 0, M::T::from_f64(3.0).unwrap());
        a.set_index(1, 1, M::T::from_f64(4.0).unwrap());

        // resizing to the current column count is a no-op
        a.resize_cols(2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.get_index(0, 0), M::T::one());

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

    /// `copy_from` with mismatched batch counts: the source broadcasts over the destination's
    /// batches, exercising the per-batch loop instead of the equal-nbatch whole-matrix copy.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_copy_from_broadcast_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let a = M::try_from_triplets(2, 2, indices, values, M::C::default()).unwrap();
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
                f::<M>(1.0),
                f::<M>(2.0),
                f::<M>(3.0),
                f::<M>(4.0),
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

    /// An owned right-hand side may be written into, which reverses the operands of `sub`.
    ///
    /// Only wired up for the backends that implement the owned-rhs combinations; move the call
    /// into `generate_dense_matrix_tests_nonbatched!` once they all do.
    pub fn test_owned_rhs_m<M>()
    where
        M: DenseMatrix,
        for<'a> &'a M: Add<M, Output = M> + Sub<M, Output = M>,
    {
        let values = vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)];
        let a = M::from_vec(
            2,
            2,
            vec![f::<M>(10.0), f::<M>(30.0), f::<M>(20.0), f::<M>(40.0)],
            Default::default(),
        );

        let b = M::from_vec(2, 2, values.clone(), Default::default());
        let c = &a - b;
        assert_eq!(c.get_index(0, 0), f::<M>(9.0));
        assert_eq!(c.get_index(1, 1), f::<M>(36.0));

        let b = M::from_vec(2, 2, values, Default::default());
        let c = &a + b;
        assert_eq!(c.get_index(0, 0), f::<M>(11.0));
        assert_eq!(c.get_index(1, 1), f::<M>(44.0));
    }

    /// Owned right-hand side with broadcasting in both directions: the right-hand side can
    /// only be written into when it already has the result's batch count.
    pub fn test_batched_owned_rhs_broadcast_m<M>(ctx: M::C)
    where
        M: Matrix + DenseMatrix,
        for<'a> &'a M: Sub<M, Output = M>,
    {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let one_batch = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let two_batches = vec![
            f::<M>(10.0),
            f::<M>(20.0),
            f::<M>(30.0),
            f::<M>(40.0),
            f::<M>(50.0),
            f::<M>(60.0),
            f::<M>(70.0),
            f::<M>(80.0),
        ];

        // lhs broadcasts over the batches of rhs, which already spans the result
        let a = M::try_from_triplets(2, 2, indices.clone(), one_batch.clone(), M::C::default())
            .unwrap();
        let b =
            M::try_from_triplets(2, 2, indices.clone(), two_batches.clone(), ctx.clone()).unwrap();
        let c = &a - b;
        assert_eq!(c.context().nbatch(), 2);
        assert_eq!(
            triplet_values(&c),
            vec![
                f::<M>(-9.0),
                f::<M>(-18.0),
                f::<M>(-27.0),
                f::<M>(-36.0),
                f::<M>(-49.0),
                f::<M>(-58.0),
                f::<M>(-67.0),
                f::<M>(-76.0),
            ]
        );

        // rhs holds a single batch, so the two-batch result has to be allocated
        let a = M::try_from_triplets(2, 2, indices.clone(), two_batches, ctx).unwrap();
        let b = M::try_from_triplets(2, 2, indices, one_batch, M::C::default()).unwrap();
        let c = &a - b;
        assert_eq!(c.context().nbatch(), 2);
        assert_eq!(
            triplet_values(&c),
            vec![
                f::<M>(9.0),
                f::<M>(18.0),
                f::<M>(27.0),
                f::<M>(36.0),
                f::<M>(49.0),
                f::<M>(58.0),
                f::<M>(67.0),
                f::<M>(76.0),
            ]
        );
    }

    /// An owned left-hand side may be written into, which must still honour broadcasting in
    /// both directions.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_owned_lhs_broadcast_m<M>(ctx: M::C)
    where
        M: Matrix + for<'a> Sub<&'a M, Output = M>,
    {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let one_batch = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let two_batches = vec![
            f::<M>(10.0),
            f::<M>(20.0),
            f::<M>(30.0),
            f::<M>(40.0),
            f::<M>(50.0),
            f::<M>(60.0),
            f::<M>(70.0),
            f::<M>(80.0),
        ];

        // rhs broadcasts over the batches of lhs
        let a =
            M::try_from_triplets(2, 2, indices.clone(), two_batches.clone(), ctx.clone()).unwrap();
        let b = M::try_from_triplets(2, 2, indices.clone(), one_batch.clone(), M::C::default())
            .unwrap();
        let c = a - &b;
        assert_eq!(c.context().nbatch(), 2);
        assert_eq!(
            triplet_values(&c),
            vec![
                f::<M>(9.0),
                f::<M>(18.0),
                f::<M>(27.0),
                f::<M>(36.0),
                f::<M>(49.0),
                f::<M>(58.0),
                f::<M>(67.0),
                f::<M>(76.0),
            ]
        );

        // lhs broadcasts over the batches of rhs
        let a = M::try_from_triplets(2, 2, indices.clone(), one_batch, M::C::default()).unwrap();
        let b = M::try_from_triplets(2, 2, indices, two_batches, ctx).unwrap();
        let c = a - &b;
        assert_eq!(c.context().nbatch(), 2);
        assert_eq!(
            triplet_values(&c),
            vec![
                f::<M>(-9.0),
                f::<M>(-18.0),
                f::<M>(-27.0),
                f::<M>(-36.0),
                f::<M>(-49.0),
                f::<M>(-58.0),
                f::<M>(-67.0),
                f::<M>(-76.0),
            ]
        );
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_scale_add_and_assign_broadcast_m<M: Matrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let x_vals = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0), f::<M>(4.0)];
        let y_vals = vec![f::<M>(10.0), f::<M>(20.0), f::<M>(30.0), f::<M>(40.0)];
        // x and y have nbatch == 1, so they broadcast over both batches of result
        let x =
            M::try_from_triplets(2, 2, indices.clone(), x_vals.clone(), M::C::default()).unwrap();
        let y = M::try_from_triplets(2, 2, indices.clone(), y_vals, M::C::default()).unwrap();
        // result carries the union sparsity (required for sparse matrices) in both batches
        let mut result_vals = x_vals.clone();
        result_vals.extend(x_vals);
        let mut result = M::try_from_triplets(2, 2, indices, result_vals, ctx).unwrap();
        result.scale_add_and_assign(&x, f::<M>(2.0), &y);
        let vals = triplet_values(&result);
        assert_eq!(
            vals,
            vec![
                f::<M>(21.0),
                f::<M>(42.0),
                f::<M>(63.0),
                f::<M>(84.0),
                f::<M>(21.0),
                f::<M>(42.0),
                f::<M>(63.0),
                f::<M>(84.0),
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

    // --- Broadcasting tests ---

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
        // x with nbatch=1, length=2 (broadcast)
        let x = vec![f::<M>(1.0), f::<M>(1.0)];
        let mut y = M::V::zeros(2, ctx);
        diff.gemv_cols(0, 2, f::<M>(1.0), &x, f::<M>(0.0), &mut y);
        // batch0: [[1,2],[4,5]] * [1,1] = [3, 9]
        // batch1: [[7,8],[10,11]] * [1,1] = [15, 21]
        assert_eq!(
            y.clone_as_vec(),
            vec![f::<M>(3.0), f::<M>(9.0), f::<M>(15.0), f::<M>(21.0)]
        );
    }

    // --- Incompatible batch tests ---

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

    /// `MatrixCommon::inner_mut` on an owned matrix; `test_matrix_common_by_ref` only reaches
    /// the generic `&M`/`&mut M` blanket impls.
    pub fn test_inner_mut<M: DenseMatrix>() {
        let mut a = M::zeros(2, 2, Default::default());
        let _ = a.inner_mut();
    }

    /// `column_mut` on an unbatched (nbatch == 1) matrix: existing coverage only calls it through
    /// batched helpers, which take a different code path than the single-batch fast path.
    pub fn test_column_mut<M: DenseMatrix>() {
        let mut a = M::from_vec(
            2,
            2,
            vec![f::<M>(1.0), f::<M>(3.0), f::<M>(2.0), f::<M>(4.0)],
            Default::default(),
        );
        a.column_mut(1).set_index(0, f::<M>(20.0));
        assert_eq!(a.get_index(0, 1), f::<M>(20.0));
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

    /// Same order-1 scan as [`test_update_backward_diff`], run on 2 independent batches with
    /// distinct data, verified via `column(..).clone_as_vec()` so both batches are checked
    /// (`get_index` only ever reads batch 0).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn test_batched_update_backward_diff<M: DenseMatrix>(ctx: M::C) {
        assert_eq!(ctx.nbatch(), 2);
        let order = 1;
        let mut a = M::from_vec(
            2,
            4,
            vec![
                // batch 0
                f::<M>(1.0),
                f::<M>(10.0),
                f::<M>(2.0),
                f::<M>(20.0),
                f::<M>(3.0),
                f::<M>(30.0),
                f::<M>(0.0),
                f::<M>(0.0),
                // batch 1
                f::<M>(5.0),
                f::<M>(50.0),
                f::<M>(6.0),
                f::<M>(60.0),
                f::<M>(7.0),
                f::<M>(70.0),
                f::<M>(0.0),
                f::<M>(0.0),
            ],
            ctx.clone(),
        );
        let d = M::V::from_vec(
            vec![f::<M>(100.0), f::<M>(1000.0), f::<M>(200.0), f::<M>(2000.0)],
            ctx,
        );

        a.update_backward_diff(order, &d);

        assert_eq!(
            a.column(3).into_owned().clone_as_vec(),
            vec![f::<M>(97.0), f::<M>(970.0), f::<M>(193.0), f::<M>(1930.0)]
        );
        assert_eq!(
            a.column(2).into_owned().clone_as_vec(),
            vec![f::<M>(100.0), f::<M>(1000.0), f::<M>(200.0), f::<M>(2000.0)]
        );
        assert_eq!(
            a.column(1).into_owned().clone_as_vec(),
            vec![f::<M>(102.0), f::<M>(1020.0), f::<M>(206.0), f::<M>(2060.0)]
        );
        assert_eq!(
            a.column(0).into_owned().clone_as_vec(),
            vec![f::<M>(103.0), f::<M>(1030.0), f::<M>(211.0), f::<M>(2110.0)]
        );
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

    pub fn test_try_from_triplets_wrong_length<M: Matrix>() {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        // one value too few: triggers the length assertion inside try_from_triplets
        let values = vec![f::<M>(1.0), f::<M>(2.0), f::<M>(3.0)];
        let _ = M::try_from_triplets(2, 2, indices, values, Default::default());
    }

    // --- Batched view-mut tests ---
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
            fn [<test_sparsity_ $suffix>]() {
                $crate::matrix::tests::test_sparsity::<$M>();
            }
            #[test]
            fn [<test_matrix_common_by_ref_ $suffix>]() {
                $crate::matrix::tests::test_matrix_common_by_ref::<$M>();
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
            fn [<test_batched_copy_from_broadcast_ $suffix>]() {
                $crate::matrix::tests::test_batched_copy_from_broadcast_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_set_column_ $suffix>]() {
                $crate::matrix::tests::test_batched_set_column_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_scale_add_ $suffix>]() {
                $crate::matrix::tests::test_batched_scale_add_and_assign_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_owned_lhs_broadcast_ $suffix>]() {
                $crate::matrix::tests::test_batched_owned_lhs_broadcast_m::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_scale_add_broadcast_ $suffix>]() {
                $crate::matrix::tests::test_batched_scale_add_and_assign_broadcast_m::<$M>($ctx2);
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
            fn [<test_column_view_ $suffix>]() {
                $crate::matrix::tests::test_column_view::<$M>();
            }
            #[test]
            fn [<test_sparsity_dense_errors_ $suffix>]() {
                $crate::matrix::tests::test_sparsity_dense_errors::<$M>();
            }
            #[test]
            fn [<test_update_backward_diff_ $suffix>]() {
                $crate::matrix::tests::test_update_backward_diff::<$M>();
            }
            #[test]
            fn [<test_gemv_cols_ $suffix>]() {
                $crate::matrix::tests::test_gemv_cols::<$M>();
            }

            #[test]
            fn [<test_gemv_cols_many_columns_ $suffix>]() {
                $crate::matrix::tests::test_gemv_cols_many_columns::<$M>();
            }

            #[test]
            fn [<test_mul_cols_by_ $suffix>]() {
                $crate::matrix::tests::test_mul_cols_by::<$M>();
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
            fn [<test_inner_mut_ $suffix>]() {
                $crate::matrix::tests::test_inner_mut::<$M>();
            }
            #[test]
            fn [<test_column_mut_ $suffix>]() {
                $crate::matrix::tests::test_column_mut::<$M>();
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
        }
    };
}

#[cfg(test)]
#[cfg_attr(not(feature = "cuda"), allow(unused_macros))]
macro_rules! generate_dense_matrix_tests_batched {
    ($suffix:ident, $M:ty, $ctx1:expr, $ctx2:expr) => {
        paste::paste! {
            #[test]
            fn [<test_batched_update_backward_diff_ $suffix>]() {
                $crate::matrix::tests::test_batched_update_backward_diff::<$M>($ctx2);
            }
            #[test]
            fn [<test_batched_gemv_cols_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_cols_m::<$M>($ctx2);
            }

            #[test]
            fn [<test_batched_mul_cols_by_ $suffix>]() {
                $crate::matrix::tests::test_batched_mul_cols_by::<$M>($ctx2);
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
            fn [<test_batched_gemv_o_broadcast_x_ $suffix>]() {
                $crate::matrix::tests::test_batched_gemv_o_broadcast_x::<$M>($ctx2);
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
