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

pub trait MatrixCommon: Sized + Debug {
    type V: Vector<T = Self::T, C = Self::C, Index: VectorIndex<C = Self::C>>;
    type T: Scalar;
    type C: Context;
    type Inner;

    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
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

pub trait MatrixOpsByValue<Rhs = Self, Output = Self>:
    MatrixCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<M, Rhs, Output> MatrixOpsByValue<Rhs, Output> for M where
    M: MatrixCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

pub trait MatrixMutOpsByValue<Rhs = Self>: MatrixCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

impl<M, Rhs> MatrixMutOpsByValue<Rhs> for M where M: MatrixCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

/// A trait allowing for references to implement matrix operations
pub trait MatrixRef<M: MatrixCommon>: Mul<Scale<M::T>, Output = M> {}
impl<RefT, M: MatrixCommon> MatrixRef<M> for RefT where RefT: Mul<Scale<M::T>, Output = M> {}

/// A mutable view of a dense matrix [Matrix]
pub trait MatrixViewMut<'a>:
    for<'b> MatrixMutOpsByValue<&'b Self>
    + for<'b> MatrixMutOpsByValue<&'b Self::View>
    + MulAssign<Scale<Self::T>>
{
    type Owned;
    type View;
    fn into_owned(self) -> Self::Owned;
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T);
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T);
}

/// A view of a dense matrix [Matrix]
pub trait MatrixView<'a>:
    for<'b> MatrixOpsByValue<&'b Self::Owned, Self::Owned> + Mul<Scale<Self::T>, Output = Self::Owned>
{
    type Owned;

    fn into_owned(self) -> Self::Owned;

    /// Perform a matrix-vector multiplication `y = self * x + beta * y`.
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    );

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);
}

/// A base matrix trait (including sparse and dense matrices)
pub trait Matrix: MatrixCommon + Mul<Scale<Self::T>, Output = Self> + Clone + 'static {
    type Sparsity: MatrixSparsity<Self>;
    type SparsityRef<'a>: MatrixSparsityRef<'a, Self>
    where
        Self: 'a;

    /// Return sparsity information (None if the matrix is dense)
    fn sparsity(&self) -> Option<Self::SparsityRef<'_>>;

    fn context(&self) -> &Self::C;

    fn is_sparse() -> bool {
        Self::zeros(1, 1, Default::default()).sparsity().is_some()
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index);

    /// Perform a matrix-vector multiplication `y = alpha * self * x + beta * y`.
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);

    /// Copy the contents of `other` into `self`
    fn copy_from(&mut self, other: &Self);

    /// Create a new matrix of shape `nrows` x `ncols` filled with zeros
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self;

    /// Create a new matrix from a sparsity pattern, the non-zero elements are not initialized
    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self;

    /// Create a new diagonal matrix from a [Vector] holding the diagonal elements
    fn from_diagonal(v: &Self::V) -> Self;

    /// sets the values of column `j` to be equal to the values in `v`
    /// For sparse matrices, only the existing non-zero elements are updated
    fn set_column(&mut self, j: IndexType, v: &Self::V);

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V);

    /// assign the values in the `data` vector to the matrix at the indices in `dst_indices` from the indices in `src_indices`
    /// dst_index can be obtained using the `get_index` method on the Sparsity type
    ///      - for dense matrices, the dst_index is the data index in column-major order
    ///      - for sparse matrices, the dst_index is the index into the data array
    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    );

    /// gather the values in the matrix `other` at the indices in `indices` to the matrix `self`
    /// for sparse matrices: the index `idx_i` in `indices` is an index into the data array for `other`, and is copied to the index `idx_i` in the data array for `self`
    /// for dense matrices: the index `idx_i` in `indices` is the data index in column-major order for `other`, and is copied to the index `idx_i` in the data array for `self` (again in column-major order)
    ///
    /// See also [Self::split] which can be used to generate these indices
    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index);

    /// Split the matrix into four submatrices, based on the indices in `algebraic_indices`
    ///
    /// M = [UL, UR]
    ///     [LL, LR]
    ///
    /// UL contains the rows and columns not in `algebraic_indices`
    /// UR contains the rows not in `algebraic_indices` and the columns in `algebraic_indices`
    /// LL contains the rows in `algebraic_indices` and the columns not in `algebraic_indices`
    /// LR contains the rows and columns in `algebraic_indices`
    ///
    /// The function returns an array of tuples, where each tuple contains a submatrix, and the indices of the original matrix that were used to create the submatrix
    /// [Self::gather] can be used in conjunction with these indices to update the submatrix
    ///
    /// The order of the submatrices in the array is as follows: [UL, UR, LL, LR]
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

    fn combine(
        ul: &Self,
        ur: &Self,
        ll: &Self,
        lr: &Self,
        algebraic_indices: &<Self::V as Vector>::Index,
    ) -> Self {
        combine(ul, ur, ll, lr, algebraic_indices)
    }

    /// Perform the assignment self = x + beta * y where x and y are matrices and beta is a scalar
    /// Panics if the sparsity of self, x, and y do not match (i.e. sparsity of self must be the union of the sparsity of x and y)
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self);

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, Self::T)>;

    /// Create a new matrix from a vector of triplets (i, j, value) where i and j are the row and column indices of the value
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, Self::T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError>;
}

/// A host matrix is a matrix type that has a host vector type
pub trait MatrixHost: Matrix<V: VectorHost> {}

impl<T: Matrix<V: VectorHost>> MatrixHost for T {}

/// A dense column-major matrix. The assumption is that the underlying matrix is stored in column-major order, so functions for taking columns views are efficient
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

    /// Perform a matrix-matrix multiplication `self = alpha * a * b + beta * self`, where `alpha` and `beta` are scalars, and `a` and `b` are matrices
    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T);

    /// Performs an axpy operation on two columns of the matrix `M[:, i] = alpha * M[:, j] + M[:, i]`
    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, i: IndexType);

    /// Get a matrix view of the columns starting at `start` and ending at `end`
    fn columns(&self, start: IndexType, end: IndexType) -> Self::View<'_>;

    /// Get a vector view of the column `i`
    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_>;

    /// Get a mutable matrix view of the columns starting at `start` and ending at `end`
    fn columns_mut(&mut self, start: IndexType, end: IndexType) -> Self::ViewMut<'_>;

    /// Get a mutable vector view of the column `i`
    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_>;

    /// Set the value at a given index
    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T);

    /// Get the value at a given index
    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T;

    /// mat_mat_mul using gemm, allocating a new matrix
    fn mat_mul(&self, b: &Self) -> Self {
        let nrows = self.nrows();
        let ncols = b.ncols();
        let mut ret = Self::zeros(nrows, ncols, self.context().clone());
        ret.gemm(Self::T::one(), self, b, Self::T::zero());
        ret
    }

    /// Resize the number of columns in the matrix. Existing data is preserved, new elements are uninitialized
    fn resize_cols(&mut self, ncols: IndexType);

    /// creates a new matrix from a vector of values, which are assumed
    /// to be in column-major order
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
