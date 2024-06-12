use std::fmt::Debug;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::scalar::Scale;
use crate::{IndexType, Scalar, Vector, VectorIndex};
use anyhow::Result;
use num_traits::{One, Zero};
use sparsity::{MatrixSparsity, MatrixSparsityRef};

#[cfg(feature = "nalgebra")]
mod dense_nalgebra_serial;

#[cfg(feature = "faer")]
mod dense_faer_serial;

#[cfg(feature = "faer")]
pub mod sparse_faer;

pub mod default_solver;
mod sparse_serial;
pub mod sparsity;

#[cfg(feature = "sundials")]
pub mod sundials;

pub trait MatrixCommon: Sized + Debug {
    type V: Vector<T = Self::T>;
    type T: Scalar;

    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
}

impl<'a, M> MatrixCommon for &'a M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;

    fn nrows(&self) -> IndexType {
        M::nrows(*self)
    }
    fn ncols(&self) -> IndexType {
        M::ncols(*self)
    }
}

impl<'a, M> MatrixCommon for &'a mut M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;

    fn ncols(&self) -> IndexType {
        M::ncols(*self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(*self)
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
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T);
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T);
}

/// A view of a dense matrix [Matrix]
pub trait MatrixView<'a>:
    for<'b> MatrixOpsByValue<&'b Self::Owned, Self::Owned>
    + Mul<Scale<Self::T>, Output = Self::Owned>
    + Clone
{
    type Owned;

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
pub trait Matrix: MatrixCommon + Mul<Scale<Self::T>, Output = Self> + Clone {
    type Sparsity: MatrixSparsity<Self>;
    type SparsityRef<'a>: MatrixSparsityRef<'a, Self>
    where
        Self: 'a;

    /// Return sparsity information (None if the matrix is dense)
    fn sparsity(&self) -> Option<Self::SparsityRef<'_>>;

    fn is_sparse() -> bool {
        Self::zeros(1, 1).sparsity().is_some()
    }

    /// Split the current matrix into four submatrices at the given indices
    fn split_at_indices(
        &self,
        indices: &<Self::V as crate::vector::Vector>::Index,
    ) -> (Self, Self, Self, Self) {
        let n = self.nrows();
        if n != self.ncols() {
            panic!("Matrix must be square");
        }
        let ni = indices.len();
        let nni = n - ni;
        let mut indices = indices.clone_as_vec();
        indices.sort();
        let cat = (0..n)
            .map(|i| indices.as_slice().binary_search(&i).is_ok())
            .collect::<Vec<_>>();
        let mut upper_indices = Vec::with_capacity(n);
        let mut lower_indices = Vec::with_capacity(n);
        let mut upper_acc = 0;
        let mut lower_acc = 0;
        for c in cat.iter() {
            lower_indices.push(lower_acc);
            upper_indices.push(upper_acc);
            if *c {
                lower_acc += 1;
            } else {
                upper_acc += 1;
            }
        }
        let mut ur_triplets = Vec::new();
        let mut ul_triplets = Vec::new();
        let mut lr_triplets = Vec::new();
        let mut ll_triplets = Vec::new();
        for (i, j, &v) in self.triplet_iter() {
            if !cat[i] && !cat[j] {
                let ii = upper_indices[i];
                let jj = upper_indices[j];
                ul_triplets.push((ii, jj, v));
            } else if !cat[i] && cat[j] {
                let ii = upper_indices[i];
                let jj = lower_indices[j];
                ur_triplets.push((ii, jj, v));
            } else if cat[i] && !cat[j] {
                let ii = lower_indices[i];
                let jj = upper_indices[j];
                ll_triplets.push((ii, jj, v));
            } else {
                let ii = lower_indices[i];
                let jj = lower_indices[j];
                lr_triplets.push((ii, jj, v));
            }
        }
        (
            Self::try_from_triplets(nni, nni, ul_triplets).unwrap(),
            Self::try_from_triplets(nni, ni, ur_triplets).unwrap(),
            Self::try_from_triplets(ni, nni, ll_triplets).unwrap(),
            Self::try_from_triplets(ni, ni, lr_triplets).unwrap(),
        )
    }

    /// Combine four matrices into a single matrix at the given indices
    fn combine_at_indices(
        ul: &Self,
        ur: &Self,
        ll: &Self,
        lr: &Self,
        indices: &<Self::V as Vector>::Index,
    ) -> Self {
        let n = ul.nrows() + ll.nrows();
        let m = ul.ncols() + ur.ncols();
        if ul.ncols() != ll.ncols()
            || ur.ncols() != lr.ncols()
            || ul.nrows() != ur.nrows()
            || ll.nrows() != lr.nrows()
        {
            panic!("Matrices must have the same shape");
        }
        let mut triplets = Vec::new();
        let mut indices = indices.clone_as_vec();
        indices.sort();
        let cat = (0..n)
            .map(|i| indices.as_slice().binary_search(&i).is_ok())
            .collect::<Vec<_>>();
        let mut upper_indices = Vec::with_capacity(n);
        let mut lower_indices = Vec::with_capacity(n);
        let mut upper_indices_short = Vec::with_capacity(n - indices.len());
        let mut lower_indices_short = Vec::with_capacity(indices.len());
        let mut upper_acc = 0;
        let mut lower_acc = 0;
        for (i, c) in cat.iter().enumerate() {
            lower_indices.push(lower_acc);
            upper_indices.push(upper_acc);
            if *c {
                lower_indices_short.push(i);
                lower_acc += 1;
            } else {
                upper_indices_short.push(i);
                upper_acc += 1;
            }
        }
        for (i, j, &v) in ul.triplet_iter() {
            let ii = upper_indices_short[i];
            let jj = upper_indices_short[j];
            triplets.push((ii, jj, v));
        }
        for (i, j, &v) in ur.triplet_iter() {
            let ii = upper_indices_short[i];
            let jj = lower_indices_short[j];
            triplets.push((ii, jj, v));
        }
        for (i, j, &v) in ll.triplet_iter() {
            let ii = lower_indices_short[i];
            let jj = upper_indices_short[j];
            triplets.push((ii, jj, v));
        }
        for (i, j, &v) in lr.triplet_iter() {
            let ii = lower_indices_short[i];
            let jj = lower_indices_short[j];
            triplets.push((ii, jj, v));
        }
        Self::try_from_triplets(n, m, triplets).unwrap()
    }

    /// Extract the diagonal of the matrix as an owned vector
    fn diagonal(&self) -> Self::V;

    /// Perform a matrix-vector multiplication `y = self * x + beta * y`.
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);

    /// Copy the contents of `other` into `self`
    fn copy_from(&mut self, other: &Self);

    /// Create a new matrix of shape `nrows` x `ncols` filled with zeros
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;

    /// Create a new matrix from a sparsity pattern, the non-zero elements are not initialized
    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
    ) -> Self;

    /// Create a new diagonal matrix from a [Vector] holding the diagonal elements
    fn from_diagonal(v: &Self::V) -> Self;

    fn set_column(&mut self, j: IndexType, v: &Self::V);

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V);

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    );

    /// Perform the assignment self = x + beta * y where x and y are matrices and beta is a scalar
    /// Panics if the sparsity of self, x, and y do not match (i.e. sparsity of self must be the union of the sparsity of x and y)
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self);

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, &Self::T)>;

    /// Create a new matrix from a vector of triplets (i, j, value) where i and j are the row and column indices of the value
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, Self::T)>,
    ) -> Result<Self>;
}

/// A dense column-major matrix. The assumption is that the underlying matrix is stored in column-major order, so functions for taking columns views are efficient
pub trait DenseMatrix:
    Matrix
    + for<'a, 'b> MatrixOpsByValue<&'b Self::View<'a>, Self>
    + for<'a, 'b> MatrixMutOpsByValue<&'b Self::View<'a>>
    + Index<(IndexType, IndexType), Output = Self::T>
    + IndexMut<(IndexType, IndexType), Output = Self::T>
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
    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, beta: Self::T, i: IndexType);

    /// Get a matrix view of the columns starting at `start` and ending at `start + ncols`
    fn columns(&self, start: IndexType, ncols: IndexType) -> Self::View<'_>;

    /// Get a vector view of the column `i`
    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_>;

    /// Get a mutable matrix view of the columns starting at `start` and ending at `start + ncols`
    fn columns_mut(&mut self, start: IndexType, ncols: IndexType) -> Self::ViewMut<'_>;

    /// Get a mutable vector view of the column `i`
    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_>;

    /// mat_mat_mul using gemm, allocating a new matrix
    fn mat_mul(&self, b: &Self) -> Self {
        let nrows = self.nrows();
        let ncols = b.ncols();
        let mut ret = Self::zeros(nrows, ncols);
        ret.gemm(Self::T::one(), self, b, Self::T::zero());
        ret
    }
}

#[cfg(test)]
mod tests {
    use crate::SparseColMat;
    use super::*;

    #[test]
    fn test_split_combine_at_indices() {
        let triplets = vec![
                (0, 0, 1.0),
                (1, 0, 5.0),
                (2, 0, 9.0),
                (3, 0, 13.0),
                (0, 1, 2.0),
                (1, 1, 6.0),
                (2, 1, 10.0),
                (3, 1, 14.0),
                (0, 2, 3.0),
                (1, 2, 7.0),
                (2, 2, 11.0),
                (3, 2, 15.0),
                (0, 3, 4.0),
                (1, 3, 8.0),
                (2, 3, 12.0),
                (3, 3, 16.0),
            ];
        let m = SparseColMat::try_from_triplets(
            4,
            4,
            triplets.clone(),
        )
        .unwrap();
        let indices = vec![0, 2];
        let (ul, ur, ll, lr) = m.split_at_indices(&indices);
        let ul_triplets = vec![(0, 0, 6.0), (1, 0, 14.0), (0, 1, 8.0), (1, 1, 16.0)];
        let ur_triplets = vec![(0, 0, 5.0), (1, 0, 13.0), (0, 1, 7.0), (1, 1, 15.0)];
        let ll_triplets = vec![(0, 0, 2.0), (1, 0, 10.0), (0, 1, 4.0), (1, 1, 12.0)];
        let lr_triplets = vec![(0, 0, 1.0), (1, 0, 9.0), (0, 1, 3.0), (1, 1, 11.0)];
        assert_eq!(ul_triplets, ul.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>());
        assert_eq!(ur_triplets, ur.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>());
        assert_eq!(ll_triplets, ll.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>());
        assert_eq!(lr_triplets, lr.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>());

        let mat = SparseColMat::combine_at_indices(&ul, &ur, &ll, &lr, &indices);
        assert_eq!(triplets, mat.triplet_iter().map(|(i, j, v)| (i, j, *v)).collect::<Vec<_>>());
    }
}


