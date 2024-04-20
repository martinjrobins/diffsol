use std::fmt::Debug;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::scalar::Scale;
use crate::{IndexType, Scalar, Vector};
use anyhow::Result;
use num_traits::{One, Zero};

#[cfg(feature = "nalgebra")]
mod dense_nalgebra_serial;

#[cfg(feature = "faer")]
mod dense_faer_serial;

pub mod default_solver;
mod sparse_serial;

#[cfg(feature = "sundials")]
pub mod sundials;

pub trait MatrixCommon: Sized + Debug {
    type V: Vector<T = Self::T>;
    type T: Scalar;

    /// Get the number of columns of the matrix
    fn nrows(&self) -> IndexType;

    /// Get the number of rows of the matrix
    fn ncols(&self) -> IndexType;
}

impl<'a, M> MatrixCommon for &'a M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;

    fn ncols(&self) -> IndexType {
        M::ncols(self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(self)
    }
}

impl<'a, M> MatrixCommon for &'a mut M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;

    fn ncols(&self) -> IndexType {
        M::ncols(self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(self)
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
pub trait MatrixRef<M: MatrixCommon>:
    MatrixOpsByValue<M, M> + for<'a> MatrixOpsByValue<&'a M, M> + Mul<Scale<M::T>, Output = M>
{
}

impl<RefT, M: MatrixCommon> MatrixRef<M> for RefT where
    RefT:
        MatrixOpsByValue<M, M> + for<'a> MatrixOpsByValue<&'a M, M> + Mul<Scale<M::T>, Output = M>
{
}

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
pub trait Matrix:
    for<'a> MatrixOpsByValue<&'a Self, Self> + Mul<Scale<Self::T>, Output = Self> + Clone
{
    /// Extract the diagonal of the matrix as an owned vector
    fn diagonal(&self) -> Self::V;

    /// Perform a matrix-vector multiplication `y = self * x + beta * y`.
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);

    /// Copy the contents of `other` into `self`
    fn copy_from(&mut self, other: &Self);

    /// Create a new matrix of shape `nrows` x `ncols` filled with zeros
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;

    /// Create a new diagonal matrix from a [Vector] holding the diagonal elements
    fn from_diagonal(v: &Self::V) -> Self;

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
