use std::ops::{Mul, MulAssign};

use anyhow::Result;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorView, DVectorViewMut};

use crate::{scalar::Scale, IndexType, Scalar};

use crate::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};

impl<'a, T: Scalar> MatrixCommon for DMatrixViewMut<'a, T> {
    type V = DVector<T>;
    type T = T;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<'a, T: Scalar> MatrixCommon for DMatrixView<'a, T> {
    type V = DVector<T>;
    type T = T;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<T: Scalar> MatrixCommon for DMatrix<T> {
    type V = DVector<T>;
    type T = T;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<'a, T: Scalar> Mul<Scale<T>> for DMatrixView<'a, T> {
    type Output = DMatrix<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * rhs.value()
    }
}

impl<'a, T: Scalar> MatrixView<'a> for DMatrixView<'a, T> {
    type Owned = DMatrix<T>;
}

impl<'a, T: Scalar> MulAssign<Scale<T>> for DMatrixViewMut<'a, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        *self *= rhs.value();
    }
}

impl<'a, T: Scalar> MatrixViewMut<'a> for DMatrixViewMut<'a, T> {
    type Owned = DMatrix<T>;
    type View = DMatrixView<'a, T>;
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
}

impl<T: Scalar> Mul<Scale<T>> for DMatrix<T> {
    type Output = DMatrix<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * rhs.value()
    }
}

impl<T: Scalar> Mul<Scale<T>> for &DMatrix<T> {
    type Output = DMatrix<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * rhs.value()
    }
}

impl<T: Scalar> DenseMatrix for DMatrix<T> {
    type View<'a> = DMatrixView<'a, T>;
    type ViewMut<'a> = DMatrixViewMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.gemv(alpha, self, x, beta);
    }

    fn column_mut(&mut self, i: IndexType) -> DVectorViewMut<'_, T> {
        self.column_mut(i)
    }

    fn columns_mut(&mut self, start: IndexType, ncols: IndexType) -> Self::ViewMut<'_> {
        self.columns_mut(start, ncols)
    }

    fn column(&self, i: IndexType) -> DVectorView<'_, T> {
        self.column(i)
    }
    fn columns(&self, start: IndexType, ncols: IndexType) -> Self::View<'_> {
        self.columns(start, ncols)
    }
}

impl<T: Scalar> Matrix for DMatrix<T> {
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
    ) -> Result<Self> {
        let mut m = Self::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(m)
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self::zeros(nrows, ncols)
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
    fn from_diagonal(v: &DVector<T>) -> Self {
        Self::from_diagonal(v)
    }
    fn diagonal(&self) -> Self::V {
        self.diagonal()
    }
}
