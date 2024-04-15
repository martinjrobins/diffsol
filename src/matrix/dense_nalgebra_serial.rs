use std::ops::{Mul, MulAssign};

use anyhow::Result;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorView, DVectorViewMut};

use crate::{scalar::Scale, IndexType, Scalar};

use crate::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};

macro_rules! impl_matrix_common {
    ($matrix_type:ty) => {
        impl<'a, T: Scalar> MatrixCommon for $matrix_type {
            type V = DVector<T>;
            type T = T;

            fn ncols(&self) -> IndexType {
                self.ncols()
            }

            fn nrows(&self) -> IndexType {
                self.nrows()
            }
        }
    };
}

impl_matrix_common!(DMatrixViewMut<'a, T>);
impl_matrix_common!(DMatrixView<'a, T>);
impl_matrix_common!(DMatrix<T>);

macro_rules! impl_mul_scale {
    ($matrix_type:ty) => {
        impl<'a, T: Scalar> Mul<Scale<T>> for $matrix_type {
            type Output = DMatrix<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }

        impl<'a, T: Scalar> Mul<Scale<T>> for &$matrix_type {
            type Output = DMatrix<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }
    };
}

impl_mul_scale!(DMatrixView<'a, T>);
impl_mul_scale!(DMatrix<T>);

impl<'a, T: Scalar> MulAssign<Scale<T>> for DMatrixViewMut<'a, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        *self *= rhs.value();
    }
}

impl<'a, T: Scalar> MatrixView<'a> for DMatrixView<'a, T> {
    type Owned = DMatrix<T>;

    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        y.gemv(alpha, self, x, beta);
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.gemv(alpha, self, x, beta);
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
    fn from_diagonal(v: &DVector<T>) -> Self {
        Self::from_diagonal(v)
    }
    fn diagonal(&self) -> Self::V {
        self.diagonal()
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.gemv(alpha, self, x, beta);
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
}

impl<T: Scalar> DenseMatrix for DMatrix<T> {
    type View<'a> = DMatrixView<'a, T>;
    type ViewMut<'a> = DMatrixViewMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
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
