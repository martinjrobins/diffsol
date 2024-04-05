use std::ops::{Mul, MulAssign};

use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::scalar::{IndexType, Scalar, Scale};
use anyhow::Result;
use faer::{linalg::matmul::matmul, Col, ColMut, ColRef, Mat, MatMut, MatRef, Parallelism};

macro_rules! impl_matrix_common {
    ($mat_type:ty) => {
        impl<'a, T: Scalar> MatrixCommon for $mat_type {
            type T = T;
            type V = Col<T>;

            fn ncols(&self) -> IndexType {
                self.ncols()
            }

            fn nrows(&self) -> IndexType {
                self.nrows()
            }
        }
    };
}

impl_matrix_common!(MatMut<'a, T>);
impl_matrix_common!(MatRef<'a, T>);
impl_matrix_common!(Mat<T>);

macro_rules! impl_mul_scale {
    ($mat_type:ty) => {
        impl<'a, T: Scalar> Mul<Scale<T>> for $mat_type {
            type Output = Mat<T>;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
                self * scale
            }
        }
    };
}
impl_mul_scale!(MatRef<'a, T>);
impl_mul_scale!(Mat<T>);
impl_mul_scale!(&Mat<T>);

impl<'a, T: Scalar> MulAssign<Scale<T>> for MatMut<'a, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        let scale: faer::Scale<T> = rhs.into();
        *self *= scale;
    }
}

impl<'a, T: Scalar> MatrixView<'a> for MatRef<'a, T> {
    type Owned = Mat<T>;
}

impl<'a, T: Scalar> MatrixViewMut<'a> for MatMut<'a, T> {
    type Owned = Mat<T>;
    type View = MatRef<'a, T>;

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        matmul(
            self.as_mut(),
            a.as_ref(),
            b.as_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        )
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        matmul(
            self.as_mut(),
            a.as_ref(),
            b.as_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        )
    }
}

impl<T: Scalar> DenseMatrix for Mat<T> {
    type View<'a> = MatRef<'a, T>;
    type ViewMut<'a> = MatMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        matmul(
            self.as_mut(),
            a.as_ref(),
            b.as_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        )
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        *y = faer::scale(alpha) * self * x + faer::scale(beta) * &*y;
    }
    fn column_mut(&mut self, i: usize) -> ColMut<'_, T> {
        self.get_mut(0..self.nrows(), i)
    }

    fn columns_mut(&mut self, start: usize, ncols: usize) -> MatMut<'_, T> {
        self.get_mut(0..self.nrows(), start..ncols)
    }

    fn column(&self, i: usize) -> ColRef<'_, T> {
        self.get(0..self.nrows(), i)
    }
    fn columns(&self, start: usize, nrows: usize) -> MatRef<'_, T> {
        self.get(0..self.nrows(), start..nrows)
    }
}

impl<T: Scalar> Matrix for Mat<T> {
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
        *self = other.clone();
    }
    fn from_diagonal(v: &Col<T>) -> Self {
        let dim = v.nrows();
        Self::from_fn(dim, dim, |i, j| if i == j { v[i] } else { T::zero() })
    }
    fn diagonal(&self) -> Self::V {
        let dim = self.nrows().min(self.ncols());
        Col::from_fn(dim, |i| self[(i, i)])
    }
}