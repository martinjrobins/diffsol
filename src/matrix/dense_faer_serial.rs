use std::ops::{Mul, MulAssign};

use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::scalar::{IndexType, Scale};
use anyhow::Result;
use faer::{linalg::matmul::matmul, Col, ColMut, ColRef, Mat, MatMut, MatRef, Parallelism};

impl<'a> MatrixCommon for MatMut<'a, f64> {
    type T = f64;
    type V = Col<f64>;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<'a> MatrixCommon for MatRef<'a, f64> {
    type T = f64;
    type V = Col<f64>;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl MatrixCommon for Mat<f64> {
    type T = f64;
    type V = Col<f64>;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<'a> Mul<Scale<f64>> for MatRef<'a, f64> {
    type Output = Mat<f64>;
    fn mul(self, rhs: Scale<f64>) -> Self::Output {
        self * faer::scale(rhs.value())
    }
}

impl<'a> MatrixView<'a> for MatRef<'a, f64> {
    type Owned = Mat<f64>;
}

impl<'a> MulAssign<Scale<f64>> for MatMut<'a, f64> {
    fn mul_assign(&mut self, rhs: Scale<f64>) {
        *self *= faer::scale(rhs.value());
    }
}

impl<'a> MatrixViewMut<'a> for MatMut<'a, f64> {
    type Owned = Mat<f64>;
    type View = MatRef<'a, f64>;

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

impl Mul<Scale<f64>> for Mat<f64> {
    type Output = Mat<f64>;
    fn mul(self, rhs: Scale<f64>) -> Self::Output {
        self * faer::scale(rhs.value())
    }
}

impl Mul<Scale<f64>> for &Mat<f64> {
    type Output = Mat<f64>;
    fn mul(self, rhs: Scale<f64>) -> Self::Output {
        self * faer::scale(rhs.value())
    }
}

impl DenseMatrix for Mat<f64> {
    type View<'a> = MatRef<'a, f64>;
    type ViewMut<'a> = MatMut<'a, f64>;

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
    fn column_mut(&mut self, i: usize) -> ColMut<'_, f64> {
        self.get_mut(0..self.nrows(), i)
    }

    fn columns_mut(&mut self, start: usize, ncols: usize) -> MatMut<'_, f64> {
        self.get_mut(0..self.nrows(), start..ncols)
    }

    fn column(&self, i: usize) -> ColRef<'_, f64> {
        self.get(0..self.nrows(), i)
    }
    fn columns(&self, start: usize, nrows: usize) -> MatRef<'_, f64> {
        self.get(0..self.nrows(), start..nrows)
    }
}

impl Matrix for Mat<f64> {
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, f64)>,
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
    fn from_diagonal(v: &Col<f64>) -> Self {
        let dim = v.nrows();
        Self::from_fn(dim, dim, |i, j| if i == j { v[i] } else { 0.0 })
    }
    fn diagonal(&self) -> Self::V {
        let dim = self.nrows().min(self.ncols());
        Col::from_fn(dim, |i| self[(i, i)])
    }
}
