use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::scalar::IndexType;
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

impl<'a> MatrixView<'a> for MatRef<'a, f64> {
    type Owned = Mat<f64>;
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

impl DenseMatrix for Mat<f64> {
    type View<'a> = MatRef<'a, f64>;
    type ViewMut<'a> = MatMut<'a, f64>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.gemv(alpha, self, x, beta);
    }
    fn column_mut(&mut self, i: IndexType) -> ColMut<'_, f64> {
        self.column_mut(i)
    }

    fn columns_mut(&mut self, start: IndexType, nrows: IndexType) -> Self::ViewMut<'_> {
        self.columns_mut(start, nrows)
    }

    fn column(&self, i: IndexType) -> ColRef<'_, f64> {
        self.column(i)
    }
    fn columns(&self, start: IndexType, nrows: IndexType) -> Self::View<'_> {
        self.columns(start, nrows)
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
        self = other.clone().as_mut();
    }
    fn from_diagonal(v: &Col<f64>) -> Self {
        let dim = v.len();
        Self::from_fn(dim, dim, |i, j| if i == j { v[i] } else { 0.0 })
    }
    fn diagonal(&self) -> Self::V {
        let dim = self.nrows().min(self.ncols());
        Col::from_fn(dim, |i| self[(i, i)])
    }
}
