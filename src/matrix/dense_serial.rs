use nalgebra::{DVector, DMatrix};
use anyhow::Result;

use crate::{Scalar, IndexType};

use super::{Matrix, MatrixView};


impl<T: Scalar> MatrixView<T, DVector<T>> for DMatrix<T> {
    fn diagonal(&self) -> DVector<T> {
        self.diagonal()
    }
    fn gemv(alpha: T, a: &Self, x: &DVector<T>, beta: T, y: &mut DVector<T>) {
        y.gemv(alpha, a, x, beta);
    }
    fn gemm(alpha: T, a: &Self, b: &Self, beta: T, c: &mut Self) {
        c.gemm(alpha, a, b, beta);
    }
    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<T: Scalar> Matrix<T, DVector<T>> for DMatrix<T> {
    type View = DMatrix<T>;
    type Row = DVector<T>;
    fn rows(&self, start: IndexType, nrows: IndexType) -> Self::View {
        self.rows(start, nrows)
    }
    fn row(&self, i: IndexType) -> Self::Row {
        self.row(i)
    }
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self> {
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
}