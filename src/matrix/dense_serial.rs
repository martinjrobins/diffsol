use nalgebra::{DVector, DMatrix, DMatrixView, DVectorView, DMatrixViewMut, DVectorViewMut};
use anyhow::Result;

use crate::{Scalar, IndexType};

use super::{Matrix, MatrixView, MatrixCommon};

impl<'a, T: Scalar> MatrixCommon<T, DVector<T>> for DMatrixView<'a, T> {
    fn diagonal(&self) -> DVector<T> {
        self.diagonal()
    }
    fn gemv(&self, alpha: T, x: &DVector<T>, beta: T, y: &mut DVector<T>) {
        y.gemv(alpha, self, x, beta);
    }
    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<'a, T: Scalar> MatrixView<'a, T, DVector<T>> for DMatrixView<'a, T> {
    type Owned = DMatrix<T>;
}

impl<T: Scalar> MatrixCommon<T, DVector<T>> for DMatrix<T> {
    fn diagonal(&self) -> DVector<T> {
        self.diagonal()
    }
    fn gemv(&self, alpha: T, x: &DVector<T>, beta: T, y: &mut DVector<T>) {
        y.gemv(alpha, self, x, beta);
    }
    
    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<T: Scalar> Matrix<T, DVector<T>> for DMatrix<T> {
    type View<'a> = DMatrixView<'a, T>;
    type Column<'a> = DVectorView<'a, T>;
    

    fn set_columns(&self, start: IndexType, nrows: IndexType, other: Self::View<'_>) {
        self.columns_mut(start, nrows).copy_from(&other);
    }

    fn set_column(&self, i: IndexType, other: &DVector<T>) {
        self.column_mut(i).copy_from(other);
    }
    
    fn add_assign_column(&self, i: IndexType, other: &DVector<T>) {
        self.set_column(i, &(self.column(i) + other));
    }

    fn column(&self, i: IndexType) -> Self::Column<'_> {
        self.column(i)
    }
    fn columns(&self, start: IndexType, nrows: IndexType) -> Self::View<'_> {
        self.columns(start, nrows)
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
    fn gemm(&mut self, alpha: T, a: &Self, b: &Self, beta: T) {
        self.gemm(alpha, a, b, beta);
    }

}