use nalgebra::{DVector, DMatrix, DMatrixView, DVectorView, DMatrixViewMut, DVectorViewMut};
use anyhow::Result;

use crate::{Scalar, IndexType};

use super::{Matrix, MatrixView, MatrixCommon, MatrixViewMut};

impl<'a, T: Scalar> MatrixCommon for DMatrixViewMut<'a, T> {
    type V = DVector<T>;
    type T = T;
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

impl<'a, T: Scalar> MatrixViewMut<'a> for DMatrixViewMut<'a, T> {
    type Owned = DMatrix<T>;
    type View = DMatrixView<'a, T>;
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
    fn gemm_ov(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::View, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
}

impl<'a, T: Scalar> MatrixCommon for DMatrixView<'a, T> {
    type V = DVector<T>;
    type T = T;
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

impl<'a, T: Scalar> MatrixView<'a> for DMatrixView<'a, T> {
    type Owned = DMatrix<T>;
}

impl<T: Scalar> MatrixCommon for DMatrix<T> {
    type V = DVector<T>;
    type T = T;
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

impl<T: Scalar> Matrix for DMatrix<T> {
    type View<'a> = DMatrixView<'a, T>;
    type ViewMut<'a> = DMatrixViewMut<'a, T>;
    
    fn column_mut(&mut self, i: IndexType) -> DVectorViewMut<'_, T> {
        self.column_mut(i)
    }
    
    fn columns_mut(&mut self, start: IndexType, nrows: IndexType) -> Self::ViewMut<'_> {
        self.columns_mut(start, nrows)
    }

    fn column(&self, i: IndexType) -> DVectorView<'_, T> {
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