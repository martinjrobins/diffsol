use nalgebra::{DVector, DMatrix};
use anyhow::Result;

use crate::{Scalar, IndexType};

use super::Matrix;


impl<T: Scalar> Matrix<T, DVector<T>> for DMatrix<T> {
    fn rows(&self) -> IndexType {
        self.nrows()
    }
    fn cols(&self) -> IndexType {
        self.ncols()
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self::zeros(nrows, ncols)
    }
    fn mul_to(&self, x: &DVector<T>, y: &mut DVector<T>) {
        self.mul_to(x, y)
    }
    fn from_diagonal(v: &DVector<T>) -> Self {
        Self::from_diagonal(v)
    }
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self> {
        let mut m = Self::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(m)
    }
    fn diagonal(&self) -> DVector<T> {
        self.diagonal()
    }
}