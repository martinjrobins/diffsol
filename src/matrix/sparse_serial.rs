use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use anyhow::Result;

use crate::{IndexType, Scalar};

use super::{Matrix, MatrixCommon};



impl<T: Scalar> MatrixCommon for CscMatrix<T> {
    type V = DVector<T>;
    type T = T;
    
    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
    
}

impl<T: Scalar> Matrix for CscMatrix<T> {
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for (i, j, v) in triplets {
            coo.push(i, j, v);
        }
        Ok(CscMatrix::from(&coo))
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self::zeros(nrows, ncols)
    }
    fn from_diagonal(v: &DVector<T>) -> Self {
        let nrows = v.len();
        let ncols = v.len();
        let mut coo = CooMatrix::<T>::new(nrows, ncols);
        for (i, &v) in v.into_iter().enumerate() {
            coo.push(i, i, v);
        }
        CscMatrix::from(&coo)
    }
    fn diagonal(&self) -> Self::V {
        let mut ret = DVector::zeros(self.nrows());
        for (i, _j, &v) in self.diagonal_as_csc().triplet_iter() {
            ret[i] = v;
        }
        ret
    }
}