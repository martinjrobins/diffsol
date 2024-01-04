use nalgebra::DVector;
use nalgebra_sparse::{CsrMatrix, CooMatrix};
use anyhow::Result;

use crate::{Scalar, IndexType};

use super::Matrix;


impl<T: Scalar> Matrix<T, DVector<T>> for CsrMatrix<T> 
{
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
        *y = self * x;
    }
    fn from_diagonal(v: &DVector<T>) -> Self {
        let row_indices = (0..v.len()).map(|i| i as IndexType).collect::<Vec<_>>();
        let col_indices = (0..v.len()).map(|i| i as IndexType).collect::<Vec<_>>();
        let values = v.iter().copied().collect::<Vec<_>>();
        let coo = CooMatrix::try_from_triplets(v.len() as IndexType, v.len() as IndexType, row_indices, col_indices, values).unwrap();
        Self::from(&coo)
    }
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self> {
        let mut row_indices = Vec::with_capacity(triplets.len());
        let mut col_indices = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        for (i, j, v) in triplets {
            row_indices.push(i);
            col_indices.push(j);
            values.push(v);
        }
        let coo = CooMatrix::try_from_triplets(nrows, ncols, row_indices, col_indices, values).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self::from(&coo))
    }
    fn diagonal(&self) -> DVector<T> {
        self.diagonal()
    }
}