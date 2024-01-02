use nalgebra::{LU, DVector, ComplexField, Dyn, DMatrix};
use anyhow::Result;

use crate::Scalar;

use super::LinearSolver;


// implement LinearSolver for LU, DMatrix and DVector from nalgebra
impl<T> LinearSolver<T, DVector<T>, DMatrix<T>> for LU<T, Dyn, Dyn>
where
    T: Scalar + ComplexField,
{
    fn new(a: DMatrix<T>) -> Self {
        Self::new(a)
    }

    fn solve(&self, b: &DVector<T>) -> Result<DVector<T>> {
        match LU::solve(self, b) {
            Some(x) => Ok(x),
            None => Err(anyhow::anyhow!("LU solve failed")),
        }
    }
}
