use crate::{IndexType, Vector, Scalar};
use anyhow::Result;

mod sparse_csr;
mod dense_serial;

pub trait Matrix<T: Scalar, V: Vector<T>>: Clone
{
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;
    fn from_diagonal(v: &V) -> Self;
    fn diagonal(&self) -> V;
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self>;
    fn rows(&self) -> IndexType;
    fn cols(&self) -> IndexType;
    fn mul_to(&self, x: &V, y: &mut V);
}
