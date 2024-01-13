use std::ops::Index;

use crate::{IndexType, Vector, Scalar, VectorView};
use anyhow::Result;

mod dense_serial;

pub trait MatrixView<T: Scalar, V: Vector<T>>: Index<(IndexType, IndexType), Output = T> + Clone {
    fn diagonal(&self) -> V;
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn gemm(alpha: T, a: &Self, b: &Self, beta: T, c: &mut Self);
    fn gemv(alpha: T, a: &Self, x: &V, beta: T, y: &mut V);
}

pub trait Matrix<T: Scalar, V: Vector<T>>: MatrixView<T, V> + Clone
{
    type View: MatrixView<T, V>;
    type Row: VectorView<T, V>;
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;
    fn from_diagonal(v: &V) -> Self;
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self>;
    fn rows(&self, start: IndexType, nrows: IndexType) -> Self::View;
    fn row(&self, i: IndexType) -> Self::Row;
}
