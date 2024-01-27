use std::ops::{Index, Add, Sub, Div, Mul, AddAssign, SubAssign, MulAssign, DivAssign};

use crate::{IndexType, Vector, Scalar};
use anyhow::Result;
use nalgebra::{ClosedAdd, ClosedSub, ClosedMul, ClosedDiv};

mod dense_serial;

pub trait MatrixCommon<T: Scalar, V: Vector<T>>: 
    Index<(IndexType, IndexType), Output = T> 
    + Sized
{
    fn diagonal(&self) -> V;
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn gemv(&self, alpha: T, x: &V, beta: T, y: &mut V);
}

pub trait MatrixViewMut<'a, T: Scalar, V: Vector<T>>: MatrixCommon<T, V> 
    + for<'b> AddAssign<&'b Self::Owned> 
    + for<'b> SubAssign<&'b Self::Owned> 
    + AddAssign<Self::Owned> 
    + AddAssign<Self::Owned> 
    + MulAssign<T> 
    + DivAssign<T> 
{
    type Owned: Matrix<T, V>;
    fn gemm_oo(&mut self, alpha: T, a: &Self::Owned, b: &Self::Owned, beta: T);
    fn gemm_ov(&mut self, alpha: T, a: &Self::Owned, b: &<Self::Owned as Matrix<T, V>>::View<'_>, beta: T);
}

pub trait MatrixView<'a, T: Scalar, V: Vector<T>>: MatrixCommon<T, V> 
    + for<'b> Add<&'b Self, Output = Self::Owned> 
    + for<'b> Sub<&'b Self, Output = Self::Owned> 
    + Add<Self, Output = Self::Owned>
    + Sub<Self, Output = Self::Owned>
    + Mul<T, Output = Self::Owned>
    + Div<T, Output = Self::Owned>
    + Clone 
{
    type Owned: Matrix<T, V>;
}

pub trait Matrix<T: Scalar, V: Vector<T>>: MatrixCommon<T, V>
    + for<'a> ClosedAdd<&'a Self, Output = Self> 
    + for<'a> ClosedSub<&'a Self, Output = Self> 
    + ClosedAdd<Self, Output = Self> 
    + ClosedSub<Self, Output = Self> 
    + ClosedMul<T, Output = Self> 
    + ClosedDiv<T, Output = Self> 
    + Clone 
{
    type View<'a>: MatrixView<'a, T, V, Owned = Self> where Self: 'a;
    type ViewMut<'a>: MatrixViewMut<'a, T, V, Owned = Self> where Self: 'a;
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;
    fn from_diagonal(v: &V) -> Self;
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, T)>) -> Result<Self>;
    fn columns(&self, start: IndexType, nrows: IndexType) -> Self::View<'_>;
    fn column(&self, i: IndexType) -> <V as Vector<T>>::View<'_>;
    fn columns_mut(&self, start: IndexType, nrows: IndexType) -> Self::ViewMut<'_>;
    fn column_mut(&self, i: IndexType) -> <V as Vector<T>>::ViewMut<'_>;
    fn add_assign_column(&self, i: IndexType, other: &V);
    fn gemm(&mut self, alpha: T, a: &Self, b: &Self, beta: T);
}
