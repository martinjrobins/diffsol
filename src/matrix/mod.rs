use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::fmt::{Debug, Display};

use crate::{IndexType, Scalar, Vector};
use anyhow::Result;

mod dense_serial;

pub trait MatrixCommon: Sized + Debug + Display
{
    type V: Vector<T = Self::T>;
    type T: Scalar;
    fn diagonal(&self) -> Self::V;
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);
}

impl <'a, M> MatrixCommon for &'a M where M: MatrixCommon {
    type T = M::T;
    type V = M::V;
    fn diagonal(&self) -> Self::V {
        M::diagonal(self)
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        M::gemv(self, alpha, x, beta, y)
    }
    fn ncols(&self) -> IndexType {
        M::ncols(self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(self)
    }
}

impl <'a, M> MatrixCommon for &'a mut M where M: MatrixCommon {
    type T = M::T;
    type V = M::V;
    fn diagonal(&self) -> Self::V {
        M::diagonal(self)
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        M::gemv(self, alpha, x, beta, y)
    }
    fn ncols(&self) -> IndexType {
        M::ncols(self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(self)
    }
}

pub trait MatrixOpsByValue<Rhs = Self, Output = Self>: MatrixCommon 
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output> 
{}

impl <M, Rhs, Output> MatrixOpsByValue<Rhs, Output> for M where M: MatrixCommon 
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output> 
{}

pub trait MatrixMutOpsByValue<Rhs = Self>: MatrixCommon 
    + AddAssign<Rhs>
    + SubAssign<Rhs> 
{}

impl <M, Rhs> MatrixMutOpsByValue<Rhs> for M where M: MatrixCommon 
    + AddAssign<Rhs>
    + SubAssign<Rhs> 
{}

pub trait MatrixMutOps<View>: 
    MatrixMutOpsByValue<Self> 
    + for<'a> MatrixMutOpsByValue<&'a Self>
    + MatrixMutOpsByValue<View> 
    + for<'a> MatrixMutOpsByValue<&'a View>
    + MulAssign<Self::T>
    + DivAssign<Self::T>
{}

impl <M, View> MatrixMutOps<View> for M 
where 
    M: MatrixMutOpsByValue<Self> 
    + for<'a> MatrixMutOpsByValue<&'a Self> 
    + MatrixMutOpsByValue<View> 
    + for<'a> MatrixMutOpsByValue<&'a View>
    + MulAssign<Self::T>
    + DivAssign<Self::T>
{}

pub trait MatrixOps<View>: 
    MatrixOpsByValue<Self> 
    + for<'a> MatrixOpsByValue<&'a Self>
    + MatrixOpsByValue<View> 
    + for<'a> MatrixOpsByValue<&'a View>
    + Mul<Self::T, Output = Self>
    + Div<Self::T, Output = Self>
{}

impl <M, View> MatrixOps<View> for M 
where 
    M: MatrixOpsByValue<Self> 
    + for<'a> MatrixOpsByValue<&'a Self> 
    + MatrixOpsByValue<View> 
    + for<'a> MatrixOpsByValue<&'a View>
    + Mul<Self::T, Output = Self>
    + Div<Self::T, Output = Self>
{}

pub trait MatrixRef<M: MatrixCommon>:
    MatrixOpsByValue<M, M>
    + for<'a> MatrixOpsByValue<M, M> 
    + Mul<M::T, Output = M>
    + Div<M::T, Output = M>
{}

impl <RefT, M: MatrixCommon> MatrixRef<M> for RefT where
    RefT: MatrixOpsByValue<M, M>
    + for<'a> MatrixOpsByValue<&'a M, M>
    + Mul<M::T, Output = M>
    + Div<M::T, Output = M>
{}


pub trait MatrixViewMut<'a>: 
    MatrixMutOps<Self::View>
{
    type Owned: Matrix<V = Self::V>;
    type View: MatrixView<'a, V = Self::V, Owned = Self::Owned, T = Self::T>;
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T);
    fn gemm_ov(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::View, beta: Self::T);
}

pub trait MatrixView<'a>: 
    MatrixRef<Self::Owned>
    + Clone 
{
    type Owned: Matrix<V = Self::V>;
}

pub trait Matrix: 
    for <'a> MatrixOps<Self::View<'a>>
    + for <'a> MatrixMutOps<Self::View<'a>>
    + Index<(IndexType, IndexType), Output = Self::T> 
    + IndexMut<(IndexType, IndexType), Output = Self::T> 
    + Clone 
{
    type View<'a>: MatrixView<'a, Owned = Self, T = Self::T> where Self: 'a;
    type ViewMut<'a>: MatrixViewMut<'a, Owned = Self, T = Self::T, View = Self::View<'a>> where Self: 'a;
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;
    fn from_diagonal(v: &Self::V) -> Self;
    fn try_from_triplets(nrows: IndexType, ncols: IndexType, triplets: Vec<(IndexType, IndexType, Self::T)>) -> Result<Self>;
    fn columns(&self, start: IndexType, nrows: IndexType) -> Self::View<'_>;
    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_>;
    fn columns_mut(&mut self, start: IndexType, nrows: IndexType) -> Self::ViewMut<'_>;
    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_>;
    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T);
}
