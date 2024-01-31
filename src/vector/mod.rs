use std::ops::{Index, IndexMut, Deref, Mul, MulAssign, Div, DivAssign, Add, Sub, AddAssign, SubAssign};
use std::fmt::{Debug, Display};
use num_traits::{Zero, One};


use crate::{Scalar, IndexType};

mod serial;

pub trait VectorIndex: Sized + Index<IndexType, Output=IndexType> + Debug + Display {
    fn len(&self) -> IndexType;
}

pub trait VectorCommon: Sized + Debug + Display {
    type T: Scalar;
}

impl <'a, V> VectorCommon for &'a V where V: VectorCommon {
    type T = V::T;
}

impl <'a, V> VectorCommon for &'a mut V where V: VectorCommon {
    type T = V::T;
}

trait VectorOpsByValue<Rhs = Self, Output = Self>: VectorCommon 
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output> 
    + Mul<Self::T, Output = Output>
    + Div<Self::T, Output = Output>
{}

impl <V, Rhs, Output> VectorOpsByValue<Rhs, Output> for V where V: VectorCommon 
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output> 
    + Mul<Self::T, Output = Output>
    + Div<Self::T, Output = Output>
{}


pub trait VectorOps<View>: 
    VectorOpsByValue<Self> 
    + for<'a> VectorOpsByValue<&'a Self>
    + VectorOpsByValue<View> 
    + for<'a> VectorOpsByValue<&'a View>
{}

impl <V, View> VectorOps<View> for V 
where 
    V: VectorOpsByValue<Self> 
    + for<'a> VectorOpsByValue<&'a Self> 
    + VectorOpsByValue<View> 
    + for<'a> VectorOpsByValue<&'a View>
{}

pub trait VectorMutOpsByValue<Rhs = Self>: VectorCommon 
    + AddAssign<Rhs>
    + SubAssign<Rhs> 
{}

impl <V, Rhs> VectorMutOpsByValue<Rhs> for V where V: VectorCommon 
    + AddAssign<Rhs>
    + SubAssign<Rhs> 
{}

pub trait VectorMutOps<View>: 
    VectorMutOpsByValue<Self>
    + VectorMutOpsByValue<View>
    + for<'a> VectorMutOpsByValue<&'a View>
    + for<'a> VectorMutOpsByValue<&'a Self>
    + MulAssign<Self::T>
    + DivAssign<Self::T>
{}

impl <V, View> VectorMutOps<View> for V 
where 
    V: VectorMutOpsByValue<Self>
    + VectorMutOpsByValue<View>
    + for<'a> VectorMutOpsByValue<&'a Self>
    + for<'a> VectorMutOpsByValue<&'a View>
    + MulAssign<Self::T>
    + DivAssign<Self::T>
{}

pub trait VectorRef<V>:
    VectorOpsByValue<V, V>
    + for<'a> VectorOpsByValue<&'a V, V> 
    + for<'a> VectorOpsByValue<V::View<'a>, V>
    + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V> 
where
    V: Vector
{}

impl <RefT, V: Vector> VectorRef<V> for RefT where
    RefT: VectorOpsByValue<V, V>
    + for<'a> VectorOpsByValue<&'a V, V>
    + for<'a> VectorOpsByValue<V::View<'a>, V>
    + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V> 
{}

pub trait VectorViewMut<'a>: 
    VectorMutOps<Self::View>
{
    type View: VectorView<'a, T = Self::T>;
    type Owned: Vector<T = Self::T, ViewMut<'a> = Self> where Self: 'a;
    fn abs(&self) -> Self::Owned;
    fn copy_from(&mut self, other: &Self::Owned);
    fn copy_from_view(&mut self, other: &<Self::Owned as Vector>::View<'_>);
}

pub trait VectorView<'a>: 
    VectorRef<Self::Owned>
    + Index<IndexType, Output=Self::T>
{
    type Owned: Vector<T = Self::T>;
    fn abs(&self) -> Self::Owned;
    fn into_owned(self) -> Self::Owned;
}


pub trait Vector:
    for <'a> VectorOps<Self::View<'a>>
    + for <'a> VectorMutOps<Self::View<'a>>
    + Index<IndexType, Output=Self::T>
    + IndexMut<IndexType, Output=Self::T>
    + Clone
{
    type View<'a>: VectorView<'a, T = Self::T, Owned = Self> where Self: 'a;
    type ViewMut<'a>: VectorViewMut<'a, T = Self::T, Owned = Self, View = Self::View<'a>> where Self: 'a;
    type Index: VectorIndex;
    fn norm(&self) -> Self::T;
    fn len(&self) -> IndexType;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn abs(&self) -> Self;
    fn from_element(nstates: usize, value: Self::T) -> Self;
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, Self::T::zero())
    }
    fn copy_from(&mut self, other: &Self);
    fn copy_from_view(&mut self, other: &Self::View<'_>);
    fn from_vec(vec: Vec<Self::T>) -> Self;
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T);
    fn add_scalar_mut(&mut self, scalar: Self::T);
    fn component_mul_assign(&mut self, other: &Self);
    fn component_div_assign(&mut self, other: &Self);
    fn map_mut<F: Fn(Self::T) -> Self::T>(&mut self, f: F);
    fn filter_indices<F: Fn(Self::T) -> bool>(&self, f: F) -> Self::Index;
    fn gather_from(&mut self, other: &Self, indices: &Self::Index);
    fn scatter_from(&mut self, other: &Self, indices: &Self::Index);
    fn assert_eq(&self, other: &Self, tol: Self::T) {
        assert_eq!(self.len(), other.len(), "Vector length mismatch: {} != {}", self.len(), other.len());
        for i in 0..self.len() {
            if num_traits::abs(self[i] - other[i]) > tol {
                eprintln!("Vector element mismatch at index {}: {} != {}", i, self[i], other[i]);
                if self.len() <= 3 {
                    eprintln!("left: {}", self);
                    eprintln!("right: {}", other);
                } else if i == 0 {
                    eprintln!("left: [{}, {}, {}, ...] != [{}, {}, {}, ...]", self[0], self[1], self[2], other[0], other[1], other[2]);
                } else if i == self.len() - 1 {
                    eprintln!("left: [..., {}, {}, {}] != [..., {}, {}, {}]", self[i-2], self[i-1], self[i], other[i-2], other[i-1], other[i]);
                } else {
                    eprintln!("left: [..., {}, {}, {}, ...] != [..., {}, {}, {}, ...]", self[i-1], self[i], self[i+1], other[i-1], other[i], other[i+1]);
                }
                break;
            }
        }
    }
}