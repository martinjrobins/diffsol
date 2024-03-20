use std::ops::{Index, IndexMut, Mul, MulAssign, Div, DivAssign, Add, Sub, AddAssign, SubAssign};
use std::fmt::{Debug, Display};
use num_traits::Zero;


use crate::{Scalar, IndexType};

mod serial;

pub trait VectorIndex: Sized + Index<IndexType, Output=IndexType> + Debug + Display {
    fn zeros(len: IndexType) -> Self;
    fn len(&self) -> IndexType;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
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

pub trait VectorOpsByValue<Rhs = Self, Output = Self>: VectorCommon 
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output> 
{}

impl <V, Rhs, Output> VectorOpsByValue<Rhs, Output> for V where V: VectorCommon 
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output> 
{}


pub trait VectorMutOpsByValue<Rhs = Self>: VectorCommon 
    + AddAssign<Rhs>
    + SubAssign<Rhs> 
{}

impl <V, Rhs> VectorMutOpsByValue<Rhs> for V where V: VectorCommon 
    + AddAssign<Rhs>
    + SubAssign<Rhs> 
{}

pub trait VectorRef<V: Vector>:
    VectorOpsByValue<V, V>
    + for<'a> VectorOpsByValue<&'a V, V> 
    + for<'a> VectorOpsByValue<V::View<'a>, V>
    + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V> 
    + Mul<V::T, Output = V>
    + Div<V::T, Output = V>
{}

impl <RefT, V: Vector> VectorRef<V> for RefT where
    RefT: VectorOpsByValue<V, V>
    + for<'a> VectorOpsByValue<&'a V, V>
    + for<'a> VectorOpsByValue<V::View<'a>, V>
    + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V> 
    + Mul<V::T, Output = V>
    + Div<V::T, Output = V>
{}

pub trait VectorViewMut<'a>: 
    VectorMutOpsByValue<Self::View>
    + VectorMutOpsByValue<Self::Owned>
    + for<'b> VectorMutOpsByValue<&'b Self::View>
    + for<'b> VectorMutOpsByValue<&'b Self::Owned>
    + MulAssign<Self::T>
    + DivAssign<Self::T>
    + Index<IndexType, Output=Self::T>
    + IndexMut<IndexType, Output=Self::T>
{
    type Owned;
    type View;
    fn abs(&self) -> Self::Owned;
    fn copy_from(&mut self, other: &Self::Owned);
    fn copy_from_view(&mut self, other: &Self::View);
}

pub trait VectorView<'a>: 
    VectorOpsByValue<Self, Self::Owned> 
    + VectorOpsByValue<Self::Owned, Self::Owned> 
    + for<'b> VectorOpsByValue<&'b Self::Owned, Self::Owned> 
    + for<'b> VectorOpsByValue<&'b Self, Self::Owned> 
    + Mul<Self::T, Output = Self::Owned>
    + Div<Self::T, Output = Self::Owned>
    + Index<IndexType, Output=Self::T>
{
    type Owned;
    fn abs(&self) -> Self::Owned;
    fn into_owned(self) -> Self::Owned;
}


pub trait Vector:
    VectorOpsByValue<Self> 
    + for<'b> VectorOpsByValue<&'b Self>
    + for<'a> VectorOpsByValue<Self::View<'a>> 
    + for<'a, 'b> VectorOpsByValue<&'b Self::View<'a>>
    + Mul<Self::T, Output = Self>
    + Div<Self::T, Output = Self>

    + VectorMutOpsByValue<Self>
    + for<'a> VectorMutOpsByValue<Self::View<'a>>
    + for<'b> VectorMutOpsByValue<&'b Self>
    + for<'a, 'b> VectorMutOpsByValue<&'b Self::View<'a>>
    + MulAssign<Self::T>
    + DivAssign<Self::T>

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
    fn as_view(&self) -> Self::View<'_>;
    fn as_view_mut(&mut self) -> Self::ViewMut<'_>;
    fn copy_from(&mut self, other: &Self);
    fn copy_from_view(&mut self, other: &Self::View<'_>);
    fn from_vec(vec: Vec<Self::T>) -> Self;
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T);
    fn add_scalar_mut(&mut self, scalar: Self::T);
    fn component_mul_assign(&mut self, other: &Self);
    fn component_div_assign(&mut self, other: &Self);
    fn filter_indices<F: Fn(Self::T) -> bool>(&self, f: F) -> Self::Index;
    fn filter(&self, indices: &Self::Index) -> Self {
        let mut result = Self::zeros(indices.len());
        result.gather_from(self, indices);
        result
    }
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
                panic!("Vector element mismatch at index {}: {} != {}", i, self[i], other[i]);
            }
        }
    }
}