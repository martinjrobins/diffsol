use std::ops::{Index, IndexMut, Deref, Mul, MulAssign, Div, DivAssign, Add, Sub, AddAssign, SubAssign};
use std::fmt::{Debug, Display};

use nalgebra::{ClosedMul, ClosedDiv, ClosedAdd, ClosedSub, Owned};

use crate::{Scalar, IndexType};

mod serial;

pub trait VectorIndex: Sized + Index<IndexType, Output=IndexType> + Debug + Display {
    fn len(&self) -> IndexType;
}

pub trait VectorCommon<T: Scalar>: Sized + Index<IndexType, Output=T> + Debug + Display {
    fn norm(&self) -> T;
    
    fn len(&self) -> IndexType;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn assert_eq(&self, other: &Self, tol: T) {
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

pub trait VectorViewMut<'a, T: Scalar>: 
    VectorCommon<T>
    + for<'b> AddAssign<&'b Self::Owned> 
    + for<'b> SubAssign<&'b Self::Owned> 
    + AddAssign<Self::Owned> 
    + AddAssign<Self::Owned> 
    + MulAssign<T> 
    + DivAssign<T> 
    + IndexMut<IndexType, Output=T> 
{
    type Owned: Vector<T>;
    fn abs(&self) -> Self::Owned;
    fn copy_from(&mut self, other: &Self::Owned);
    fn copy_from_view(&mut self, other: &<Self::Owned as Vector<T>>::View<'_>);
}

pub trait VectorView<'a, T: Scalar>: 
    VectorCommon<T>
    + for<'b> Add<&'b Self, Output = Self::Owned> 
    + for<'b> Sub<&'b Self, Output = Self::Owned> 
    + Add<Self, Output = Self::Owned> 
    + Sub<Self, Output = Self::Owned> 
    + Mul<T, Output = Self::Owned>
    + Div<T, Output = Self::Owned>
{
    type Owned: Vector<T>;
    fn abs(&self) -> Self::Owned;
    fn into_owned(self) -> Self::Owned;
}

pub trait Vector<T: Scalar>:
    VectorCommon<T>
    + for<'a> ClosedAdd<&'a Self, Output = Self> 
    + for<'a> ClosedSub<&'a Self, Output = Self> 
    + for<'a> ClosedAdd<Self::View<'a>, Output = Self> 
    + for<'a> ClosedSub<Self::View<'a>, Output = Self> 
    + ClosedAdd<Self, Output = Self> 
    + ClosedSub<Self, Output = Self> 
    + ClosedMul<T, Output = Self> 
    + ClosedDiv<T, Output = Self> 
    + IndexMut<IndexType, Output=T> 
    + Clone
where
    for <'a> &'a Self: Add<Self, Output = Self>,
    for <'a> &'a Self: Sub<Self, Output = Self>,
    for <'a> &'a Self: Mul<T, Output = Self>,
    for <'a> &'a Self: Div<T, Output = Self>,
    for <'a, 'b> &'a Self: Add<&'b Self, Output = Self>,
    for <'a, 'b> &'a Self: Sub<&'b Self, Output = Self>,
{
    type View<'a>: VectorView<'a, T, Owned = Self> where Self: 'a;
    type ViewMut<'a>: VectorViewMut<'a, T, Owned = Self> where Self: 'a;
    type Index: VectorIndex;
    fn abs(&self) -> Self;
    fn from_element(nstates: usize, value: T) -> Self;
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, T::zero())
    }
    fn copy_from(&mut self, other: &Self);
    fn copy_from_view(&mut self, other: &Self::View<'_>);
    fn from_vec(vec: Vec<T>) -> Self;
    fn axpy(&mut self, alpha: T, x: &Self, beta: T);
    fn add_scalar_mut(&mut self, scalar: T);
    fn component_mul_assign(&mut self, other: &Self);
    fn component_div_assign(&mut self, other: &Self);
    fn map_mut<F: Fn(T) -> T>(&mut self, f: F);
    fn filter_indices<F: Fn(T) -> bool>(&self, f: F) -> Self::Index;
    fn gather_from(&mut self, other: &Self, indices: &Self::Index);
    fn scatter_from(&mut self, other: &Self, indices: &Self::Index);
}

// create macro to impl generic op for reference to vector
// macro takes a single  op name (e.g Add, Sub, Mul, Div) and its method name (e.g. add, sub, mul, div) and impliments the op for reference to vector
macro_rules! impl_op_for_ref {
    ($op:ident, $method:ident) => (
        impl <'a, T: Scalar, V: Vector<T>> $op<V> for &'a V 
        {
            type Output = V;
            fn $method(self, other: &'a V) -> V {
                V::$method(self, other)
            }
        }
    )
}
impl_op_for_ref!(Add, add);
impl_op_for_ref!(Sub, sub);




macro_rules! impl_mul_for_ref {
    ($($t:ty),*) => ($(
        impl <'a, T: Scalar, V: Vector<T>> Add<V> for &'a V {
            type Output = V;
            fn mul(self, other: $t) -> V {
                V::mul(self, &V::from_element(self.len(), other))
            }
        }
    )*)
}

macro_rules! impl_add_for_ref {
    ($($t:ty),*) => ($(
        impl <'a, T: Scalar, V: Vector<T>> Add<$t> for &'a V {
            type Output = V;
            fn add(self, other: $t) -> V {
                V::add(self, &V::from_element(self.len(), other))
            }
        }
    )*)
}




impl <'a, T: Scalar, V: Vector<T>> Add<&'a V> for &'a V {
    type Output = V;
    fn add(self, other: &'a V) -> V {
        V::add(self, other)
    }
}

// impl sub for reference to vector
impl <'a, T: Scalar, V: Vector<T>> Sub<&'a V> for &'a V {
    type Output = V;
    fn sub(self, other: &'a V) -> V {
        V::sub(self, other)
    }
}

impl <'a, T: Scalar, V: Vector<T>> VectorCommon<T> for &'a V {
    fn norm(&self) -> T {
        V::norm(self)
    }
    fn len(&self) -> IndexType {
        V::len(self)
    }
}
