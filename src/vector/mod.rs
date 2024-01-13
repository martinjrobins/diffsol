use std::ops::{Index, IndexMut, Add, Sub};
use std::fmt::{Debug, Display};

use nalgebra::{ClosedMul, ClosedDiv, ClosedAdd, ClosedSub};

use crate::{Scalar, IndexType};

mod serial;

pub trait VectorView<T: Scalar, V: Vector<T>>: 
    Index<IndexType, Output=T> 
    + IndexMut<IndexType, Output=T> 
    + Debug + Display + Clone
    + for<'a> ClosedAdd<&'a Self> 
    + for<'a> ClosedSub<&'a Self> 
    + ClosedMul<T, Output=Self> 
    + ClosedDiv<T, Output=Self>
{
    fn len(&self) -> IndexType;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn axpy(&mut self, alpha: T, x: &V);
    fn norm(&self) -> T;
    fn abs(&self) -> Self;
    fn add_scalar_mut(&mut self, scalar: T);
    fn component_mul_assign(&mut self, other: &Self);
    fn component_div_assign(&mut self, other: &Self);
    fn map_mut<F: Fn(T) -> T>(&mut self, f: F);
    fn assert_eq(&self, other: &Self, tol: T) {
        assert_eq!(self.len(), other.len(), "Vector length mismatch: {} != {}", self.len(), other.len());
        for i in 0..self.len() {
            if T::abs(&(self[i] - other[i])) > tol {
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

pub trait Vector<T: Scalar>: VectorView<T, Self>
{
    fn from_element(nstates: usize, value: T) -> Self;
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, T::zero())
    }
    fn from_vec(vec: Vec<T>) -> Self;
}



