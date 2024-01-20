use nalgebra::{DVector, DVectorView, DVectorViewMut};

use crate::{Scalar, IndexType};

use super::{Vector, VectorView, VectorCommon, VectorViewMut, VectorIndex};

impl VectorIndex for DVector<IndexType> {
    fn len(&self) -> crate::IndexType {
        self.len()
    }
}

impl<T: Scalar> VectorCommon<T> for DVector<T> {
    fn len(&self) -> crate::IndexType {
        self.len()
    }
    fn norm(&self) -> T {
        self.norm()
    }
}

impl<'a, T: Scalar> VectorCommon<T> for DVectorView<'a, T> {
    fn len(&self) -> crate::IndexType {
        self.len()
    }
    fn norm(&self) -> T {
        self.norm()
    }
}

impl<'a, T: Scalar> VectorCommon<T> for DVectorViewMut<'a, T> {
    fn len(&self) -> crate::IndexType {
        self.len()
    }
    fn norm(&self) -> T {
        self.norm()
    }
}


impl<'a, T: Scalar> VectorView<'a, T> for DVectorView<'a, T> {
    type Owned = DVector<T>;
    fn abs(&self) -> DVector<T> {
        self.abs()
    }
    fn into_owned(self) -> Self::Owned {
        self.into_owned()
    }
}

impl<'a, T: Scalar> VectorViewMut<'a, T> for DVectorViewMut<'a, T> {
    type Owned = DVector<T>;
    fn abs(&self) -> DVector<T> {
        self.abs()
    }
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &<Self::Owned as Vector<T>>::View<'_>) {
        self.copy_from(other);
    }
}

impl<T: Scalar> Vector<T> for DVector<T> {
    type View<'a> = DVectorView<'a, T>;
    type ViewMut<'a> = DVectorViewMut<'a, T>;
    type Index = DVector<IndexType>;
    fn abs(&self) -> Self {
        self.abs()
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.copy_from(other);
    }
    fn from_element(nstates: usize, value: T) -> Self {
        Self::from_element(nstates, value)
    }
    fn from_vec(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
    fn zeros(nstates: usize) -> Self {
        Self::zeros(nstates)
    }
    fn add_scalar_mut(&mut self, scalar: T) {
        self.add_scalar_mut(scalar);
    }
    fn axpy(&mut self, alpha: T, x: &Self, beta: T) {
        self.axpy(alpha, x, beta);
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.component_div_assign(other);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.component_mul_assign(other);
    }
    fn map_mut<F: Fn(T) -> T>(&mut self, f: F) {
        self.map_mut(f);
    }
    fn filter_indices<F: Fn(T) -> bool>(&self, f: F) -> Self::Index {
        self.filter_indices(f)
    }
    fn gather_from(&mut self, other: &Self, indices: &Self::Index) {
        self.gather_from(other, indices);
    }
    fn scatter_from(&mut self, other: &Self, indices: &Self::Index) {
        self.scatter_from(other, indices);
    }
}


// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs() {
        let v = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let v_abs = v.abs();
        assert_eq!(v_abs, DVector::from_vec(vec![1.0, 2.0, 3.0]));
    }
}