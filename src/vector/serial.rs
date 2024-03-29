use nalgebra::{DVector, DVectorView, DVectorViewMut};

use crate::{IndexType, Scalar};

use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

impl VectorIndex for DVector<IndexType> {
    fn zeros(len: IndexType) -> Self {
        DVector::from_element(len, 0)
    }
    fn len(&self) -> crate::IndexType {
        self.len()
    }
}

impl<T: Scalar> VectorCommon for DVector<T> {
    type T = T;
}

impl<'a, T: Scalar> VectorCommon for DVectorView<'a, T> {
    type T = T;
}

impl<'a, T: Scalar> VectorCommon for DVectorViewMut<'a, T> {
    type T = T;
}

impl<'a, T: Scalar> VectorView<'a> for DVectorView<'a, T> {
    type Owned = DVector<T>;
    fn abs(&self) -> DVector<T> {
        self.abs()
    }
    fn into_owned(self) -> Self::Owned {
        self.into_owned()
    }
}

impl<'a, T: Scalar> VectorViewMut<'a> for DVectorViewMut<'a, T> {
    type Owned = DVector<T>;
    type View = DVectorView<'a, T>;
    fn abs(&self) -> DVector<T> {
        self.abs()
    }
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.copy_from(other);
    }
}

impl<T: Scalar> Vector for DVector<T> {
    type View<'a> = DVectorView<'a, T>;
    type ViewMut<'a> = DVectorViewMut<'a, T>;
    type Index = DVector<IndexType>;
    fn len(&self) -> IndexType {
        self.len()
    }
    fn norm(&self) -> Self::T {
        self.norm()
    }
    fn abs(&self) -> Self {
        self.abs()
    }
    fn as_view(&self) -> Self::View<'_> {
        self.as_view()
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        self.as_view_mut()
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
    fn exp(&self) -> Self {
        self.map(|x| x.exp())
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
    fn filter_indices<F: Fn(T) -> bool>(&self, f: F) -> Self::Index {
        let mut indices = vec![];
        for (i, &x) in self.iter().enumerate() {
            if f(x) {
                indices.push(i as IndexType);
            }
        }
        Self::Index::from_vec(indices)
    }
    fn gather_from(&mut self, other: &Self, indices: &Self::Index) {
        for (i, &index) in indices.iter().enumerate() {
            self[i] = other[index];
        }
    }
    fn scatter_from(&mut self, other: &Self, indices: &Self::Index) {
        for (i, &index) in indices.iter().enumerate() {
            self[index] = other[i];
        }
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
