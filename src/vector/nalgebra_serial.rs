use std::ops::{Div, Mul, MulAssign};

use nalgebra::{DVector, DVectorView, DVectorViewMut};

use crate::{IndexType, Scalar, Scale};

use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

macro_rules! impl_op_for_dvector_struct {
    ($struct:ident, $trait_name:ident, $func_name:ident) => {
        impl<'a, T: Scalar> $trait_name<Scale<T>> for $struct<'a, T> {
            type Output = DVector<T>;
            fn $func_name(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }
    };
}

impl_op_for_dvector_struct!(DVectorView, Mul, mul);
impl_op_for_dvector_struct!(DVectorViewMut, Mul, mul);

impl VectorIndex for DVector<IndexType> {
    fn zeros(len: IndexType) -> Self {
        DVector::from_element(len, 0)
    }
    fn len(&self) -> crate::IndexType {
        self.len()
    }
}

macro_rules! impl_vector_common {
    ($vector_type:ty) => {
        impl<'a, T: Scalar> VectorCommon for $vector_type {
            type T = T;
        }
    };
}

impl_vector_common!(DVector<T>);
impl_vector_common!(DVectorView<'a, T>);
impl_vector_common!(DVectorViewMut<'a, T>);

impl<'a, T: Scalar> VectorView<'a> for DVectorView<'a, T> {
    type Owned = DVector<T>;
    fn abs(&self) -> DVector<T> {
        self.abs()
    }
    fn into_owned(self) -> Self::Owned {
        self.into_owned()
    }
}

macro_rules! impl_mul_scale_vector {
    ($vector_type:ty) => {
        impl<T: Scalar> Mul<Scale<T>> for $vector_type {
            type Output = DVector<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }

        impl<T: Scalar> Mul<Scale<T>> for &$vector_type {
            type Output = DVector<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }
    };
}

impl_mul_scale_vector!(DVector<T>);
macro_rules! impl_mul_assign_scale_vector {
    ($vector_type:ty) => {
        impl<'a, T: Scalar> MulAssign<Scale<T>> for $vector_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                *self *= rhs.value();
            }
        }
    };
}

impl_mul_assign_scale_vector!(DVector<T>);
impl_mul_assign_scale_vector!(DVectorViewMut<'a, T>);

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

impl<T: Scalar> Div<Scale<T>> for DVector<T> {
    type Output = DVector<T>;
    fn div(self, rhs: Scale<T>) -> Self::Output {
        self / rhs.value()
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
