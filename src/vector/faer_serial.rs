use std::ops::{Div, Mul, MulAssign};

use faer::{unzipped, zipped, Col, ColMut, ColRef, Mat};

use crate::{scalar::Scale, IndexType, Scalar};

use crate::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

use super::DefaultDenseMatrix;

impl<T: Scalar> DefaultDenseMatrix for Col<T> {
    type M = Mat<T>;
}

macro_rules! impl_op_for_faer_struct {
    ($struct:ident, $trait_name:ident, $func_name:ident) => {
        impl<'a, T: Scalar> $trait_name<Scale<T>> for $struct<'a, T> {
            type Output = Col<T>;

            fn $func_name(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
                self * scale
            }
        }
    };
}

impl_op_for_faer_struct!(ColRef, Mul, mul);
impl_op_for_faer_struct!(ColMut, Mul, mul);

macro_rules! impl_mul_scale {
    ($col_type:ty) => {
        impl<T: Scalar> Mul<Scale<T>> for $col_type {
            type Output = Col<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
                self * scale
            }
        }

        impl<T: Scalar> Mul<Scale<T>> for &$col_type {
            type Output = Col<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
                self * scale
            }
        }
    };
}

macro_rules! impl_div_scale {
    ($col_type:ty) => {
        impl<'a, T: Scalar> Div<Scale<T>> for $col_type {
            type Output = Col<T>;
            fn div(self, rhs: Scale<T>) -> Self::Output {
                zipped!(self).map(|unzipped!(xi)| *xi / rhs.value())
            }
        }
    };
}

impl_mul_scale!(Col<T>);
impl_div_scale!(faer::Col<T>);

macro_rules! impl_mul_assign_scale {
    ($col_type:ty) => {
        impl<'a, T: Scalar> MulAssign<Scale<T>> for $col_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale: faer::Scale<T> = rhs.into();
                *self *= scale;
            }
        }
    };
}

impl_mul_assign_scale!(ColMut<'a, T>);
impl_mul_assign_scale!(Col<T>);

impl<T: Scalar> Vector for Col<T> {
    type View<'a> = ColRef<'a, T>;
    type ViewMut<'a> = ColMut<'a, T>;
    type Index = Vec<IndexType>;
    fn len(&self) -> IndexType {
        self.nrows()
    }
    fn norm(&self) -> T {
        self.norm_l2()
    }
    fn abs(&self) -> Self {
        zipped!(self).map(|unzipped!(xi)| xi.faer_abs())
    }
    fn as_view(&self) -> Self::View<'_> {
        self.as_ref()
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        self.as_mut()
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other)
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.copy_from(other)
    }
    fn from_element(nstates: usize, value: Self::T) -> Self {
        Col::from_vec(vec![value; nstates])
    }
    fn from_vec(vec: Vec<Self::T>) -> Self {
        Col::from_fn(vec.len(), |i| vec[i])
    }
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, T::zero())
    }
    fn add_scalar_mut(&mut self, scalar: Self::T) {
        zipped!(self.as_mut()).for_each(|unzipped!(mut s)| *s += scalar)
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        *self = &*self * faer::scale(beta) + x * faer::scale(alpha);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        *self = &*self * faer::scale(beta) + x * faer::scale(alpha);
    }
    fn exp(&self) -> Self {
        zipped!(self).map(|unzipped!(xi)| xi.exp())
    }
    fn component_mul_assign(&mut self, other: &Self) {
        zipped!(self.as_mut(), other.as_view()).for_each(|unzipped!(mut s, o)| *s *= *o);
    }
    fn component_div_assign(&mut self, other: &Self) {
        zipped!(self.as_mut(), other.as_view()).for_each(|unzipped!(mut s, o)| *s /= *o);
    }
    fn filter_indices<F: Fn(Self::T) -> bool>(&self, f: F) -> Self::Index {
        let mut indices = vec![];
        for i in 0..self.len() {
            if f(self[i]) {
                indices.push(i as IndexType);
            }
        }
        indices
    }
    fn binary_fold<B, F>(&self, other: &Self, init: B, f: F) -> B
    where
        F: Fn(B, Self::T, Self::T, IndexType) -> B,
    {
        let mut acc = init;
        for i in 0..self.len() {
            acc = f(acc, self[i], other[i], i);
        }
        acc
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
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        for &index in indices {
            self[index] = value;
        }
    }
}

impl VectorIndex for Vec<IndexType> {
    fn zeros(len: IndexType) -> Self {
        vec![0; len]
    }
    fn len(&self) -> IndexType {
        self.len() as IndexType
    }
    fn from_slice(slice: &[IndexType]) -> Self {
        slice.to_vec()
    }
}

macro_rules! impl_vector_common {
    ($vector_type:ty) => {
        impl<'a, T: Scalar> VectorCommon for $vector_type {
            type T = T;
        }
    };
}

impl_vector_common!(Col<T>);
impl_vector_common!(ColRef<'a, T>);
impl_vector_common!(ColMut<'a, T>);

impl<'a, T: Scalar> VectorView<'a> for ColRef<'a, T> {
    type Owned = Col<T>;
    fn abs(&self) -> Col<T> {
        zipped!(self).map(|unzipped!(xi)| xi.faer_abs())
    }
    fn into_owned(self) -> Col<T> {
        self.to_owned()
    }
}

impl<'a, T: Scalar> VectorViewMut<'a> for ColMut<'a, T> {
    type Owned = Col<T>;
    type View = ColRef<'a, T>;
    fn abs(&self) -> Col<T> {
        zipped!(self).map(|unzipped!(xi)| xi.faer_abs())
    }
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.copy_from(other);
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::scale;

    #[test]
    fn test_abs() {
        let v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let v_abs = v.abs();
        assert_eq!(v_abs, Col::from_vec(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_mult() {
        let v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let s = scale(2.0);
        let r = Col::from_vec(vec![2.0, -4.0, 6.0]);
        assert_eq!(v * s, r);
    }

    #[test]
    fn test_mul_assign() {
        let mut v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let s = scale(2.0);
        let r = Col::from_vec(vec![2.0, -4.0, 6.0]);
        v.mul_assign(s);
        assert_eq!(v, r);
    }
}
