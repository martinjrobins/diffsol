use std::ops::{Div, Mul, MulAssign};

use faer::{unzipped, zipped, Col, ColMut, ColRef};

use crate::{scalar::Scalar, scalar::Scale, IndexType};

use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};
impl<'a, T: Scalar> Mul<Scale<T>> for ColRef<'a, f64> {
    type Output = Col<f64>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * faer::scale(rhs.value().into())
    }
}

impl<'a, T: Scalar> Mul<Scale<T>> for ColMut<'a, f64> {
    type Output = Col<f64>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * faer::scale(rhs.value().into())
    }
}

impl<'a, T: Scalar> Mul<Scale<T>> for Col<f64> {
    type Output = Col<f64>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * faer::scale(rhs.value().into())
    }
}
impl<'a, T: Scalar> MulAssign<Scale<T>> for ColMut<'a, f64> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        let scale = faer::scale(rhs.value().into());
        *self *= scale;
    }
}
impl<T: Scalar> MulAssign<Scale<T>> for Col<f64> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        *self = &*self * faer::scale(rhs.value().into())
    }
}

impl<'a, T: Scalar> Div<crate::scalar::Scale<T>> for faer::Col<f64> {
    type Output = faer::Col<f64>;
    fn div(self, rhs: crate::scalar::Scale<T>) -> Self::Output {
        self
        // zipped!(self.as_ref()).map(|unzipped!(xi)| xi / rhs.value())
        // self * faer::scale(rhs.value().into())
    }
}

impl Vector for Col<f64> {
    type View<'a> = ColRef<'a, f64>;
    type ViewMut<'a> = ColMut<'a, f64>;
    type Index = Vec<IndexType>;
    fn len(&self) -> IndexType {
        self.nrows()
    }
    fn norm(&self) -> f64 {
        self.norm_l2()
    }
    fn abs(&self) -> Self {
        zipped!(self).map(|unzipped!(xi)| xi.abs())
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
        Self::from_element(nstates, 0.0)
    }
    fn add_scalar_mut(&mut self, scalar: Self::T) {
        zipped!(self.as_mut()).for_each(|unzipped!(mut s)| *s += scalar)
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
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

impl VectorIndex for Vec<IndexType> {
    fn zeros(len: IndexType) -> Self {
        vec![0; len as usize]
    }
    fn len(&self) -> IndexType {
        self.len() as IndexType
    }
}

impl VectorCommon for Col<f64> {
    type T = f64;
}
impl<'a> VectorCommon for ColRef<'a, f64> {
    type T = f64;
}
impl<'a> VectorCommon for ColMut<'a, f64> {
    type T = f64;
}

impl<'a> VectorView<'a> for ColRef<'a, f64> {
    type Owned = Col<f64>;
    fn abs(&self) -> Col<f64> {
        zipped!(self).map(|unzipped!(xi)| xi.abs())
    }
    fn into_owned(self) -> Col<f64> {
        self.to_owned()
    }
    fn scalar_mul(&self, rhs: Self::T) -> Self::Owned {
        self * faer::scale(rhs)
    }
}

impl<'a> VectorViewMut<'a> for ColMut<'a, f64> {
    type Owned = Col<f64>;
    type View = ColRef<'a, f64>;
    fn abs(&self) -> Col<f64> {
        zipped!(self).map(|unzipped!(xi)| xi.abs())
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

    #[test]
    fn test_abs() {
        let v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let v_abs = v.abs();
        assert_eq!(v_abs, Col::from_vec(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_mult() {
        let v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let s = crate::scalar::scale(2.0);
        let r = Col::from_vec(vec![2.0, -4.0, 6.0]);
        assert_eq!(v * s, r);
    }

    #[test]
    fn test_mul_assign() {
        let mut v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let s = crate::scalar::scale(2.0);
        let r = Col::from_vec(vec![2.0, -4.0, 6.0]);
        v.mul_assign(s);
        assert_eq!(v, r);
    }
}
