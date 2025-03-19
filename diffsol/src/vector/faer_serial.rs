use std::ops::{Div, Mul, MulAssign};
use std::slice;

use faer::{unzip, zip, Col, ColMut, ColRef, Mat};

use crate::{scalar::Scale, IndexType, Scalar, Vector};

use crate::{VectorCommon, VectorHost, VectorIndex, VectorView, VectorViewMut};

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
                let inv_rhs = T::one() / rhs.value();
                self * faer::Scale(inv_rhs)
            }
        }
    };
}

impl_mul_scale!(Col<T>);
impl_div_scale!(Col<T>);

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

impl<T: Scalar> VectorHost for Col<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        unsafe { slice::from_raw_parts_mut(self.as_ptr_mut(), self.len()) }
    }
    fn as_slice(&self) -> &[Self::T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T: Scalar> Vector for Col<T> {
    type View<'a> = ColRef<'a, T>;
    type ViewMut<'a> = ColMut<'a, T>;
    type Index = Vec<IndexType>;
    fn len(&self) -> IndexType {
        self.nrows()
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        self[index]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        self[index] = value;
    }
    fn norm(&self, k: i32) -> T {
        match k {
            1 => self.norm_l1(),
            2 => self.norm_l2(),
            _ => self
                .iter()
                .fold(T::zero(), |acc, x| acc + x.pow(k))
                .pow(T::one() / T::from(k as f64)),
        }
    }

    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = unsafe { y.get_unchecked(i) };
            let ai = unsafe { atol.get_unchecked(i) };
            let xi = unsafe { self.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from(self.len() as f64)
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
    fn fill(&mut self, value: Self::T) {
        self.iter_mut().for_each(|s| *s = value);
    }
    fn from_element(nstates: usize, value: Self::T) -> Self {
        Col::from_vec(vec![value; nstates])
    }
    fn from_vec(vec: Vec<Self::T>) -> Self {
        Col::from_fn(vec.len(), |i| vec[i])
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.iter().cloned().collect()
    }
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, T::zero())
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        zip!(self.as_mut(), x.as_view()).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        zip!(self.as_mut(), x).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        zip!(self.as_mut(), other.as_view()).for_each(|unzip!(s, o)| *s *= *o);
    }
    fn component_div_assign(&mut self, other: &Self) {
        zip!(self.as_mut(), other.as_view()).for_each(|unzip!(s, o)| *s /= *o);
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let mut max_frac = T::zero();
        let mut max_frac_index = -1;
        let mut found_root = false;
        assert_eq!(self.len(), g1.len(), "Vector lengths do not match");
        for i in 0..self.len() {
            let g0 = unsafe { *self.get_unchecked(i) };
            let g1 = unsafe { *g1.get_unchecked(i) };
            if g1 == T::zero() {
                found_root = true;
            }
            if g0 * g1 < T::zero() {
                let frac = (g1 / (g1 - g0)).abs();
                if frac > max_frac {
                    max_frac = frac;
                    max_frac_index = i as i32;
                }
            }
        }
        (found_root, max_frac, max_frac_index)
    }

    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        for i in indices {
            self[*i] = value;
        }
    }

    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        for i in indices {
            self[*i] = other[*i];
        }
    }

    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        for (s, o) in self.iter_mut().zip(indices.iter()) {
            *s = other[*o];
        }
    }

    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        for (s, o) in self.iter().zip(indices.iter()) {
            other[*o] = *s;
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
    fn from_vec(v: Vec<IndexType>) -> Self {
        v
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.clone()
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
    fn into_owned(self) -> Col<T> {
        self.to_owned()
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.nrows() || y.nrows() != atol.nrows() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.nrows() {
            let yi = unsafe { y.get_unchecked(i) };
            let ai = unsafe { atol.get_unchecked(i) };
            let xi = unsafe { self.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from(self.nrows() as f64)
    }
}

impl<'a, T: Scalar> VectorViewMut<'a> for ColMut<'a, T> {
    type Owned = Col<T>;
    type View = ColRef<'a, T>;
    type Index = Vec<IndexType>;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.copy_from(other);
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        zip!(self.as_mut(), x.as_view()).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::scale;

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

    #[test]
    fn test_error_norm() {
        let v = Col::from_vec(vec![1.0, -2.0, 3.0]);
        let y = Col::from_vec(vec![1.0, 2.0, 3.0]);
        let atol = Col::from_vec(vec![0.1, 0.2, 0.3]);
        let rtol = 0.1;
        let mut tmp = y.clone() * scale(rtol);
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.squared_norm_l2() / 3.0;
        assert!(
            (v.squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.squared_norm(&y, &atol, rtol),
            errorn_check
        );
        assert!(
            (v.as_ref().squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.as_ref().squared_norm(&y, &atol, rtol),
            errorn_check
        );
    }

    #[test]
    fn test_root_finding() {
        super::super::tests::test_root_finding::<Col<f64>>();
    }
}
