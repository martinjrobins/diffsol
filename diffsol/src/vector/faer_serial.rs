use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::slice;

use faer::{unzip, zip, Col, ColMut, ColRef};

use crate::{scalar::Scale, FaerContext, IndexType, Scalar, Vector};

use crate::{FaerMat, VectorCommon, VectorHost, VectorIndex, VectorView, VectorViewMut};

use super::utils::*;
use super::DefaultDenseMatrix;

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVec<T: Scalar> {
    pub(crate) data: Col<T>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecIndex {
    pub(crate) data: Vec<IndexType>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecRef<'a, T: Scalar> {
    pub(crate) data: ColRef<'a, T>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, PartialEq)]
pub struct FaerVecMut<'a, T: Scalar> {
    pub(crate) data: ColMut<'a, T>,
    pub(crate) context: FaerContext,
}

impl<T: Scalar> From<Col<T>> for FaerVec<T> {
    fn from(data: Col<T>) -> Self {
        Self {
            data,
            context: FaerContext::default(),
        }
    }
}

impl<T: Scalar> DefaultDenseMatrix for FaerVec<T> {
    type M = FaerMat<T>;
}

impl_vector_common!(FaerVec<T>, FaerContext, Col<T>);
impl_vector_common_ref!(FaerVecRef<'a, T>, FaerContext, ColRef<'a, T>);
impl_vector_common_ref!(FaerVecMut<'a, T>, FaerContext, ColMut<'a, T>);

macro_rules! impl_mul_scalar {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: Scalar> Mul<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: $scalar = rhs.into();
                Self::Output {
                    data: &self.data * scale,
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($lhs:ty, $out:ty, $scalar:expr) => {
        impl<'a, T: Scalar> Div<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            fn div(self, rhs: Scale<T>) -> Self::Output {
                let inv_rhs: T = T::one() / rhs.value();
                let scale = faer::Scale(inv_rhs);
                Self::Output {
                    data: &self.data * scale,
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($col_type:ty, $scalar:ty) => {
        impl<'a, T: Scalar> MulAssign<Scale<T>> for $col_type {
            #[inline]
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale = faer::Scale(rhs.value());
                self.data *= scale;
            }
        }
    };
}

impl_mul_scalar!(FaerVec<T>, FaerVec<T>, faer::Scale<T>);
impl_mul_scalar!(&FaerVec<T>, FaerVec<T>, faer::Scale<T>);
impl_mul_scalar!(FaerVecRef<'_, T>, FaerVec<T>, faer::Scale<T>);
impl_mul_scalar!(FaerVecMut<'_, T>, FaerVec<T>, faer::Scale<T>);
impl_div_scalar!(FaerVec<T>, FaerVec<T>, faer::Scale::<T>);
impl_mul_assign_scalar!(FaerVecMut<'a, T>, faer::Scale<T>);
impl_mul_assign_scalar!(FaerVec<T>, faer::Scale<T>);

impl_sub_assign!(FaerVec<T>, FaerVec<T>);
impl_sub_assign!(FaerVec<T>, &FaerVec<T>);
impl_sub_assign!(FaerVec<T>, FaerVecRef<'_, T>);
impl_sub_assign!(FaerVec<T>, &FaerVecRef<'_, T>);

impl_sub_assign!(FaerVecMut<'_, T>, FaerVec<T>);
impl_sub_assign!(FaerVecMut<'_, T>, &FaerVec<T>);
impl_sub_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>);
impl_sub_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>);

impl_add_assign!(FaerVec<T>, FaerVec<T>);
impl_add_assign!(FaerVec<T>, &FaerVec<T>);
impl_add_assign!(FaerVec<T>, FaerVecRef<'_, T>);
impl_add_assign!(FaerVec<T>, &FaerVecRef<'_, T>);

impl_add_assign!(FaerVecMut<'_, T>, FaerVec<T>);
impl_add_assign!(FaerVecMut<'_, T>, &FaerVec<T>);
impl_add_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>);
impl_add_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>);

impl_sub_both_ref!(&FaerVec<T>, &FaerVec<T>, FaerVec<T>);
impl_sub_rhs!(&FaerVec<T>, FaerVec<T>, FaerVec<T>);
impl_sub_both_ref!(&FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_sub_both_ref!(&FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_sub_lhs!(FaerVec<T>, FaerVec<T>, FaerVec<T>);
impl_sub_lhs!(FaerVec<T>, &FaerVec<T>, FaerVec<T>);
impl_sub_lhs!(FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_sub_lhs!(FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_sub_rhs!(FaerVecRef<'_, T>, FaerVec<T>, FaerVec<T>);
impl_sub_both_ref!(FaerVecRef<'_, T>, &FaerVec<T>, FaerVec<T>);
impl_sub_both_ref!(FaerVecRef<'_, T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_sub_both_ref!(FaerVecRef<'_, T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_add_both_ref!(&FaerVec<T>, &FaerVec<T>, FaerVec<T>);
impl_add_rhs!(&FaerVec<T>, FaerVec<T>, FaerVec<T>);
impl_add_both_ref!(&FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_add_both_ref!(&FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_add_lhs!(FaerVec<T>, FaerVec<T>, FaerVec<T>);
impl_add_lhs!(FaerVec<T>, &FaerVec<T>, FaerVec<T>);
impl_add_lhs!(FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_add_lhs!(FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_add_rhs!(FaerVecRef<'_, T>, FaerVec<T>, FaerVec<T>);
impl_add_both_ref!(FaerVecRef<'_, T>, &FaerVec<T>, FaerVec<T>);
impl_add_both_ref!(FaerVecRef<'_, T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_add_both_ref!(FaerVecRef<'_, T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_index!(FaerVec<T>);
impl_index_mut!(FaerVec<T>);
impl_index!(FaerVecRef<'_, T>);

impl<T: Scalar> VectorHost for FaerVec<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr_mut(), self.len()) }
    }
    fn as_slice(&self) -> &[Self::T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len()) }
    }
}

impl<T: Scalar> Vector for FaerVec<T> {
    type View<'a> = FaerVecRef<'a, T>;
    type ViewMut<'a> = FaerVecMut<'a, T>;
    type Index = FaerVecIndex;
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn len(&self) -> IndexType {
        self.data.nrows()
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        self.data[index]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        self.data[index] = value;
    }
    fn norm(&self, k: i32) -> T {
        match k {
            1 => self.data.norm_l1(),
            2 => self.data.norm_l2(),
            _ => self
                .data
                .iter()
                .fold(T::zero(), |acc, x| acc + x.pow(k))
                .pow(T::one() / T::from_f64(k as f64).unwrap()),
        }
    }

    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = unsafe { y.data.get_unchecked(i) };
            let ai = unsafe { atol.data.get_unchecked(i) };
            let xi = unsafe { self.data.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from_f64(self.len() as f64).unwrap()
    }
    fn as_view(&self) -> Self::View<'_> {
        FaerVecRef {
            data: self.data.as_ref(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        FaerVecMut {
            data: self.data.as_mut(),
            context: self.context,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data)
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.data.copy_from(&other.data)
    }
    fn fill(&mut self, value: Self::T) {
        self.data.iter_mut().for_each(|s| *s = value);
    }
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self {
        let data = Col::from_fn(nstates, |_| value);
        FaerVec { data, context: ctx }
    }
    fn from_vec(vec: Vec<Self::T>, ctx: Self::C) -> Self {
        let data = Col::from_fn(vec.len(), |i| vec[i]);
        FaerVec { data, context: ctx }
    }
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self {
        let data = Col::from_fn(slice.len(), |i| slice[i]);
        FaerVec { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data.iter().cloned().collect()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        Self::from_element(nstates, T::zero(), ctx)
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        zip!(self.data.as_mut(), x.data.as_ref())
            .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        zip!(self.data.as_mut(), x.data).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        zip!(self.data.as_mut(), other.data.as_ref()).for_each(|unzip!(s, o)| *s *= *o);
    }
    fn component_div_assign(&mut self, other: &Self) {
        zip!(self.data.as_mut(), other.data.as_ref()).for_each(|unzip!(s, o)| *s /= *o);
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let mut max_frac = T::zero();
        let mut max_frac_index = -1;
        let mut found_root = false;
        assert_eq!(self.len(), g1.len(), "Vector lengths do not match");
        for i in 0..self.len() {
            let g0 = unsafe { *self.data.get_unchecked(i) };
            let g1 = unsafe { *g1.data.get_unchecked(i) };
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
        for i in indices.data.iter() {
            self[*i] = value;
        }
    }

    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        for i in indices.data.iter() {
            self[*i] = other[*i];
        }
    }

    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        for (s, o) in self.data.iter_mut().zip(indices.data.iter()) {
            *s = other[*o];
        }
    }

    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        for (s, o) in self.data.iter().zip(indices.data.iter()) {
            other[*o] = *s;
        }
    }
}

impl VectorIndex for FaerVecIndex {
    type C = FaerContext;
    fn zeros(len: IndexType, ctx: Self::C) -> Self {
        Self {
            data: vec![0; len],
            context: ctx,
        }
    }
    fn len(&self) -> IndexType {
        self.data.len() as IndexType
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        Self {
            data: v,
            context: ctx,
        }
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.data.clone()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl<'a, T: Scalar> VectorView<'a> for FaerVecRef<'a, T> {
    type Owned = FaerVec<T>;
    fn into_owned(self) -> FaerVec<T> {
        FaerVec {
            data: self.data.to_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.data.nrows() || y.data.nrows() != atol.data.nrows() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.data.nrows() {
            let yi = unsafe { y.data.get_unchecked(i) };
            let ai = unsafe { atol.data.get_unchecked(i) };
            let xi = unsafe { self.data.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from_f64(self.data.nrows() as f64).unwrap()
    }
}

impl<'a, T: Scalar> VectorViewMut<'a> for FaerVecMut<'a, T> {
    type Owned = FaerVec<T>;
    type View = FaerVecRef<'a, T>;
    type Index = FaerVecIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.data.copy_from(&other.data);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.data.copy_from(&other.data);
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        zip!(self.data.as_mut(), x.data.as_ref())
            .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::scale;

    #[test]
    fn test_mult() {
        let v = FaerVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let s = scale(2.0);
        let r = FaerVec::from_vec(vec![2.0, -4.0, 6.0], Default::default());
        assert_eq!(v * s, r);
    }

    #[test]
    fn test_mul_assign() {
        let mut v = FaerVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let s = scale(2.0);
        let r = FaerVec::from_vec(vec![2.0, -4.0, 6.0], Default::default());
        v.mul_assign(s);
        assert_eq!(v, r);
    }

    #[test]
    fn test_error_norm() {
        let v: FaerVec<f64> = FaerVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let y = FaerVec::from_vec(vec![1.0, 2.0, 3.0], Default::default());
        let atol = FaerVec::from_vec(vec![0.1, 0.2, 0.3], Default::default());
        let rtol = 0.1;
        let mut tmp = y.clone() * scale(rtol);
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.data.squared_norm_l2() / 3.0;
        assert!(
            (v.squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.squared_norm(&y, &atol, rtol),
            errorn_check
        );
        assert!(
            (v.squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.squared_norm(&y, &atol, rtol),
            errorn_check
        );
    }

    #[test]
    fn test_root_finding() {
        super::super::tests::test_root_finding::<FaerVec<f64>>();
    }

    #[test]
    fn test_from_slice() {
        let slice = [1.0, 2.0, 3.0];
        let v = FaerVec::from_slice(&slice, Default::default());
        assert_eq!(v.clone_as_vec(), slice);
    }

    #[test]
    fn test_into() {
        let col: Col<f64> = Col::from_fn(3, |i| (i + 1) as f64);
        let v: FaerVec<f64> = col.into();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }
}
