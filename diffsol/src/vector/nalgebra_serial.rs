use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::utils::*;
use nalgebra::{DVector, DVectorView, DVectorViewMut, LpNorm};

use crate::{IndexType, NalgebraContext, NalgebraMat, Scalar, Scale, VectorHost};

use super::{DefaultDenseMatrix, Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraIndex {
    pub(crate) data: DVector<IndexType>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVec<T: Scalar> {
    pub(crate) data: DVector<T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVecRef<'a, T: Scalar> {
    pub(crate) data: DVectorView<'a, T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraVecMut<'a, T: Scalar> {
    pub(crate) data: DVectorViewMut<'a, T>,
    pub(crate) context: NalgebraContext,
}

impl<T: Scalar> From<DVector<T>> for NalgebraVec<T> {
    fn from(data: DVector<T>) -> Self {
        Self {
            data,
            context: NalgebraContext,
        }
    }
}

impl<T: Scalar> DefaultDenseMatrix for NalgebraVec<T> {
    type M = NalgebraMat<T>;
}

impl_vector_common!(NalgebraVec<T>, NalgebraContext, DVector<T>);
impl_vector_common_ref!(NalgebraVecRef<'a, T>, NalgebraContext, DVectorView<'a, T>);
impl_vector_common_ref!(
    NalgebraVecMut<'a, T>,
    NalgebraContext,
    DVectorViewMut<'a, T>
);

macro_rules! impl_mul_scalar {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: Scalar> Mul<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: $scalar = rhs.value();
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
                Self::Output {
                    data: self.data * inv_rhs,
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
                let scale = rhs.value();
                self.data *= scale;
            }
        }
    };
}

impl_mul_scalar!(NalgebraVec<T>, NalgebraVec<T>, T);
impl_mul_scalar!(&NalgebraVec<T>, NalgebraVec<T>, T);
impl_mul_scalar!(NalgebraVecRef<'_, T>, NalgebraVec<T>, T);
impl_mul_scalar!(NalgebraVecMut<'_, T>, NalgebraVec<T>, T);
impl_div_scalar!(NalgebraVec<T>, NalgebraVec<T>, T);
impl_mul_assign_scalar!(NalgebraVecMut<'a, T>, T);
impl_mul_assign_scalar!(NalgebraVec<T>, T);

impl_sub_assign!(NalgebraVec<T>, NalgebraVec<T>);
impl_sub_assign!(NalgebraVec<T>, &NalgebraVec<T>);
impl_sub_assign!(NalgebraVec<T>, NalgebraVecRef<'_, T>);
impl_sub_assign!(NalgebraVec<T>, &NalgebraVecRef<'_, T>);

impl_sub_assign!(NalgebraVecMut<'_, T>, NalgebraVec<T>);
impl_sub_assign!(NalgebraVecMut<'_, T>, &NalgebraVec<T>);
impl_sub_assign!(NalgebraVecMut<'_, T>, NalgebraVecRef<'_, T>);
impl_sub_assign!(NalgebraVecMut<'_, T>, &NalgebraVecRef<'_, T>);

impl_add_assign!(NalgebraVec<T>, NalgebraVec<T>);
impl_add_assign!(NalgebraVec<T>, &NalgebraVec<T>);
impl_add_assign!(NalgebraVec<T>, NalgebraVecRef<'_, T>);
impl_add_assign!(NalgebraVec<T>, &NalgebraVecRef<'_, T>);

impl_add_assign!(NalgebraVecMut<'_, T>, NalgebraVec<T>);
impl_add_assign!(NalgebraVecMut<'_, T>, &NalgebraVec<T>);
impl_add_assign!(NalgebraVecMut<'_, T>, NalgebraVecRef<'_, T>);
impl_add_assign!(NalgebraVecMut<'_, T>, &NalgebraVecRef<'_, T>);

impl_sub_both_ref!(&NalgebraVec<T>, &NalgebraVec<T>, NalgebraVec<T>);
impl_sub_rhs!(&NalgebraVec<T>, NalgebraVec<T>, NalgebraVec<T>);
impl_sub_both_ref!(&NalgebraVec<T>, NalgebraVecRef<'_, T>, NalgebraVec<T>);
impl_sub_both_ref!(&NalgebraVec<T>, &NalgebraVecRef<'_, T>, NalgebraVec<T>);

impl_sub_lhs!(NalgebraVec<T>, NalgebraVec<T>, NalgebraVec<T>);
impl_sub_lhs!(NalgebraVec<T>, &NalgebraVec<T>, NalgebraVec<T>);
impl_sub_lhs!(NalgebraVec<T>, NalgebraVecRef<'_, T>, NalgebraVec<T>);
impl_sub_lhs!(NalgebraVec<T>, &NalgebraVecRef<'_, T>, NalgebraVec<T>);

impl_sub_rhs!(NalgebraVecRef<'_, T>, NalgebraVec<T>, NalgebraVec<T>);
impl_sub_both_ref!(NalgebraVecRef<'_, T>, &NalgebraVec<T>, NalgebraVec<T>);
impl_sub_both_ref!(NalgebraVecRef<'_, T>, NalgebraVecRef<'_, T>, NalgebraVec<T>);
impl_sub_both_ref!(
    NalgebraVecRef<'_, T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>
);

impl_add_both_ref!(&NalgebraVec<T>, &NalgebraVec<T>, NalgebraVec<T>);
impl_add_rhs!(&NalgebraVec<T>, NalgebraVec<T>, NalgebraVec<T>);
impl_add_both_ref!(&NalgebraVec<T>, NalgebraVecRef<'_, T>, NalgebraVec<T>);
impl_add_both_ref!(&NalgebraVec<T>, &NalgebraVecRef<'_, T>, NalgebraVec<T>);

impl_add_lhs!(NalgebraVec<T>, NalgebraVec<T>, NalgebraVec<T>);
impl_add_lhs!(NalgebraVec<T>, &NalgebraVec<T>, NalgebraVec<T>);
impl_add_lhs!(NalgebraVec<T>, NalgebraVecRef<'_, T>, NalgebraVec<T>);
impl_add_lhs!(NalgebraVec<T>, &NalgebraVecRef<'_, T>, NalgebraVec<T>);

impl_add_rhs!(NalgebraVecRef<'_, T>, NalgebraVec<T>, NalgebraVec<T>);
impl_add_both_ref!(NalgebraVecRef<'_, T>, &NalgebraVec<T>, NalgebraVec<T>);
impl_add_both_ref!(NalgebraVecRef<'_, T>, NalgebraVecRef<'_, T>, NalgebraVec<T>);
impl_add_both_ref!(
    NalgebraVecRef<'_, T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>
);

impl_index!(NalgebraVec<T>);
impl_index_mut!(NalgebraVec<T>);

impl_index!(NalgebraVecRef<'_, T>);

impl VectorIndex for NalgebraIndex {
    type C = NalgebraContext;
    fn zeros(len: IndexType, ctx: Self::C) -> Self {
        let data = DVector::from_element(len, 0);
        Self { data, context: ctx }
    }
    fn len(&self) -> crate::IndexType {
        self.data.len()
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        let data = DVector::from_vec(v);
        Self { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.data.iter().copied().collect()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl<'a, T: Scalar> VectorView<'a> for NalgebraVecRef<'a, T> {
    type Owned = NalgebraVec<T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.into_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.data.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.data.len() {
            let yi = unsafe { y.data.get_unchecked(i) };
            let ai = unsafe { atol.data.get_unchecked(i) };
            let xi = unsafe { self.data.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from_f64(self.data.len() as f64).unwrap()
    }
}

impl<'a, T: Scalar> VectorViewMut<'a> for NalgebraVecMut<'a, T> {
    type Owned = NalgebraVec<T>;
    type View = NalgebraVecRef<'a, T>;
    type Index = NalgebraIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.data.copy_from(&other.data);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.data.copy_from(&other.data);
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        self.data.axpy(alpha, &x.data, beta);
    }
}

impl<T: Scalar> VectorHost for NalgebraVec<T> {
    fn as_slice(&self) -> &[Self::T] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        self.data.as_mut_slice()
    }
}

impl<T: Scalar> Vector for NalgebraVec<T> {
    type View<'a> = NalgebraVecRef<'a, T>;
    type ViewMut<'a> = NalgebraVecMut<'a, T>;
    type Index = NalgebraIndex;
    fn len(&self) -> IndexType {
        self.data.len()
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn norm(&self, k: i32) -> Self::T {
        self.data.apply_norm(&LpNorm(k))
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        self.data[index]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        self.data[index] = value;
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
        Self::View {
            data: self.data.as_view(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        Self::ViewMut {
            data: self.data.as_view_mut(),
            context: self.context,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
    }
    fn fill(&mut self, value: Self::T) {
        self.data.iter_mut().for_each(|x: &mut _| *x = value);
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.data.copy_from(&other.data);
    }
    fn from_element(nstates: usize, value: T, ctx: Self::C) -> Self {
        let data = DVector::from_element(nstates, value);
        Self { data, context: ctx }
    }
    fn from_vec(vec: Vec<T>, ctx: Self::C) -> Self {
        let data = DVector::from_vec(vec);
        Self { data, context: ctx }
    }
    fn from_slice(slice: &[T], ctx: Self::C) -> Self {
        let data = DVector::from_column_slice(slice);
        Self { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data.iter().copied().collect()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let data = DVector::zeros(nstates);
        Self { data, context: ctx }
    }
    fn axpy(&mut self, alpha: T, x: &Self, beta: T) {
        self.data.axpy(alpha, &x.data, beta);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        self.data.axpy(alpha, &x.data, beta);
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.data.component_div_assign(&other.data);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.data.component_mul_assign(&other.data);
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

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_norm() {
        let v = NalgebraVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let y = NalgebraVec::from_vec(vec![1.0, 2.0, 3.0], Default::default());
        let atol = NalgebraVec::from_vec(vec![0.1, 0.2, 0.3], Default::default());
        let rtol = 0.1;
        let mut tmp = y.clone() * Scale(rtol);
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.data.norm_squared() / 3.0;
        assert_eq!(v.squared_norm(&y, &atol, rtol), errorn_check);
        let vview = v.as_view();
        assert_eq!(
            VectorView::squared_norm(&vview, &y, &atol, rtol),
            errorn_check
        );
    }

    #[test]
    fn test_root_finding() {
        super::super::tests::test_root_finding::<NalgebraVec<f64>>();
    }

    #[test]
    fn test_from_slice() {
        let slice = [1.0, 2.0, 3.0];
        let v = NalgebraVec::from_slice(&slice, Default::default());
        assert_eq!(v.clone_as_vec(), slice);
    }

    #[test]
    fn test_into() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v: NalgebraVec<f64> = vec.into();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }
}
