use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::slice;

use faer::{unzip, zip, Col, ColMut, ColRef};

use crate::{scalar::Scale, Context, FaerContext, FaerScalar, IndexType, Scalar, Vector};

use crate::{FaerMat, VectorCommon, VectorHost, VectorIndex, VectorView, VectorViewMut};

use super::utils::*;
use super::DefaultDenseMatrix;

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVec<T: FaerScalar> {
    pub(crate) data: Vec<Col<T>>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecIndex {
    pub(crate) data: Vec<IndexType>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecRef<'a, T: FaerScalar> {
    pub(crate) data: Vec<ColRef<'a, T>>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, PartialEq)]
pub struct FaerVecMut<'a, T: FaerScalar> {
    pub(crate) data: Vec<ColMut<'a, T>>,
    pub(crate) context: FaerContext,
}

impl<T: FaerScalar> From<Col<T>> for FaerVec<T> {
    fn from(data: Col<T>) -> Self {
        Self {
            data: vec![data],
            context: FaerContext::default(),
        }
    }
}

impl<T: FaerScalar> FaerVec<T> {
    pub fn check_for_nan(&self, label: &str) -> bool {
        for data in &self.data {
            for i in 0..data.nrows() {
                if unsafe { data.get_unchecked(i) }.is_nan() {
                    eprintln!("{}: NaN at index {}", label, i);
                    return true;
                }
            }
        }
        false
    }
}

impl<T: FaerScalar> DefaultDenseMatrix for FaerVec<T> {
    type M = FaerMat<T>;
}

impl_vector_common!(FaerVec<T>, FaerContext, Vec<Col<T>>, FaerScalar);
impl_vector_common_ref!(
    FaerVecRef<'a, T>,
    FaerContext,
    Vec<ColRef<'a, T>>,
    FaerScalar
);
impl_vector_common_ref!(
    FaerVecMut<'a, T>,
    FaerContext,
    Vec<ColMut<'a, T>>,
    FaerScalar
);

macro_rules! impl_mul_scalar {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: FaerScalar> Mul<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: $scalar = rhs.into();
                Self::Output {
                    data: self.data.iter().map(|data| data * scale).collect(),
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($lhs:ty, $out:ty, $scalar:expr) => {
        impl<'a, T: FaerScalar> Div<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: Scale<T>) -> Self::Output {
                let inv_rhs: T = T::one() / rhs.value();
                let scale = faer::Scale(inv_rhs);
                Self::Output {
                    data: self.data.iter().map(|data| data * scale).collect(),
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($col_type:ty, $scalar:ty) => {
        impl<'a, T: FaerScalar> MulAssign<Scale<T>> for $col_type {
            #[inline]
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale = faer::Scale(rhs.value());
                for data in &mut self.data {
                    *data *= scale;
                }
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

impl_sub_assign!(FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_assign!(FaerVec<T>, &FaerVec<T>, FaerScalar);
impl_sub_assign!(FaerVec<T>, FaerVecRef<'_, T>, FaerScalar);
impl_sub_assign!(FaerVec<T>, &FaerVecRef<'_, T>, FaerScalar);

impl_sub_assign!(FaerVecMut<'_, T>, FaerVec<T>, FaerScalar);
impl_sub_assign!(FaerVecMut<'_, T>, &FaerVec<T>, FaerScalar);
impl_sub_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>, FaerScalar);
impl_sub_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>, FaerScalar);

impl_add_assign!(FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_assign!(FaerVec<T>, &FaerVec<T>, FaerScalar);
impl_add_assign!(FaerVec<T>, FaerVecRef<'_, T>, FaerScalar);
impl_add_assign!(FaerVec<T>, &FaerVecRef<'_, T>, FaerScalar);

impl_add_assign!(FaerVecMut<'_, T>, FaerVec<T>, FaerScalar);
impl_add_assign!(FaerVecMut<'_, T>, &FaerVec<T>, FaerScalar);
impl_add_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>, FaerScalar);
impl_add_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>, FaerScalar);

impl_sub_both_ref!(&FaerVec<T>, &FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_rhs!(&FaerVec<T>, FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_both_ref!(&FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);
impl_sub_both_ref!(&FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);

impl_sub_lhs!(FaerVec<T>, FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_lhs!(FaerVec<T>, &FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_lhs!(FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);
impl_sub_lhs!(FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);

impl_sub_rhs!(FaerVecRef<'_, T>, FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_both_ref!(FaerVecRef<'_, T>, &FaerVec<T>, FaerVec<T>, FaerScalar);
impl_sub_both_ref!(FaerVecRef<'_, T>, FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);
impl_sub_both_ref!(
    FaerVecRef<'_, T>,
    &FaerVecRef<'_, T>,
    FaerVec<T>,
    FaerScalar
);

impl_add_both_ref!(&FaerVec<T>, &FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_rhs!(&FaerVec<T>, FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_both_ref!(&FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);
impl_add_both_ref!(&FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);

impl_add_lhs!(FaerVec<T>, FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_lhs!(FaerVec<T>, &FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_lhs!(FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);
impl_add_lhs!(FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);

impl_add_rhs!(FaerVecRef<'_, T>, FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_both_ref!(FaerVecRef<'_, T>, &FaerVec<T>, FaerVec<T>, FaerScalar);
impl_add_both_ref!(FaerVecRef<'_, T>, FaerVecRef<'_, T>, FaerVec<T>, FaerScalar);
impl_add_both_ref!(
    FaerVecRef<'_, T>,
    &FaerVecRef<'_, T>,
    FaerVec<T>,
    FaerScalar
);

impl_index!(FaerVec<T>, FaerScalar);
impl_index_mut!(FaerVec<T>, FaerScalar);
impl_index!(FaerVecRef<'_, T>, FaerScalar);

impl<T: FaerScalar> VectorHost for FaerVec<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        assert_eq!(self.context.nbatch(), 1);
        unsafe { slice::from_raw_parts_mut(self.data[0].as_ptr_mut(), self.len()) }
    }
    fn as_slice(&self) -> &[Self::T] {
        assert_eq!(self.context.nbatch(), 1);
        unsafe { slice::from_raw_parts(self.data[0].as_ptr(), self.len()) }
    }
}

impl<T: FaerScalar> Vector for FaerVec<T> {
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
        self.data[0].nrows()
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        assert_eq!(
            self.context.nbatch(),
            1,
            "get_index not supported for batched vectors"
        );
        self.data[0][index]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        for data in &mut self.data {
            data[index] = value;
        }
    }
    fn norm(&self, k: i32) -> T {
        match k {
            1 => self
                .data
                .iter()
                .map(|data| data.norm_l1())
                .fold(T::zero(), |a, b| a.max(b)),
            2 => self
                .data
                .iter()
                .map(|data| data.norm_l2())
                .fold(T::zero(), |a, b| a.max(b)),
            _ => self
                .data
                .iter()
                .map(|data| {
                    data.iter()
                        .fold(T::zero(), |acc, x| acc + x.pow(k))
                        .pow(T::one() / T::from_f64(k as f64).unwrap())
                })
                .fold(T::zero(), |a, b| a.max(b)),
        }
    }

    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "squared_norm");
        self.context
            .assert_compatible_nbatch(atol.context.nbatch(), "squared_norm");
        self.data
            .iter()
            .enumerate()
            .map(|(b, x)| {
                x.iter()
                    .zip(y.data[b % y.data.len()].iter())
                    .zip(atol.data[b % atol.data.len()].iter())
                    .fold(T::zero(), |a, ((x, y), atol)| {
                        let t = *x / (y.abs() * rtol + *atol);
                        a + t * t
                    })
                    / T::from_f64(x.nrows() as f64).unwrap()
            })
            .fold(T::zero(), |a, b| a.max(b))
    }
    fn as_view(&self) -> Self::View<'_> {
        FaerVecRef {
            data: self.data.iter().map(|data| data.as_ref()).collect(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        FaerVecMut {
            data: self.data.iter_mut().map(|data| data.as_mut()).collect(),
            context: self.context,
        }
    }
    fn get_batch(&self, batch: usize) -> Self::View<'_> {
        assert!(batch < self.data.len());
        FaerVecRef {
            data: vec![self.data[batch].as_ref()],
            context: FaerContext::default(),
        }
    }
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_> {
        assert!(batch < self.data.len());
        FaerVecMut {
            data: vec![self.data[batch].as_mut()],
            context: FaerContext::default(),
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[b % other.data.len()]);
        }
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from_view");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.copy_from(other.data[b % other.data.len()]);
        }
    }
    fn fill(&mut self, value: Self::T) {
        for data in &mut self.data {
            data.fill(value);
        }
    }
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self {
        let data = (0..ctx.nbatch())
            .map(|_| Col::from_fn(nstates, |_| value))
            .collect();
        FaerVec { data, context: ctx }
    }
    fn from_vec(vec: Vec<Self::T>, ctx: Self::C) -> Self {
        assert!(
            vec.len() % ctx.nbatch() == 0,
            "vector length must be divisible by nbatch"
        );
        let n = vec.len() / ctx.nbatch();
        let data = if n == 0 {
            (0..ctx.nbatch()).map(|_| Col::zeros(0)).collect()
        } else {
            vec.chunks(n).map(|v| Col::from_fn(n, |i| v[i])).collect()
        };
        FaerVec { data, context: ctx }
    }
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self {
        assert!(
            slice.len() % ctx.nbatch() == 0,
            "vector length must be divisible by nbatch"
        );
        let n = slice.len() / ctx.nbatch();
        let data = if n == 0 {
            (0..ctx.nbatch()).map(|_| Col::zeros(0)).collect()
        } else {
            slice.chunks(n).map(|v| Col::from_fn(n, |i| v[i])).collect()
        };
        FaerVec { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data
            .iter()
            .flat_map(|data| data.iter().copied())
            .collect()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        Self::from_element(nstates, T::zero(), ctx)
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy");
        for (b, data) in self.data.iter_mut().enumerate() {
            zip!(data.as_mut(), x.data[b % x.data.len()].as_ref())
                .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
        }
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy_v");
        for (b, data) in self.data.iter_mut().enumerate() {
            zip!(data.as_mut(), x.data[b % x.data.len()])
                .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
        }
    }
    fn batched_axpy(&mut self, alpha: &[Self::T], x: &Self, beta: Self::T) {
        assert_eq!(
            alpha.len(),
            self.context.nbatch(),
            "alpha.len() must equal nbatch"
        );
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "batched_axpy");
        for (b, data) in self.data.iter_mut().enumerate() {
            zip!(data.as_mut(), x.data[b % x.data.len()].as_ref())
                .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha[b]);
        }
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "component_mul_assign");
        for (b, data) in self.data.iter_mut().enumerate() {
            zip!(data.as_mut(), other.data[b % other.data.len()].as_ref())
                .for_each(|unzip!(s, o)| *s *= *o);
        }
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "component_div_assign");
        for (b, data) in self.data.iter_mut().enumerate() {
            zip!(data.as_mut(), other.data[b % other.data.len()].as_ref())
                .for_each(|unzip!(s, o)| *s /= *o);
        }
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let mut max_frac = T::zero();
        let mut max_frac_index = -1;
        let mut found_root = false;
        assert_eq!(self.len(), g1.len(), "Vector lengths do not match");
        self.context
            .assert_compatible_nbatch(g1.context.nbatch(), "root_finding");
        let mut batch_result = None;
        for batch in 0..self.data.len() {
            let mut found = false;
            let mut frac = T::zero();
            let mut index = -1;
            for i in 0..self.len() {
                let g0 = self.data[batch][i];
                let g1 = g1.data[batch % g1.data.len()][i];
                if g1 == T::zero() {
                    found = true;
                }
                if g0 * g1 < T::zero() {
                    let value = (g1 / (g1 - g0)).abs();
                    if value > frac {
                        frac = value;
                        index = i as i32;
                    }
                }
            }
            if let Some(result) = batch_result {
                assert_eq!(
                    result,
                    (found, frac, index),
                    "root finding results differ across batches"
                );
            } else {
                batch_result = Some((found, frac, index));
            }
            found_root = found;
            max_frac = frac;
            max_frac_index = index;
        }
        (found_root, max_frac, max_frac_index)
    }

    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        for data in &mut self.data {
            for i in &indices.data {
                data[*i] = value;
            }
        }
    }

    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from_indices");
        for (batch, data) in self.data.iter_mut().enumerate() {
            for i in &indices.data {
                data[*i] = other.data[batch % other.data.len()][*i];
            }
        }
    }

    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "gather");
        for (batch, data) in self.data.iter_mut().enumerate() {
            for (i, o) in indices.data.iter().enumerate() {
                data[i] = other.data[batch % other.data.len()][*o];
            }
        }
    }

    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "scatter");
        let other_nbatch = other.data.len();
        for (batch, data) in self.data.iter().enumerate() {
            for (i, o) in indices.data.iter().enumerate() {
                other.data[batch % other_nbatch][*o] = data[i];
            }
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

impl<'a, T: FaerScalar> VectorView<'a> for FaerVecRef<'a, T> {
    type Owned = FaerVec<T>;
    fn get_index(&self, index: IndexType) -> Self::T {
        assert_eq!(self.context.nbatch(), 1, "get_index requires nbatch == 1");
        self.data[0][index]
    }
    fn into_owned(self) -> FaerVec<T> {
        FaerVec {
            data: self.data.into_iter().map(|data| data.to_owned()).collect(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "squared_norm");
        self.context
            .assert_compatible_nbatch(atol.context.nbatch(), "squared_norm");
        self.data
            .iter()
            .enumerate()
            .map(|(b, x)| {
                x.iter()
                    .zip(y.data[b % y.data.len()].iter())
                    .zip(atol.data[b % atol.data.len()].iter())
                    .fold(T::zero(), |acc, ((x, y), atol)| {
                        let term = *x / (y.abs() * rtol + *atol);
                        acc + term * term
                    })
                    / T::from_f64(x.nrows() as f64).unwrap()
            })
            .fold(T::zero(), |a, b| a.max(b))
    }
}

impl<'a, T: FaerScalar> VectorViewMut<'a> for FaerVecMut<'a, T> {
    type Owned = FaerVec<T>;
    type View = FaerVecRef<'a, T>;
    type Index = FaerVecIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[batch % other.data.len()]);
        }
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from_view");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.copy_from(other.data[batch % other.data.len()]);
        }
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        for data in &mut self.data {
            data[index] = value;
        }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy");
        for (batch, data) in self.data.iter_mut().enumerate() {
            zip!(data.as_mut(), x.data[batch % x.data.len()].as_ref())
                .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
        }
    }
}

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
        let errorn_check = r.data[0].squared_norm_l2() / 3.0;
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

    super::super::generate_vector_tests_nonbatched!(faer, FaerVec<f64>);
    super::super::generate_vector_tests_batched!(
        faer,
        FaerVec<f64>,
        FaerContext::with_nbatch(2),
        FaerContext::with_nbatch(3)
    );
}
