use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::utils::*;
use nalgebra::{DVector, DVectorView, DVectorViewMut, LpNorm};

use crate::{
    Context, IndexType, NalgebraContext, NalgebraMat, NalgebraScalar, Scalar, Scale, VectorHost,
};

use super::{DefaultDenseMatrix, Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraIndex {
    pub(crate) data: DVector<IndexType>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVec<T: NalgebraScalar> {
    pub(crate) data: Vec<DVector<T>>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVecRef<'a, T: NalgebraScalar> {
    pub(crate) data: Vec<DVectorView<'a, T>>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraVecMut<'a, T: NalgebraScalar> {
    pub(crate) data: Vec<DVectorViewMut<'a, T>>,
    pub(crate) context: NalgebraContext,
}

impl<T: NalgebraScalar> From<DVector<T>> for NalgebraVec<T> {
    fn from(data: DVector<T>) -> Self {
        Self {
            data: vec![data],
            context: NalgebraContext::default(),
        }
    }
}

impl<T: NalgebraScalar> DefaultDenseMatrix for NalgebraVec<T> {
    type M = NalgebraMat<T>;
}

impl_vector_common!(
    NalgebraVec<T>,
    NalgebraContext,
    Vec<DVector<T>>,
    NalgebraScalar
);
impl_vector_common_ref!(
    NalgebraVecRef<'a, T>,
    NalgebraContext,
    Vec<DVectorView<'a, T>>,
    NalgebraScalar
);
impl_vector_common_ref!(
    NalgebraVecMut<'a, T>,
    NalgebraContext,
    Vec<DVectorViewMut<'a, T>>,
    NalgebraScalar
);

macro_rules! impl_mul_scalar {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: NalgebraScalar> Mul<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: $scalar = rhs.value();
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
        impl<'a, T: NalgebraScalar> Div<Scale<T>> for $lhs {
            type Output = $out;
            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: Scale<T>) -> Self::Output {
                let inv_rhs: T = T::one() / rhs.value();
                Self::Output {
                    data: self.data.iter().map(|data| data * inv_rhs).collect(),
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($col_type:ty, $scalar:ty) => {
        impl<'a, T: NalgebraScalar> MulAssign<Scale<T>> for $col_type {
            #[inline]
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale = rhs.value();
                for data in &mut self.data {
                    *data *= scale;
                }
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

impl_sub_assign!(NalgebraVec<T>, NalgebraVec<T>, NalgebraScalar);
impl_sub_assign!(NalgebraVec<T>, &NalgebraVec<T>, NalgebraScalar);
impl_sub_assign!(NalgebraVec<T>, NalgebraVecRef<'_, T>, NalgebraScalar);
impl_sub_assign!(NalgebraVec<T>, &NalgebraVecRef<'_, T>, NalgebraScalar);

impl_sub_assign!(NalgebraVecMut<'_, T>, NalgebraVec<T>, NalgebraScalar);
impl_sub_assign!(NalgebraVecMut<'_, T>, &NalgebraVec<T>, NalgebraScalar);
impl_sub_assign!(NalgebraVecMut<'_, T>, NalgebraVecRef<'_, T>, NalgebraScalar);
impl_sub_assign!(
    NalgebraVecMut<'_, T>,
    &NalgebraVecRef<'_, T>,
    NalgebraScalar
);

impl_add_assign!(NalgebraVec<T>, NalgebraVec<T>, NalgebraScalar);
impl_add_assign!(NalgebraVec<T>, &NalgebraVec<T>, NalgebraScalar);
impl_add_assign!(NalgebraVec<T>, NalgebraVecRef<'_, T>, NalgebraScalar);
impl_add_assign!(NalgebraVec<T>, &NalgebraVecRef<'_, T>, NalgebraScalar);

impl_add_assign!(NalgebraVecMut<'_, T>, NalgebraVec<T>, NalgebraScalar);
impl_add_assign!(NalgebraVecMut<'_, T>, &NalgebraVec<T>, NalgebraScalar);
impl_add_assign!(NalgebraVecMut<'_, T>, NalgebraVecRef<'_, T>, NalgebraScalar);
impl_add_assign!(
    NalgebraVecMut<'_, T>,
    &NalgebraVecRef<'_, T>,
    NalgebraScalar
);

impl_sub_both_ref!(
    &NalgebraVec<T>,
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_rhs!(
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_both_ref!(
    &NalgebraVec<T>,
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_both_ref!(
    &NalgebraVec<T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);

impl_sub_lhs!(
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_lhs!(
    NalgebraVec<T>,
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_lhs!(
    NalgebraVec<T>,
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_lhs!(
    NalgebraVec<T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);

impl_sub_rhs!(
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_both_ref!(
    NalgebraVecRef<'_, T>,
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_both_ref!(
    NalgebraVecRef<'_, T>,
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_sub_both_ref!(
    NalgebraVecRef<'_, T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);

impl_add_both_ref!(
    &NalgebraVec<T>,
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_rhs!(
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_both_ref!(
    &NalgebraVec<T>,
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_both_ref!(
    &NalgebraVec<T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);

impl_add_lhs!(
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_lhs!(
    NalgebraVec<T>,
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_lhs!(
    NalgebraVec<T>,
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_lhs!(
    NalgebraVec<T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);

impl_add_rhs!(
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_both_ref!(
    NalgebraVecRef<'_, T>,
    &NalgebraVec<T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_both_ref!(
    NalgebraVecRef<'_, T>,
    NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);
impl_add_both_ref!(
    NalgebraVecRef<'_, T>,
    &NalgebraVecRef<'_, T>,
    NalgebraVec<T>,
    NalgebraScalar
);

impl_index!(NalgebraVec<T>, NalgebraScalar);
impl_index_mut!(NalgebraVec<T>, NalgebraScalar);

impl_index!(NalgebraVecRef<'_, T>, NalgebraScalar);

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

impl<'a, T: NalgebraScalar> VectorView<'a> for NalgebraVecRef<'a, T> {
    type Owned = NalgebraVec<T>;

    fn get_index(&self, index: IndexType) -> Self::T {
        assert_eq!(self.context.nbatch(), 1, "get_index requires nbatch == 1");
        self.data[0][index]
    }

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self
                .data
                .into_iter()
                .map(|data| data.into_owned())
                .collect(),
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
                    / T::from_f64(x.len() as f64).unwrap()
            })
            .fold(T::zero(), |a, b| a.max(b))
    }
}

impl<'a, T: NalgebraScalar> VectorViewMut<'a> for NalgebraVecMut<'a, T> {
    type Owned = NalgebraVec<T>;
    type View = NalgebraVecRef<'a, T>;
    type Index = NalgebraIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[b % other.data.len()]);
        }
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from_view");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[b % other.data.len()]);
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
        for (b, data) in self.data.iter_mut().enumerate() {
            data.axpy(alpha, &x.data[b % x.data.len()], beta);
        }
    }
}

impl<T: NalgebraScalar> VectorHost for NalgebraVec<T> {
    fn as_slice(&self) -> &[Self::T] {
        assert_eq!(self.context.nbatch(), 1, "as_slice requires nbatch == 1");
        self.data[0].as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        assert_eq!(
            self.context.nbatch(),
            1,
            "as_mut_slice requires nbatch == 1"
        );
        self.data[0].as_mut_slice()
    }
}

impl<T: NalgebraScalar> Vector for NalgebraVec<T> {
    type View<'a> = NalgebraVecRef<'a, T>;
    type ViewMut<'a> = NalgebraVecMut<'a, T>;
    type Index = NalgebraIndex;
    fn len(&self) -> IndexType {
        self.data[0].len()
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn norm(&self, k: i32) -> Self::T {
        self.data
            .iter()
            .map(|data| data.apply_norm(&LpNorm(k)))
            .fold(T::zero(), |a, b| a.max(b))
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
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "squared_norm");
        self.context
            .assert_compatible_nbatch(atol.context.nbatch(), "squared_norm");
        self.data
            .iter()
            .enumerate()
            .map(|(b, x)| {
                let y = &y.data[b % y.data.len()];
                let atol = &atol.data[b % atol.data.len()];
                x.iter()
                    .zip(y.iter())
                    .zip(atol.iter())
                    .fold(T::zero(), |acc, ((x, y), atol)| {
                        let term = *x / (y.abs() * rtol + *atol);
                        acc + term * term
                    })
                    / T::from_f64(x.len() as f64).unwrap()
            })
            .fold(T::zero(), |a, b| a.max(b))
    }
    fn as_view(&self) -> Self::View<'_> {
        Self::View {
            data: self.data.iter().map(|data| data.as_view()).collect(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        Self::ViewMut {
            data: self
                .data
                .iter_mut()
                .map(|data| data.as_view_mut())
                .collect(),
            context: self.context,
        }
    }
    fn get_batch(&self, batch: usize) -> Self::View<'_> {
        assert!(batch < self.data.len());
        Self::View {
            data: vec![self.data[batch].as_view()],
            context: NalgebraContext::default(),
        }
    }
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_> {
        assert!(batch < self.data.len());
        Self::ViewMut {
            data: vec![self.data[batch].as_view_mut()],
            context: NalgebraContext::default(),
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[b % other.data.len()]);
        }
    }
    fn fill(&mut self, value: Self::T) {
        for data in &mut self.data {
            data.fill(value);
        }
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from_view");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[b % other.data.len()]);
        }
    }
    fn from_element(nstates: usize, value: T, ctx: Self::C) -> Self {
        let data = (0..ctx.nbatch())
            .map(|_| DVector::from_element(nstates, value))
            .collect();
        Self { data, context: ctx }
    }
    fn from_vec(vec: Vec<T>, ctx: Self::C) -> Self {
        assert!(
            vec.len().is_multiple_of(ctx.nbatch()),
            "vector length must be divisible by nbatch"
        );
        let n = vec.len() / ctx.nbatch();
        let data = if n == 0 {
            (0..ctx.nbatch()).map(|_| DVector::zeros(0)).collect()
        } else {
            vec.chunks(n).map(DVector::from_column_slice).collect()
        };
        Self { data, context: ctx }
    }
    fn from_slice(slice: &[T], ctx: Self::C) -> Self {
        assert!(
            slice.len().is_multiple_of(ctx.nbatch()),
            "vector length must be divisible by nbatch"
        );
        let n = slice.len() / ctx.nbatch();
        let data = if n == 0 {
            (0..ctx.nbatch()).map(|_| DVector::zeros(0)).collect()
        } else {
            slice.chunks(n).map(DVector::from_column_slice).collect()
        };
        Self { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data
            .iter()
            .flat_map(|data| data.iter().copied())
            .collect()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let data = (0..ctx.nbatch()).map(|_| DVector::zeros(nstates)).collect();
        Self { data, context: ctx }
    }
    fn axpy(&mut self, alpha: T, x: &Self, beta: T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.axpy(alpha, &x.data[b % x.data.len()], beta);
        }
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy_v");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.axpy(alpha, &x.data[b % x.data.len()], beta);
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
            data.axpy(alpha[b], &x.data[b % x.data.len()], beta);
        }
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "component_div_assign");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.component_div_assign(&other.data[b % other.data.len()]);
        }
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "component_mul_assign");
        for (b, data) in self.data.iter_mut().enumerate() {
            data.component_mul_assign(&other.data[b % other.data.len()]);
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
        let errorn_check = r.data[0].norm_squared() / 3.0;
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

    super::super::generate_vector_tests_nonbatched!(nalgebra, NalgebraVec<f64>);
    super::super::generate_vector_tests_batched!(
        nalgebra,
        NalgebraVec<f64>,
        NalgebraContext::with_nbatch(2),
        NalgebraContext::with_nbatch(3)
    );
}
