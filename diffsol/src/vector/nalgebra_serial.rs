use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::utils::*;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, DVector, LpNorm};

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
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVecRef<'a, T: NalgebraScalar> {
    pub(crate) data: DMatrixView<'a, T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraVecMut<'a, T: NalgebraScalar> {
    pub(crate) data: DMatrixViewMut<'a, T>,
    pub(crate) context: NalgebraContext,
}

impl<T: NalgebraScalar> From<DVector<T>> for NalgebraVec<T> {
    fn from(data: DVector<T>) -> Self {
        let n = data.len();
        let data = DMatrix::from_iterator(n, 1, data.iter().copied());
        Self {
            data,
            context: NalgebraContext::default(),
        }
    }
}

impl<T: NalgebraScalar> DefaultDenseMatrix for NalgebraVec<T> {
    type M = NalgebraMat<T>;
}

impl_vector_common!(NalgebraVec<T>, NalgebraContext, DMatrix<T>, NalgebraScalar);
impl_vector_common_ref!(
    NalgebraVecRef<'a, T>,
    NalgebraContext,
    DMatrixView<'a, T>,
    NalgebraScalar
);
impl_vector_common_ref!(
    NalgebraVecMut<'a, T>,
    NalgebraContext,
    DMatrixViewMut<'a, T>,
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
                    data: &self.data * scale,
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
        impl<'a, T: NalgebraScalar> MulAssign<Scale<T>> for $col_type {
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
        assert!(
            self.context.nbatch() == 1,
            "get_index is not supported for batched vector views (nbatch > 1)"
        );
        self.data[(index, 0)]
    }

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.into_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let nbatch = self.context.nbatch();
        let nstates = self.data.nrows();
        let atol_nbatch = atol.context.nbatch();
        if y.len() != nstates || atol.len() != nstates {
            panic!("Vector lengths do not match");
        }
        let mut max_norm = T::zero();
        for b in 0..nbatch {
            let mut acc = T::zero();
            let atol_b = if atol_nbatch > 1 { b } else { 0 };
            for i in 0..nstates {
                let xi = self.data[(i, b)];
                let yi = y.data[(i, b)];
                let ai = atol.data[(i, atol_b)];
                let term = xi / (yi.abs() * rtol + ai);
                acc += term * term;
            }
            let norm = acc / Self::T::from_f64(nstates as f64).unwrap();
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }
}

impl<'a, T: NalgebraScalar> VectorViewMut<'a> for NalgebraVecMut<'a, T> {
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
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            *si = *si * beta + *xi * alpha;
        }
    }
}

impl<T: NalgebraScalar> VectorHost for NalgebraVec<T> {
    fn as_slice(&self) -> &[Self::T] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        self.data.as_mut_slice()
    }
}

impl<T: NalgebraScalar> Vector for NalgebraVec<T> {
    type View<'a> = NalgebraVecRef<'a, T>;
    type ViewMut<'a> = NalgebraVecMut<'a, T>;
    type Index = NalgebraIndex;
    fn len(&self) -> IndexType {
        self.data.nrows()
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn norm(&self, k: i32) -> Self::T {
        let nbatch = self.context.nbatch();
        if nbatch == 1 {
            return self.data.column(0).apply_norm(&LpNorm(k));
        }
        let mut max_norm = T::zero();
        for b in 0..nbatch {
            let norm = self.data.column(b).apply_norm(&LpNorm(k));
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        assert!(
            self.context.nbatch() == 1,
            "get_index is not supported for batched vectors (nbatch > 1)"
        );
        self.data[(index, 0)]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        for b in 0..nbatch {
            self.data[(index, b)] = value;
        }
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let nbatch = self.context.nbatch();
        let nstates = self.len();
        let atol_nbatch = atol.context.nbatch();
        if y.len() != nstates || atol.len() != nstates {
            panic!("Vector lengths do not match");
        }
        let mut max_norm = T::zero();
        let self_slice = self.data.as_slice();
        let y_slice = y.data.as_slice();
        let atol_slice = atol.data.as_slice();
        for b in 0..nbatch {
            let mut acc = T::zero();
            let atol_b = if atol_nbatch > 1 { b } else { 0 };
            for i in 0..nstates {
                let xi = unsafe { self_slice.get_unchecked(b * nstates + i) };
                let yi = unsafe { y_slice.get_unchecked(b * nstates + i) };
                let ai = unsafe { atol_slice.get_unchecked(atol_b * nstates + i) };
                let term = *xi / (yi.abs() * rtol + *ai);
                acc += term * term;
            }
            let norm = acc / Self::T::from_f64(nstates as f64).unwrap();
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
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
    fn get_batch(&self, batch: usize) -> Self::View<'_> {
        Self::View {
            data: self.data.columns(batch, 1),
            context: self.context.clone_with_nbatch(1),
        }
    }
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_> {
        Self::ViewMut {
            data: self.data.columns_mut(batch, 1),
            context: self.context.clone_with_nbatch(1),
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
        let data = DMatrix::from_element(nstates, ctx.nbatch(), value);
        Self { data, context: ctx }
    }
    fn from_vec(vec: Vec<T>, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert!(
            vec.len() % nbatch == 0,
            "vec length {} must be divisible by nbatch {}",
            vec.len(),
            nbatch
        );
        let nstates = vec.len() / nbatch;
        let data = DMatrix::from_vec(nstates, nbatch, vec);
        Self { data, context: ctx }
    }
    fn from_slice(slice: &[T], ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert!(
            slice.len() % nbatch == 0,
            "slice length {} must be divisible by nbatch {}",
            slice.len(),
            nbatch
        );
        let nstates = slice.len() / nbatch;
        let data = DMatrix::from_column_slice(nstates, nbatch, slice);
        Self { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data.as_slice().to_vec()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let data = DMatrix::zeros(nstates, ctx.nbatch());
        Self { data, context: ctx }
    }
    fn axpy(&mut self, alpha: T, x: &Self, beta: T) {
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            *si = *si * beta + *xi * alpha;
        }
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            *si = *si * beta + *xi * alpha;
        }
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.data.component_div_assign(&other.data);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.data.component_mul_assign(&other.data);
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let nbatch = self.context.nbatch();
        let nstates = self.len();
        assert_eq!(nstates, g1.len(), "Vector lengths do not match");
        let self_slice = self.data.as_slice();
        let g1_slice = g1.data.as_slice();
        let mut first_result: Option<(bool, Self::T, i32)> = None;
        for b in 0..nbatch {
            let mut max_frac = T::zero();
            let mut max_frac_index: i32 = -1;
            let mut found_root = false;
            for i in 0..nstates {
                let g0 = unsafe { *self_slice.get_unchecked(b * nstates + i) };
                let g1v = unsafe { *g1_slice.get_unchecked(b * nstates + i) };
                if g1v == T::zero() {
                    found_root = true;
                }
                if g0 * g1v < T::zero() {
                    let frac = (g1v / (g1v - g0)).abs();
                    if frac > max_frac {
                        max_frac = frac;
                        max_frac_index = i as i32;
                    }
                }
            }
            let result = (found_root, max_frac, max_frac_index);
            if let Some(ref first) = first_result {
                if first.0 != result.0 || first.2 != result.2 {
                    panic!(
                        "Root finding results differ across batches: batch 0 = {:?}, batch {} = {:?}",
                        first, b, result
                    );
                }
            } else {
                first_result = Some(result);
            }
        }
        first_result.unwrap()
    }

    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        let nbatch = self.context.nbatch();
        for b in 0..nbatch {
            for i in indices.data.iter() {
                self.data[(*i, b)] = value;
            }
        }
    }

    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        let nbatch = self.context.nbatch();
        for b in 0..nbatch {
            for i in indices.data.iter() {
                self.data[(*i, b)] = other.data[(*i, b)];
            }
        }
    }

    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        let nstates = self.len();
        let nbatch = self.context.nbatch();
        assert_eq!(nstates, indices.len(), "Vector lengths do not match");
        for b in 0..nbatch {
            for (i, o) in indices.data.iter().enumerate() {
                self.data[(i, b)] = other.data[(*o, b)];
            }
        }
    }

    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        let nstates = self.len();
        let nbatch = self.context.nbatch();
        assert_eq!(nstates, indices.len(), "Vector lengths do not match");
        for b in 0..nbatch {
            for (i, o) in indices.data.iter().enumerate() {
                other.data[(*o, b)] = self.data[(i, b)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    super::super::generate_vector_tests!(
        nalgebra,
        NalgebraVec<f64>,
        NalgebraContext::with_nbatch(2),
        NalgebraContext::with_nbatch(3)
    );

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
    fn test_into() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v: NalgebraVec<f64> = vec.into();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }
}
