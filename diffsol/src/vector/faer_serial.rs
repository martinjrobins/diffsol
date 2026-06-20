use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::slice;

use faer::reborrow::ReborrowMut;
use faer::{unzip, zip, Col, Mat, MatMut, MatRef};

use crate::{scalar::Scale, Context, FaerContext, FaerScalar, IndexType, Scalar, Vector};

use crate::{FaerMat, VectorCommon, VectorHost, VectorIndex, VectorView, VectorViewMut};

use super::utils::*;
use super::DefaultDenseMatrix;

/// Dense vector backed by a faer [`Mat`].
///
/// # Data layout with batching
///
/// When `nbatch > 1`, data is stored as an `(nstates, nbatch)` [`Mat`] in
/// **column-major** order. Each column corresponds to one batch, so batch *b*
/// occupies column *b*.  Linear indexing (e.g. `from_vec`) visits batch 0
/// elements first, then batch 1, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct FaerVec<T: FaerScalar> {
    pub(crate) data: Mat<T>,
    pub(crate) context: FaerContext,
}

/// Stores integer indices used by gather/scatter/assign-at-indices operations.
/// Indices are shared across all batches.
#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecIndex {
    pub(crate) data: Vec<IndexType>,
    pub(crate) context: FaerContext,
}

/// Immutable reference to a [`FaerVec`].
///
/// The underlying [`MatRef`] has shape `(nstates, nbatch)` with one column
/// per batch, matching the layout of [`FaerVec`].
#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecRef<'a, T: FaerScalar> {
    pub(crate) data: MatRef<'a, T>,
    pub(crate) context: FaerContext,
}

/// Mutable reference to a [`FaerVec`].
///
/// The underlying [`MatMut`] has shape `(nstates, nbatch)` with one column
/// per batch, matching the layout of [`FaerVec`].
#[derive(Debug, PartialEq)]
pub struct FaerVecMut<'a, T: FaerScalar> {
    pub(crate) data: MatMut<'a, T>,
    pub(crate) context: FaerContext,
}

impl<T: FaerScalar> From<Col<T>> for FaerVec<T> {
    fn from(data: Col<T>) -> Self {
        let nrows = data.nrows();
        let mat = Mat::from_fn(nrows, 1, |i, _| data[i]);
        Self {
            data: mat,
            context: FaerContext::default(),
        }
    }
}

macro_rules! faer_squared_norm_body {
    ($self:ident, $y:ident, $atol:ident, $rtol:ident, $nstates_expr:expr, $T:ty) => {{
        let nbatch = $self.context.nbatch();
        let nstates = $nstates_expr;
        let atol_nbatch = $atol.context.nbatch();
        if $y.len() != nstates || $atol.len() != nstates {
            panic!("Vector lengths do not match");
        }
        let nstates_t = <$T as num_traits::FromPrimitive>::from_f64(nstates as f64).unwrap();
        if nbatch == 1 && atol_nbatch == 1 {
            let mut acc = <$T as num_traits::Zero>::zero();
            zip!($self.data.col(0), $y.data.col(0), $atol.data.col(0)).for_each(
                |unzip!(xi, yi, ai)| {
                    let denom = yi.abs() * $rtol + *ai;
                    let term = *xi / denom;
                    acc += term * term;
                },
            );
            return acc / nstates_t;
        }
        let mut max_norm = <$T as num_traits::Zero>::zero();
        for b in 0..nbatch {
            let atol_b = if atol_nbatch > 1 { b } else { 0 };
            let mut acc = <$T as num_traits::Zero>::zero();
            zip!($self.data.col(b), $y.data.col(b), $atol.data.col(atol_b)).for_each(
                |unzip!(xi, yi, ai)| {
                    let denom = yi.abs() * $rtol + *ai;
                    let term = *xi / denom;
                    acc += term * term;
                },
            );
            let norm = acc / nstates_t;
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }};
}

impl<T: FaerScalar> FaerVec<T> {
    pub fn check_for_nan(&self, label: &str) -> bool {
        for b in 0..self.data.ncols() {
            for i in 0..self.data.nrows() {
                if self.data[(i, b)].is_nan() {
                    eprintln!("{}: NaN at batch {} index {}", label, b, i);
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

impl_vector_common!(FaerVec<T>, FaerContext, Mat<T>, FaerScalar);
impl_vector_common_ref!(FaerVecRef<'a, T>, FaerContext, MatRef<'a, T>, FaerScalar);
impl_vector_common_ref!(FaerVecMut<'a, T>, FaerContext, MatMut<'a, T>, FaerScalar);

macro_rules! impl_mul_scalar {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: FaerScalar> Mul<Scale<T>> for $lhs {
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
        impl<'a, T: FaerScalar> Div<Scale<T>> for $lhs {
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
        impl<'a, T: FaerScalar> MulAssign<Scale<T>> for $col_type {
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

macro_rules! impl_faer_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: FaerScalar> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let self_ncols = self.data.ncols();
                let rhs_ncols = rhs.data.ncols();
                if self_ncols == rhs_ncols {
                    self.data -= &rhs.data;
                } else if rhs_ncols == 1 {
                    let rhs_col = rhs.data.col(0);
                    for b in 0..self_ncols {
                        let mut col = self.data.as_mut().col_mut(b);
                        col -= rhs_col;
                    }
                } else {
                    panic!(
                        "incompatible nbatch in sub_assign: self={}, rhs={}",
                        self_ncols, rhs_ncols
                    );
                }
            }
        }
    };
}

macro_rules! impl_faer_add_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: FaerScalar> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                let self_ncols = self.data.ncols();
                let rhs_ncols = rhs.data.ncols();
                if self_ncols == rhs_ncols {
                    self.data += &rhs.data;
                } else if rhs_ncols == 1 {
                    let rhs_col = rhs.data.col(0);
                    for b in 0..self_ncols {
                        let mut col = self.data.as_mut().col_mut(b);
                        col += rhs_col;
                    }
                } else {
                    panic!(
                        "incompatible nbatch in add_assign: self={}, rhs={}",
                        self_ncols, rhs_ncols
                    );
                }
            }
        }
    };
}

macro_rules! faer_copy_from_body {
    ($self_data:expr, $other:ident, $name:expr) => {{
        let self_ncols = $self_data.ncols();
        let other_ncols = $other.data.ncols();
        if self_ncols == other_ncols {
            $self_data.copy_from(&$other.data);
        } else if other_ncols == 1 {
            let src = $other.data.col(0);
            for b in 0..self_ncols {
                $self_data.col_mut(b).copy_from(&src);
            }
        } else {
            panic!(
                "incompatible nbatch in {}: self={}, other={}",
                $name, self_ncols, other_ncols
            );
        }
    }};
}

macro_rules! faer_axpy_body {
    ($self_data:expr, $x:ident, $alpha:expr, $beta:expr, $name:expr) => {{
        let self_ncols = $self_data.ncols();
        let x_ncols = $x.data.ncols();
        if self_ncols == x_ncols {
            for b in 0..self_ncols {
                zip!($self_data.col_mut(b), $x.data.col(b))
                    .for_each(|unzip!(si, xi)| *si = *si * $beta + *xi * $alpha);
            }
        } else if x_ncols == 1 {
            let x_col = $x.data.col(0);
            for b in 0..self_ncols {
                zip!($self_data.col_mut(b), x_col)
                    .for_each(|unzip!(si, xi)| *si = *si * $beta + *xi * $alpha);
            }
        } else {
            panic!(
                "incompatible nbatch in {}: self={}, x={}",
                $name, self_ncols, x_ncols
            );
        }
    }};
}

macro_rules! faer_component_op_body {
    ($self:ident, $other:ident, $op:tt, $name:expr) => {{
        let self_ncols = $self.data.ncols();
        let other_ncols = $other.data.ncols();
        if self_ncols == other_ncols {
            zip!($self.data.as_mut(), $other.data.as_ref())
                .for_each(|unzip!(s, o)| *s $op *o);
        } else if other_ncols == 1 {
            let other_col = $other.data.col(0);
            for b in 0..self_ncols {
                zip!($self.data.col_mut(b), other_col)
                    .for_each(|unzip!(s, o)| *s $op *o);
            }
        } else {
            panic!(
                "incompatible nbatch in {}: self={}, other={}",
                $name, self_ncols, other_ncols
            );
        }
    }};
}

impl_faer_sub_assign!(FaerVec<T>, FaerVec<T>);
impl_faer_sub_assign!(FaerVec<T>, &FaerVec<T>);
impl_faer_sub_assign!(FaerVec<T>, FaerVecRef<'_, T>);
impl_faer_sub_assign!(FaerVec<T>, &FaerVecRef<'_, T>);

impl_faer_sub_assign!(FaerVecMut<'_, T>, FaerVec<T>);
impl_faer_sub_assign!(FaerVecMut<'_, T>, &FaerVec<T>);
impl_faer_sub_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>);
impl_faer_sub_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>);

impl_faer_add_assign!(FaerVec<T>, FaerVec<T>);
impl_faer_add_assign!(FaerVec<T>, &FaerVec<T>);
impl_faer_add_assign!(FaerVec<T>, FaerVecRef<'_, T>);
impl_faer_add_assign!(FaerVec<T>, &FaerVecRef<'_, T>);

impl_faer_add_assign!(FaerVecMut<'_, T>, FaerVec<T>);
impl_faer_add_assign!(FaerVecMut<'_, T>, &FaerVec<T>);
impl_faer_add_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>);
impl_faer_add_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>);

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

impl<T: Scalar + FaerScalar> Index<IndexType> for FaerVec<T> {
    type Output = T;
    fn index(&self, index: IndexType) -> &Self::Output {
        assert!(
            self.context.nbatch() == 1,
            "Index is not supported for batched vectors (nbatch > 1)"
        );
        &self.data[(index, 0)]
    }
}

impl<T: Scalar + FaerScalar> IndexMut<IndexType> for FaerVec<T> {
    fn index_mut(&mut self, index: IndexType) -> &mut Self::Output {
        assert!(
            self.context.nbatch() == 1,
            "IndexMut is not supported for batched vectors (nbatch > 1)"
        );
        &mut self.data[(index, 0)]
    }
}

impl<T: Scalar + FaerScalar> Index<IndexType> for FaerVecRef<'_, T> {
    type Output = T;
    fn index(&self, index: IndexType) -> &Self::Output {
        assert!(
            self.context.nbatch() == 1,
            "Index is not supported for batched vector views (nbatch > 1)"
        );
        &self.data[(index, 0)]
    }
}

impl<T: FaerScalar> VectorHost for FaerVec<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr_mut(), self.total_len()) }
    }
    fn as_slice(&self) -> &[Self::T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.total_len()) }
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
        self.data.nrows()
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
    fn norm(&self, k: i32) -> T {
        let nbatch = self.context.nbatch();
        if nbatch == 1 {
            let col = self.data.col(0);
            return match k {
                1 => col.norm_l1(),
                2 => col.norm_l2(),
                _ => col.iter().fold(T::zero(), |acc, x| acc + x.pow(k))
                    .pow(T::one() / T::from_f64(k as f64).unwrap()),
            };
        }
        let mut max_norm = T::zero();
        for b in 0..self.context.nbatch() {
            let col = self.data.col(b);
            let norm = match k {
                1 => col.norm_l1(),
                2 => col.norm_l2(),
                _ => col
                    .iter()
                    .fold(T::zero(), |acc, x| acc + x.pow(k))
                    .pow(T::one() / T::from_f64(k as f64).unwrap()),
            };
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }

    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        faer_squared_norm_body!(self, y, atol, rtol, self.len(), Self::T)
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
    fn get_batch(&self, batch: usize) -> Self::View<'_> {
        assert!(batch < self.context.nbatch(), "batch index out of bounds");
        FaerVecRef {
            data: self
                .data
                .as_ref()
                .get(0..self.data.nrows(), batch..batch + 1),
            context: self.context.clone_with_nbatch(1),
        }
    }
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_> {
        assert!(batch < self.context.nbatch(), "batch index out of bounds");
        let nrows = self.data.nrows();
        FaerVecMut {
            data: self.data.as_mut().get_mut(0..nrows, batch..batch + 1),
            context: self.context.clone_with_nbatch(1),
        }
    }
    fn copy_from(&mut self, other: &Self) {
        faer_copy_from_body!(self.data, other, "copy_from")
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        faer_copy_from_body!(self.data, other, "copy_from_view")
    }
    fn fill(&mut self, value: Self::T) {
        zip!(self.data.as_mut()).for_each(|unzip!(s)| *s = value);
    }
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = Mat::from_fn(nstates, nbatch, |_, _| value);
        FaerVec { data, context: ctx }
    }
    fn from_vec(vec: Vec<Self::T>, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert!(
            vec.len() % nbatch == 0,
            "vec length {} must be divisible by nbatch {}",
            vec.len(),
            nbatch
        );
        let nstates = vec.len() / nbatch;
        let data = Mat::from_fn(nstates, nbatch, |i, b| vec[b * nstates + i]);
        FaerVec { data, context: ctx }
    }
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert!(
            slice.len() % nbatch == 0,
            "slice length {} must be divisible by nbatch {}",
            slice.len(),
            nbatch
        );
        let nstates = slice.len() / nbatch;
        let data = Mat::from_fn(nstates, nbatch, |i, b| slice[b * nstates + i]);
        FaerVec { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        (0..self.data.ncols())
            .flat_map(|b| self.data.col(b).iter().copied())
            .collect()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = Mat::zeros(nstates, nbatch);
        FaerVec { data, context: ctx }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        faer_axpy_body!(self.data, x, alpha, beta, "axpy")
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        faer_axpy_body!(self.data, x, alpha, beta, "axpy_v")
    }
    fn batched_axpy(&mut self, alpha: &[Self::T], x: &Self, beta: Self::T) {
        let self_ncols = self.data.ncols();
        let x_ncols = x.data.ncols();
        assert_eq!(
            alpha.len(),
            self_ncols,
            "batched_axpy: alpha.len() must equal self.nbatch()"
        );
        if self_ncols == x_ncols {
            for b in 0..self_ncols {
                zip!(self.data.col_mut(b), x.data.col(b))
                    .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha[b]);
            }
        } else if x_ncols == 1 {
            let x_col = x.data.col(0);
            for b in 0..self_ncols {
                zip!(self.data.col_mut(b), x_col)
                    .for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha[b]);
            }
        } else {
            panic!(
                "incompatible nbatch in batched_axpy: self={}, x={}",
                self_ncols, x_ncols
            );
        }
    }
    fn component_mul_assign(&mut self, other: &Self) {
        faer_component_op_body!(self, other, *=, "component_mul_assign")
    }
    fn component_div_assign(&mut self, other: &Self) {
        faer_component_op_body!(self, other, /=, "component_div_assign")
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let nbatch = self.context.nbatch();
        let nstates = self.len();
        assert_eq!(nstates, g1.len(), "Vector lengths do not match");
        if nbatch == 1 {
            let s_col = self.data.col(0);
            let g_col = g1.data.col(0);
            let mut max_frac = T::zero();
            let mut max_frac_index: i32 = -1;
            let mut found_root = false;
            for i in 0..nstates {
                let g0 = unsafe { *s_col.get_unchecked(i) };
                let g1v = unsafe { *g_col.get_unchecked(i) };
                if g1v == T::zero() { found_root = true; }
                if g0 * g1v < T::zero() {
                    let frac = (g1v / (g1v - g0)).abs();
                    if frac > max_frac { max_frac = frac; max_frac_index = i as i32; }
                }
            }
            return (found_root, max_frac, max_frac_index);
        }
        let mut first_result: Option<(bool, Self::T, i32)> = None;
        for b in 0..nbatch {
            let mut max_frac = T::zero();
            let mut max_frac_index: i32 = -1;
            let mut found_root = false;
            for i in 0..nstates {
                let g0 = self.data[(i, b)];
                let g1v = g1.data[(i, b)];
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
        assert!(
            self.context.nbatch() == 1,
            "get_index is not supported for batched vector views (nbatch > 1)"
        );
        self.data[(index, 0)]
    }
    fn into_owned(self) -> FaerVec<T> {
        FaerVec {
            data: self.data.to_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        faer_squared_norm_body!(self, y, atol, rtol, self.data.nrows(), Self::T)
    }
}

impl<'a, T: FaerScalar> VectorViewMut<'a> for FaerVecMut<'a, T> {
    type Owned = FaerVec<T>;
    type View = FaerVecRef<'a, T>;
    type Index = FaerVecIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        faer_copy_from_body!(self.data.rb_mut(), other, "VectorViewMut::copy_from")
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        faer_copy_from_body!(self.data.rb_mut(), other, "VectorViewMut::copy_from_view")
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        for b in 0..nbatch {
            self.data[(index, b)] = value;
        }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        faer_axpy_body!(self.data.rb_mut(), x, alpha, beta, "VectorViewMut::axpy")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::scale;

    super::super::generate_vector_tests!(
        faer,
        FaerVec<f64>,
        FaerContext::with_nbatch(2),
        FaerContext::with_nbatch(3)
    );

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
        let errorn_check = r.data.col(0).squared_norm_l2() / 3.0;
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
