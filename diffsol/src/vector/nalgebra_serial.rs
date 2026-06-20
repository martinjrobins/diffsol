use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::utils::*;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, DVector, LpNorm};

use crate::{
    Context, IndexType, NalgebraContext, NalgebraMat, NalgebraScalar, Scalar, Scale, VectorHost,
};

use super::{DefaultDenseMatrix, Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

/// Stores integer indices used by gather/scatter/assign-at-indices operations.
/// Indices are shared across all batches.
#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraIndex {
    pub(crate) data: DVector<IndexType>,
    pub(crate) context: NalgebraContext,
}

/// Dense vector backed by nalgebra's [`DMatrix`].
///
/// # Data layout with batching
///
/// When `nbatch > 1`, data is stored as an `(nstates, nbatch)` DMatrix in
/// **column-major** order. Each column corresponds to one batch, so batch *b*
/// occupies column *b*.  Linear indexing (e.g. `from_vec`) visits batch 0
/// elements first, then batch 1, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVec<T: NalgebraScalar> {
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}

/// Immutable reference to a [`NalgebraVec`].
///
/// The underlying [`DMatrixView`] has shape `(nstates, nbatch)` with one
/// column per batch, matching the layout of [`NalgebraVec`].
#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVecRef<'a, T: NalgebraScalar> {
    pub(crate) data: DMatrixView<'a, T>,
    pub(crate) context: NalgebraContext,
}

/// Mutable reference to a [`NalgebraVec`].
///
/// The underlying [`DMatrixViewMut`] has shape `(nstates, nbatch)` with one
/// column per batch, matching the layout of [`NalgebraVec`].
#[derive(Debug, PartialEq)]
pub struct NalgebraVecMut<'a, T: NalgebraScalar> {
    pub(crate) data: DMatrixViewMut<'a, T>,
    pub(crate) context: NalgebraContext,
}

impl<T: NalgebraScalar> From<DVector<T>> for NalgebraVec<T> {
    fn from(data: DVector<T>) -> Self {
        let n = data.len();
        let data = DMatrix::from_vec(n, 1, data.as_slice().to_vec());
        Self {
            data,
            context: NalgebraContext::default(),
        }
    }
}

macro_rules! nalgebra_squared_norm_body {
    ($self:ident, $y:ident, $atol:ident, $rtol:ident, $nstates_expr:expr, $T:ty) => {{
        let nbatch = $self.context.nbatch();
        let nstates = $nstates_expr;
        let atol_nbatch = $atol.context.nbatch();
        if $y.len() != nstates || $atol.len() != nstates {
            panic!("Vector lengths do not match");
        }
        let nstates_t = <$T as num_traits::FromPrimitive>::from_f64(nstates as f64).unwrap();
        if nbatch == 1 && atol_nbatch == 1 {
            let acc = $self
                .data
                .column(0)
                .iter()
                .zip($y.data.column(0).iter())
                .zip($atol.data.column(0).iter())
                .fold(<$T as num_traits::Zero>::zero(), |acc, ((xi, yi), ai)| {
                    let term = *xi / (yi.abs() * $rtol + *ai);
                    acc + term * term
                });
            return acc / nstates_t;
        }
        let mut max_norm = <$T as num_traits::Zero>::zero();
        for b in 0..nbatch {
            let atol_b = if atol_nbatch > 1 { b } else { 0 };
            let acc = $self
                .data
                .column(b)
                .iter()
                .zip($y.data.column(b).iter())
                .zip($atol.data.column(atol_b).iter())
                .fold(<$T as num_traits::Zero>::zero(), |acc, ((xi, yi), ai)| {
                    let term = *xi / (yi.abs() * $rtol + *ai);
                    acc + term * term
                });
            let norm = acc / nstates_t;
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }};
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

macro_rules! impl_nalgebra_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: NalgebraScalar> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let self_ncols = self.data.ncols();
                let rhs_ncols = rhs.data.ncols();
                if self_ncols == rhs_ncols {
                    self.data -= &rhs.data;
                } else if rhs_ncols == 1 {
                    let rhs_col = rhs.data.column(0);
                    for b in 0..self_ncols {
                        let mut col = self.data.column_mut(b);
                        col -= &rhs_col;
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

macro_rules! impl_nalgebra_add_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: NalgebraScalar> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                let self_ncols = self.data.ncols();
                let rhs_ncols = rhs.data.ncols();
                if self_ncols == rhs_ncols {
                    self.data += &rhs.data;
                } else if rhs_ncols == 1 {
                    let rhs_col = rhs.data.column(0);
                    for b in 0..self_ncols {
                        let mut col = self.data.column_mut(b);
                        col += &rhs_col;
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

macro_rules! nalgebra_copy_from_body {
    ($self:ident, $other:ident, $name:expr) => {{
        let self_ncols = $self.data.ncols();
        let other_ncols = $other.data.ncols();
        if self_ncols == other_ncols {
            $self.data.copy_from(&$other.data);
        } else if other_ncols == 1 {
            let src = $other.data.column(0);
            for b in 0..self_ncols {
                $self.data.column_mut(b).copy_from(&src);
            }
        } else {
            panic!(
                "incompatible nbatch in {}: self={}, other={}",
                $name, self_ncols, other_ncols
            );
        }
    }};
}

macro_rules! nalgebra_axpy_body {
    ($self:ident, $x:ident, $alpha:expr, $beta:expr, $name:expr) => {{
        let self_ncols = $self.data.ncols();
        let x_ncols = $x.data.ncols();
        if self_ncols == x_ncols {
            for b in 0..self_ncols {
                $self
                    .data
                    .column_mut(b)
                    .axpy($alpha, &$x.data.column(b), $beta);
            }
        } else if x_ncols == 1 {
            let x_col = $x.data.column(0);
            for b in 0..self_ncols {
                $self.data.column_mut(b).axpy($alpha, &x_col, $beta);
            }
        } else {
            panic!(
                "incompatible nbatch in {}: self={}, x={}",
                $name, self_ncols, x_ncols
            );
        }
    }};
}

macro_rules! nalgebra_component_op_body {
    ($self:ident, $other:ident, $method:ident, $name:expr) => {{
        let self_ncols = $self.data.ncols();
        let other_ncols = $other.data.ncols();
        if self_ncols == other_ncols {
            $self.data.$method(&$other.data);
        } else if other_ncols == 1 {
            let other_col = $other.data.column(0);
            for b in 0..self_ncols {
                $self.data.column_mut(b).$method(&other_col);
            }
        } else {
            panic!(
                "incompatible nbatch in {}: self={}, other={}",
                $name, self_ncols, other_ncols
            );
        }
    }};
}

impl_nalgebra_sub_assign!(NalgebraVec<T>, NalgebraVec<T>);
impl_nalgebra_sub_assign!(NalgebraVec<T>, &NalgebraVec<T>);
impl_nalgebra_sub_assign!(NalgebraVec<T>, NalgebraVecRef<'_, T>);
impl_nalgebra_sub_assign!(NalgebraVec<T>, &NalgebraVecRef<'_, T>);

impl_nalgebra_sub_assign!(NalgebraVecMut<'_, T>, NalgebraVec<T>);
impl_nalgebra_sub_assign!(NalgebraVecMut<'_, T>, &NalgebraVec<T>);
impl_nalgebra_sub_assign!(NalgebraVecMut<'_, T>, NalgebraVecRef<'_, T>);
impl_nalgebra_sub_assign!(NalgebraVecMut<'_, T>, &NalgebraVecRef<'_, T>);

impl_nalgebra_add_assign!(NalgebraVec<T>, NalgebraVec<T>);
impl_nalgebra_add_assign!(NalgebraVec<T>, &NalgebraVec<T>);
impl_nalgebra_add_assign!(NalgebraVec<T>, NalgebraVecRef<'_, T>);
impl_nalgebra_add_assign!(NalgebraVec<T>, &NalgebraVecRef<'_, T>);

impl_nalgebra_add_assign!(NalgebraVecMut<'_, T>, NalgebraVec<T>);
impl_nalgebra_add_assign!(NalgebraVecMut<'_, T>, &NalgebraVec<T>);
impl_nalgebra_add_assign!(NalgebraVecMut<'_, T>, NalgebraVecRef<'_, T>);
impl_nalgebra_add_assign!(NalgebraVecMut<'_, T>, &NalgebraVecRef<'_, T>);

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
        nalgebra_squared_norm_body!(self, y, atol, rtol, self.data.nrows(), Self::T)
    }
}

impl<'a, T: NalgebraScalar> VectorViewMut<'a> for NalgebraVecMut<'a, T> {
    type Owned = NalgebraVec<T>;
    type View = NalgebraVecRef<'a, T>;
    type Index = NalgebraIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        nalgebra_copy_from_body!(self, other, "VectorViewMut::copy_from")
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        nalgebra_copy_from_body!(self, other, "VectorViewMut::copy_from_view")
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        for b in 0..nbatch {
            self.data[(index, b)] = value;
        }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        nalgebra_axpy_body!(self, x, alpha, beta, "VectorViewMut::axpy")
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
            return self.data.apply_norm(&LpNorm(k));
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
        nalgebra_squared_norm_body!(self, y, atol, rtol, self.len(), Self::T)
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
        nalgebra_copy_from_body!(self, other, "copy_from")
    }
    fn fill(&mut self, value: Self::T) {
        self.data.fill(value);
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        nalgebra_copy_from_body!(self, other, "copy_from_view")
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
        nalgebra_axpy_body!(self, x, alpha, beta, "axpy")
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        nalgebra_axpy_body!(self, x, alpha, beta, "axpy_v")
    }
    fn batched_axpy(&mut self, alpha: &[T], x: &Self, beta: T) {
        let self_ncols = self.data.ncols();
        let x_ncols = x.data.ncols();
        assert_eq!(
            alpha.len(),
            self_ncols,
            "batched_axpy: alpha.len() must equal self.nbatch()"
        );
        if self_ncols == x_ncols {
            for b in 0..self_ncols {
                self.data
                    .column_mut(b)
                    .axpy(alpha[b], &x.data.column(b), beta);
            }
        } else if x_ncols == 1 {
            let x_col = x.data.column(0);
            for b in 0..self_ncols {
                self.data.column_mut(b).axpy(alpha[b], &x_col, beta);
            }
        } else {
            panic!(
                "incompatible nbatch in batched_axpy: self={}, x={}",
                self_ncols, x_ncols
            );
        }
    }
    fn component_div_assign(&mut self, other: &Self) {
        nalgebra_component_op_body!(self, other, component_div_assign, "component_div_assign")
    }
    fn component_mul_assign(&mut self, other: &Self) {
        nalgebra_component_op_body!(self, other, component_mul_assign, "component_mul_assign")
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let nbatch = self.context.nbatch();
        let nstates = self.len();
        assert_eq!(nstates, g1.len(), "Vector lengths do not match");
        let self_slice = self.data.as_slice();
        let g1_slice = g1.data.as_slice();
        if nbatch == 1 {
            let mut max_frac = T::zero();
            let mut max_frac_index: i32 = -1;
            let mut found_root = false;
            for i in 0..nstates {
                let g0 = unsafe { *self_slice.get_unchecked(i) };
                let g1v = unsafe { *g1_slice.get_unchecked(i) };
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
