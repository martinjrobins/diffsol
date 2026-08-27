use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{Const, DMatrix, DVector, Dyn, LpNorm, MatrixView, MatrixViewMut};

use crate::context::{broadcast_batch, cold_call};
use crate::{Context, IndexType, NalgebraContext, NalgebraMat, NalgebraScalar, Scale, VectorHost};

use super::{DefaultDenseMatrix, Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraIndex {
    pub(crate) data: DVector<IndexType>,
    pub(crate) context: NalgebraContext,
}

/// A batched vector, stored as one column per batch.
///
/// Invariant: `data.ncols() == context.nbatch()` and `data.nrows() == len()`, so
/// `data.as_slice()` matches the CUDA layout `[batch0 states..., batch1 states..., ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVec<T: NalgebraScalar> {
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVecRef<'a, T: NalgebraScalar> {
    pub(crate) data: MatrixView<'a, T, Dyn, Dyn, Const<1>, Dyn>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraVecMut<'a, T: NalgebraScalar> {
    pub(crate) data: MatrixViewMut<'a, T, Dyn, Dyn, Const<1>, Dyn>,
    pub(crate) context: NalgebraContext,
}

/// Column feeding batch `batch` of a `nbatch`-batch destination, broadcasting a narrower
/// vector over contiguous groups of destination batches (see [`broadcast_batch`]).
macro_rules! batch_method {
    () => {
        #[inline]
        pub(crate) fn batch(&self, batch: usize, nbatch: usize) -> usize {
            broadcast_batch(batch, self.data.ncols(), nbatch)
        }
    };
}
impl<T: NalgebraScalar> NalgebraVec<T> {
    batch_method!();
}
impl<T: NalgebraScalar> NalgebraVecRef<'_, T> {
    batch_method!();
}

impl<T: NalgebraScalar> From<DVector<T>> for NalgebraVec<T> {
    fn from(data: DVector<T>) -> Self {
        Self {
            data: DMatrix::from_column_slice(data.len(), 1, data.as_slice()),
            context: NalgebraContext::default(),
        }
    }
}

impl<T: NalgebraScalar> DefaultDenseMatrix for NalgebraVec<T> {
    type M = NalgebraMat<T>;
}

macro_rules! common {
    ($t:ty, $inner:ty) => {
        impl<T: NalgebraScalar> VectorCommon for $t {
            type T = T;
            type C = NalgebraContext;
            type Inner = $inner;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common!(NalgebraVec<T>, DMatrix<T>);
macro_rules! common_ref {
    ($t:ty, $inner:ty) => {
        impl<'a, T: NalgebraScalar> VectorCommon for $t {
            type T = T;
            type C = NalgebraContext;
            type Inner = $inner;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common_ref!(
    NalgebraVecRef<'a, T>,
    MatrixView<'a, T, Dyn, Dyn, Const<1>, Dyn>
);
common_ref!(
    NalgebraVecMut<'a, T>,
    MatrixViewMut<'a, T, Dyn, Dyn, Const<1>, Dyn>
);

macro_rules! vec_binary {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $op:tt, $binary:tt) => {
        impl<T: NalgebraScalar> $trait<$rhs> for $lhs {
            type Output = NalgebraVec<T>;

            fn $method(self, rhs: $rhs) -> Self::Output {
                // neither operand is owned, so the result is allocated at the left-hand
                // side's batch count and `rhs` broadcasts into it
                self.context
                    .assert_broadcastable_into(rhs.context.nbatch(), stringify!($method));
                if self.data.ncols() == rhs.data.ncols() {
                    return NalgebraVec {
                        data: &self.data $binary &rhs.data,
                        context: self.context,
                    };
                }
                let nb = self.data.ncols();
                let mut data = DMatrix::zeros(self.data.nrows(), nb);
                for b in 0..nb {
                    let mut column = data.column_mut(b);
                    column.copy_from(&self.data.column(b));
                    column $op &rhs.data.column(rhs.batch(b, nb));
                }
                NalgebraVec {
                    data,
                    context: self.context,
                }
            }
        }
    };
}
macro_rules! vec_assign {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $op:tt) => {
        impl<T: NalgebraScalar> $trait<$rhs> for $lhs {
            fn $method(&mut self, rhs: $rhs) {
                self.context
                    .assert_broadcastable_into(rhs.context.nbatch(), stringify!($method));
                if self.data.ncols() == rhs.data.ncols() {
                    self.data $op &rhs.data;
                    return;
                }
                let nb = self.data.ncols();
                for b in 0..nb {
                    let mut column = self.data.column_mut(b);
                    column $op &rhs.data.column(rhs.batch(b, nb));
                }
            }
        }
    };
}

macro_rules! copy_from_data {
    ($self:ident, $other:ident, $method:literal) => {
        $self
            .context
            .assert_broadcastable_into($other.context.nbatch(), $method);
        assert_eq!(
            $self.data.nrows(),
            $other.data.nrows(),
            "copy_from row mismatch"
        );
        // the unbatched path is column-wise on purpose: nalgebra's whole-matrix copy_from is
        // ~30% slower than copying the single column (see lin_alg_ops copy_from/nalgebra)
        if $self.context.nbatch() == 1 && $other.context.nbatch() == 1 {
            $self.data.column_mut(0).copy_from(&$other.data.column(0));
            return;
        }
        if $self.data.ncols() == $other.data.ncols() {
            $self.data.copy_from(&$other.data);
            return;
        }
        let nb = $self.data.ncols();
        cold_call(|| {
            for b in 0..nb {
                $self
                    .data
                    .column_mut(b)
                    .copy_from(&$other.data.column($other.batch(b, nb)));
            }
        });
    };
}

/// `self_b = alpha_b * x_b + beta * self_b` for every batch of `self`, broadcasting a
/// single-batch `x`.  Shared by the owned vector and its mutable view, with `x` either an
/// owned vector or a view, and `alpha` either one scalar or one value per batch.
macro_rules! axpy_data {
    ($self:ident, $x:ident, $beta:expr, $op:literal, |$batch:ident| $alpha:expr) => {{
        $self
            .context
            .assert_broadcastable_into($x.context.nbatch(), $op);
        let nb = $self.data.ncols();
        // the unbatched path keeps constant column indices and no loop, which is worth
        // ~30% here (see lin_alg_ops axpy/nalgebra/100)
        if nb == 1 {
            let $batch = 0;
            let alpha = $alpha;
            $self
                .data
                .column_mut(0)
                .axpy(alpha, &$x.data.column(0), $beta);
            return;
        }
        cold_call(|| {
            for $batch in 0..nb {
                let alpha = $alpha;
                $self.data.column_mut($batch).axpy(
                    alpha,
                    &$x.data.column($x.batch($batch, nb)),
                    $beta,
                );
            }
        });
    }};
}

/// Weighted error norm of every batch, reduced by taking the maximum.
macro_rules! squared_norm_data {
    ($self:ident, $y:ident, $atol:ident, $rtol:ident) => {{
        let nrows = $self.data.nrows();
        assert!(
            nrows == $y.data.nrows() && nrows == $atol.data.nrows(),
            "squared_norm row mismatch"
        );
        let nstates = T::from_f64(nrows as f64).unwrap();
        // rows of a column are contiguous whatever the batch stride, so each batch is a slice
        let batch_norm = |xb: usize, yb: usize, atolb: usize| {
            let x = $self.data.column(xb);
            let y = $y.data.column(yb);
            let atol = $atol.data.column(atolb);
            let norm = x
                .as_slice()
                .iter()
                .zip(y.as_slice().iter().zip(atol.as_slice().iter()))
                .fold(T::zero(), |norm, (&x, (&y, &atol))| {
                    let term = x.algebraic_div(y.abs().algebraic_mul($rtol).algebraic_add(atol));
                    norm.algebraic_add(term.algebraic_mul(term))
                });
            norm / nstates
        };
        // the reduction runs over `self`'s batches and broadcasts `y` and `atol` over them
        let nb = $self.data.ncols();
        // the unbatched case keeps constant column indices, which codegens better than the
        // loop variable below (and folds away the broadcast arithmetic).  Checking the other
        // two operands here as well costs ~0.9ns of a 5.3ns call, so it stays a single test.
        if nb == 1 {
            return batch_norm(0, 0, 0);
        }
        $self
            .context
            .assert_broadcastable_into($y.context.nbatch(), "squared_norm");
        $self
            .context
            .assert_broadcastable_into($atol.context.nbatch(), "squared_norm");
        cold_call(|| {
            let mut max_norm = T::zero();
            for b in 0..nb {
                max_norm = max_norm.max(batch_norm(b, $y.batch(b, nb), $atol.batch(b, nb)));
            }
            max_norm
        })
    }};
}

/// `rhs` is the owned operand, so it is the destination.
///
/// A commutative op is just the in-place op with the operands swapped.  A non-commutative one
/// cannot be: `rhs -= lhs` computes `rhs - lhs`, so it writes `combine(&mut rhs_i, lhs_i)`
/// instead, which gets `lhs - rhs` in the same single pass.
macro_rules! vec_binary_owned_rhs {
    (commutes, $trait:ident, $method:ident, $assign_trait:ident, $assign:ident, $lhs:ty,
     $op:tt, $combine:expr) => {
        impl<T: NalgebraScalar> $trait<NalgebraVec<T>> for $lhs {
            type Output = NalgebraVec<T>;

            fn $method(self, mut rhs: NalgebraVec<T>) -> Self::Output {
                $assign_trait::$assign(&mut rhs, self);
                rhs
            }
        }
    };
    (noncommutes, $trait:ident, $method:ident, $assign_trait:ident, $assign:ident, $lhs:ty, $op:tt, $combine:expr) => {
        impl<T: NalgebraScalar> $trait<NalgebraVec<T>> for $lhs {
            type Output = NalgebraVec<T>;

            fn $method(self, mut rhs: NalgebraVec<T>) -> Self::Output {
                rhs.context
                    .assert_broadcastable_into(self.context.nbatch(), stringify!($method));
                if self.data.ncols() == rhs.data.ncols() {
                    rhs.data.zip_apply(&self.data, $combine);
                    return rhs;
                }
                let nb = rhs.data.ncols();
                for b in 0..nb {
                    let lhs = self.data.column(self.batch(b, nb));
                    rhs.data.column_mut(b).zip_apply(&lhs, $combine);
                }
                rhs
            }
        }
    };
}

/// `self` is the owned operand, so it is the destination -- which makes the in-place op the
/// whole implementation, broadcast check included.
macro_rules! vec_binary_owned_lhs {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign:ident, $rhs:ty) => {
        impl<T: NalgebraScalar> $trait<$rhs> for NalgebraVec<T> {
            type Output = NalgebraVec<T>;

            fn $method(mut self, rhs: $rhs) -> Self::Output {
                $assign_trait::$assign(&mut self, rhs);
                self
            }
        }
    };
}

macro_rules! binary_set {
    ($trait:ident,$method:ident,$assign_trait:ident,$assign:ident,$commutes:ident,
     $op:tt,$binary:tt,$combine:expr) => {
        vec_binary_owned_lhs!($trait, $method, $assign_trait, $assign, NalgebraVec<T>);
        vec_binary_owned_lhs!($trait, $method, $assign_trait, $assign, &NalgebraVec<T>);
        vec_binary_owned_lhs!(
            $trait,
            $method,
            $assign_trait,
            $assign,
            NalgebraVecRef<'_, T>
        );
        vec_binary_owned_lhs!(
            $trait,
            $method,
            $assign_trait,
            $assign,
            &NalgebraVecRef<'_, T>
        );
        vec_binary_owned_rhs!(
            $commutes,
            $trait,
            $method,
            $assign_trait,
            $assign,
            NalgebraVecRef<'_, T>,
            $op,
            $combine
        );
        vec_binary_owned_rhs!(
            $commutes,
            $trait,
            $method,
            $assign_trait,
            $assign,
            &NalgebraVec<T>,
            $op,
            $combine
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            &NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            &NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            &NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            &NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
    };
}
binary_set!(Add, add, AddAssign, add_assign, commutes, +=, +, |rhs, lhs| *rhs += lhs);
binary_set!(Sub, sub, SubAssign, sub_assign, noncommutes, -=, -, |rhs, lhs| *rhs = lhs - *rhs);
macro_rules! assign_set {
    ($trait:ident,$method:ident,$op:tt) => {
        vec_assign!($trait, $method, NalgebraVec<T>, NalgebraVec<T>, $op);
        vec_assign!($trait, $method, NalgebraVec<T>, &NalgebraVec<T>, $op);
        vec_assign!($trait, $method, NalgebraVec<T>, NalgebraVecRef<'_, T>, $op);
        vec_assign!($trait, $method, NalgebraVec<T>, &NalgebraVecRef<'_, T>, $op);
        vec_assign!($trait, $method, NalgebraVecMut<'_, T>, NalgebraVec<T>, $op);
        vec_assign!($trait, $method, NalgebraVecMut<'_, T>, &NalgebraVec<T>, $op);
        vec_assign!(
            $trait,
            $method,
            NalgebraVecMut<'_, T>,
            NalgebraVecRef<'_, T>,
            $op
        );
        vec_assign!(
            $trait,
            $method,
            NalgebraVecMut<'_, T>,
            &NalgebraVecRef<'_, T>,
            $op
        );
    };
}
assign_set!(AddAssign, add_assign, +=);
assign_set!(SubAssign, sub_assign, -=);

impl<T: NalgebraScalar> Mul<Scale<T>> for NalgebraVec<T> {
    type Output = Self;
    fn mul(mut self, rhs: Scale<T>) -> Self {
        self.data *= rhs.value();
        self
    }
}
macro_rules! scale_ref {
    ($t:ty) => {
        impl<T: NalgebraScalar> Mul<Scale<T>> for $t {
            type Output = NalgebraVec<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                NalgebraVec {
                    data: self.data.clone_owned() * rhs.value(),
                    context: self.context,
                }
            }
        }
    };
}
scale_ref!(&NalgebraVec<T>);
scale_ref!(NalgebraVecRef<'_, T>);
impl<T: NalgebraScalar> Mul<Scale<T>> for NalgebraVecMut<'_, T> {
    type Output = NalgebraVec<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        NalgebraVec {
            data: self.data.into_owned() * rhs.value(),
            context: self.context,
        }
    }
}
impl<T: NalgebraScalar> Div<Scale<T>> for NalgebraVec<T> {
    type Output = Self;
    fn div(self, rhs: Scale<T>) -> Self {
        Self {
            data: self.data * (T::one() / rhs.value()),
            context: self.context,
        }
    }
}
impl<T: NalgebraScalar> MulAssign<Scale<T>> for NalgebraVec<T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        self.data *= rhs.value();
    }
}
impl<T: NalgebraScalar> MulAssign<Scale<T>> for NalgebraVecMut<'_, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        self.data *= rhs.value();
    }
}
macro_rules! index {
    ($t:ty) => {
        impl<T: NalgebraScalar> Index<IndexType> for $t {
            type Output = T;
            fn index(&self, i: IndexType) -> &T {
                &self.data[(i, 0)]
            }
        }
    };
}
index!(NalgebraVec<T>);
index!(NalgebraVecRef<'_, T>);
impl<T: NalgebraScalar> IndexMut<IndexType> for NalgebraVec<T> {
    fn index_mut(&mut self, i: IndexType) -> &mut T {
        &mut self.data[(i, 0)]
    }
}

impl VectorIndex for NalgebraIndex {
    type C = NalgebraContext;
    fn zeros(len: IndexType, ctx: Self::C) -> Self {
        Self {
            data: DVector::zeros(len),
            context: ctx,
        }
    }
    fn len(&self) -> IndexType {
        self.data.len()
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        Self {
            data: DVector::from_vec(v),
            context: ctx,
        }
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
    fn get_index(&self, i: IndexType) -> T {
        assert_eq!(self.context.nbatch(), 1, "get_index requires nbatch == 1");
        self.data[(i, 0)]
    }
    fn into_owned(self) -> Self::Owned {
        NalgebraVec {
            data: self.data.into_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: T) -> T {
        squared_norm_data!(self, y, atol, rtol)
    }
}
impl<'a, T: NalgebraScalar> VectorViewMut<'a> for NalgebraVecMut<'a, T> {
    type Owned = NalgebraVec<T>;
    type View = NalgebraVecRef<'a, T>;
    type Index = NalgebraIndex;
    fn copy_from(&mut self, o: &Self::Owned) {
        copy_from_data!(self, o, "copy_from");
    }
    fn copy_from_view(&mut self, o: &Self::View) {
        copy_from_data!(self, o, "copy_from_view");
    }
    fn set_index(&mut self, i: IndexType, v: T) {
        self.data.row_mut(i).fill(v);
    }
    fn axpy(&mut self, a: T, x: &Self::Owned, beta: T) {
        axpy_data!(self, x, beta, "axpy", |_batch| a)
    }
}
impl<T: NalgebraScalar> VectorHost for NalgebraVec<T> {
    fn as_slice(&self) -> &[T] {
        assert_eq!(self.context.nbatch(), 1);
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        assert_eq!(self.context.nbatch(), 1);
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
    fn norm(&self, k: i32) -> T {
        // LpNorm is a generic elementwise loop, `norm` is the optimised L2 path
        (0..self.data.ncols())
            .map(|b| {
                let column = self.data.column(b);
                if k == 2 {
                    column.norm()
                } else {
                    column.apply_norm(&LpNorm(k))
                }
            })
            .fold(T::zero(), |a, b| a.max(b))
    }
    fn get_index(&self, i: IndexType) -> T {
        assert_eq!(
            self.context.nbatch(),
            1,
            "get_index not supported for batched vectors"
        );
        self.data[(i, 0)]
    }
    fn set_index(&mut self, i: IndexType, v: T) {
        self.data.row_mut(i).fill(v);
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: T) -> T {
        squared_norm_data!(self, y, atol, rtol)
    }
    fn as_view(&self) -> Self::View<'_> {
        NalgebraVecRef {
            data: self.data.as_view(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        NalgebraVecMut {
            data: self.data.as_view_mut(),
            context: self.context,
        }
    }
    fn get_batch(&self, b: usize) -> Self::View<'_> {
        NalgebraVecRef {
            data: self.data.columns(b, 1),
            context: NalgebraContext::default(),
        }
    }
    fn get_batch_mut(&mut self, b: usize) -> Self::ViewMut<'_> {
        NalgebraVecMut {
            data: self.data.columns_mut(b, 1),
            context: NalgebraContext::default(),
        }
    }
    fn copy_from(&mut self, o: &Self) {
        copy_from_data!(self, o, "copy_from");
    }
    fn fill(&mut self, v: T) {
        self.data.fill(v)
    }
    fn copy_from_view(&mut self, o: &Self::View<'_>) {
        copy_from_data!(self, o, "copy_from_view");
    }
    fn from_element(n: usize, v: T, ctx: Self::C) -> Self {
        Self {
            data: DMatrix::from_element(n, ctx.nbatch(), v),
            context: ctx,
        }
    }
    fn from_vec(v: Vec<T>, ctx: Self::C) -> Self {
        assert!(
            v.len().is_multiple_of(ctx.nbatch()),
            "vector length must be divisible by nbatch"
        );
        Self {
            data: DMatrix::from_vec(v.len() / ctx.nbatch(), ctx.nbatch(), v),
            context: ctx,
        }
    }
    fn from_slice(v: &[T], ctx: Self::C) -> Self {
        Self::from_vec(v.to_vec(), ctx)
    }
    fn clone_as_vec(&self) -> Vec<T> {
        self.data.iter().copied().collect()
    }
    fn zeros(n: usize, ctx: Self::C) -> Self {
        Self {
            data: DMatrix::zeros(n, ctx.nbatch()),
            context: ctx,
        }
    }
    fn axpy(&mut self, a: T, x: &Self, beta: T) {
        axpy_data!(self, x, beta, "axpy", |_batch| a)
    }
    fn axpy_v(&mut self, a: T, x: &Self::View<'_>, beta: T) {
        axpy_data!(self, x, beta, "axpy_v", |_batch| a)
    }
    fn batched_axpy(&mut self, a: &[T], x: &Self, beta: T) {
        assert_eq!(
            a.len(),
            self.context.nbatch(),
            "alpha.len() must equal nbatch"
        );
        axpy_data!(self, x, beta, "batched_axpy", |batch| a[batch])
    }
    fn component_div_assign(&mut self, o: &Self) {
        self.context
            .assert_broadcastable_into(o.context.nbatch(), "component_div_assign");
        assert_eq!(
            self.data.nrows(),
            o.data.nrows(),
            "component_div_assign row mismatch"
        );
        if self.data.ncols() == o.data.ncols() {
            self.data.component_div_assign(&o.data);
            return;
        }
        let nb = self.data.ncols();
        for c in 0..nb {
            self.data
                .column_mut(c)
                .component_div_assign(&o.data.column(o.batch(c, nb)));
        }
    }
    fn component_mul_assign(&mut self, o: &Self) {
        self.context
            .assert_broadcastable_into(o.context.nbatch(), "component_mul_assign");
        assert_eq!(
            self.data.nrows(),
            o.data.nrows(),
            "component_mul_assign row mismatch"
        );
        // the elementwise unbatched path beats nalgebra's component_mul_assign by ~25%
        // (see lin_alg_ops component_mul_assign/nalgebra)
        if self.context.nbatch() == 1 {
            for i in 0..self.data.nrows() {
                let lhs = unsafe { self.data.get_unchecked_mut((i, 0)) };
                let rhs = unsafe { *o.data.get_unchecked((i, 0)) };
                *lhs *= rhs;
            }
            return;
        }
        if self.data.ncols() == o.data.ncols() {
            self.data.component_mul_assign(&o.data);
            return;
        }
        let nb = self.data.ncols();
        for c in 0..nb {
            self.data
                .column_mut(c)
                .component_mul_assign(&o.data.column(o.batch(c, nb)));
        }
    }
    fn root_finding(&self, g1: &Self) -> (bool, T, i32) {
        self.context
            .assert_broadcastable_into(g1.context.nbatch(), "root_finding");
        assert_eq!(self.len(), g1.len(), "Vector lengths do not match");
        let mut out = None;
        // the scan runs over `self`'s batches and broadcasts `g1` over them
        let nb = self.data.ncols();
        for b in 0..nb {
            let mut found = false;
            let mut frac = T::zero();
            let mut idx = -1;
            let g0_column = self.data.column(b);
            let g1_column = g1.data.column(g1.batch(b, nb));
            for (i, (&g0, &g)) in g0_column
                .as_slice()
                .iter()
                .zip(g1_column.as_slice())
                .enumerate()
            {
                if g == T::zero() {
                    found = true
                }
                if g0 * g < T::zero() {
                    let q = (g / (g - g0)).abs();
                    if q > frac {
                        frac = q;
                        idx = i as i32
                    }
                }
            }
            if let Some(x) = out {
                assert_eq!(
                    x,
                    (found, frac, idx),
                    "root finding results differ across batches"
                )
            }
            out = Some((found, frac, idx));
        }
        out.unwrap_or((false, T::zero(), -1))
    }
    fn assign_at_indices(&mut self, idx: &Self::Index, v: T) {
        for b in 0..self.data.ncols() {
            for i in idx.data.iter() {
                self.data[(*i, b)] = v
            }
        }
    }
    fn copy_from_indices(&mut self, o: &Self, idx: &Self::Index) {
        self.context
            .assert_broadcastable_into(o.context.nbatch(), "copy_from_indices");
        let nb = self.data.ncols();
        for b in 0..nb {
            for i in idx.data.iter() {
                self.data[(*i, b)] = o.data[(*i, o.batch(b, nb))]
            }
        }
    }
    fn gather(&mut self, o: &Self, idx: &Self::Index) {
        assert_eq!(self.len(), idx.len());
        self.context
            .assert_broadcastable_into(o.context.nbatch(), "gather");
        let nb = self.data.ncols();
        for b in 0..nb {
            for (i, j) in idx.data.iter().enumerate() {
                self.data[(i, b)] = o.data[(*j, o.batch(b, nb))]
            }
        }
    }
    fn scatter(&self, idx: &Self::Index, o: &mut Self) {
        assert_eq!(self.len(), idx.len());
        // `o` is the destination here, so its batch count governs the loop
        o.context
            .assert_broadcastable_into(self.data.ncols(), "scatter");
        let nb = o.data.ncols();
        for b in 0..nb {
            let src = broadcast_batch(b, self.data.ncols(), nb);
            for (i, j) in idx.data.iter().enumerate() {
                o.data[(*j, b)] = self.data[(i, src)]
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
        let errorn_check = r.data.column(0).norm_squared() / 3.0;
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
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v: NalgebraVec<f64> = vector.into();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_host_only() {
        super::super::tests::test_host_only::<NalgebraVec<f64>>();
    }

    super::super::generate_vector_tests_nonbatched!(nalgebra, NalgebraVec<f64>);
    super::super::generate_vector_tests_batched!(
        nalgebra,
        NalgebraVec<f64>,
        NalgebraContext::with_nbatch(2),
        NalgebraContext::with_nbatch(3)
    );
}
