use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{unzip, zip, Col, Mat, MatMut, MatRef};

use crate::{scalar::Scale, Context, FaerContext, FaerScalar, IndexType, Vector, VectorHost};

use crate::{FaerMat, VectorCommon, VectorIndex, VectorView, VectorViewMut};

use super::DefaultDenseMatrix;

/// A batched vector, stored as one column per batch.
///
/// Invariant: `data.ncols() == context.nbatch()` and `data.nrows() == len()`.  Note that
/// faer pads its column stride for alignment, so the batches are *not* contiguous with each
/// other; only a single column is.
#[derive(Debug, Clone, PartialEq)]
pub struct FaerVec<T: FaerScalar> {
    pub(crate) data: Mat<T>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecIndex {
    pub(crate) data: Vec<IndexType>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaerVecRef<'a, T: FaerScalar> {
    pub(crate) data: MatRef<'a, T>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, PartialEq)]
pub struct FaerVecMut<'a, T: FaerScalar> {
    pub(crate) data: MatMut<'a, T>,
    pub(crate) context: FaerContext,
}

/// Column holding `batch`, broadcasting a single-batch vector over all batches.
macro_rules! batch_method {
    () => {
        #[inline]
        pub(crate) fn batch(&self, batch: usize) -> usize {
            batch % self.data.ncols()
        }
    };
}
impl<T: FaerScalar> FaerVec<T> {
    batch_method!();

    pub fn check_for_nan(&self, label: &str) -> bool {
        for b in 0..self.data.ncols() {
            let column = self.data.rb().col(b);
            for i in 0..column.nrows() {
                if unsafe { *column.get_unchecked(i) }.is_nan() {
                    eprintln!("{}: NaN at index {} of batch {}", label, i, b);
                    return true;
                }
            }
        }
        false
    }
}
impl<T: FaerScalar> FaerVecRef<'_, T> {
    batch_method!();
}

impl<T: FaerScalar> From<Col<T>> for FaerVec<T> {
    fn from(data: Col<T>) -> Self {
        Self {
            data: Mat::from_fn(data.nrows(), 1, |i, _| data[i]),
            context: FaerContext::default(),
        }
    }
}

impl<T: FaerScalar> DefaultDenseMatrix for FaerVec<T> {
    type M = FaerMat<T>;
}

macro_rules! common {
    ($t:ty, $inner:ty) => {
        impl<T: FaerScalar> VectorCommon for $t {
            type T = T;
            type C = FaerContext;
            type Inner = $inner;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common!(FaerVec<T>, Mat<T>);
macro_rules! common_ref {
    ($t:ty, $inner:ty) => {
        impl<'a, T: FaerScalar> VectorCommon for $t {
            type T = T;
            type C = FaerContext;
            type Inner = $inner;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common_ref!(FaerVecRef<'a, T>, MatRef<'a, T>);
common_ref!(FaerVecMut<'a, T>, MatMut<'a, T>);

macro_rules! vec_binary {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $op:tt, $binary:tt) => {
        impl<T: FaerScalar> $trait<$rhs> for $lhs {
            type Output = FaerVec<T>;

            // the `%` below is broadcast indexing over batches, not arithmetic on the
            // operands, which is what the lint is looking for
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn $method(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                // the unbatched path goes through the columns on purpose: faer's whole-matrix
                // operators are ~45% slower for a single column (see lin_alg_ops
                // add_ref_ref/faer)
                if self.data.ncols() == 1 && rhs.data.ncols() == 1 {
                    let mut data = Mat::zeros(self.data.nrows(), 1);
                    zip!(
                        data.rb_mut().col_mut(0),
                        self.data.rb().col(0),
                        rhs.data.rb().col(0)
                    )
                    .for_each(|unzip!(o, l, r)| *o = *l $binary *r);
                    return FaerVec {
                        data,
                        context: self.context,
                    };
                }
                if self.data.ncols() == rhs.data.ncols() {
                    return FaerVec {
                        data: self.data.rb() $binary rhs.data.rb(),
                        context: self.context,
                    };
                }
                let nb = self.data.ncols().max(rhs.data.ncols());
                let mut data = Mat::zeros(self.data.nrows(), nb);
                for b in 0..nb {
                    let mut column = data.rb_mut().col_mut(b);
                    column.copy_from(self.data.rb().col(b % self.data.ncols()));
                    column $op rhs.data.rb().col(b % rhs.data.ncols());
                }
                FaerVec {
                    data,
                    context: if self.data.ncols() == nb {
                        self.context
                    } else {
                        rhs.context
                    },
                }
            }
        }
    };
}
macro_rules! vec_assign {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $op:tt) => {
        impl<T: FaerScalar> $trait<$rhs> for $lhs {
            // the `%` below is broadcast indexing over batches, not arithmetic on the
            // operands, which is what the lint is looking for
            #[allow(clippy::suspicious_op_assign_impl)]
            fn $method(&mut self, rhs: $rhs) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                // the whole-matrix operator only pays off past one column: faer's 2-D
                // iteration costs ~1.5x the single-column path on short vectors (see
                // lin_alg_ops add_assign/faer/2)
                if self.data.ncols() == rhs.data.ncols() && self.data.ncols() > 1 {
                    self.data $op rhs.data.rb();
                    return;
                }
                for b in 0..self.data.ncols() {
                    let mut column = self.data.rb_mut().col_mut(b);
                    column $op rhs.data.rb().col(b % rhs.data.ncols());
                }
            }
        }
    };
}

macro_rules! copy_from_data {
    ($self:ident, $other:ident, $method:literal) => {
        $self
            .context
            .assert_compatible_nbatch($other.context.nbatch(), $method);
        assert_eq!(
            $self.data.nrows(),
            $other.data.nrows(),
            "copy_from row mismatch"
        );
        // the unbatched path is column-wise on purpose: faer's whole-matrix copy_from is
        // ~25% slower than copying the single column (see lin_alg_ops copy_from/faer)
        if $self.data.ncols() == 1 && $other.data.ncols() == 1 {
            $self
                .data
                .rb_mut()
                .col_mut(0)
                .copy_from($other.data.rb().col(0));
            return;
        }
        if $self.data.ncols() == $other.data.ncols() {
            $self.data.rb_mut().copy_from($other.data.rb());
            return;
        }
        for b in 0..$self.data.ncols() {
            $self
                .data
                .rb_mut()
                .col_mut(b)
                .copy_from($other.data.rb().col(b % $other.data.ncols()));
        }
    };
}

/// `self_b = alpha_b * x_b + beta * self_b` for every batch of `self`, broadcasting a
/// single-batch `x`.  Shared by the owned vector and its mutable view, with `x` either an
/// owned vector or a view, and `alpha` either one scalar or one value per batch.
macro_rules! axpy_data {
    ($self:ident, $x:ident, $beta:expr, $op:literal, |$batch:ident| $alpha:expr) => {{
        $self
            .context
            .assert_compatible_nbatch($x.context.nbatch(), $op);
        for $batch in 0..$self.data.ncols() {
            let alpha = $alpha;
            zip!(
                $self.data.rb_mut().col_mut($batch),
                $x.data.rb().col($x.batch($batch))
            )
            .for_each(|unzip!(s, xi)| *s = *s * $beta + *xi * alpha);
        }
    }};
}

/// Weighted error norm of every batch, reduced by taking the maximum.
macro_rules! squared_norm_data {
    ($self:ident, $y:ident, $atol:ident, $rtol:ident) => {{
        $self
            .context
            .assert_compatible_nbatch($y.context.nbatch(), "squared_norm");
        $self
            .context
            .assert_compatible_nbatch($atol.context.nbatch(), "squared_norm");
        let nrows = $self.data.nrows();
        assert!(
            nrows == $y.data.nrows() && nrows == $atol.data.nrows(),
            "squared_norm row mismatch"
        );
        let nstates = T::from_f64(nrows as f64).unwrap();
        // rows of a column are contiguous whatever the batch stride, so each batch is a slice
        let batch_norm = |xb: usize, yb: usize, atolb: usize| {
            let x = $self
                .data
                .rb()
                .col(xb)
                .try_as_col_major()
                .unwrap()
                .as_slice();
            let y = $y.data.rb().col(yb).try_as_col_major().unwrap().as_slice();
            let atol = $atol
                .data
                .rb()
                .col(atolb)
                .try_as_col_major()
                .unwrap()
                .as_slice();
            let norm = x.iter().zip(y.iter().zip(atol.iter())).fold(
                T::zero(),
                |norm, (&x, (&y, &atol))| {
                    let term = x.algebraic_div(y.abs().algebraic_mul($rtol).algebraic_add(atol));
                    norm.algebraic_add(term.algebraic_mul(term))
                },
            );
            norm / nstates
        };
        // the unbatched case keeps a constant column index, which codegens better than the
        // loop variable below (and folds away `batch`'s modulus)
        if $self.data.ncols() == 1 {
            return batch_norm(0, $y.batch(0), $atol.batch(0));
        }
        let mut max_norm = T::zero();
        for b in 0..$self.data.ncols() {
            max_norm = max_norm.max(batch_norm(b, $y.batch(b), $atol.batch(b)));
        }
        max_norm
    }};
}

/// `rhs` is owned, so the result is written into its allocation whenever `rhs` already holds
/// as many batches as the result: `combine(&mut rhs_i, lhs_i)` writes the result in place.
macro_rules! vec_binary_owned_rhs {
    ($trait:ident, $method:ident, $lhs:ty, $op:tt, $combine:expr) => {
        impl<T: FaerScalar> $trait<FaerVec<T>> for $lhs {
            type Output = FaerVec<T>;

            fn $method(self, mut rhs: FaerVec<T>) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                if self.data.ncols() == rhs.data.ncols() {
                    zip!(rhs.data.rb_mut(), self.data.rb())
                        .for_each(|unzip!(r, l)| $combine(r, *l));
                    return rhs;
                }
                if rhs.data.ncols() > self.data.ncols() {
                    for b in 0..rhs.data.ncols() {
                        let lhs = self.data.rb().col(self.batch(b));
                        zip!(rhs.data.rb_mut().col_mut(b), lhs)
                            .for_each(|unzip!(r, l)| $combine(r, *l));
                    }
                    return rhs;
                }
                // rhs holds fewer batches than the result, so it cannot be written into
                let nb = self.data.ncols();
                let mut data = Mat::zeros(self.data.nrows(), nb);
                for b in 0..nb {
                    let mut column = data.rb_mut().col_mut(b);
                    column.copy_from(self.data.rb().col(b));
                    column $op rhs.data.rb().col(rhs.batch(b));
                }
                FaerVec {
                    data,
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! vec_binary_owned_lhs {
    ($trait:ident, $method:ident, $rhs:ty, $op:tt) => {
        impl<T: FaerScalar> $trait<$rhs> for FaerVec<T> {
            type Output = FaerVec<T>;

            // the `%` below is broadcast indexing over batches, not arithmetic on the
            // operands, which is what the lint is looking for
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn $method(mut self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                // matching batch counts need no broadcast, so the whole matrix goes at once
                // (past one column, where faer's 2-D iteration costs more than it saves)
                if self.data.ncols() == rhs.data.ncols() && self.data.ncols() > 1 {
                    self.data $op rhs.data.rb();
                    return self;
                }
                if self.data.ncols() >= rhs.data.ncols() {
                    for b in 0..self.data.ncols() {
                        let mut column = self.data.rb_mut().col_mut(b);
                        column $op rhs.data.rb().col(b % rhs.data.ncols());
                    }
                    return self;
                }
                let nb = rhs.data.ncols();
                let mut data = Mat::zeros(self.data.nrows(), nb);
                for b in 0..nb {
                    let mut column = data.rb_mut().col_mut(b);
                    column.copy_from(self.data.rb().col(b % self.data.ncols()));
                    column $op rhs.data.rb().col(b);
                }
                FaerVec {
                    data,
                    context: rhs.context,
                }
            }
        }
    };
}
macro_rules! binary_set {
    ($trait:ident,$method:ident,$op:tt,$binary:tt,$combine:expr) => {
        vec_binary_owned_lhs!($trait, $method, FaerVec<T>, $op);
        vec_binary_owned_lhs!($trait, $method, &FaerVec<T>, $op);
        vec_binary_owned_lhs!($trait, $method, FaerVecRef<'_, T>, $op);
        vec_binary_owned_lhs!($trait, $method, &FaerVecRef<'_, T>, $op);
        vec_binary_owned_rhs!($trait, $method, FaerVecRef<'_, T>, $op, $combine);
        vec_binary_owned_rhs!($trait, $method, &FaerVec<T>, $op, $combine);
        vec_binary!(
            $trait,
            $method,
            FaerVecRef<'_, T>,
            &FaerVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            FaerVecRef<'_, T>,
            FaerVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            FaerVecRef<'_, T>,
            &FaerVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!($trait, $method, &FaerVec<T>, &FaerVec<T>, $op, $binary);
        vec_binary!(
            $trait,
            $method,
            &FaerVec<T>,
            FaerVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &FaerVec<T>,
            &FaerVecRef<'_, T>,
            $op,
            $binary
        );
    };
}
binary_set!(Add, add, +=, +, |rhs: &mut T, lhs: T| *rhs += lhs);
binary_set!(Sub, sub, -=, -, |rhs: &mut T, lhs: T| *rhs = lhs - *rhs);
macro_rules! assign_set {
    ($trait:ident,$method:ident,$op:tt) => {
        vec_assign!($trait, $method, FaerVec<T>, FaerVec<T>, $op);
        vec_assign!($trait, $method, FaerVec<T>, &FaerVec<T>, $op);
        vec_assign!($trait, $method, FaerVec<T>, FaerVecRef<'_, T>, $op);
        vec_assign!($trait, $method, FaerVec<T>, &FaerVecRef<'_, T>, $op);
        vec_assign!($trait, $method, FaerVecMut<'_, T>, FaerVec<T>, $op);
        vec_assign!($trait, $method, FaerVecMut<'_, T>, &FaerVec<T>, $op);
        vec_assign!($trait, $method, FaerVecMut<'_, T>, FaerVecRef<'_, T>, $op);
        vec_assign!($trait, $method, FaerVecMut<'_, T>, &FaerVecRef<'_, T>, $op);
    };
}
assign_set!(AddAssign, add_assign, +=);
assign_set!(SubAssign, sub_assign, -=);

impl<T: FaerScalar> Mul<Scale<T>> for FaerVec<T> {
    type Output = Self;
    fn mul(mut self, rhs: Scale<T>) -> Self {
        self.data *= faer::Scale(rhs.value());
        self
    }
}
macro_rules! scale_ref {
    ($t:ty) => {
        impl<T: FaerScalar> Mul<Scale<T>> for $t {
            type Output = FaerVec<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                // ponytail: a scaled copy of an unbatched vector is ~1.5x the old
                // `Col`-of-batches cost (lin_alg_ops scalar_mul/faer).  faer's 2-D
                // iteration machinery looks like the floor: an unbatched fast path over
                // the single column measured no better with `Mat::zeros` + `zip!` and
                // ~1.8x worse with `Mat::from_fn`.  Revisit if faer grows a way to fill a
                // fresh `Mat` in one vectorised pass.
                FaerVec {
                    data: self.data.rb() * faer::Scale(rhs.value()),
                    context: self.context,
                }
            }
        }
    };
}
scale_ref!(&FaerVec<T>);
scale_ref!(FaerVecRef<'_, T>);
scale_ref!(FaerVecMut<'_, T>);
impl<T: FaerScalar> Div<Scale<T>> for FaerVec<T> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(mut self, rhs: Scale<T>) -> Self {
        // self is owned, so scale it in place rather than allocating a result
        self.data *= faer::Scale(T::one() / rhs.value());
        self
    }
}
macro_rules! mul_assign_scale {
    ($t:ty) => {
        impl<T: FaerScalar> MulAssign<Scale<T>> for $t {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                if self.data.ncols() == 1 {
                    let mut column = self.data.rb_mut().col_mut(0);
                    column *= faer::Scale(rhs.value());
                    return;
                }
                self.data *= faer::Scale(rhs.value());
            }
        }
    };
}
mul_assign_scale!(FaerVec<T>);
mul_assign_scale!(FaerVecMut<'_, T>);
macro_rules! index {
    ($t:ty) => {
        impl<T: FaerScalar> Index<IndexType> for $t {
            type Output = T;
            fn index(&self, i: IndexType) -> &T {
                &self.data[(i, 0)]
            }
        }
    };
}
index!(FaerVec<T>);
index!(FaerVecRef<'_, T>);
impl<T: FaerScalar> IndexMut<IndexType> for FaerVec<T> {
    fn index_mut(&mut self, i: IndexType) -> &mut T {
        &mut self.data[(i, 0)]
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
        self.data.len()
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
    fn get_index(&self, i: IndexType) -> T {
        assert_eq!(self.context.nbatch(), 1, "get_index requires nbatch == 1");
        self.data[(i, 0)]
    }
    fn into_owned(self) -> Self::Owned {
        FaerVec {
            data: self.data.to_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: T) -> T {
        squared_norm_data!(self, y, atol, rtol)
    }
}
impl<'a, T: FaerScalar> VectorViewMut<'a> for FaerVecMut<'a, T> {
    type Owned = FaerVec<T>;
    type View = FaerVecRef<'a, T>;
    type Index = FaerVecIndex;
    fn copy_from(&mut self, o: &Self::Owned) {
        copy_from_data!(self, o, "copy_from");
    }
    fn copy_from_view(&mut self, o: &Self::View) {
        copy_from_data!(self, o, "copy_from_view");
    }
    fn set_index(&mut self, i: IndexType, v: T) {
        if self.data.ncols() == 1 {
            self.data[(i, 0)] = v;
            return;
        }
        self.data.rb_mut().row_mut(i).fill(v);
    }
    fn axpy(&mut self, a: T, x: &Self::Owned, beta: T) {
        axpy_data!(self, x, beta, "axpy", |_batch| a)
    }
}
impl<T: FaerScalar> VectorHost for FaerVec<T> {
    fn as_slice(&self) -> &[T] {
        assert_eq!(self.context.nbatch(), 1);
        self.data.col_as_slice(0)
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        assert_eq!(self.context.nbatch(), 1);
        self.data.col_as_slice_mut(0)
    }
}
impl<T: FaerScalar> Vector for FaerVec<T> {
    type View<'a> = FaerVecRef<'a, T>;
    type ViewMut<'a> = FaerVecMut<'a, T>;
    type Index = FaerVecIndex;
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
        (0..self.data.ncols())
            .map(|b| {
                let column = self.data.rb().col(b);
                match k {
                    1 => column.norm_l1(),
                    2 => column.norm_l2(),
                    _ => column
                        .iter()
                        .fold(T::zero(), |acc, x| acc.algebraic_add(x.abs().pow(k)))
                        .pow(T::one() / T::from_f64(k as f64).unwrap()),
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
        if self.data.ncols() == 1 {
            self.data[(i, 0)] = v;
            return;
        }
        self.data.rb_mut().row_mut(i).fill(v);
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: T) -> T {
        squared_norm_data!(self, y, atol, rtol)
    }
    fn as_view(&self) -> Self::View<'_> {
        FaerVecRef {
            data: self.data.rb(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        FaerVecMut {
            data: self.data.rb_mut(),
            context: self.context,
        }
    }
    fn get_batch(&self, b: usize) -> Self::View<'_> {
        FaerVecRef {
            data: self.data.rb().subcols(b, 1),
            context: FaerContext::default(),
        }
    }
    fn get_batch_mut(&mut self, b: usize) -> Self::ViewMut<'_> {
        FaerVecMut {
            data: self.data.rb_mut().subcols_mut(b, 1),
            context: FaerContext::default(),
        }
    }
    fn copy_from(&mut self, o: &Self) {
        copy_from_data!(self, o, "copy_from");
    }
    fn fill(&mut self, v: T) {
        if self.data.ncols() == 1 {
            self.data.rb_mut().col_mut(0).fill(v);
            return;
        }
        self.data.rb_mut().fill(v)
    }
    fn copy_from_view(&mut self, o: &Self::View<'_>) {
        copy_from_data!(self, o, "copy_from_view");
    }
    fn from_element(n: usize, v: T, ctx: Self::C) -> Self {
        Self {
            data: Mat::from_fn(n, ctx.nbatch(), |_, _| v),
            context: ctx,
        }
    }
    fn from_vec(v: Vec<T>, ctx: Self::C) -> Self {
        Self::from_slice(v.as_slice(), ctx)
    }
    fn from_slice(v: &[T], ctx: Self::C) -> Self {
        assert!(
            v.len().is_multiple_of(ctx.nbatch()),
            "vector length must be divisible by nbatch"
        );
        let n = v.len() / ctx.nbatch();
        Self {
            data: Mat::from_fn(n, ctx.nbatch(), |i, b| v[b * n + i]),
            context: ctx,
        }
    }
    fn clone_as_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.data.nrows() * self.data.ncols());
        for b in 0..self.data.ncols() {
            out.extend(self.data.rb().col(b).iter().copied());
        }
        out
    }
    fn zeros(n: usize, ctx: Self::C) -> Self {
        Self {
            data: Mat::zeros(n, ctx.nbatch()),
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
            .assert_compatible_nbatch(o.context.nbatch(), "component_div_assign");
        assert_eq!(
            self.data.nrows(),
            o.data.nrows(),
            "component_div_assign row mismatch"
        );
        if self.data.ncols() == o.data.ncols() {
            zip!(self.data.rb_mut(), o.data.rb()).for_each(|unzip!(s, o)| *s /= *o);
            return;
        }
        for c in 0..self.data.ncols() {
            zip!(self.data.rb_mut().col_mut(c), o.data.rb().col(o.batch(c)))
                .for_each(|unzip!(s, o)| *s /= *o);
        }
    }
    fn component_mul_assign(&mut self, o: &Self) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "component_mul_assign");
        assert_eq!(
            self.data.nrows(),
            o.data.nrows(),
            "component_mul_assign row mismatch"
        );
        if self.data.ncols() == o.data.ncols() {
            zip!(self.data.rb_mut(), o.data.rb()).for_each(|unzip!(s, o)| *s *= *o);
            return;
        }
        for c in 0..self.data.ncols() {
            zip!(self.data.rb_mut().col_mut(c), o.data.rb().col(o.batch(c)))
                .for_each(|unzip!(s, o)| *s *= *o);
        }
    }
    fn root_finding(&self, g1: &Self) -> (bool, T, i32) {
        self.context
            .assert_compatible_nbatch(g1.context.nbatch(), "root_finding");
        assert_eq!(self.len(), g1.len(), "Vector lengths do not match");
        let mut out = None;
        for b in 0..self.data.ncols() {
            let mut found = false;
            let mut frac = T::zero();
            let mut idx = -1;
            let g0_column = self.data.rb().col(b);
            let g1_column = g1.data.rb().col(g1.batch(b));
            for (i, (&g0, &g)) in g0_column
                .try_as_col_major()
                .unwrap()
                .as_slice()
                .iter()
                .zip(g1_column.try_as_col_major().unwrap().as_slice())
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
            .assert_compatible_nbatch(o.context.nbatch(), "copy_from_indices");
        for b in 0..self.data.ncols() {
            for i in idx.data.iter() {
                self.data[(*i, b)] = o.data[(*i, o.batch(b))]
            }
        }
    }
    fn gather(&mut self, o: &Self, idx: &Self::Index) {
        assert_eq!(self.len(), idx.len());
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "gather");
        for b in 0..self.data.ncols() {
            for (i, j) in idx.data.iter().enumerate() {
                self.data[(i, b)] = o.data[(*j, o.batch(b))]
            }
        }
    }
    fn scatter(&self, idx: &Self::Index, o: &mut Self) {
        assert_eq!(self.len(), idx.len());
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "scatter");
        for b in 0..self.data.ncols() {
            for (i, j) in idx.data.iter().enumerate() {
                let batch = o.batch(b);
                o.data[(*j, batch)] = self.data[(i, b)]
            }
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
        let errorn_check = r.data.rb().col(0).squared_norm_l2() / 3.0;
        assert!(
            (v.squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.squared_norm(&y, &atol, rtol),
            errorn_check
        );
        let vview = v.as_view();
        assert!((VectorView::squared_norm(&vview, &y, &atol, rtol) - errorn_check).abs() < 1e-10);
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

    /// The generic suite only covers `norm(1)` and `norm(2)`; an odd `k` has to take the
    /// magnitude of each element before raising it, which the previous impl skipped.
    #[test]
    fn test_norm_odd_k_uses_magnitudes() {
        let v = FaerVec::from_vec(vec![-3.0, 4.0], Default::default());
        let expected: f64 = (27.0f64 + 64.0).powf(1.0 / 3.0);
        assert!((v.norm(3) - expected).abs() < 1e-12, "{}", v.norm(3));
    }

    /// A zero-length vector has no columns to fill, so the constructors must not divide by
    /// or index past it.
    #[test]
    fn test_empty() {
        let ctx = FaerContext::with_nbatch(2);
        for v in [
            FaerVec::<f64>::zeros(0, ctx),
            FaerVec::<f64>::from_vec(vec![], ctx),
            FaerVec::<f64>::from_element(0, 1.0, ctx),
        ] {
            assert_eq!(v.len(), 0);
            assert!(v.is_empty());
            assert_eq!(v.clone_as_vec(), Vec::<f64>::new());
        }
    }

    #[test]
    fn test_host_only() {
        super::super::tests::test_host_only::<FaerVec<f64>>();
    }

    super::super::generate_vector_tests_nonbatched!(faer, FaerVec<f64>);
    super::super::generate_vector_tests_batched!(
        faer,
        FaerVec<f64>,
        FaerContext::with_nbatch(2),
        FaerContext::with_nbatch(3)
    );
}
