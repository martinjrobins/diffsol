use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use crate::error::LaError;
use crate::{scalar::Scale, Context, FaerScalar, Vector};
use crate::{
    DenseMatrix, FaerContext, FaerLU, FaerVec, FaerVecIndex, FaerVecMut, FaerVecRef, Matrix,
    MatrixCommon, MatrixView, MatrixViewMut, VectorIndex,
};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{
    get_global_parallelism, linalg::matmul::matmul, unzip, zip, Accum, Mat, MatMut, MatRef,
};

/// A batched matrix, stored as `nbatch` side-by-side blocks of `logical_ncols` columns.
///
/// Invariant: `data.ncols() == logical_ncols() * context.nbatch()`, so batch `b` column `j`
/// lives at physical column `col(b, j)`.
#[derive(Clone, Debug, PartialEq)]
pub struct FaerMat<T: FaerScalar> {
    pub(crate) data: Mat<T>,
    pub(crate) context: FaerContext,
}
/// A view of columns `start..end` of every batch of a [`FaerMat`].
///
/// A batch's columns are not contiguous with the next batch's, so `data` stays the whole
/// matrix and the column range is carried in `start`/`end` instead of being sliced out.
#[derive(Clone, Debug, PartialEq)]
pub struct FaerMatRef<'a, T: FaerScalar> {
    pub(crate) data: MatRef<'a, T>,
    pub(crate) context: FaerContext,
    pub(crate) start: usize,
    pub(crate) end: usize,
}
/// Mutable counterpart of [`FaerMatRef`].
#[derive(Debug, PartialEq)]
pub struct FaerMatMut<'a, T: FaerScalar> {
    pub(crate) data: MatMut<'a, T>,
    pub(crate) context: FaerContext,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Column arithmetic shared by both view types: `logical_ncols` is the per-batch column
/// count of the whole matrix, `ncols_local` the width of this view, and `col` maps a
/// (batch, local column) pair to a physical column, broadcasting a single-batch operand.
macro_rules! mat_methods {
    () => {
        fn logical_ncols(&self) -> usize {
            self.data.ncols() / self.context.nbatch()
        }
        fn ncols_local(&self) -> usize {
            self.end - self.start
        }
        fn col(&self, b: usize, j: usize) -> usize {
            (b % self.context.nbatch()) * self.logical_ncols() + self.start + j
        }
    };
}
impl<T: FaerScalar> FaerMat<T> {
    fn logical_ncols(&self) -> usize {
        self.data.ncols() / self.context.nbatch()
    }
    fn col(&self, b: usize, j: usize) -> usize {
        (b % self.context.nbatch()) * self.logical_ncols() + j
    }
}
impl<T: FaerScalar> FaerMatRef<'_, T> {
    mat_methods!();
}
impl<T: FaerScalar> FaerMatMut<'_, T> {
    mat_methods!();
}
impl<T: FaerScalar> DefaultSolver for FaerMat<T> {
    type LS = FaerLU<T>;
}

impl<T: FaerScalar> MatrixCommon for FaerMat<T> {
    type T = T;
    type V = FaerVec<T>;
    type C = FaerContext;
    type Inner = Mat<T>;
    fn nrows(&self) -> usize {
        self.data.nrows()
    }
    fn ncols(&self) -> usize {
        self.logical_ncols()
    }
    fn inner(&self) -> &Self::Inner {
        &self.data
    }
}
macro_rules! common_ref {
    ($t:ty,$inner:ty) => {
        impl<'a, T: FaerScalar> MatrixCommon for $t {
            type T = T;
            type V = FaerVec<T>;
            type C = FaerContext;
            type Inner = $inner;
            fn nrows(&self) -> usize {
                self.data.nrows()
            }
            fn ncols(&self) -> usize {
                self.ncols_local()
            }
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common_ref!(FaerMatRef<'a, T>, MatRef<'a, T>);
common_ref!(FaerMatMut<'a, T>, MatMut<'a, T>);

macro_rules! binary {
    ($tr:ident, $fn:ident, $l:ty, $r:ty, $op:tt, $binary:tt) => {
        impl<T: FaerScalar> $tr<$r> for $l {
            type Output = FaerMat<T>;

            fn $fn(self, rhs: $r) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nb = self.context.nbatch().max(rhs.context.nbatch());
                let nc = self.ncols();
                if self.context.nbatch() == rhs.context.nbatch()
                    && self.data.ncols() == nc * nb
                    && rhs.data.ncols() == nc * nb
                {
                    return FaerMat {
                        data: self.data.rb() $binary rhs.data.rb(),
                        context: self.context,
                    };
                }
                let mut data = Mat::zeros(self.nrows(), nc * nb);
                for b in 0..nb {
                    let mut columns = data.rb_mut().subcols_mut(b * nc, nc);
                    columns.copy_from(self.data.rb().subcols(self.col(b, 0), nc));
                    columns $op rhs.data.rb().subcols(rhs.col(b, 0), nc);
                }
                FaerMat {
                    data,
                    context: if self.context.nbatch() == nb {
                        self.context
                    } else {
                        rhs.context
                    },
                }
            }
        }
    };
}
binary!(Add, add, FaerMatRef<'_, T>, &FaerMat<T>, +=, +);
binary!(Sub, sub, FaerMatRef<'_, T>, &FaerMat<T>, -=, -);
/// `self` is owned, so the result is written into its columns whenever it already holds as
/// many batches as the result.
macro_rules! binary_owned_lhs {
    ($tr:ident, $fn:ident, $rhs:ty, $op:tt) => {
        impl<T: FaerScalar> $tr<$rhs> for FaerMat<T> {
            type Output = FaerMat<T>;

            fn $fn(mut self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nc = self.ncols();
                if self.context.nbatch() >= rhs.context.nbatch() {
                    for b in 0..self.context.nbatch() {
                        let c = self.col(b, 0);
                        let mut columns = self.data.rb_mut().subcols_mut(c, nc);
                        columns $op rhs.data.rb().subcols(rhs.col(b, 0), nc);
                    }
                    return self;
                }
                // self holds fewer batches than the result, so it cannot be written into
                let nb = rhs.context.nbatch();
                let mut data = Mat::zeros(self.nrows(), nc * nb);
                for b in 0..nb {
                    let mut columns = data.rb_mut().subcols_mut(b * nc, nc);
                    columns.copy_from(self.data.rb().subcols(self.col(b, 0), nc));
                    columns $op rhs.data.rb().subcols(rhs.col(b, 0), nc);
                }
                FaerMat {
                    data,
                    context: rhs.context,
                }
            }
        }
    };
}
binary_owned_lhs!(Add, add, &FaerMat<T>, +=);
binary_owned_lhs!(Add, add, &FaerMatRef<'_, T>, +=);
binary_owned_lhs!(Sub, sub, &FaerMat<T>, -=);
binary_owned_lhs!(Sub, sub, &FaerMatRef<'_, T>, -=);

/// `rhs` is owned, so the result is written into its columns whenever it already holds as
/// many batches as the result: `combine(&mut rhs_ij, lhs_ij)`.
macro_rules! binary_owned_rhs {
    ($tr:ident, $fn:ident, $lhs:ty, $op:tt, $combine:expr) => {
        impl<T: FaerScalar> $tr<FaerMat<T>> for $lhs {
            type Output = FaerMat<T>;

            fn $fn(self, mut rhs: FaerMat<T>) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nc = self.ncols();
                if rhs.context.nbatch() >= self.context.nbatch() {
                    for b in 0..rhs.context.nbatch() {
                        let c = rhs.col(b, 0);
                        let columns = rhs.data.rb_mut().subcols_mut(c, nc);
                        zip!(columns, self.data.rb().subcols(self.col(b, 0), nc))
                            .for_each(|unzip!(r, l)| $combine(r, *l));
                    }
                    return rhs;
                }
                // rhs holds fewer batches than the result, so it cannot be written into
                let nb = self.context.nbatch();
                let mut data = Mat::zeros(self.nrows(), nc * nb);
                for b in 0..nb {
                    let mut columns = data.rb_mut().subcols_mut(b * nc, nc);
                    columns.copy_from(self.data.rb().subcols(self.col(b, 0), nc));
                    columns $op rhs.data.rb().subcols(rhs.col(b, 0), nc);
                }
                FaerMat {
                    data,
                    context: self.context,
                }
            }
        }
    };
}
binary_owned_rhs!(Add, add, &FaerMat<T>, +=, |rhs: &mut T, lhs: T| *rhs += lhs);
binary_owned_rhs!(Add, add, FaerMatRef<'_, T>, +=, |rhs: &mut T, lhs: T| *rhs += lhs);
binary_owned_rhs!(Sub, sub, &FaerMat<T>, -=, |rhs: &mut T, lhs: T| *rhs = lhs - *rhs);
binary_owned_rhs!(
    Sub,
    sub,
    FaerMatRef<'_, T>,
    -=,
    |rhs: &mut T, lhs: T| *rhs = lhs - *rhs
);
macro_rules! assign {
    ($tr:ident, $fn:ident, $l:ty, $r:ty, $op:tt) => {
        impl<T: FaerScalar> $tr<$r> for $l {
            fn $fn(&mut self, rhs: $r) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nb = self.context.nbatch();
                let nc = self.ncols();
                if nb == rhs.context.nbatch()
                    && self.data.ncols() == nc * nb
                    && rhs.data.ncols() == nc * nb
                {
                    self.data $op rhs.data.rb();
                    return;
                }
                for b in 0..nb {
                    let c = self.col(b, 0);
                    let mut columns = self.data.rb_mut().subcols_mut(c, nc);
                    columns $op rhs.data.rb().subcols(rhs.col(b, 0), nc);
                }
            }
        }
    };
}
assign!(AddAssign, add_assign, FaerMat<T>, &FaerMat<T>, +=);
assign!(AddAssign, add_assign, FaerMat<T>, &FaerMatRef<'_, T>, +=);
assign!(
    AddAssign,
    add_assign,
    FaerMatMut<'_, T>,
    &FaerMatRef<'_, T>,
    +=
);
assign!(
    AddAssign,
    add_assign,
    FaerMatMut<'_, T>,
    &FaerMatMut<'_, T>,
    +=
);
assign!(SubAssign, sub_assign, FaerMat<T>, &FaerMat<T>, -=);
assign!(SubAssign, sub_assign, FaerMat<T>, &FaerMatRef<'_, T>, -=);
assign!(
    SubAssign,
    sub_assign,
    FaerMatMut<'_, T>,
    &FaerMatRef<'_, T>,
    -=
);
assign!(
    SubAssign,
    sub_assign,
    FaerMatMut<'_, T>,
    &FaerMatMut<'_, T>,
    -=
);
impl<T: FaerScalar> Mul<Scale<T>> for FaerMat<T> {
    type Output = Self;
    fn mul(mut self, r: Scale<T>) -> Self {
        self.data *= faer::Scale(r.value());
        self
    }
}
macro_rules! scale_ref {
    ($t:ty) => {
        impl<T: FaerScalar> Mul<Scale<T>> for $t {
            type Output = FaerMat<T>;
            fn mul(self, r: Scale<T>) -> Self::Output {
                if self.data.ncols() == self.ncols() * self.context.nbatch() {
                    return FaerMat {
                        data: self.data.rb() * faer::Scale(r.value()),
                        context: self.context,
                    };
                }
                let mut out = FaerMat::zeros(self.nrows(), self.ncols(), self.context);
                let nc = out.ncols();
                for b in 0..out.context.nbatch() {
                    let c = out.col(b, 0);
                    out.data
                        .rb_mut()
                        .subcols_mut(c, nc)
                        .copy_from(self.data.rb().subcols(self.col(b, 0), nc));
                }
                out.data *= faer::Scale(r.value());
                out
            }
        }
    };
}
scale_ref!(&FaerMat<T>);
scale_ref!(FaerMatRef<'_, T>);
impl<T: FaerScalar> MulAssign<Scale<T>> for FaerMatMut<'_, T> {
    fn mul_assign(&mut self, r: Scale<T>) {
        if self.data.ncols() == self.ncols() * self.context.nbatch() {
            self.data *= faer::Scale(r.value());
            return;
        }
        let nc = self.ncols();
        for b in 0..self.context.nbatch() {
            let c = self.col(b, 0);
            let mut columns = self.data.rb_mut().subcols_mut(c, nc);
            columns *= faer::Scale(r.value());
        }
    }
}
macro_rules! ind {
    ($t:ty) => {
        impl<T: FaerScalar> Index<(usize, usize)> for $t {
            type Output = T;
            fn index(&self, x: (usize, usize)) -> &T {
                &self.data[(x.0, self.col(0, x.1))]
            }
        }
    };
}
ind!(FaerMat<T>);
ind!(FaerMatRef<'_, T>);
impl<T: FaerScalar> IndexMut<(usize, usize)> for FaerMat<T> {
    fn index_mut(&mut self, x: (usize, usize)) -> &mut T {
        let c = self.col(0, x.1);
        &mut self.data[(x.0, c)]
    }
}

/// `y_b = a * self_b * x_b + beta * y_b` for every batch of `y`, broadcasting single-batch
/// operands.  Shared by the owned matrix and its views: both expose `col` and `ncols`, and
/// `x` may be an owned vector or a view.
macro_rules! gemv_data {
    ($self:ident, $a:ident, $x:ident, $beta:ident, $y:ident, $op:literal) => {{
        $y.context
            .assert_compatible_nbatch($self.context.nbatch(), $op);
        $y.context
            .assert_compatible_nbatch($x.context.nbatch(), $op);
        let nc = $self.ncols();
        for b in 0..$y.data.ncols() {
            let mut column = $y.data.rb_mut().col_mut(b);
            let accum = if $beta.is_zero() {
                Accum::Replace
            } else {
                if !$beta.is_one() {
                    column *= faer::Scale($beta);
                }
                Accum::Add
            };
            matmul(
                column,
                accum,
                $self.data.rb().subcols($self.col(b, 0), nc),
                $x.data.rb().col($x.batch(b)),
                $a,
                get_global_parallelism(),
            );
        }
    }};
}

/// Copies columns `start..end` of every batch of a view into a fresh owned matrix.  Shared by
/// both view types, which expose `col` and `ncols` alike.
macro_rules! into_owned_data {
    ($self:ident) => {{
        let nc = $self.ncols();
        let mut out = FaerMat::zeros($self.nrows(), nc, $self.context);
        for b in 0..$self.context.nbatch() {
            let c = out.col(b, 0);
            out.data
                .rb_mut()
                .subcols_mut(c, nc)
                .copy_from($self.data.rb().subcols($self.col(b, 0), nc));
        }
        out
    }};
}

/// `self_b = a * x_b * y_b + beta * self_b` for every batch of `self`, broadcasting
/// single-batch operands.  Shared by the owned matrix and its mutable view, with `x` and `y`
/// either owned matrices or views: all of them expose `col` and `ncols`.
macro_rules! gemm_data {
    ($self:ident, $a:ident, $x:ident, $y:ident, $beta:ident, $op:literal) => {{
        $self
            .context
            .assert_compatible_nbatch($x.context.nbatch(), $op);
        $self
            .context
            .assert_compatible_nbatch($y.context.nbatch(), $op);
        let (nc, xc, yc) = ($self.ncols(), $x.ncols(), $y.ncols());
        for b in 0..$self.context.nbatch() {
            let c = $self.col(b, 0);
            let mut columns = $self.data.rb_mut().subcols_mut(c, nc);
            columns *= faer::Scale($beta);
            matmul(
                columns,
                Accum::Add,
                $x.data.rb().subcols($x.col(b, 0), xc),
                $y.data.rb().subcols($y.col(b, 0), yc),
                $a,
                get_global_parallelism(),
            );
        }
    }};
}

impl<'a, T: FaerScalar> MatrixView<'a> for FaerMatRef<'a, T> {
    type Owned = FaerMat<T>;
    fn into_owned(self) -> Self::Owned {
        into_owned_data!(self)
    }
    fn gemv(&self, a: T, x: &FaerVec<T>, beta: T, y: &mut FaerVec<T>) {
        gemv_data!(self, a, x, beta, y, "gemv")
    }
}
impl<'a, T: FaerScalar> MatrixViewMut<'a> for FaerMatMut<'a, T> {
    type Owned = FaerMat<T>;
    type View = FaerMatRef<'a, T>;
    fn into_owned(self) -> Self::Owned {
        into_owned_data!(self)
    }
    fn gemm_oo(&mut self, a: T, x: &Self::Owned, y: &Self::Owned, beta: T) {
        gemm_data!(self, a, x, y, beta, "gemm_oo")
    }
    fn gemm_vo(&mut self, a: T, x: &Self::View, y: &Self::Owned, beta: T) {
        gemm_data!(self, a, x, y, beta, "gemm_vo")
    }
}

impl<T: FaerScalar> Matrix for FaerMat<T> {
    type Sparsity = Dense<Self>;
    type SparsityRef<'a> = DenseRef<'a, Self>;
    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        None
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn set_data_with_indices(&mut self, dst: &FaerVecIndex, src: &FaerVecIndex, data: &FaerVec<T>) {
        for (d, s) in dst.data.iter().zip(src.data.iter()) {
            let i = d % self.nrows();
            let j = d / self.nrows();
            for b in 0..self.context.nbatch() {
                let c = self.col(b, j);
                self.data[(i, c)] = data.data[(*s, data.batch(b))]
            }
        }
    }
    fn gather(&mut self, o: &Self, idx: &FaerVecIndex) {
        for b in 0..self.context.nbatch() {
            for (d, s) in idx.data.iter().enumerate() {
                let i = d % self.nrows();
                let j = d / self.nrows();
                let oi = s % o.nrows();
                let oj = s / o.nrows();
                let c = self.col(b, j);
                self.data[(i, c)] = o.data[(oi, o.col(b, oj))]
            }
        }
    }
    fn partition_indices_by_zero_diagonal(&self) -> (FaerVecIndex, FaerVecIndex) {
        let (mut z, mut nz) = (Vec::new(), Vec::new());
        for i in 0..self.nrows() {
            if self.data[(i, self.col(0, i))].is_zero() {
                z.push(i)
            } else {
                nz.push(i)
            }
        }
        (
            FaerVecIndex::from_vec(z, self.context),
            FaerVecIndex::from_vec(nz, self.context),
        )
    }
    fn add_column_to_vector(&self, j: usize, v: &mut FaerVec<T>) {
        v.context
            .assert_compatible_nbatch(self.context.nbatch(), "add_column_to_vector");
        for b in 0..v.context.nbatch() {
            zip!(
                v.data.rb_mut().col_mut(b),
                self.data.rb().col(self.col(b, j))
            )
            .for_each(|unzip!(v, column)| *v += *column);
        }
    }
    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (usize, usize)> + '_,
        impl Iterator<Item = T> + '_,
    ) {
        let pos: Vec<_> = (0..self.ncols())
            .flat_map(|j| (0..self.nrows()).map(move |i| (i, j)))
            .collect();
        let val: Vec<_> = (0..self.context.nbatch())
            .flat_map(|b| {
                pos.iter()
                    .map(move |&(i, j)| self.data[(i, self.col(b, j))])
            })
            .collect();
        (pos.into_iter(), val.into_iter())
    }
    fn try_from_triplets(
        nr: usize,
        nc: usize,
        idx: Vec<(usize, usize)>,
        v: Vec<T>,
        ctx: Self::C,
    ) -> Result<Self, LaError> {
        assert_eq!(v.len(), idx.len() * ctx.nbatch());
        let mut out = Self::zeros(nr, nc, ctx);
        for b in 0..ctx.nbatch() {
            for (k, &(i, j)) in idx.iter().enumerate() {
                let c = out.col(b, j);
                out.data[(i, c)] = v[b * idx.len() + k]
            }
        }
        Ok(out)
    }
    fn zeros(nr: usize, nc: usize, ctx: Self::C) -> Self {
        Self {
            data: Mat::zeros(nr, nc * ctx.nbatch()),
            context: ctx,
        }
    }
    fn from_diagonal(v: &FaerVec<T>) -> Self {
        let mut out = Self::zeros(v.len(), v.len(), v.context);
        let nc = out.ncols();
        for b in 0..v.context.nbatch() {
            let c = out.col(b, 0);
            out.data
                .rb_mut()
                .subcols_mut(c, nc)
                .diagonal_mut()
                .column_vector_mut()
                .copy_from(v.data.rb().col(b));
        }
        out
    }
    fn gemv(&self, a: T, x: &FaerVec<T>, beta: T, y: &mut FaerVec<T>) {
        gemv_data!(self, a, x, beta, y, "gemv")
    }
    fn copy_from(&mut self, o: &Self) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "copy_from");
        if self.context.nbatch() == o.context.nbatch() {
            self.data.rb_mut().copy_from(o.data.rb());
            return;
        }
        let nc = self.ncols();
        for b in 0..self.context.nbatch() {
            let c = self.col(b, 0);
            self.data
                .rb_mut()
                .subcols_mut(c, nc)
                .copy_from(o.data.rb().subcols(o.col(b, 0), nc));
        }
    }
    fn set_column(&mut self, j: usize, v: &FaerVec<T>) {
        self.context
            .assert_compatible_nbatch(v.context.nbatch(), "set_column");
        for b in 0..self.context.nbatch() {
            let c = self.col(b, j);
            self.data
                .rb_mut()
                .col_mut(c)
                .copy_from(v.data.rb().col(v.batch(b)));
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, b: T, y: &Self) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "scale_add_and_assign");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "scale_add_and_assign");
        let nc = self.ncols();
        for batch in 0..self.context.nbatch() {
            let c = self.col(batch, 0);
            zip!(
                self.data.rb_mut().subcols_mut(c, nc),
                x.data.rb().subcols(x.col(batch, 0), nc),
                y.data.rb().subcols(y.col(batch, 0), nc)
            )
            .for_each(|unzip!(s, x, y)| *s = *x + b * *y);
        }
    }
    fn new_from_sparsity(nr: usize, nc: usize, _: Option<Self::Sparsity>, ctx: Self::C) -> Self {
        Self::zeros(nr, nc, ctx)
    }
}
impl<T: FaerScalar> DenseMatrix for FaerMat<T> {
    type View<'a> = FaerMatRef<'a, T>;
    type ViewMut<'a> = FaerMatMut<'a, T>;
    fn gemm(&mut self, a: T, x: &Self, y: &Self, b: T) {
        gemm_data!(self, a, x, y, b, "gemm")
    }
    fn resize_cols(&mut self, nc: usize) {
        let old = self.ncols();
        if old == nc {
            return;
        }
        let keep = old.min(nc);
        let mut d = Mat::zeros(self.nrows(), nc * self.context.nbatch());
        for b in 0..self.context.nbatch() {
            d.rb_mut()
                .subcols_mut(b * nc, keep)
                .copy_from(self.data.rb().subcols(self.col(b, 0), keep));
        }
        self.data = d
    }
    fn get_index(&self, i: usize, j: usize) -> T {
        self.data[(i, self.col(0, j))]
    }
    fn weighted_column_sum(
        &self,
        start: usize,
        end: usize,
        weights: Option<&[T]>,
        y: &mut FaerVec<T>,
    ) {
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "weighted_column_sum");
        assert!(end <= self.logical_ncols(), "column range out of bounds");
        assert!(
            weights.is_none_or(|w| w.len() == end - start),
            "weights length must match the column range"
        );
        if start >= end {
            y.fill(T::zero());
            return;
        }
        for b in 0..y.data.ncols() {
            for (k, j) in (start..end).enumerate() {
                let w = weights.map_or(T::one(), |weights| weights[k]);
                let src = self.data.rb().col(self.col(b, j));
                let dst = y.data.rb_mut().col_mut(b);
                if k == 0 {
                    zip!(dst, src).for_each(|unzip!(y, x)| *y = w.algebraic_mul(*x));
                } else {
                    zip!(dst, src)
                        .for_each(|unzip!(y, x)| *y = y.algebraic_add(w.algebraic_mul(*x)));
                }
            }
        }
    }
    fn from_vec(nr: usize, nc: usize, d: Vec<T>, ctx: Self::C) -> Self {
        assert_eq!(d.len(), nr * nc * ctx.nbatch());
        Self {
            data: Mat::from_fn(nr, nc * ctx.nbatch(), |i, j| d[j * nr + i]),
            context: ctx,
        }
    }
    fn column_mut(&mut self, i: usize) -> FaerVecMut<'_, T> {
        let nc = self.ncols();
        assert!(i < nc, "column index out of bounds");
        if self.context.nbatch() == 1 {
            return FaerVecMut {
                data: self.data.rb_mut().subcols_mut(i, 1),
                context: self.context,
            };
        }
        let (nr, nb) = (self.nrows(), self.context.nbatch());
        // column i of every batch: batch b holds it at physical column b * nc + i, so the
        // batches are `nc` columns apart in the underlying (padded) storage
        let stride = self.data.col_stride() * nc as isize;
        let data = unsafe {
            MatMut::from_raw_parts_mut(
                self.data.rb_mut().ptr_at_mut(0, i),
                nr,
                nb,
                self.data.row_stride(),
                stride,
            )
        };
        FaerVecMut {
            data,
            context: self.context,
        }
    }
    fn columns_mut(&mut self, s: usize, e: usize) -> Self::ViewMut<'_> {
        FaerMatMut {
            data: self.data.rb_mut(),
            context: self.context,
            start: s,
            end: e,
        }
    }
    fn set_index(&mut self, i: usize, j: usize, v: T) {
        for b in 0..self.context.nbatch() {
            let c = self.col(b, j);
            self.data[(i, c)] = v;
        }
    }
    fn column(&self, i: usize) -> FaerVecRef<'_, T> {
        let nc = self.ncols();
        assert!(i < nc, "column index out of bounds");
        if self.context.nbatch() == 1 {
            return FaerVecRef {
                data: self.data.rb().subcols(i, 1),
                context: self.context,
            };
        }
        let (nr, nb) = (self.nrows(), self.context.nbatch());
        // column i of every batch: batch b holds it at physical column b * nc + i, so the
        // batches are `nc` columns apart in the underlying (padded) storage
        let stride = self.data.col_stride() * nc as isize;
        let data = unsafe {
            MatRef::from_raw_parts(
                self.data.rb().ptr_at(0, i),
                nr,
                nb,
                self.data.row_stride(),
                stride,
            )
        };
        FaerVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, s: usize, e: usize) -> Self::View<'_> {
        FaerMatRef {
            data: self.data.rb(),
            context: self.context,
            start: s,
            end: e,
        }
    }
    fn update_backward_diff(&mut self, order: usize, d: &FaerVec<T>) {
        assert!(order + 2 < self.logical_ncols(), "order out of bounds");
        self.context
            .assert_compatible_nbatch(d.context.nbatch(), "update_backward_diff");
        for b in 0..self.context.nbatch() {
            let base = self.col(b, 0);
            let d_col = d.batch(b);
            // diff[:, order+2] = d - diff[:, order+1]
            {
                let src = base + order + 1;
                let dst = base + order + 2;
                let (left, right) = self.data.rb_mut().split_at_col_mut(dst);
                zip!(right.col_mut(0), d.data.rb().col(d_col), left.col(src))
                    .for_each(|unzip!(out, dval, sval)| *out = *dval - *sval);
            }
            // for i in (0..=order+1).rev(): diff[:, i] += diff[:, i+1]
            for i in (0..=order + 1).rev() {
                let dst = base + i;
                let src = base + i + 1;
                let (left, right) = self.data.rb_mut().split_at_col_mut(src);
                zip!(left.col_mut(dst), right.col(0)).for_each(|unzip!(d, s)| *d += *s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_backward_diff() {
        super::super::tests::test_update_backward_diff::<FaerMat<f64>>();
    }

    #[test]
    fn test_index_operator() {
        let mut mat = FaerMat::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0], FaerContext::default());
        assert_eq!(mat[(1, 0)], 3.0);
        mat[(1, 0)] = 30.0;
        assert_eq!(mat.get_index(1, 0), 30.0);
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<FaerMat<f64>>();
    }

    #[test]
    fn test_resize_cols() {
        super::super::tests::test_resize_cols::<FaerMat<f64>>();
    }

    #[test]
    fn test_owned_rhs() {
        super::super::tests::test_owned_rhs_m::<FaerMat<f64>>();
    }

    #[test]
    fn test_batched_owned_rhs_broadcast() {
        super::super::tests::test_batched_owned_rhs_broadcast_m::<FaerMat<f64>>(
            FaerContext::with_nbatch(2),
        );
    }

    super::super::generate_matrix_tests_nonbatched!(faer, FaerMat<f64>);
    super::super::generate_matrix_tests_batched!(
        faer,
        FaerMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );
    super::super::generate_dense_matrix_tests_nonbatched!(faer, FaerMat<f64>);
    super::super::generate_dense_matrix_tests_batched!(
        faer,
        FaerMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );
}
