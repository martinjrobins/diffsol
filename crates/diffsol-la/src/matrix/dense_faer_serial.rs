use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use crate::context::broadcast_batch;
use crate::error::LaError;
use crate::{scalar::Scale, Context, FaerScalar, Vector};
use crate::{
    DenseMatrix, FaerContext, FaerLU, FaerVec, FaerVecIndex, FaerVecMut, FaerVecRef, Matrix,
    MatrixCommon, VectorIndex,
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
impl<T: FaerScalar> FaerMat<T> {
    fn logical_ncols(&self) -> usize {
        self.data.ncols() / self.context.nbatch()
    }
    /// Physical column of logical column `j` in this matrix's own batch `b`.
    fn col(&self, b: usize, j: usize) -> usize {
        debug_assert!(b < self.context.nbatch(), "batch out of bounds");
        b * self.logical_ncols() + j
    }
    /// Physical column feeding batch `b` of a `nbatch`-batch destination, broadcasting this
    /// matrix over contiguous groups of destination batches (see [`broadcast_batch`]).
    fn col_bcast(&self, b: usize, j: usize, nbatch: usize) -> usize {
        self.col(broadcast_batch(b, self.context.nbatch(), nbatch), j)
    }
}
/// Scalars of staging buffer `mul_cols_by` uses per row tile. The tile is
/// `MUL_COLS_TILE / ncols` rows, so a narrow range gets a tall tile and a wide one a short
/// tile, and either way the staged values stay in L1.
const MUL_COLS_TILE: usize = 512;

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
/// `self` is the owned operand, so it is the destination -- which makes the in-place op the
/// whole implementation, broadcast check and whole-matrix fast path included.
macro_rules! binary_owned_lhs {
    ($tr:ident, $fn:ident, $assign_tr:ident, $assign:ident, $rhs:ty) => {
        impl<T: FaerScalar> $tr<$rhs> for FaerMat<T> {
            type Output = FaerMat<T>;

            fn $fn(mut self, rhs: $rhs) -> Self::Output {
                $assign_tr::$assign(&mut self, rhs);
                self
            }
        }
    };
}
binary_owned_lhs!(Add, add, AddAssign, add_assign, &FaerMat<T>);
binary_owned_lhs!(Sub, sub, SubAssign, sub_assign, &FaerMat<T>);

/// `rhs` is the owned operand, so it is the destination.
///
/// A commutative op is just the in-place op with the operands swapped.  A non-commutative one
/// cannot be: `rhs -= lhs` computes `rhs - lhs`, so it writes `combine(&mut rhs_ij, lhs_ij)`
/// instead, which gets `lhs - rhs` in the same single pass.
macro_rules! binary_owned_rhs {
    (commutes, $tr:ident, $fn:ident, $assign_tr:ident, $assign:ident, $lhs:ty, $op:tt,
     $combine:expr) => {
        impl<T: FaerScalar> $tr<FaerMat<T>> for $lhs {
            type Output = FaerMat<T>;

            fn $fn(self, mut rhs: FaerMat<T>) -> Self::Output {
                $assign_tr::$assign(&mut rhs, self);
                rhs
            }
        }
    };
    (noncommutes, $tr:ident, $fn:ident, $assign_tr:ident, $assign:ident, $lhs:ty, $op:tt, $combine:expr) => {
        impl<T: FaerScalar> $tr<FaerMat<T>> for $lhs {
            type Output = FaerMat<T>;

            fn $fn(self, mut rhs: FaerMat<T>) -> Self::Output {
                rhs.context
                    .assert_broadcastable_into(self.context.nbatch(), stringify!($fn));
                let nc = self.ncols();
                let nb = rhs.context.nbatch();
                for b in 0..nb {
                    let c = rhs.col(b, 0);
                    let columns = rhs.data.rb_mut().subcols_mut(c, nc);
                    zip!(
                        columns,
                        self.data.rb().subcols(self.col_bcast(b, 0, nb), nc)
                    )
                    .for_each(|unzip!(r, l)| $combine(r, *l));
                }
                rhs
            }
        }
    };
}
binary_owned_rhs!(
    commutes, Add, add, AddAssign, add_assign, &FaerMat<T>, +=, |rhs: &mut T, lhs: T| *rhs += lhs
);
binary_owned_rhs!(
    noncommutes, Sub, sub, SubAssign, sub_assign, &FaerMat<T>, -=,
    |rhs: &mut T, lhs: T| *rhs = lhs - *rhs
);
macro_rules! assign {
    ($tr:ident, $fn:ident, $l:ty, $r:ty, $op:tt) => {
        impl<T: FaerScalar> $tr<$r> for $l {
            fn $fn(&mut self, rhs: $r) {
                self.context
                    .assert_broadcastable_into(rhs.context.nbatch(), stringify!($fn));
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
                    columns $op rhs.data.rb().subcols(rhs.col_bcast(b, 0, nb), nc);
                }
            }
        }
    };
}
assign!(AddAssign, add_assign, FaerMat<T>, &FaerMat<T>, +=);
assign!(SubAssign, sub_assign, FaerMat<T>, &FaerMat<T>, -=);
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
                // every batch is scaled by the same factor, and the batches tile the columns,
                // so this is one whole-matrix multiply whatever the batch count
                FaerMat {
                    data: self.data.rb() * faer::Scale(r.value()),
                    context: self.context,
                }
            }
        }
    };
}
scale_ref!(&FaerMat<T>);
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
impl<T: FaerScalar> IndexMut<(usize, usize)> for FaerMat<T> {
    fn index_mut(&mut self, x: (usize, usize)) -> &mut T {
        let c = self.col(0, x.1);
        &mut self.data[(x.0, c)]
    }
}

/// `y_b = a * self_b * x_b + beta * y_b` for every batch of `y`, broadcasting single-batch
/// operands. `x` is a full vector operand of length `ncols`.
macro_rules! gemv_data {
    ($self:ident, $a:ident, $x:ident, $beta:ident, $y:ident, $op:literal) => {{
        $y.context
            .assert_broadcastable_into($self.context.nbatch(), $op);
        $y.context
            .assert_broadcastable_into($x.context.nbatch(), $op);
        let nc = $self.ncols();
        let nb = $y.data.ncols();
        for b in 0..nb {
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
                $self.data.rb().subcols($self.col_bcast(b, 0, nb), nc),
                $x.data.rb().col($x.batch(b, nb)),
                $a,
                get_global_parallelism(),
            );
        }
    }};
}

/// `y_b = a * self_b[:, start..start + nc] * x + beta * y_b` for every batch of `y`, with `x` a
/// host coefficient slice shared by all batches. `ColRef::from_slice` wraps it without copying,
/// so faer's own `matmul` still does the work.
macro_rules! gemv_cols_data {
    ($self:ident, $start:expr, $nc:expr, $a:ident, $x:ident, $beta:ident, $y:ident, $op:literal) => {{
        $y.context
            .assert_broadcastable_into($self.context.nbatch(), $op);
        let (start, nc) = ($start, $nc);
        // only a debug assert: `ncols()` on the owned matrix is a division, and this runs once
        // per Runge-Kutta stage.  An out-of-range range still panics in release, inside faer's
        // `subcols`.
        debug_assert!(
            start + nc <= $self.ncols(),
            concat!($op, ": column range out of bounds")
        );
        // an empty column range contributes nothing, leaving y = beta * y
        if nc == 0 {
            if $beta.is_zero() {
                $y.fill(T::zero());
            } else if !$beta.is_one() {
                for b in 0..$y.data.ncols() {
                    let mut column = $y.data.rb_mut().col_mut(b);
                    column *= faer::Scale($beta);
                }
            }
        } else {
            let x = faer::ColRef::from_slice(&$x[..nc]);
            let nb = $y.data.ncols();
            for b in 0..nb {
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
                    $self.data.rb().subcols($self.col_bcast(b, start, nb), nc),
                    x,
                    $a,
                    get_global_parallelism(),
                );
            }
        }
    }};
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
        self.context
            .assert_broadcastable_into(data.context.nbatch(), "set_data_with_indices");
        let nb = self.context.nbatch();
        for (d, s) in dst.data.iter().zip(src.data.iter()) {
            let i = d % self.nrows();
            let j = d / self.nrows();
            for b in 0..nb {
                let c = self.col(b, j);
                self.data[(i, c)] = data.data[(*s, data.batch(b, nb))]
            }
        }
    }
    fn gather(&mut self, o: &Self, idx: &FaerVecIndex) {
        self.context
            .assert_broadcastable_into(o.context.nbatch(), "gather");
        let nb = self.context.nbatch();
        for b in 0..nb {
            for (d, s) in idx.data.iter().enumerate() {
                let i = d % self.nrows();
                let j = d / self.nrows();
                let oi = s % o.nrows();
                let oj = s / o.nrows();
                let c = self.col(b, j);
                self.data[(i, c)] = o.data[(oi, o.col_bcast(b, oj, nb))]
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
            .assert_broadcastable_into(self.context.nbatch(), "add_column_to_vector");
        let nb = v.context.nbatch();
        for b in 0..nb {
            zip!(
                v.data.rb_mut().col_mut(b),
                self.data.rb().col(self.col_bcast(b, j, nb))
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
            .assert_broadcastable_into(o.context.nbatch(), "copy_from");
        if self.context.nbatch() == o.context.nbatch() {
            self.data.rb_mut().copy_from(o.data.rb());
            return;
        }
        let nc = self.ncols();
        let nb = self.context.nbatch();
        for b in 0..nb {
            let c = self.col(b, 0);
            self.data
                .rb_mut()
                .subcols_mut(c, nc)
                .copy_from(o.data.rb().subcols(o.col_bcast(b, 0, nb), nc));
        }
    }
    fn set_column(&mut self, j: usize, v: &FaerVec<T>) {
        self.context
            .assert_broadcastable_into(v.context.nbatch(), "set_column");
        let nb = self.context.nbatch();
        for b in 0..nb {
            let c = self.col(b, j);
            self.data
                .rb_mut()
                .col_mut(c)
                .copy_from(v.data.rb().col(v.batch(b, nb)));
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, b: T, y: &Self) {
        self.context
            .assert_broadcastable_into(x.context.nbatch(), "scale_add_and_assign");
        self.context
            .assert_broadcastable_into(y.context.nbatch(), "scale_add_and_assign");
        let nc = self.ncols();
        let nb = self.context.nbatch();
        for batch in 0..nb {
            let c = self.col(batch, 0);
            zip!(
                self.data.rb_mut().subcols_mut(c, nc),
                x.data.rb().subcols(x.col_bcast(batch, 0, nb), nc),
                y.data.rb().subcols(y.col_bcast(batch, 0, nb), nc)
            )
            .for_each(|unzip!(s, x, y)| *s = *x + b * *y);
        }
    }
    fn new_from_sparsity(nr: usize, nc: usize, _: Option<Self::Sparsity>, ctx: Self::C) -> Self {
        Self::zeros(nr, nc, ctx)
    }
}
impl<T: FaerScalar> DenseMatrix for FaerMat<T> {
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
    fn gemv_cols(&self, start: usize, end: usize, alpha: T, x: &[T], beta: T, y: &mut FaerVec<T>) {
        assert!(start <= end, "gemv_cols: column range start > end");
        assert!(
            end - start <= crate::matrix::MAX_SMALL_COLS,
            "gemv_cols: column range exceeds MAX_SMALL_COLS"
        );
        assert!(
            x.len() >= end - start,
            "gemv_cols: x must hold at least end - start values"
        );
        gemv_cols_data!(self, start, end - start, alpha, x, beta, y, "gemv_cols")
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
    fn mul_cols_by(&mut self, ncols: usize, rhs: &[T]) {
        assert!(
            ncols <= self.logical_ncols(),
            "mul_cols_by: column range out of bounds"
        );
        assert_eq!(
            rhs.len(),
            ncols * ncols,
            "mul_cols_by: rhs must hold ncols * ncols values"
        );
        if ncols == 0 {
            return;
        }
        assert!(
            ncols <= crate::matrix::MAX_SMALL_COLS,
            "mul_cols_by: ncols exceeds MAX_SMALL_COLS"
        );
        let nrows = self.nrows();
        let nbatch = self.context.nbatch();
        // A column is contiguous, so a tile of consecutive rows is a contiguous run of each
        // column: stage the tile's old values, then overwrite in place. Tall for a narrow
        // range, short for a wide one, so the tile always fits the same buffer.
        let tile = (MUL_COLS_TILE / ncols).max(1);
        let mut buf = [T::zero(); MUL_COLS_TILE];
        for b in 0..nbatch {
            let base = self.col(b, 0);
            let mut row = 0;
            while row < nrows {
                let rows = tile.min(nrows - row);
                for j in 0..ncols {
                    let src = self.data.rb().col(base + j);
                    for i in 0..rows {
                        buf[j * rows + i] = src[row + i];
                    }
                }
                for j in 0..ncols {
                    let w = &rhs[j * ncols..][..ncols];
                    let mut dst = self.data.rb_mut().col_mut(base + j);
                    // accumulate column by column so every access is a contiguous run
                    for i in 0..rows {
                        dst[row + i] = buf[i] * w[0];
                    }
                    for l in 1..ncols {
                        let src = &buf[l * rows..][..rows];
                        for i in 0..rows {
                            dst[row + i] += src[i] * w[l];
                        }
                    }
                }
                row += rows;
            }
        }
    }

    fn update_backward_diff(&mut self, order: usize, d: &FaerVec<T>) {
        assert!(order + 2 < self.logical_ncols(), "order out of bounds");
        self.context
            .assert_broadcastable_into(d.context.nbatch(), "update_backward_diff");
        let nb = self.context.nbatch();
        for b in 0..nb {
            let base = self.col(b, 0);
            let d_col = d.batch(b, nb);
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

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_batched_owned_rhs_narrow() {
        super::super::tests::test_batched_owned_rhs_narrow_m::<FaerMat<f64>>(
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
