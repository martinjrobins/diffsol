use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

use nalgebra::{
    Const, DMatrix, DVectorView, MatrixView as NaMatrixView, MatrixViewMut as NaMatrixViewMut,
};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use crate::context::broadcast_batch;
use crate::error::LaError;
use crate::{scalar::Scale, Context, NalgebraScalar, Vector};
use crate::{
    DenseMatrix, Matrix, MatrixCommon, NalgebraContext, NalgebraLU, NalgebraVec, NalgebraVecMut,
    NalgebraVecRef, VectorIndex,
};

/// A batched matrix, stored as `nbatch` side-by-side blocks of `logical_ncols` columns.
///
/// Invariant: `data.ncols() == logical_ncols() * context.nbatch()`, so batch `b` column `j`
/// lives at physical column `col(b, j)`.
#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMat<T: NalgebraScalar> {
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}
impl<T: NalgebraScalar> NalgebraMat<T> {
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

impl<T: NalgebraScalar> DefaultSolver for NalgebraMat<T> {
    type LS = NalgebraLU<T>;
}

impl<T: NalgebraScalar> MatrixCommon for NalgebraMat<T> {
    type T = T;
    type V = NalgebraVec<T>;
    type C = NalgebraContext;
    type Inner = DMatrix<T>;
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
/// `self` is owned, so the result is written into its columns whenever it already holds as
/// many batches as the result.
macro_rules! binary_owned_lhs {
    ($tr:ident, $fn:ident, $rhs:ty, $op:tt) => {
        impl<T: NalgebraScalar> $tr<$rhs> for NalgebraMat<T> {
            type Output = NalgebraMat<T>;

            fn $fn(mut self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nc = self.ncols();
                if self.context.nbatch() >= rhs.context.nbatch() {
                    let nb = self.context.nbatch();
                    for b in 0..nb {
                        let mut columns = self.data.columns_mut(self.col(b, 0), nc);
                        columns $op &rhs.data.columns(rhs.col_bcast(b, 0, nb), nc);
                    }
                    return self;
                }
                // self holds fewer batches than the result, so it cannot be written into
                let nb = rhs.context.nbatch();
                let mut data = DMatrix::zeros(self.nrows(), nc * nb);
                for b in 0..nb {
                    let mut columns = data.columns_mut(b * nc, nc);
                    columns.copy_from(&self.data.columns(self.col_bcast(b, 0, nb), nc));
                    columns $op &rhs.data.columns(rhs.col(b, 0), nc);
                }
                NalgebraMat {
                    data,
                    context: rhs.context,
                }
            }
        }
    };
}
binary_owned_lhs!(Add, add, &NalgebraMat<T>, +=);
binary_owned_lhs!(Sub, sub, &NalgebraMat<T>, -=);

/// `rhs` is owned, so the result is written into its columns whenever it already holds as
/// many batches as the result: `combine(&mut rhs_ij, lhs_ij)`.
macro_rules! binary_owned_rhs {
    ($tr:ident, $fn:ident, $lhs:ty, $op:tt, $combine:expr) => {
        impl<T: NalgebraScalar> $tr<NalgebraMat<T>> for $lhs {
            type Output = NalgebraMat<T>;

            fn $fn(self, mut rhs: NalgebraMat<T>) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nc = self.ncols();
                if rhs.context.nbatch() >= self.context.nbatch() {
                    let nb = rhs.context.nbatch();
                    for b in 0..nb {
                        let mut columns = rhs.data.columns_mut(rhs.col(b, 0), nc);
                        columns
                            .zip_apply(&self.data.columns(self.col_bcast(b, 0, nb), nc), $combine);
                    }
                    return rhs;
                }
                // rhs holds fewer batches than the result, so it cannot be written into
                let nb = self.context.nbatch();
                let mut data = DMatrix::zeros(self.nrows(), nc * nb);
                for b in 0..nb {
                    let mut columns = data.columns_mut(b * nc, nc);
                    columns.copy_from(&self.data.columns(self.col(b, 0), nc));
                    columns $op &rhs.data.columns(rhs.col_bcast(b, 0, nb), nc);
                }
                NalgebraMat {
                    data,
                    context: self.context,
                }
            }
        }
    };
}
binary_owned_rhs!(Add, add, &NalgebraMat<T>, +=, |rhs: &mut T, lhs: T| *rhs += lhs);
binary_owned_rhs!(Sub, sub, &NalgebraMat<T>, -=, |rhs: &mut T, lhs: T| *rhs = lhs - *rhs);
macro_rules! assign {
    ($tr:ident, $fn:ident, $l:ty, $r:ty, $op:tt) => {
        impl<T: NalgebraScalar> $tr<$r> for $l {
            fn $fn(&mut self, rhs: $r) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nb = self.context.nbatch();
                let nc = self.ncols();
                if nb == rhs.context.nbatch()
                    && self.data.ncols() == nc * nb
                    && rhs.data.ncols() == nc * nb
                {
                    self.data $op &rhs.data;
                    return;
                }
                for b in 0..nb {
                    let mut columns = self.data.columns_mut(self.col(b, 0), nc);
                    columns $op &rhs.data.columns(rhs.col_bcast(b, 0, nb), nc);
                }
            }
        }
    };
}
assign!(AddAssign, add_assign, NalgebraMat<T>, &NalgebraMat<T>, +=);
assign!(SubAssign, sub_assign, NalgebraMat<T>, &NalgebraMat<T>, -=);
impl<T: NalgebraScalar> Mul<Scale<T>> for NalgebraMat<T> {
    type Output = Self;
    fn mul(mut self, r: Scale<T>) -> Self {
        self.data *= r.value();
        self
    }
}
macro_rules! scale_ref {
    ($t:ty) => {
        impl<T: NalgebraScalar> Mul<Scale<T>> for $t {
            type Output = NalgebraMat<T>;
            fn mul(self, r: Scale<T>) -> Self::Output {
                // every batch is scaled by the same factor, and the batches tile the columns,
                // so this is one whole-matrix multiply whatever the batch count
                NalgebraMat {
                    data: &self.data * r.value(),
                    context: self.context,
                }
            }
        }
    };
}
scale_ref!(&NalgebraMat<T>);
macro_rules! ind {
    ($t:ty) => {
        impl<T: NalgebraScalar> Index<(usize, usize)> for $t {
            type Output = T;
            fn index(&self, x: (usize, usize)) -> &T {
                &self.data[(x.0, self.col(0, x.1))]
            }
        }
    };
}
ind!(NalgebraMat<T>);
impl<T: NalgebraScalar> IndexMut<(usize, usize)> for NalgebraMat<T> {
    fn index_mut(&mut self, x: (usize, usize)) -> &mut T {
        let c = self.col(0, x.1);
        &mut self.data[(x.0, c)]
    }
}

/// `y_b = a * self_b * x_b + beta * y_b` for every batch of `y`, broadcasting single-batch
/// operands. `x` is a full vector operand of length `ncols`.
///
/// nalgebra's own `gemv` is a per-column `axcpy` loop, so it traverses `y` once per column.
/// A single flat pass accumulating every column into `y` measures ~5-8% faster once `nrows`
/// reaches a few hundred, but slower below that (the `stage_accumulate` benchmark: 19.4ns
/// against 16.5ns at 2 rows), and the solvers' hot matrices are small — so this stays on
/// nalgebra's path throughout.
macro_rules! gemv_data {
    ($self:ident, $a:ident, $x:ident, $beta:ident, $y:ident, $op:literal) => {{
        $y.context
            .assert_compatible_nbatch($self.context.nbatch(), $op);
        $y.context
            .assert_compatible_nbatch($x.context.nbatch(), $op);
        $self
            .context
            .assert_compatible_nbatch($x.context.nbatch(), $op);
        let nc = $self.ncols();
        // the unbatched case keeps constant column indices, which codegens better than the
        // loop variable below
        if $y.data.ncols() == 1 {
            $y.data.column_mut(0).gemv(
                $a,
                &$self.data.columns($self.col(0, 0), nc),
                &$x.data.column(0),
                $beta,
            );
            return;
        }
        let nb = $y.data.ncols();
        for b in 0..nb {
            $y.data.column_mut(b).gemv(
                $a,
                &$self.data.columns($self.col_bcast(b, 0, nb), nc),
                &$x.data.column($x.batch(b, nb)),
                $beta,
            );
        }
    }};
}

/// `y_b = a * self_b[:, start..start + nc] * x + beta * y_b` for every batch of `y`, with `x` a
/// host coefficient slice shared by all batches. `DVectorView::from_slice` wraps it without
/// copying, so nalgebra's own `gemv` still does the work.
macro_rules! gemv_cols_data {
    ($self:ident, $start:expr, $nc:expr, $a:ident, $x:ident, $beta:ident, $y:ident, $op:literal) => {{
        $y.context
            .assert_compatible_nbatch($self.context.nbatch(), $op);
        let (start, nc) = ($start, $nc);
        // only a debug assert: `ncols()` on the owned matrix is a division, and this runs once
        // per Runge-Kutta stage.  An out-of-range range still panics in release, inside
        // nalgebra's `columns`.
        debug_assert!(
            start + nc <= $self.ncols(),
            concat!($op, ": column range out of bounds")
        );
        let x = DVectorView::from_slice(&$x[..nc], nc);
        if $y.data.ncols() == 1 {
            $y.data
                .column_mut(0)
                .gemv($a, &$self.data.columns($self.col(0, start), nc), &x, $beta);
            return;
        }
        let nb = $y.data.ncols();
        for b in 0..nb {
            $y.data.column_mut(b).gemv(
                $a,
                &$self.data.columns($self.col_bcast(b, start, nb), nc),
                &x,
                $beta,
            );
        }
    }};
}

impl<T: NalgebraScalar> Matrix for NalgebraMat<T> {
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
    fn set_data_with_indices(
        &mut self,
        dst: &crate::vector::nalgebra_serial::NalgebraIndex,
        src: &crate::vector::nalgebra_serial::NalgebraIndex,
        data: &NalgebraVec<T>,
    ) {
        self.context
            .assert_compatible_nbatch(data.context.nbatch(), "set_data_with_indices");
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
    fn gather(&mut self, o: &Self, idx: &crate::vector::nalgebra_serial::NalgebraIndex) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "gather");
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
    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (
        crate::vector::nalgebra_serial::NalgebraIndex,
        crate::vector::nalgebra_serial::NalgebraIndex,
    ) {
        let (mut z, mut nz) = (Vec::new(), Vec::new());
        for i in 0..self.nrows() {
            if self.data[(i, self.col(0, i))].is_zero() {
                z.push(i)
            } else {
                nz.push(i)
            }
        }
        (
            crate::vector::nalgebra_serial::NalgebraIndex::from_vec(z, self.context),
            crate::vector::nalgebra_serial::NalgebraIndex::from_vec(nz, self.context),
        )
    }
    fn add_column_to_vector(&self, j: usize, v: &mut NalgebraVec<T>) {
        v.context
            .assert_compatible_nbatch(self.context.nbatch(), "add_column_to_vector");
        if self.context.nbatch() == 1 && v.context.nbatch() == 1 {
            v.data
                .column_mut(0)
                .axpy(T::one(), &self.data.column(j), T::one());
            return;
        }
        let nb = v.context.nbatch();
        for b in 0..nb {
            v.data.column_mut(b).axpy(
                T::one(),
                &self.data.column(self.col_bcast(b, j, nb)),
                T::one(),
            );
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
            data: DMatrix::zeros(nr, nc * ctx.nbatch()),
            context: ctx,
        }
    }
    fn from_diagonal(v: &NalgebraVec<T>) -> Self {
        if v.context.nbatch() == 1 {
            return Self {
                data: DMatrix::from_diagonal(&v.data.column(0)),
                context: v.context,
            };
        }
        let mut out = Self::zeros(v.len(), v.len(), v.context);
        for b in 0..v.context.nbatch() {
            out.data
                .columns_mut(out.col(b, 0), out.ncols())
                .set_diagonal(&v.data.column(b));
        }
        out
    }
    fn gemv(&self, a: T, x: &NalgebraVec<T>, beta: T, y: &mut NalgebraVec<T>) {
        gemv_data!(self, a, x, beta, y, "gemv")
    }
    fn copy_from(&mut self, o: &Self) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "copy_from");
        if self.context.nbatch() == o.context.nbatch() {
            self.data.copy_from(&o.data);
            return;
        }
        let nb = self.context.nbatch();
        for b in 0..nb {
            let nc = self.ncols();
            self.data
                .columns_mut(self.col(b, 0), nc)
                .copy_from(&o.data.columns(o.col_bcast(b, 0, nb), nc));
        }
    }
    fn set_column(&mut self, j: usize, v: &NalgebraVec<T>) {
        self.context
            .assert_compatible_nbatch(v.context.nbatch(), "set_column");
        // unbatched fast path: worth ~4x on lin_alg_ops set_column/nalgebra/10
        if self.context.nbatch() == 1 && v.context.nbatch() == 1 {
            self.data.column_mut(j).copy_from(&v.data.column(0));
            return;
        }
        let nb = self.context.nbatch();
        for b in 0..nb {
            self.data
                .column_mut(self.col(b, j))
                .copy_from(&v.data.column(v.batch(b, nb)));
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, b: T, y: &Self) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "scale_add_and_assign");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "scale_add_and_assign");
        let nc = self.ncols();
        let nb = self.context.nbatch();
        for batch in 0..nb {
            let mut columns = self.data.columns_mut(self.col(batch, 0), nc);
            columns.copy_from(&x.data.columns(x.col_bcast(batch, 0, nb), nc));
            columns.zip_apply(&y.data.columns(y.col_bcast(batch, 0, nb), nc), |s, y| {
                *s += b * y;
            });
        }
    }
    fn new_from_sparsity(nr: usize, nc: usize, _: Option<Self::Sparsity>, ctx: Self::C) -> Self {
        Self::zeros(nr, nc, ctx)
    }
}
impl<T: NalgebraScalar> DenseMatrix for NalgebraMat<T> {
    fn resize_cols(&mut self, nc: usize) {
        let old = self.ncols();
        if old == nc {
            return;
        }
        let mut d = DMatrix::zeros(self.nrows(), nc * self.context.nbatch());
        for b in 0..self.context.nbatch() {
            d.columns_mut(b * nc, old.min(nc))
                .copy_from(&self.data.columns(self.col(b, 0), old.min(nc)));
        }
        self.data = d
    }
    fn get_index(&self, i: usize, j: usize) -> T {
        self.data[(i, self.col(0, j))]
    }
    fn gemv_cols(
        &self,
        start: usize,
        end: usize,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut NalgebraVec<T>,
    ) {
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
            data: DMatrix::from_vec(nr, nc * ctx.nbatch(), d),
            context: ctx,
        }
    }
    fn column_mut(&mut self, i: usize) -> NalgebraVecMut<'_, T> {
        if self.context.nbatch() == 1 {
            return NalgebraVecMut {
                data: self.data.columns_mut(i, 1),
                context: self.context,
            };
        }
        let nr = self.nrows();
        let nc = self.ncols();
        let nb = self.context.nbatch();
        assert!(i < nc, "column index out of bounds");
        // column i of every batch: batch b holds it at offset (b * nc + i) * nr
        let data = NaMatrixViewMut::from_slice_with_strides_generic(
            &mut self.data.as_mut_slice()[i * nr..],
            nalgebra::Dyn(nr),
            nalgebra::Dyn(nb),
            Const,
            nalgebra::Dyn(nr * nc),
        );
        NalgebraVecMut {
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
    fn column(&self, i: usize) -> NalgebraVecRef<'_, T> {
        if self.context.nbatch() == 1 {
            return NalgebraVecRef {
                data: self.data.columns(i, 1),
                context: self.context,
            };
        }
        let nr = self.nrows();
        let nc = self.ncols();
        let nb = self.context.nbatch();
        assert!(i < nc, "column index out of bounds");
        // column i of every batch: batch b holds it at offset (b * nc + i) * nr
        let data = NaMatrixView::from_slice_with_strides_generic(
            &self.data.as_slice()[i * nr..],
            nalgebra::Dyn(nr),
            nalgebra::Dyn(nb),
            Const,
            nalgebra::Dyn(nr * nc),
        );
        NalgebraVecRef {
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
        let logical_ncols = self.logical_ncols();
        // A column is contiguous, so a tile of consecutive rows is a contiguous run of each
        // column: stage the tile's old values (the multiply cannot alias its input and
        // output), then overwrite in place.
        let tile = (MUL_COLS_TILE / ncols).max(1);
        let mut buf = [T::zero(); MUL_COLS_TILE];
        let data = self.data.as_mut_slice();
        for b in 0..nbatch {
            let base = b * logical_ncols * nrows;
            let mut row = 0;
            while row < nrows {
                let rows = tile.min(nrows - row);
                for j in 0..ncols {
                    let src = &data[base + j * nrows + row..][..rows];
                    buf[j * rows..][..rows].copy_from_slice(src);
                }
                // One dot product per output element, rather than accumulating a column at a
                // time. Two alternatives were (a) an explicit per-column axpy pass, and (b) handing the staged
                // tile to nalgebra's `gemm`. Both performed worse for the very small rhs matrix
                for j in 0..ncols {
                    let w = &rhs[j * ncols..][..ncols];
                    let dst = base + j * nrows + row;
                    for i in 0..rows {
                        let mut acc = T::zero();
                        for l in 0..ncols {
                            acc += buf[l * rows + i] * w[l];
                        }
                        data[dst + i] = acc;
                    }
                }
                row += rows;
            }
        }
    }

    fn update_backward_diff(&mut self, order: usize, d: &NalgebraVec<T>) {
        assert!(order + 2 < self.logical_ncols(), "order out of bounds");
        self.context
            .assert_compatible_nbatch(d.context.nbatch(), "update_backward_diff");
        let nb = self.context.nbatch();
        for b in 0..nb {
            let base = self.col(b, 0);
            let d_col = d.batch(b, nb);
            // diff[:, order+2] = d - diff[:, order+1]
            {
                let (src, mut dst) = self.data.columns_range_pair_mut(
                    base + order + 1..base + order + 2,
                    base + order + 2..base + order + 3,
                );
                dst.column_mut(0).copy_from(&d.data.column(d_col));
                dst.column_mut(0).axpy(-T::one(), &src.column(0), T::one());
            }
            // for i in (0..=order+1).rev(): diff[:, i] += diff[:, i+1]
            for i in (0..=order + 1).rev() {
                let (mut dst, src) = self
                    .data
                    .columns_range_pair_mut(base + i..base + i + 1, base + i + 1..base + i + 2);
                dst.column_mut(0).axpy(T::one(), &src.column(0), T::one());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_backward_diff() {
        super::super::tests::test_update_backward_diff::<NalgebraMat<f64>>();
    }

    #[test]
    fn test_index_operator() {
        let mut mat =
            NalgebraMat::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0], NalgebraContext::default());
        assert_eq!(mat[(1, 0)], 3.0);
        mat[(1, 0)] = 30.0;
        assert_eq!(mat.get_index(1, 0), 30.0);
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<NalgebraMat<f64>>();
    }

    #[test]
    fn test_resize_cols() {
        super::super::tests::test_resize_cols::<NalgebraMat<f64>>();
    }

    #[test]
    fn test_owned_rhs() {
        super::super::tests::test_owned_rhs_m::<NalgebraMat<f64>>();
    }

    #[test]
    fn test_batched_owned_rhs_broadcast() {
        super::super::tests::test_batched_owned_rhs_broadcast_m::<NalgebraMat<f64>>(
            NalgebraContext::with_nbatch(2),
        );
    }

    super::super::generate_matrix_tests_nonbatched!(nalgebra, NalgebraMat<f64>);
    super::super::generate_matrix_tests_batched!(
        nalgebra,
        NalgebraMat<f64>,
        NalgebraContext::default(),
        NalgebraContext::with_nbatch(2)
    );
    super::super::generate_dense_matrix_tests_nonbatched!(nalgebra, NalgebraMat<f64>);
    super::super::generate_dense_matrix_tests_batched!(
        nalgebra,
        NalgebraMat<f64>,
        NalgebraContext::default(),
        NalgebraContext::with_nbatch(2)
    );
}
