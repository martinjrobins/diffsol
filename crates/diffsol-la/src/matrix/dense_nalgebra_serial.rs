use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{Const, DMatrix, Dyn, MatrixView as NaMatrixView, MatrixViewMut as NaMatrixViewMut};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use crate::error::LaError;
use crate::{scalar::Scale, Context, NalgebraScalar, Vector};
use crate::{
    DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut, NalgebraContext, NalgebraLU,
    NalgebraVec, NalgebraVecMut, NalgebraVecRef, VectorIndex,
};

#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMat<T: NalgebraScalar> {
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}
#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMatRef<'a, T: NalgebraScalar> {
    pub(crate) data: NaMatrixView<'a, T, Dyn, Dyn, Const<1>, Dyn>,
    pub(crate) context: NalgebraContext,
    pub(crate) start: usize,
    pub(crate) end: usize,
}
#[derive(Debug, PartialEq)]
pub struct NalgebraMatMut<'a, T: NalgebraScalar> {
    pub(crate) data: NaMatrixViewMut<'a, T, Dyn, Dyn, Const<1>, Dyn>,
    pub(crate) context: NalgebraContext,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

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
impl<T: NalgebraScalar> NalgebraMat<T> {
    fn logical_ncols(&self) -> usize {
        self.data.ncols() / self.context.nbatch()
    }
    fn col(&self, b: usize, j: usize) -> usize {
        (b % self.context.nbatch()) * self.logical_ncols() + j
    }
}
impl<T: NalgebraScalar> NalgebraMatRef<'_, T> {
    mat_methods!();
}
impl<T: NalgebraScalar> NalgebraMatMut<'_, T> {
    mat_methods!();
}
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
macro_rules! common_ref {
    ($t:ty,$inner:ty) => {
        impl<'a, T: NalgebraScalar> MatrixCommon for $t {
            type T = T;
            type V = NalgebraVec<T>;
            type C = NalgebraContext;
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
common_ref!(
    NalgebraMatRef<'a, T>,
    NaMatrixView<'a, T, Dyn, Dyn, Const<1>, Dyn>
);
common_ref!(
    NalgebraMatMut<'a, T>,
    NaMatrixViewMut<'a, T, Dyn, Dyn, Const<1>, Dyn>
);

macro_rules! binary {
    ($tr:ident, $fn:ident, $l:ty, $r:ty, $op:tt) => {
        impl<T: NalgebraScalar> $tr<$r> for $l {
            type Output = NalgebraMat<T>;

            fn $fn(self, rhs: $r) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                let nb = self.context.nbatch().max(rhs.context.nbatch());
                let nc = self.ncols();
                let mut data = DMatrix::zeros(self.nrows(), nc * nb);
                for b in 0..nb {
                    let mut columns = data.columns_mut(b * nc, nc);
                    columns.copy_from(&self.data.columns(self.col(b, 0), nc));
                    columns $op &rhs.data.columns(rhs.col(b, 0), nc);
                }
                NalgebraMat {
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
binary!(Add, add, NalgebraMat<T>, &NalgebraMat<T>, +=);
binary!(Add, add, NalgebraMat<T>, &NalgebraMatRef<'_, T>, +=);
binary!(Add, add, NalgebraMatRef<'_, T>, &NalgebraMat<T>, +=);
binary!(Sub, sub, NalgebraMat<T>, &NalgebraMat<T>, -=);
binary!(Sub, sub, NalgebraMat<T>, &NalgebraMatRef<'_, T>, -=);
binary!(Sub, sub, NalgebraMatRef<'_, T>, &NalgebraMat<T>, -=);
macro_rules! assign {
    ($tr:ident, $fn:ident, $l:ty, $r:ty, $op:tt) => {
        impl<T: NalgebraScalar> $tr<$r> for $l {
            fn $fn(&mut self, rhs: $r) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($fn));
                for b in 0..self.context.nbatch() {
                    let nc = self.ncols();
                    let mut columns = self.data.columns_mut(self.col(b, 0), nc);
                    columns $op &rhs.data.columns(rhs.col(b, 0), nc);
                }
            }
        }
    };
}
assign!(AddAssign, add_assign, NalgebraMat<T>, &NalgebraMat<T>, +=);
assign!(AddAssign, add_assign, NalgebraMat<T>, &NalgebraMatRef<'_, T>, +=);
assign!(
    AddAssign,
    add_assign,
    NalgebraMatMut<'_, T>,
    &NalgebraMatRef<'_, T>,
    +=
);
assign!(
    AddAssign,
    add_assign,
    NalgebraMatMut<'_, T>,
    &NalgebraMatMut<'_, T>,
    +=
);
assign!(SubAssign, sub_assign, NalgebraMat<T>, &NalgebraMat<T>, -=);
assign!(SubAssign, sub_assign, NalgebraMat<T>, &NalgebraMatRef<'_, T>, -=);
assign!(
    SubAssign,
    sub_assign,
    NalgebraMatMut<'_, T>,
    &NalgebraMatRef<'_, T>,
    -=
);
assign!(
    SubAssign,
    sub_assign,
    NalgebraMatMut<'_, T>,
    &NalgebraMatMut<'_, T>,
    -=
);
macro_rules! scale {
    ($t:ty) => {
        impl<T: NalgebraScalar> Mul<Scale<T>> for $t {
            type Output = NalgebraMat<T>;
            fn mul(self, r: Scale<T>) -> Self::Output {
                let mut out = NalgebraMat::zeros(self.nrows(), self.ncols(), self.context);
                for b in 0..out.context.nbatch() {
                    let nc = out.ncols();
                    out.data
                        .columns_mut(out.col(b, 0), nc)
                        .copy_from(&self.data.columns(self.col(b, 0), nc));
                    out.data.columns_mut(out.col(b, 0), nc).scale_mut(r.value());
                }
                out
            }
        }
    };
}
scale!(NalgebraMat<T>);
scale!(&NalgebraMat<T>);
scale!(NalgebraMatRef<'_, T>);
impl<T: NalgebraScalar> MulAssign<Scale<T>> for NalgebraMatMut<'_, T> {
    fn mul_assign(&mut self, r: Scale<T>) {
        for b in 0..self.context.nbatch() {
            let nc = self.ncols();
            self.data
                .columns_mut(self.col(b, 0), nc)
                .scale_mut(r.value());
        }
    }
}
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
ind!(NalgebraMatRef<'_, T>);
impl<T: NalgebraScalar> IndexMut<(usize, usize)> for NalgebraMat<T> {
    fn index_mut(&mut self, x: (usize, usize)) -> &mut T {
        let c = self.col(0, x.1);
        &mut self.data[(x.0, c)]
    }
}

impl<'a, T: NalgebraScalar> MatrixView<'a> for NalgebraMatRef<'a, T> {
    type Owned = NalgebraMat<T>;
    fn into_owned(self) -> Self::Owned {
        let mut out = NalgebraMat::zeros(self.nrows(), self.ncols(), self.context);
        for b in 0..self.context.nbatch() {
            let nc = self.ncols();
            out.data
                .columns_mut(out.col(b, 0), nc)
                .copy_from(&self.data.columns(self.col(b, 0), nc));
        }
        out
    }
    fn gemv_v(&self, a: T, x: &NalgebraVecRef<'_, T>, beta: T, y: &mut NalgebraVec<T>) {
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv_v");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv_v");
        for b in 0..y.context.nbatch() {
            y.data.column_mut(b).gemv(
                a,
                &self.data.columns(self.col(b, 0), self.ncols()),
                &x.data.column(b % x.data.ncols()),
                beta,
            );
        }
    }
    fn gemv_o(&self, a: T, x: &NalgebraVec<T>, beta: T, y: &mut NalgebraVec<T>) {
        self.gemv_v(a, &x.as_view(), beta, y)
    }
}
impl<'a, T: NalgebraScalar> MatrixViewMut<'a> for NalgebraMatMut<'a, T> {
    type Owned = NalgebraMat<T>;
    type View = NalgebraMatRef<'a, T>;
    fn into_owned(self) -> Self::Owned {
        NalgebraMatRef {
            data: self.data.as_view(),
            context: self.context,
            start: self.start,
            end: self.end,
        }
        .into_owned()
    }
    fn gemm_oo(&mut self, a: T, x: &Self::Owned, y: &Self::Owned, beta: T) {
        self.gemm(a, &x.columns(0, x.ncols()), &y.columns(0, y.ncols()), beta)
    }
    fn gemm_vo(&mut self, a: T, x: &Self::View, y: &Self::Owned, beta: T) {
        self.gemm(a, x, &y.columns(0, y.ncols()), beta)
    }
}
impl<T: NalgebraScalar> NalgebraMatMut<'_, T> {
    fn gemm(&mut self, a: T, x: &NalgebraMatRef<'_, T>, y: &NalgebraMatRef<'_, T>, beta: T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemm");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "gemm");
        for b in 0..self.context.nbatch() {
            let nc = self.ncols();
            self.data.columns_mut(self.col(b, 0), nc).gemm(
                a,
                &x.data.columns(x.col(b, 0), x.ncols()),
                &y.data.columns(y.col(b, 0), y.ncols()),
                beta,
            );
        }
    }
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
        for (d, s) in dst.data.iter().zip(src.data.iter()) {
            let i = d % self.nrows();
            let j = d / self.nrows();
            for b in 0..self.context.nbatch() {
                let c = self.col(b, j);
                self.data[(i, c)] = data.data[(*s, b % data.data.ncols())]
            }
        }
    }
    fn gather(&mut self, o: &Self, idx: &crate::vector::nalgebra_serial::NalgebraIndex) {
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
        for b in 0..v.context.nbatch() {
            v.data
                .column_mut(b)
                .axpy(T::one(), &self.data.column(self.col(b, j)), T::one());
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
        let mut out = Self::zeros(v.len(), v.len(), v.context);
        for b in 0..v.context.nbatch() {
            out.data
                .columns_mut(out.col(b, 0), out.ncols())
                .set_diagonal(&v.data.column(b));
        }
        out
    }
    fn gemv(&self, a: T, x: &NalgebraVec<T>, beta: T, y: &mut NalgebraVec<T>) {
        self.columns(0, self.ncols()).gemv_o(a, x, beta, y)
    }
    fn copy_from(&mut self, o: &Self) {
        for b in 0..self.context.nbatch() {
            let nc = self.ncols();
            self.data
                .columns_mut(self.col(b, 0), nc)
                .copy_from(&o.data.columns(o.col(b, 0), nc));
        }
    }
    fn set_column(&mut self, j: usize, v: &NalgebraVec<T>) {
        for b in 0..self.context.nbatch() {
            self.data
                .column_mut(self.col(b, j))
                .copy_from(&v.data.column(b % v.data.ncols()));
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, b: T, y: &Self) {
        self.data.copy_from(&x.data);
        self.data += y.data.clone() * b;
    }
    fn new_from_sparsity(nr: usize, nc: usize, _: Option<Self::Sparsity>, ctx: Self::C) -> Self {
        Self::zeros(nr, nc, ctx)
    }
}
impl<T: NalgebraScalar> DenseMatrix for NalgebraMat<T> {
    type View<'a> = NalgebraMatRef<'a, T>;
    type ViewMut<'a> = NalgebraMatMut<'a, T>;
    fn gemm(&mut self, a: T, x: &Self, y: &Self, b: T) {
        self.columns_mut(0, self.ncols()).gemm_oo(a, x, y, b)
    }
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
        let data = unsafe {
            NaMatrixViewMut::from_slice_with_strides_generic_unchecked(
                self.data.as_mut_slice(),
                i * nr,
                nalgebra::Dyn(nr),
                nalgebra::Dyn(nb),
                Const,
                nalgebra::Dyn(nr * nc),
            )
        };
        NalgebraVecMut {
            data,
            context: self.context,
        }
    }
    fn columns_mut(&mut self, s: usize, e: usize) -> Self::ViewMut<'_> {
        NalgebraMatMut {
            data: self.data.as_view_mut(),
            context: self.context,
            start: s,
            end: e,
        }
    }
    fn set_index(&mut self, i: usize, j: usize, v: T) {
        for b in 0..self.context.nbatch() {
            self.data.column_mut(self.col(b, j)).row_mut(i).fill(v);
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
        let data = unsafe {
            NaMatrixView::from_slice_with_strides_generic_unchecked(
                self.data.as_slice(),
                i * nr,
                nalgebra::Dyn(nr),
                nalgebra::Dyn(nb),
                Const,
                nalgebra::Dyn(nr * nc),
            )
        };
        NalgebraVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, s: usize, e: usize) -> Self::View<'_> {
        NalgebraMatRef {
            data: self.data.as_view(),
            context: self.context,
            start: s,
            end: e,
        }
    }
    fn column_axpy(&mut self, a: T, j: usize, i: usize) {
        assert_ne!(i, j, "Column index cannot be the same");
        assert!(i < self.ncols(), "Column index out of bounds");
        assert!(j < self.ncols(), "Column index out of bounds");
        for b in 0..self.context.nbatch() {
            let src = self.col(b, j);
            let dst = self.col(b, i);
            if src < dst {
                let (src, mut dst) = self.data.columns_range_pair_mut(src..src + 1, dst..dst + 1);
                dst.column_mut(0).axpy(a, &src.column(0), T::one());
            } else {
                let (mut dst, src) = self.data.columns_range_pair_mut(dst..dst + 1, src..src + 1);
                dst.column_mut(0).axpy(a, &src.column(0), T::one());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_axpy() {
        super::super::tests::test_column_axpy::<NalgebraMat<f64>>();
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<NalgebraMat<f64>>();
    }

    #[test]
    fn test_resize_cols() {
        super::super::tests::test_resize_cols::<NalgebraMat<f64>>();
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
