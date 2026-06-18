use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};

use crate::{scalar::Scale, Context, IndexType, NalgebraScalar, Scalar, Vector};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use super::utils::*;
use crate::error::DiffsolError;
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
    pub(crate) data: DMatrixView<'a, T>,
    pub(crate) context: NalgebraContext,
    pub(crate) batch_stride: usize,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraMatMut<'a, T: NalgebraScalar> {
    pub(crate) data: DMatrixViewMut<'a, T>,
    pub(crate) context: NalgebraContext,
    pub(crate) batch_stride: usize,
}

impl<T: NalgebraScalar> DefaultSolver for NalgebraMat<T> {
    type LS = NalgebraLU<T>;
}

impl<'a, T: NalgebraScalar> MatrixCommon for NalgebraMatMut<'a, T> {
    type T = T;
    type V = NalgebraVec<T>;
    type C = NalgebraContext;
    type Inner = DMatrixViewMut<'a, T>;

    fn nrows(&self) -> IndexType {
        self.data.nrows()
    }
    fn ncols(&self) -> IndexType {
        self.data.ncols() - (Context::nbatch(&self.context) - 1) * self.batch_stride
    }
    fn inner(&self) -> &Self::Inner {
        &self.data
    }
}

impl<'a, T: NalgebraScalar> MatrixCommon for NalgebraMatRef<'a, T> {
    type T = T;
    type V = NalgebraVec<T>;
    type C = NalgebraContext;
    type Inner = DMatrixView<'a, T>;

    fn nrows(&self) -> IndexType {
        self.data.nrows()
    }
    fn ncols(&self) -> IndexType {
        self.data.ncols() - (Context::nbatch(&self.context) - 1) * self.batch_stride
    }
    fn inner(&self) -> &Self::Inner {
        &self.data
    }
}
impl_matrix_common!(
    NalgebraMat<T>,
    NalgebraVec<T>,
    NalgebraContext,
    DMatrix<T>,
    NalgebraScalar
);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: NalgebraScalar> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale = rhs.value();
                Self::Output {
                    data: &self.data * scale,
                    context: self.context,
                }
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($mat_type:ty) => {
        impl<T: NalgebraScalar> MulAssign<Scale<T>> for $mat_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale = rhs.value();
                self.data *= scale;
            }
        }
    };
}

impl_mul_scalar!(NalgebraMatRef<'_, T>, NalgebraMat<T>);
impl_mul_scalar!(NalgebraMat<T>, NalgebraMat<T>);
impl_mul_scalar!(&NalgebraMat<T>, NalgebraMat<T>);

impl_mul_assign_scalar!(NalgebraMatMut<'_, T>);

impl_add!(
    NalgebraMat<T>,
    &NalgebraMat<T>,
    NalgebraMat<T>,
    NalgebraScalar
);
impl_add!(
    NalgebraMat<T>,
    &NalgebraMatRef<'_, T>,
    NalgebraMat<T>,
    NalgebraScalar
);
impl_add!(
    NalgebraMatRef<'_, T>,
    &NalgebraMat<T>,
    NalgebraMat<T>,
    NalgebraScalar
);

impl_sub!(
    NalgebraMat<T>,
    &NalgebraMat<T>,
    NalgebraMat<T>,
    NalgebraScalar
);
impl_sub!(
    NalgebraMat<T>,
    &NalgebraMatRef<'_, T>,
    NalgebraMat<T>,
    NalgebraScalar
);
impl_sub!(
    NalgebraMatRef<'_, T>,
    &NalgebraMat<T>,
    NalgebraMat<T>,
    NalgebraScalar
);

impl_add_assign!(NalgebraMat<T>, &NalgebraMat<T>, NalgebraScalar);
impl_add_assign!(NalgebraMat<T>, &NalgebraMatRef<'_, T>, NalgebraScalar);
impl_add_assign!(
    NalgebraMatMut<'_, T>,
    &NalgebraMatRef<'_, T>,
    NalgebraScalar
);
impl_add_assign!(
    NalgebraMatMut<'_, T>,
    &NalgebraMatMut<'_, T>,
    NalgebraScalar
);

impl_sub_assign!(NalgebraMat<T>, &NalgebraMat<T>, NalgebraScalar);
impl_sub_assign!(NalgebraMat<T>, &NalgebraMatRef<'_, T>, NalgebraScalar);
impl_sub_assign!(
    NalgebraMatMut<'_, T>,
    &NalgebraMatRef<'_, T>,
    NalgebraScalar
);
impl_sub_assign!(
    NalgebraMatMut<'_, T>,
    &NalgebraMatMut<'_, T>,
    NalgebraScalar
);

impl_index!(NalgebraMat<T>, NalgebraScalar);
impl_index!(NalgebraMatRef<'_, T>, NalgebraScalar);
impl_index_mut!(NalgebraMat<T>, NalgebraScalar);

impl<'a, T: NalgebraScalar> MatrixView<'a> for NalgebraMatRef<'a, T> {
    type Owned = NalgebraMat<T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.into_owned(),
            context: self.context,
        }
    }

    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let stride = self.batch_stride;
        let x_nbatch = x.data.ncols();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv_v");
        for b in 0..nbatch {
            let a_view = self.data.columns(b * stride, ncols);
            let x_col = if x_nbatch == 1 {
                x.data.column(0)
            } else {
                x.data.column(b)
            };
            let mut y_col = y.data.column_mut(b);
            y_col.gemv(alpha, &a_view, &x_col, beta);
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let stride = self.batch_stride;
        let x_nbatch = x.data.ncols();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv_o");
        for b in 0..nbatch {
            let a_view = self.data.columns(b * stride, ncols);
            let x_col = if x_nbatch == 1 {
                x.data.column(0)
            } else {
                x.data.column(b)
            };
            let mut y_col = y.data.column_mut(b);
            y_col.gemv(alpha, &a_view, &x_col, beta);
        }
    }
}

impl<'a, T: NalgebraScalar> MatrixViewMut<'a> for NalgebraMatMut<'a, T> {
    type Owned = NalgebraMat<T>;
    type View = NalgebraMatRef<'a, T>;
    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.into_owned(),
            context: self.context,
        }
    }
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let self_ncols = self.ncols();
        let self_stride = self.batch_stride;
        let a_ncols = a.ncols();
        let a_nbatch = a.context.nbatch();
        let b_ncols = b.ncols();
        let b_nbatch = b.context.nbatch();
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_oo_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_oo_b");
        for bi in 0..nbatch {
            let mut self_view = self.data.columns_mut(bi * self_stride, self_ncols);
            let a_view = if a_nbatch == 1 {
                a.data.columns(0, a_ncols)
            } else {
                a.data.columns(bi * a_ncols, a_ncols)
            };
            let b_view = if b_nbatch == 1 {
                b.data.columns(0, b_ncols)
            } else {
                b.data.columns(bi * b_ncols, b_ncols)
            };
            self_view.gemm(alpha, &a_view, &b_view, beta);
        }
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let self_ncols = self.ncols();
        let self_stride = self.batch_stride;
        let a_ncols = a.ncols();
        let a_stride = a.batch_stride;
        let a_nbatch = a.context.nbatch();
        let b_ncols = b.ncols();
        let b_nbatch = b.context.nbatch();
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_vo_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_vo_b");
        for bi in 0..nbatch {
            let mut self_view = self.data.columns_mut(bi * self_stride, self_ncols);
            let a_view = if a_nbatch == 1 {
                a.data.columns(0, a_ncols)
            } else {
                a.data.columns(bi * a_stride, a_ncols)
            };
            let b_view = if b_nbatch == 1 {
                b.data.columns(0, b_ncols)
            } else {
                b.data.columns(bi * b_ncols, b_ncols)
            };
            self_view.gemm(alpha, &a_view, &b_view, beta);
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

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        let nbatch = self.context.nbatch();
        let nrows = self.nrows();
        let ncols = self.ncols();
        for b in 0..nbatch {
            for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
                let i = dst_i % nrows;
                let j = dst_i / nrows;
                self.data[(i, b * ncols + j)] = data.data[(*src_i, b)];
            }
        }
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        let nrows = self.nrows();
        let other_nrows = other.nrows();
        let ncols = self.ncols();
        let other_ncols = other.ncols();
        let nbatch = self.context.nbatch();
        assert_eq!(indices.len(), nrows * ncols);
        for b in 0..nbatch {
            let mut idx = indices.data.iter().peekable();
            for j in 0..ncols {
                let other_col_idx = *idx.peek().unwrap() / other_nrows;
                for i in 0..nrows {
                    let other_i = idx.next().unwrap() % other_nrows;
                    self.data[(i, b * ncols + j)] =
                        other.data[(other_i, b * other_ncols + other_col_idx)];
                }
            }
        }
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let mut zero_diagonal_indices = Vec::new();
        let mut non_zero_diagonal_indices = Vec::new();
        for i in 0..self.nrows() {
            if self.data[(i, i)].is_zero() {
                zero_diagonal_indices.push(i);
            } else {
                non_zero_diagonal_indices.push(i);
            }
        }
        (
            <Self::V as Vector>::Index::from_vec(zero_diagonal_indices, self.context),
            <Self::V as Vector>::Index::from_vec(non_zero_diagonal_indices, self.context),
        )
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        v.add_assign(&self.column(j));
    }

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let ncols = self.ncols();
        let nrows = self.nrows();
        let nbatch = self.context.nbatch();
        let indices = (0..ncols).flat_map(move |j| (0..nrows).map(move |i| (i, j)));
        let values = (0..nbatch).flat_map(move |b| {
            (0..ncols).flat_map(move |j| (0..nrows).map(move |i| self.data[(i, b * ncols + j)]))
        });
        (indices, values)
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
        values: Vec<T>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let nbatch = ctx.nbatch();
        let nnz = indices.len();
        assert_eq!(values.len(), nnz * nbatch);
        let mut m = DMatrix::zeros(nrows, ncols * nbatch);
        for b in 0..nbatch {
            for (k, &(i, j)) in indices.iter().enumerate() {
                m[(i, b * ncols + j)] = values[b * nnz + k];
            }
        }
        Ok(Self {
            data: m,
            context: ctx,
        })
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = DMatrix::zeros(nrows, ncols * nbatch);
        Self { data, context: ctx }
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let nbatch = v.context().nbatch();
        let nstates = v.len();
        let mut data = DMatrix::zeros(nstates, nstates * nbatch);
        for b in 0..nbatch {
            for i in 0..nstates {
                data[(i, b * nstates + i)] = v.data[(i, b)];
            }
        }
        Self {
            data,
            context: *v.context(),
        }
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let x_nbatch = x.data.ncols();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv");
        for b in 0..nbatch {
            let a_view = self.data.columns(b * ncols, ncols);
            let x_col = if x_nbatch == 1 {
                x.data.column(0)
            } else {
                x.data.column(b)
            };
            let mut y_col = y.data.column_mut(b);
            y_col.gemv(alpha, &a_view, &x_col, beta);
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        for b in 0..nbatch {
            self.data
                .column_mut(b * ncols + j)
                .copy_from(&v.data.column(b));
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        self.copy_from(y);
        self.data.mul_assign(beta);
        self.add_assign(x);
    }
    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        _sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self {
        Self::zeros(nrows, ncols, ctx)
    }
}

impl<T: NalgebraScalar> DenseMatrix for NalgebraMat<T> {
    type View<'a> = NalgebraMatRef<'a, T>;
    type ViewMut<'a> = NalgebraMatMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        let nbatch = self.context.nbatch();
        if nbatch == 1 {
            self.data.gemm(alpha, &a.data, &b.data, beta);
            return;
        }
        let self_ncols = self.ncols();
        let a_ncols = a.ncols();
        let a_nbatch = a.context.nbatch();
        let b_ncols = b.ncols();
        let b_nbatch = b.context.nbatch();
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_b");
        for bi in 0..nbatch {
            let mut self_view = self.data.columns_mut(bi * self_ncols, self_ncols);
            let a_view = if a_nbatch == 1 {
                a.data.columns(0, a_ncols)
            } else {
                a.data.columns(bi * a_ncols, a_ncols)
            };
            let b_view = if b_nbatch == 1 {
                b.data.columns(0, b_ncols)
            } else {
                b.data.columns(bi * b_ncols, b_ncols)
            };
            self_view.gemm(alpha, &a_view, &b_view, beta);
        }
    }

    fn resize_cols(&mut self, ncols: IndexType) {
        let old_ncols = self.ncols();
        if ncols == old_ncols {
            return;
        }
        let nbatch = self.context.nbatch();
        if nbatch == 1 {
            self.data.resize_horizontally_mut(ncols, Self::T::zero());
            return;
        }
        let nrows = self.nrows();
        let copy_ncols = ncols.min(old_ncols);
        let mut new_data = DMatrix::zeros(nrows, ncols * nbatch);
        for b in 0..nbatch {
            let old_start = b * old_ncols;
            let new_start = b * ncols;
            for j in 0..copy_ncols {
                new_data
                    .column_mut(new_start + j)
                    .copy_from(&self.data.column(old_start + j));
            }
        }
        self.data = new_data;
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[(i, j)]
    }

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = DMatrix::from_vec(nrows, ncols * nbatch, data);
        Self { data, context: ctx }
    }

    fn column_mut(&mut self, j: IndexType) -> <Self::V as Vector>::ViewMut<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let data = self.data.columns_with_step_mut(j, nbatch, ncols - 1);
        NalgebraVecMut {
            data,
            context: self.context,
        }
    }

    fn columns_mut(&mut self, start: IndexType, end: IndexType) -> Self::ViewMut<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let sub_width = end - start;
        let total_raw = (nbatch - 1) * ncols + sub_width;
        let data = self.data.columns_mut(start, total_raw);
        NalgebraMatMut {
            data,
            context: self.context,
            batch_stride: ncols,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.data[(i, j)] = value;
    }

    fn column(&self, j: IndexType) -> <Self::V as Vector>::View<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let data = self.data.columns_with_step(j, nbatch, ncols - 1);
        NalgebraVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, start: IndexType, end: IndexType) -> Self::View<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let sub_width = end - start;
        let total_raw = (nbatch - 1) * ncols + sub_width;
        let data = self.data.columns(start, total_raw);
        NalgebraMatRef {
            data,
            context: self.context,
            batch_stride: ncols,
        }
    }
    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, i: IndexType) {
        if i > self.ncols() {
            panic!("Column index out of bounds");
        }
        if j > self.ncols() {
            panic!("Column index out of bounds");
        }
        if i == j {
            panic!("Column index cannot be the same");
        }
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let nrows = self.nrows();
        for b in 0..nbatch {
            for k in 0..nrows {
                let ci = b * ncols + i;
                let cj = b * ncols + j;
                let value = unsafe {
                    *self.data.get_unchecked((k, ci)) + alpha * *self.data.get_unchecked((k, cj))
                };
                unsafe {
                    *self.data.get_unchecked_mut((k, ci)) = value;
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    super::super::generate_matrix_tests!(
        nalgebra,
        NalgebraMat<f64>,
        NalgebraContext::default(),
        NalgebraContext::with_nbatch(2)
    );
}
