use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::default_solver::DefaultSolver;
use super::utils::*;
use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::error::DiffsolError;
use crate::scalar::{IndexType, Scalar, Scale};
use crate::VectorIndex;
use crate::{Dense, DenseRef, FaerContext, FaerScalar, FaerVec, Vector, VectorViewMut};
use crate::{FaerLU, FaerVecMut, FaerVecRef};

use faer::reborrow::ReborrowMut;
use faer::{linalg::matmul::matmul, Mat, MatMut, MatRef};
use faer::{unzip, zip, Accum};

use crate::Context;

#[derive(Clone, Debug, PartialEq)]
pub struct FaerMat<T: FaerScalar> {
    pub(crate) data: Mat<T>,
    pub(crate) context: FaerContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaerMatRef<'a, T: FaerScalar> {
    pub(crate) data: MatRef<'a, T>,
    pub(crate) context: FaerContext,
    pub(crate) batch_stride: usize,
}

#[derive(Debug, PartialEq)]
pub struct FaerMatMut<'a, T: FaerScalar> {
    pub(crate) data: MatMut<'a, T>,
    pub(crate) context: FaerContext,
    pub(crate) batch_stride: usize,
}

impl<T: FaerScalar> DefaultSolver for FaerMat<T> {
    type LS = FaerLU<T>;
}

impl<'a, T: Scalar + FaerScalar> MatrixCommon for FaerMatMut<'a, T> {
    type T = T;
    type V = FaerVec<T>;
    type C = FaerContext;
    type Inner = MatMut<'a, T>;

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

impl<'a, T: Scalar + FaerScalar> MatrixCommon for FaerMatRef<'a, T> {
    type T = T;
    type V = FaerVec<T>;
    type C = FaerContext;
    type Inner = MatRef<'a, T>;

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

impl_matrix_common!(FaerMat<T>, FaerVec<T>, FaerContext, Mat<T>, FaerScalar);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: FaerScalar> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
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
        impl<T: FaerScalar> MulAssign<Scale<T>> for $mat_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale: faer::Scale<T> = rhs.into();
                self.data *= scale;
            }
        }
    };
}

impl_mul_scalar!(FaerMatRef<'_, T>, FaerMat<T>);
impl_mul_scalar!(FaerMat<T>, FaerMat<T>);
impl_mul_scalar!(&FaerMat<T>, FaerMat<T>);

impl_mul_assign_scalar!(FaerMatMut<'_, T>);

impl_add!(FaerMat<T>, &FaerMat<T>, FaerMat<T>, FaerScalar);
impl_add!(FaerMat<T>, &FaerMatRef<'_, T>, FaerMat<T>, FaerScalar);
impl_add!(FaerMatRef<'_, T>, &FaerMat<T>, FaerMat<T>, FaerScalar);

impl_sub!(FaerMat<T>, &FaerMat<T>, FaerMat<T>, FaerScalar);
impl_sub!(FaerMat<T>, &FaerMatRef<'_, T>, FaerMat<T>, FaerScalar);
impl_sub!(FaerMatRef<'_, T>, &FaerMat<T>, FaerMat<T>, FaerScalar);

impl_add_assign!(FaerMat<T>, &FaerMat<T>, FaerScalar);
impl_add_assign!(FaerMat<T>, &FaerMatRef<'_, T>, FaerScalar);
impl_add_assign!(FaerMatMut<'_, T>, &FaerMatRef<'_, T>, FaerScalar);
impl_add_assign!(FaerMatMut<'_, T>, &FaerMatMut<'_, T>, FaerScalar);

impl_sub_assign!(FaerMat<T>, &FaerMat<T>, FaerScalar);
impl_sub_assign!(FaerMat<T>, &FaerMatRef<'_, T>, FaerScalar);
impl_sub_assign!(FaerMatMut<'_, T>, &FaerMatRef<'_, T>, FaerScalar);
impl_sub_assign!(FaerMatMut<'_, T>, &FaerMatMut<'_, T>, FaerScalar);

impl_index!(FaerMat<T>, FaerScalar);
impl_index!(FaerMatRef<'_, T>, FaerScalar);
impl_index_mut!(FaerMat<T>, FaerScalar);

impl<'a, T: FaerScalar> MatrixView<'a> for FaerMatRef<'a, T> {
    type Owned = FaerMat<T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.to_owned(),
            context: self.context,
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let self_nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let stride = self.batch_stride;
        let x_nbatch = x.data.ncols();
        let max_nbatch = self_nbatch.max(x_nbatch);
        self.context.assert_compatible_nbatch(x_nbatch, "gemv_o");
        for b in 0..max_nbatch {
            let a_view = if self_nbatch == 1 {
                self.data.get(0..self.nrows(), 0..ncols)
            } else {
                self.data.get(0..self.nrows(), b * stride..b * stride + ncols)
            };
            let x_col = if x_nbatch == 1 {
                x.data.col(0)
            } else {
                x.data.col(b)
            };
            let mut y_col = y.data.col_mut(b);
            y_col *= faer::Scale(beta);
            matmul(
                y_col.as_mat_mut(),
                Accum::Add,
                a_view,
                x_col.as_mat(),
                alpha,
                self.context.par,
            );
        }
    }
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        let self_nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let stride = self.batch_stride;
        let x_nbatch = x.data.ncols();
        let max_nbatch = self_nbatch.max(x_nbatch);
        self.context.assert_compatible_nbatch(x_nbatch, "gemv_v");
        for b in 0..max_nbatch {
            let a_view = if self_nbatch == 1 {
                self.data.get(0..self.nrows(), 0..ncols)
            } else {
                self.data.get(0..self.nrows(), b * stride..b * stride + ncols)
            };
            let x_col = if x_nbatch == 1 {
                x.data.col(0)
            } else {
                x.data.col(b)
            };
            let mut y_col = y.data.col_mut(b);
            y_col *= faer::Scale(beta);
            matmul(
                y_col.as_mat_mut(),
                Accum::Add,
                a_view,
                x_col.as_mat(),
                alpha,
                self.context.par,
            );
        }
    }
}

impl<'a, T: FaerScalar> MatrixViewMut<'a> for FaerMatMut<'a, T> {
    type Owned = FaerMat<T>;
    type View = FaerMatRef<'a, T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.to_owned(),
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
            let mut self_view = self.data.rb_mut().get_mut(
                0..a.nrows(),
                bi * self_stride..bi * self_stride + self_ncols,
            );
            let a_view = if a_nbatch == 1 {
                a.data.get(0..a.nrows(), 0..a_ncols)
            } else {
                a.data
                    .get(0..a.nrows(), bi * a_ncols..bi * a_ncols + a_ncols)
            };
            let b_view = if b_nbatch == 1 {
                b.data.get(0..b.nrows(), 0..b_ncols)
            } else {
                b.data
                    .get(0..b.nrows(), bi * b_ncols..bi * b_ncols + b_ncols)
            };
            self_view *= faer::Scale(beta);
            matmul(
                self_view,
                Accum::Add,
                a_view,
                b_view,
                alpha,
                self.context.par,
            );
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
            let mut self_view = self.data.rb_mut().get_mut(
                0..a.nrows(),
                bi * self_stride..bi * self_stride + self_ncols,
            );
            let a_view = if a_nbatch == 1 {
                a.data.get(0..a.nrows(), 0..a_ncols)
            } else {
                a.data
                    .get(0..a.nrows(), bi * a_stride..bi * a_stride + a_ncols)
            };
            let b_view = if b_nbatch == 1 {
                b.data.get(0..b.nrows(), 0..b_ncols)
            } else {
                b.data
                    .get(0..b.nrows(), bi * b_ncols..bi * b_ncols + b_ncols)
            };
            self_view *= faer::Scale(beta);
            matmul(
                self_view,
                Accum::Add,
                a_view,
                b_view,
                alpha,
                self.context.par,
            );
        }
    }
}

impl<T: FaerScalar> DenseMatrix for FaerMat<T> {
    type View<'a> = FaerMatRef<'a, T>;
    type ViewMut<'a> = FaerMatMut<'a, T>;

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = Mat::from_fn(nrows, ncols * nbatch, |i, j| data[i + j * nrows]);
        Self { data, context: ctx }
    }

    fn resize_cols(&mut self, ncols: IndexType) {
        let old_ncols = self.ncols();
        if ncols == old_ncols {
            return;
        }
        let nbatch = self.context.nbatch();
        if nbatch == 1 {
            self.data.resize_with(self.nrows(), ncols, |_, _| T::zero());
            return;
        }
        let nrows = self.nrows();
        let copy_ncols = ncols.min(old_ncols);
        let mut new_data = Mat::zeros(nrows, ncols * nbatch);
        for b in 0..nbatch {
            new_data
                .get_mut(0..nrows, b * ncols..b * ncols + copy_ncols)
                .copy_from(
                    self.data
                        .get(0..nrows, b * old_ncols..b * old_ncols + copy_ncols),
                );
        }
        self.data = new_data;
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[(i, j)]
    }

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        let nbatch = self.context.nbatch();
        if nbatch == 1 {
            self.data.mul_assign(faer::Scale(beta));
            matmul(
                self.data.as_mut(),
                Accum::Add,
                a.data.as_ref(),
                b.data.as_ref(),
                alpha,
                self.context.par,
            );
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
            let mut self_view = self.data.get_mut(
                0..self.nrows(),
                bi * self_ncols..bi * self_ncols + self_ncols,
            );
            let a_view = if a_nbatch == 1 {
                a.data.get(0..a.nrows(), 0..a_ncols)
            } else {
                a.data
                    .get(0..a.nrows(), bi * a_ncols..bi * a_ncols + a_ncols)
            };
            let b_view = if b_nbatch == 1 {
                b.data.get(0..b.nrows(), 0..b_ncols)
            } else {
                b.data
                    .get(0..b.nrows(), bi * b_ncols..bi * b_ncols + b_ncols)
            };
            self_view *= faer::Scale(beta);
            matmul(
                self_view,
                Accum::Add,
                a_view,
                b_view,
                alpha,
                self.context.par,
            );
        }
    }

    fn column_mut(&mut self, j: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let nrows = self.nrows();
        if nbatch == 1 {
            let data = self.data.get_mut(0..nrows, j..j + 1);
            return FaerVecMut {
                data,
                context: self.context,
            };
        }
        let col_stride = self.data.col_stride();
        // SAFETY: ptr_at_mut(0, j) points within the Mat's allocation. The strided
        // view accesses physical columns j, j+ncols, ..., j+(nbatch-1)*ncols — all
        // within the nrows * ncols * nbatch element allocation. row_stride=1 matches
        // column-major layout. Exclusive access is guaranteed by &mut self.
        unsafe {
            let ptr = self.data.as_mut().ptr_at_mut(0, j);
            let data =
                MatMut::from_raw_parts_mut(ptr, nrows, nbatch, 1, ncols as isize * col_stride);
            FaerVecMut {
                data,
                context: self.context,
            }
        }
    }

    fn columns_mut(&mut self, start: usize, end: usize) -> Self::ViewMut<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let sub_width = end - start;
        let total_raw = (nbatch - 1) * ncols + sub_width;
        let data = self
            .data
            .get_mut(0..self.data.nrows(), start..start + total_raw);
        FaerMatMut {
            data,
            context: self.context,
            batch_stride: ncols,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.data[(i, j)] = value;
    }

    fn column(&self, j: usize) -> <Self::V as Vector>::View<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let nrows = self.nrows();
        if nbatch == 1 {
            let data = self.data.get(0..nrows, j..j + 1);
            return FaerVecRef {
                data,
                context: self.context,
            };
        }
        let col_stride = self.data.col_stride();
        // SAFETY: ptr_at(0, j) points within the Mat's allocation. The strided
        // view accesses physical columns j, j+ncols, ..., j+(nbatch-1)*ncols — all
        // within the nrows * ncols * nbatch element allocation. row_stride=1 matches
        // column-major layout.
        unsafe {
            let ptr = self.data.as_ref().ptr_at(0, j);
            let data = MatRef::from_raw_parts(ptr, nrows, nbatch, 1, ncols as isize * col_stride);
            FaerVecRef {
                data,
                context: self.context,
            }
        }
    }

    fn columns(&self, start: usize, end: usize) -> Self::View<'_> {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let sub_width = end - start;
        let total_raw = (nbatch - 1) * ncols + sub_width;
        let data = self.data.get(0..self.nrows(), start..start + total_raw);
        FaerMatRef {
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
                    *self.data.get_unchecked(k, ci) + alpha * *self.data.get_unchecked(k, cj)
                };
                unsafe {
                    *self.data.get_mut_unchecked(k, ci) = value;
                };
            }
        }
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
        let mut m = Mat::zeros(nrows, ncols * nbatch);
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
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let self_nbatch = self.context.nbatch();
        let ncols = self.ncols();
        let x_nbatch = x.data.ncols();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv");
        let max_nbatch = self_nbatch.max(x_nbatch);
        for b in 0..max_nbatch {
            let a_view = if self_nbatch == 1 {
                self.data.get(0..self.nrows(), 0..ncols)
            } else {
                self.data.get(0..self.nrows(), b * ncols..b * ncols + ncols)
            };
            let x_col = if x_nbatch == 1 {
                x.data.col(0)
            } else {
                x.data.col(b)
            };
            let mut y_col = y.data.col_mut(b);
            y_col *= faer::Scale(beta);
            matmul(
                y_col.as_mat_mut(),
                Accum::Add,
                a_view,
                x_col.as_mat(),
                alpha,
                self.context.par,
            );
        }
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = Mat::zeros(nrows, ncols * nbatch);
        Self { data, context: ctx }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let nbatch = v.context().nbatch();
        let nstates = v.len();
        let mut data = Mat::zeros(nstates, nstates * nbatch);
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
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        let nbatch = self.context.nbatch();
        let ncols = self.ncols();
        for b in 0..nbatch {
            self.data.col_mut(b * ncols + j).copy_from(v.data.col(b));
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        zip!(self.data.as_mut(), x.data.as_ref(), y.data.as_ref())
            .for_each(|unzip!(s, x, y)| *s = *x + beta * *y);
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

#[cfg(test)]
mod tests {
    use super::*;

    super::super::generate_matrix_tests!(
        faer,
        FaerMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );

    super::super::generate_dense_matrix_tests!(
        faer,
        FaerMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );
}
