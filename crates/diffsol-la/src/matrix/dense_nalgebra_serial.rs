use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};

use crate::{scalar::Scale, Context, IndexType, NalgebraScalar, Scalar, Vector};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use super::utils::*;
use crate::error::LaError;
use crate::{
    DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut, NalgebraContext, NalgebraLU,
    NalgebraVec, NalgebraVecMut, NalgebraVecRef, VectorIndex,
};

#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMat<T: NalgebraScalar> {
    pub(crate) data: Vec<DMatrix<T>>,
    pub(crate) context: NalgebraContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMatRef<'a, T: NalgebraScalar> {
    pub(crate) data: Vec<DMatrixView<'a, T>>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraMatMut<'a, T: NalgebraScalar> {
    pub(crate) data: Vec<DMatrixViewMut<'a, T>>,
    pub(crate) context: NalgebraContext,
}

impl<T: NalgebraScalar> DefaultSolver for NalgebraMat<T> {
    type LS = NalgebraLU<T>;
}

impl_matrix_common_ref!(
    NalgebraMatMut<'a, T>,
    NalgebraVec<T>,
    NalgebraContext,
    Vec<DMatrixViewMut<'a, T>>,
    NalgebraScalar
);
impl_matrix_common_ref!(
    NalgebraMatRef<'a, T>,
    NalgebraVec<T>,
    NalgebraContext,
    Vec<DMatrixView<'a, T>>,
    NalgebraScalar
);
impl_matrix_common!(
    NalgebraMat<T>,
    NalgebraVec<T>,
    NalgebraContext,
    Vec<DMatrix<T>>,
    NalgebraScalar
);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: NalgebraScalar> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale = rhs.value();
                Self::Output {
                    data: self.data.iter().map(|data| data * scale).collect(),
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
                for data in &mut self.data {
                    *data *= scale;
                }
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
            data: self
                .data
                .into_iter()
                .map(|data| data.into_owned())
                .collect(),
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
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv_v");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv_v");
        for (b, y) in y.data.iter_mut().enumerate() {
            y.gemv(
                alpha,
                &self.data[b % self.data.len()],
                &x.data[b % x.data.len()],
                beta,
            );
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv_o");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv_o");
        for (b, y) in y.data.iter_mut().enumerate() {
            y.gemv(
                alpha,
                &self.data[b % self.data.len()],
                &x.data[b % x.data.len()],
                beta,
            );
        }
    }
}

impl<'a, T: NalgebraScalar> MatrixViewMut<'a> for NalgebraMatMut<'a, T> {
    type Owned = NalgebraMat<T>;
    type View = NalgebraMatRef<'a, T>;
    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self
                .data
                .into_iter()
                .map(|data| data.into_owned())
                .collect(),
            context: self.context,
        }
    }
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(a.context.nbatch(), "gemm_oo");
        self.context
            .assert_compatible_nbatch(b.context.nbatch(), "gemm_oo");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.gemm(
                alpha,
                &a.data[batch % a.data.len()],
                &b.data[batch % b.data.len()],
                beta,
            );
        }
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(a.context.nbatch(), "gemm_vo");
        self.context
            .assert_compatible_nbatch(b.context.nbatch(), "gemm_vo");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.gemm(
                alpha,
                &a.data[batch % a.data.len()],
                &b.data[batch % b.data.len()],
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
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
            let i = dst_i % self.nrows();
            let j = dst_i / self.nrows();
            for (batch, matrix) in self.data.iter_mut().enumerate() {
                matrix[(i, j)] = data.data[batch % data.data.len()][*src_i];
            }
        }
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        assert_eq!(indices.len(), self.nrows() * self.ncols());
        if self.nrows() == 0 || self.ncols() == 0 {
            return;
        }
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "gather");
        for (batch, matrix) in self.data.iter_mut().enumerate() {
            let other = &other.data[batch % other.data.len()];
            for (dst, src) in indices.data.iter().enumerate() {
                matrix[dst] = other[*src];
            }
        }
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let mut zero_diagonal_indices = Vec::new();
        let mut non_zero_diagonal_indices = Vec::new();
        for i in 0..self.nrows() {
            if self.data[0][(i, i)].is_zero() {
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
        v.context
            .assert_compatible_nbatch(self.context.nbatch(), "add_column_to_vector");
        for (batch, v) in v.data.iter_mut().enumerate() {
            v.axpy(
                T::one(),
                &self.data[batch % self.data.len()].column(j),
                T::one(),
            );
        }
    }

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let n = self.ncols();
        let m = self.nrows();
        let indices: Vec<_> = (0..n)
            .flat_map(move |j| (0..m).map(move |i| (i, j)))
            .collect();
        let values: Vec<_> = self
            .data
            .iter()
            .flat_map(|data| indices.iter().map(|&(i, j)| data[(i, j)]))
            .collect();
        (indices.into_iter(), values.into_iter())
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
        values: Vec<Self::T>,
        ctx: Self::C,
    ) -> Result<Self, LaError> {
        assert_eq!(values.len(), indices.len() * ctx.nbatch());
        let data = values
            .chunks(indices.len())
            .map(|values| {
                let mut m = DMatrix::zeros(nrows, ncols);
                for ((i, j), v) in indices.iter().zip(values) {
                    m[(*i, *j)] = *v;
                }
                m
            })
            .collect();
        Ok(Self { data, context: ctx })
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = (0..ctx.nbatch())
            .map(|_| DMatrix::zeros(nrows, ncols))
            .collect();
        Self { data, context: ctx }
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let data = v.data.iter().map(DMatrix::from_diagonal).collect();
        Self {
            data,
            context: *v.context(),
        }
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv");
        for (batch, y) in y.data.iter_mut().enumerate() {
            y.gemv(
                alpha,
                &self.data[batch % self.data.len()],
                &x.data[batch % x.data.len()],
                beta,
            );
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[batch % other.data.len()]);
        }
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        self.context
            .assert_compatible_nbatch(v.context.nbatch(), "set_column");
        let nrows = self.nrows();
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.as_mut_slice()[j * nrows..(j + 1) * nrows]
                .copy_from_slice(v.data[batch % v.data.len()].as_slice());
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "scale_add_and_assign");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "scale_add_and_assign");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&y.data[batch % y.data.len()]);
            *data *= beta;
            *data += &x.data[batch % x.data.len()];
        }
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
        self.context
            .assert_compatible_nbatch(a.context.nbatch(), "gemm");
        self.context
            .assert_compatible_nbatch(b.context.nbatch(), "gemm");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.gemm(
                alpha,
                &a.data[batch % a.data.len()],
                &b.data[batch % b.data.len()],
                beta,
            );
        }
    }

    fn resize_cols(&mut self, ncols: IndexType) {
        if ncols == self.ncols() {
            return;
        }
        for data in &mut self.data {
            data.resize_horizontally_mut(ncols, Self::T::zero());
        }
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[0][(i, j)]
    }

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        assert_eq!(data.len(), nrows * ncols * ctx.nbatch());
        let data = data
            .chunks(nrows * ncols)
            .map(|data| DMatrix::from_vec(nrows, ncols, data.to_vec()))
            .collect();
        Self { data, context: ctx }
    }

    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_> {
        let data = self
            .data
            .iter_mut()
            .map(|data| data.column_mut(i))
            .collect();
        NalgebraVecMut {
            data,
            context: self.context,
        }
    }

    fn columns_mut(&mut self, start: IndexType, end: IndexType) -> Self::ViewMut<'_> {
        let data = self
            .data
            .iter_mut()
            .map(|data| data.columns_mut(start, end - start))
            .collect();
        NalgebraMatMut {
            data,
            context: self.context,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        for data in &mut self.data {
            data[(i, j)] = value;
        }
    }

    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_> {
        let data = self.data.iter().map(|data| data.column(i)).collect();
        NalgebraVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, start: IndexType, end: IndexType) -> Self::View<'_> {
        let data = self
            .data
            .iter()
            .map(|data| data.columns(start, end - start))
            .collect();
        NalgebraMatRef {
            data,
            context: self.context,
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
        for data in &mut self.data {
            let nrows = data.nrows();
            if i < j {
                let (left, right) = data.as_mut_slice().split_at_mut(j * nrows);
                for (dst, src) in left[i * nrows..(i + 1) * nrows]
                    .iter_mut()
                    .zip(&right[..nrows])
                {
                    *dst += alpha * *src;
                }
            } else {
                let (left, right) = data.as_mut_slice().split_at_mut(i * nrows);
                for (dst, src) in right[..nrows]
                    .iter_mut()
                    .zip(&left[j * nrows..(j + 1) * nrows])
                {
                    *dst += alpha * *src;
                }
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
