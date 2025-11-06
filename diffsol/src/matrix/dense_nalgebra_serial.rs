use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};

use crate::{scalar::Scale, IndexType, Scalar, Vector};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use super::utils::*;
use crate::error::DiffsolError;
use crate::{
    DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut, NalgebraContext, NalgebraLU,
    NalgebraVec, NalgebraVecMut, NalgebraVecRef, VectorIndex,
};

#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMat<T: Scalar> {
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NalgebraMatRef<'a, T: Scalar> {
    pub(crate) data: DMatrixView<'a, T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraMatMut<'a, T: Scalar> {
    pub(crate) data: DMatrixViewMut<'a, T>,
    pub(crate) context: NalgebraContext,
}

impl<T: Scalar> DefaultSolver for NalgebraMat<T> {
    type LS = NalgebraLU<T>;
}

impl_matrix_common_ref!(
    NalgebraMatMut<'a, T>,
    NalgebraVec<T>,
    NalgebraContext,
    DMatrixViewMut<'a, T>
);
impl_matrix_common_ref!(
    NalgebraMatRef<'a, T>,
    NalgebraVec<T>,
    NalgebraContext,
    DMatrixView<'a, T>
);
impl_matrix_common!(NalgebraMat<T>, NalgebraVec<T>, NalgebraContext, DMatrix<T>);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: Scalar> Mul<Scale<T>> for $mat_type {
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
        impl<T: Scalar> MulAssign<Scale<T>> for $mat_type {
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

impl_add!(NalgebraMat<T>, &NalgebraMat<T>, NalgebraMat<T>);
impl_add!(NalgebraMat<T>, &NalgebraMatRef<'_, T>, NalgebraMat<T>);
impl_add!(NalgebraMatRef<'_, T>, &NalgebraMat<T>, NalgebraMat<T>);

impl_sub!(NalgebraMat<T>, &NalgebraMat<T>, NalgebraMat<T>);
impl_sub!(NalgebraMat<T>, &NalgebraMatRef<'_, T>, NalgebraMat<T>);
impl_sub!(NalgebraMatRef<'_, T>, &NalgebraMat<T>, NalgebraMat<T>);

impl_add_assign!(NalgebraMat<T>, &NalgebraMat<T>);
impl_add_assign!(NalgebraMat<T>, &NalgebraMatRef<'_, T>);
impl_add_assign!(NalgebraMatMut<'_, T>, &NalgebraMatRef<'_, T>);
impl_add_assign!(NalgebraMatMut<'_, T>, &NalgebraMatMut<'_, T>);

impl_sub_assign!(NalgebraMat<T>, &NalgebraMat<T>);
impl_sub_assign!(NalgebraMat<T>, &NalgebraMatRef<'_, T>);
impl_sub_assign!(NalgebraMatMut<'_, T>, &NalgebraMatRef<'_, T>);
impl_sub_assign!(NalgebraMatMut<'_, T>, &NalgebraMatMut<'_, T>);

impl_index!(NalgebraMat<T>);
impl_index!(NalgebraMatRef<'_, T>);
impl_index_mut!(NalgebraMat<T>);

impl<'a, T: Scalar> MatrixView<'a> for NalgebraMatRef<'a, T> {
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
        y.data.gemv(alpha, &self.data, &x.data, beta);
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.data.gemv(alpha, &self.data, &x.data, beta);
    }
}

impl<'a, T: Scalar> MatrixViewMut<'a> for NalgebraMatMut<'a, T> {
    type Owned = NalgebraMat<T>;
    type View = NalgebraMatRef<'a, T>;
    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.into_owned(),
            context: self.context,
        }
    }
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.data.gemm(alpha, &a.data, &b.data, beta);
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.data.gemm(alpha, &a.data, &b.data, beta);
    }
}

impl<T: Scalar> Matrix for NalgebraMat<T> {
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
        for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
            let i = dst_i % self.nrows();
            let j = dst_i / self.nrows();
            self.data[(i, j)] = data[*src_i];
        }
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        assert_eq!(indices.len(), self.nrows() * self.ncols());
        let mut idx = indices.data.iter().peekable();
        for j in 0..self.ncols() {
            let other_col = other.data.column(*idx.peek().unwrap() / other.nrows());
            for self_ij in self.data.column_mut(j).iter_mut() {
                let other_i = idx.next().unwrap() % other.nrows();
                *self_ij = other_col[other_i];
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

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, Self::T)> {
        let n = self.ncols();
        let m = self.nrows();
        (0..n).flat_map(move |j| (0..m).map(move |i| (i, j, self.data[(i, j)])))
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let mut m = DMatrix::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(Self {
            data: m,
            context: ctx,
        })
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = DMatrix::zeros(nrows, ncols);
        Self { data, context: ctx }
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let data = DMatrix::from_diagonal(&v.data);
        Self {
            data,
            context: *v.context(),
        }
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.data.gemv(alpha, &self.data, &x.data, beta);
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        self.data.column_mut(j).copy_from(&v.data);
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

impl<T: Scalar> DenseMatrix for NalgebraMat<T> {
    type View<'a> = NalgebraMatRef<'a, T>;
    type ViewMut<'a> = NalgebraMatMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.data.gemm(alpha, &a.data, &b.data, beta);
    }

    fn resize_cols(&mut self, ncols: IndexType) {
        if ncols == self.ncols() {
            return;
        }
        self.data.resize_horizontally_mut(ncols, Self::T::zero());
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[(i, j)]
    }

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        let data = DMatrix::from_vec(nrows, ncols, data);
        Self { data, context: ctx }
    }

    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_> {
        let data = self.data.column_mut(i);
        NalgebraVecMut {
            data,
            context: self.context,
        }
    }

    fn columns_mut(&mut self, start: IndexType, end: IndexType) -> Self::ViewMut<'_> {
        let data = self.data.columns_mut(start, end - start);
        NalgebraMatMut {
            data,
            context: self.context,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.data[(i, j)] = value;
    }

    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_> {
        let data = self.data.column(i);
        NalgebraVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, start: IndexType, end: IndexType) -> Self::View<'_> {
        let data = self.data.columns(start, end - start);
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
        for k in 0..self.nrows() {
            let value = unsafe {
                *self.data.get_unchecked((k, i)) + alpha * *self.data.get_unchecked((k, j))
            };
            unsafe {
                *self.data.get_unchecked_mut((k, i)) = value;
            };
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
}
