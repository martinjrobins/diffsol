use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::default_solver::DefaultSolver;
use super::utils::*;
use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::error::DiffsolError;
use crate::scalar::{IndexType, Scalar, Scale};
use crate::VectorIndex;
use crate::{Dense, DenseRef, FaerContext, FaerVec, Vector, VectorViewMut};
use crate::{FaerLU, FaerVecMut, FaerVecRef};

use faer::{get_global_parallelism, unzip, zip, Accum};
use faer::{linalg::matmul::matmul, Mat, MatMut, MatRef};

#[derive(Clone, Debug, PartialEq)]
pub struct FaerMat<T: Scalar> {
    pub(crate) data: Mat<T>,
    pub(crate) context: FaerContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaerMatRef<'a, T: Scalar> {
    pub(crate) data: MatRef<'a, T>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, PartialEq)]
pub struct FaerMatMut<'a, T: Scalar> {
    pub(crate) data: MatMut<'a, T>,
    pub(crate) context: FaerContext,
}

impl<T: Scalar> DefaultSolver for FaerMat<T> {
    type LS = FaerLU<T>;
}

impl_matrix_common_ref!(FaerMatMut<'a, T>, FaerVec<T>, FaerContext, MatMut<'a, T>);
impl_matrix_common_ref!(FaerMatRef<'a, T>, FaerVec<T>, FaerContext, MatRef<'a, T>);
impl_matrix_common!(FaerMat<T>, FaerVec<T>, FaerContext, Mat<T>);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: Scalar> Mul<Scale<T>> for $mat_type {
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
        impl<T: Scalar> MulAssign<Scale<T>> for $mat_type {
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

impl_add!(FaerMat<T>, &FaerMat<T>, FaerMat<T>);
impl_add!(FaerMat<T>, &FaerMatRef<'_, T>, FaerMat<T>);
impl_add!(FaerMatRef<'_, T>, &FaerMat<T>, FaerMat<T>);

impl_sub!(FaerMat<T>, &FaerMat<T>, FaerMat<T>);
impl_sub!(FaerMat<T>, &FaerMatRef<'_, T>, FaerMat<T>);
impl_sub!(FaerMatRef<'_, T>, &FaerMat<T>, FaerMat<T>);

impl_add_assign!(FaerMat<T>, &FaerMat<T>);
impl_add_assign!(FaerMat<T>, &FaerMatRef<'_, T>);
impl_add_assign!(FaerMatMut<'_, T>, &FaerMatRef<'_, T>);
impl_add_assign!(FaerMatMut<'_, T>, &FaerMatMut<'_, T>);

impl_sub_assign!(FaerMat<T>, &FaerMat<T>);
impl_sub_assign!(FaerMat<T>, &FaerMatRef<'_, T>);
impl_sub_assign!(FaerMatMut<'_, T>, &FaerMatRef<'_, T>);
impl_sub_assign!(FaerMatMut<'_, T>, &FaerMatMut<'_, T>);

impl_index!(FaerMat<T>);
impl_index!(FaerMatRef<'_, T>);
impl_index_mut!(FaerMat<T>);

impl<'a, T: Scalar> MatrixView<'a> for FaerMatRef<'a, T> {
    type Owned = FaerMat<T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.to_owned(),
            context: self.context,
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.mul_assign(Scale(beta));
        matmul(
            y.data.as_mut(),
            Accum::Add,
            self.data.as_ref(),
            x.data.as_ref(),
            alpha,
            get_global_parallelism(),
        );
    }
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        y.mul_assign(Scale(beta));
        matmul(
            y.data.as_mut(),
            Accum::Add,
            self.data.as_ref(),
            x.data.as_ref(),
            alpha,
            get_global_parallelism(),
        );
    }
}

impl<'a, T: Scalar> MatrixViewMut<'a> for FaerMatMut<'a, T> {
    type Owned = FaerMat<T>;
    type View = FaerMatRef<'a, T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.to_owned(),
            context: self.context,
        }
    }

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.mul_assign(Scale(beta));
        matmul(
            self.data.as_mut(),
            Accum::Add,
            a.data.as_ref(),
            b.data.as_ref(),
            alpha,
            get_global_parallelism(),
        )
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.mul_assign(Scale(beta));
        matmul(
            self.data.as_mut(),
            Accum::Add,
            a.data.as_ref(),
            b.data.as_ref(),
            alpha,
            get_global_parallelism(),
        )
    }
}

impl<T: Scalar> DenseMatrix for FaerMat<T> {
    type View<'a> = FaerMatRef<'a, T>;
    type ViewMut<'a> = FaerMatMut<'a, T>;

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        let data = Mat::from_fn(nrows, ncols, |i, j| data[i + j * nrows]);
        Self { data, context: ctx }
    }

    fn resize_cols(&mut self, ncols: IndexType) {
        if ncols == self.ncols() {
            return;
        }
        let nrows = self.nrows();
        self.data.resize_with(nrows, ncols, |_, _| T::zero());
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[(i, j)]
    }

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.data.mul_assign(faer::Scale(beta));
        matmul(
            self.data.as_mut(),
            Accum::Add,
            a.data.as_ref(),
            b.data.as_ref(),
            alpha,
            get_global_parallelism(),
        )
    }
    fn column_mut(&mut self, i: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let data = self.data.get_mut(0..self.nrows(), i);
        FaerVecMut {
            data,
            context: self.context,
        }
    }

    fn columns_mut(&mut self, start: usize, end: usize) -> Self::ViewMut<'_> {
        let data = self.data.get_mut(0..self.data.nrows(), start..end);
        FaerMatMut {
            data,
            context: self.context,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.data[(i, j)] = value;
    }

    fn column(&self, i: usize) -> <Self::V as Vector>::View<'_> {
        let data = self.data.get(0..self.data.nrows(), i);
        FaerVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, start: usize, end: usize) -> Self::View<'_> {
        let data = self.data.get(0..self.nrows(), start..end);
        FaerMatRef {
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
            let value =
                unsafe { *self.data.get_unchecked(k, i) + alpha * *self.data.get_unchecked(k, j) };
            unsafe { *self.data.get_mut_unchecked(k, i) = value };
        }
    }
}

impl<T: Scalar> Matrix for FaerMat<T> {
    type Sparsity = Dense<Self>;
    type SparsityRef<'a> = DenseRef<'a, Self>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        None
    }

    fn context(&self) -> &Self::C {
        &self.context
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        assert_eq!(indices.len(), self.nrows() * self.ncols());
        let mut idx = indices.data.iter().peekable();
        for j in 0..self.ncols() {
            let other_col = other.data.col(*idx.peek().unwrap() / other.nrows());
            for self_ij in self.data.col_mut(j).iter_mut() {
                let other_i = idx.next().unwrap() % other.nrows();
                *self_ij = other_col[other_i];
            }
        }
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

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        v.add_assign(&self.column(j));
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, Self::T)> {
        (0..self.ncols())
            .flat_map(move |j| (0..self.nrows()).map(move |i| (i, j, self.data[(i, j)])))
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let mut m = Mat::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(Self {
            data: m,
            context: ctx,
        })
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.mul_assign(Scale(beta));
        matmul(
            y.data.as_mut(),
            Accum::Add,
            self.data.as_ref(),
            x.data.as_ref(),
            alpha,
            get_global_parallelism(),
        );
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = Mat::zeros(nrows, ncols);
        Self { data, context: ctx }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let dim = v.len();
        let data = Mat::from_fn(dim, dim, |i, j| if i == j { v[i] } else { T::zero() });
        Self {
            data,
            context: *v.context(),
        }
    }
    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let diagonal = self.data.diagonal().column_vector();
        let (zero_indices, nonzero_indices) = diagonal.iter().enumerate().fold(
            (Vec::new(), Vec::new()),
            |(mut zero_indices, mut nonzero_indices), (i, &v)| {
                if v.is_zero() {
                    zero_indices.push(i);
                } else {
                    nonzero_indices.push(i);
                }
                (zero_indices, nonzero_indices)
            },
        );
        (
            <Self::V as Vector>::Index::from_vec(zero_indices, self.context),
            <Self::V as Vector>::Index::from_vec(nonzero_indices, self.context),
        )
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        self.column_mut(j).copy_from(v);
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

    #[test]
    fn test_column_axpy() {
        super::super::tests::test_column_axpy::<FaerMat<f64>>();
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<FaerMat<f64>>();
    }

    #[test]
    fn test_resize_cols() {
        super::super::tests::test_resize_cols::<FaerMat<f64>>();
    }
}
