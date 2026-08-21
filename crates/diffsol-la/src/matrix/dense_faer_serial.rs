use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use super::default_solver::DefaultSolver;
use super::utils::*;
use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::error::LaError;
use crate::scalar::{IndexType, Scalar, Scale};
use crate::VectorIndex;
use crate::{Context, Dense, DenseRef, FaerContext, FaerScalar, FaerVec, Vector};
use crate::{FaerLU, FaerVecMut, FaerVecRef};

use faer::{get_global_parallelism, unzip, zip, Accum};
use faer::{linalg::matmul::matmul, Mat, MatMut, MatRef};

#[derive(Clone, Debug, PartialEq)]
pub struct FaerMat<T: FaerScalar> {
    pub(crate) data: Vec<Mat<T>>,
    pub(crate) context: FaerContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaerMatRef<'a, T: FaerScalar> {
    pub(crate) data: Vec<MatRef<'a, T>>,
    pub(crate) context: FaerContext,
}

#[derive(Debug, PartialEq)]
pub struct FaerMatMut<'a, T: FaerScalar> {
    pub(crate) data: Vec<MatMut<'a, T>>,
    pub(crate) context: FaerContext,
}

impl<T: FaerScalar> DefaultSolver for FaerMat<T> {
    type LS = FaerLU<T>;
}

impl_matrix_common_ref!(
    FaerMatMut<'a, T>,
    FaerVec<T>,
    FaerContext,
    Vec<MatMut<'a, T>>,
    FaerScalar
);
impl_matrix_common_ref!(
    FaerMatRef<'a, T>,
    FaerVec<T>,
    FaerContext,
    Vec<MatRef<'a, T>>,
    FaerScalar
);
impl_matrix_common!(FaerMat<T>, FaerVec<T>, FaerContext, Vec<Mat<T>>, FaerScalar);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: FaerScalar> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
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
        impl<T: FaerScalar> MulAssign<Scale<T>> for $mat_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let scale: faer::Scale<T> = rhs.into();
                for data in &mut self.data {
                    *data *= scale;
                }
            }
        }
    };
}

impl_mul_scalar!(FaerMatRef<'_, T>, FaerMat<T>);
impl_mul_scalar!(FaerMat<T>, FaerMat<T>);
impl_mul_scalar!(&FaerMat<T>, FaerMat<T>);

impl_mul_assign_scalar!(FaerMatMut<'_, T>);

/// `self` is owned, so the result is written into its batches whenever it already holds as
/// many as the result: `*self_b $op rhs_b`.
macro_rules! matrix_binary_owned_lhs {
    ($trait:ident, $method:ident, $rhs:ty, $op:tt, $binary:tt) => {
        impl<T: FaerScalar> $trait<$rhs> for FaerMat<T> {
            type Output = FaerMat<T>;

            fn $method(mut self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                let rhs_nbatch = rhs.data.len();
                if self.data.len() >= rhs_nbatch {
                    for (batch, data) in self.data.iter_mut().enumerate() {
                        *data $op &rhs.data[batch % rhs_nbatch];
                    }
                    return self;
                }
                // self holds fewer batches than the result, so it cannot be written into
                let self_nbatch = self.data.len();
                FaerMat {
                    data: (0..rhs_nbatch)
                        .map(|batch| &self.data[batch % self_nbatch] $binary &rhs.data[batch])
                        .collect(),
                    context: rhs.context,
                }
            }
        }
    };
}

/// `rhs` is owned, so the result is written into its batches whenever it already holds as
/// many as the result: `combine(&mut rhs_b, lhs_b)`.
macro_rules! matrix_binary_owned_rhs {
    ($trait:ident, $method:ident, $lhs:ty, $op:tt, $combine:expr) => {
        impl<T: FaerScalar> $trait<FaerMat<T>> for $lhs {
            type Output = FaerMat<T>;

            fn $method(self, mut rhs: FaerMat<T>) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                let self_nbatch = self.data.len();
                if rhs.data.len() >= self_nbatch {
                    for (batch, data) in rhs.data.iter_mut().enumerate() {
                        zip!(data.as_mut(), self.data[batch % self_nbatch].as_ref())
                            .for_each(|unzip!(rhs, lhs)| $combine(rhs, *lhs));
                    }
                    return rhs;
                }
                // rhs holds fewer batches than the result, so it cannot be written into
                let rhs_nbatch = rhs.data.len();
                FaerMat {
                    data: (0..self_nbatch)
                        .map(|batch| {
                            let mut data = self.data[batch].as_ref().to_owned();
                            data $op &rhs.data[batch % rhs_nbatch];
                            data
                        })
                        .collect(),
                    context: self.context,
                }
            }
        }
    };
}

matrix_binary_owned_lhs!(Add, add, &FaerMat<T>, +=, +);
matrix_binary_owned_lhs!(Add, add, &FaerMatRef<'_, T>, +=, +);
matrix_binary_owned_lhs!(Sub, sub, &FaerMat<T>, -=, -);
matrix_binary_owned_lhs!(Sub, sub, &FaerMatRef<'_, T>, -=, -);

matrix_binary_owned_rhs!(Add, add, &FaerMat<T>, +=, |rhs: &mut T, lhs: T| *rhs += lhs);
matrix_binary_owned_rhs!(Add, add, FaerMatRef<'_, T>, +=, |rhs: &mut T, lhs: T| *rhs += lhs);
matrix_binary_owned_rhs!(Sub, sub, &FaerMat<T>, -=, |rhs: &mut T, lhs: T| *rhs = lhs - *rhs);
matrix_binary_owned_rhs!(
    Sub,
    sub,
    FaerMatRef<'_, T>,
    -=,
    |rhs: &mut T, lhs: T| *rhs = lhs - *rhs
);

impl_add!(FaerMatRef<'_, T>, &FaerMat<T>, FaerMat<T>, FaerScalar);
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
            data: self.data.iter().map(|data| data.to_owned()).collect(),
            context: self.context,
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv_o");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv_o");
        for (b, y) in y.data.iter_mut().enumerate() {
            *y *= faer::Scale(beta);
            matmul(
                y.as_mut(),
                Accum::Add,
                self.data[b % self.data.len()].as_ref(),
                x.data[b % x.data.len()].as_ref(),
                alpha,
                get_global_parallelism(),
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
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv_v");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv_v");
        for (b, y) in y.data.iter_mut().enumerate() {
            *y *= faer::Scale(beta);
            matmul(
                y.as_mut(),
                Accum::Add,
                self.data[b % self.data.len()].as_ref(),
                x.data[b % x.data.len()],
                alpha,
                get_global_parallelism(),
            );
        }
    }
}

impl<'a, T: FaerScalar> MatrixViewMut<'a> for FaerMatMut<'a, T> {
    type Owned = FaerMat<T>;
    type View = FaerMatRef<'a, T>;

    fn into_owned(self) -> Self::Owned {
        Self::Owned {
            data: self.data.iter().map(|data| data.to_owned()).collect(),
            context: self.context,
        }
    }

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(a.context.nbatch(), "gemm_oo");
        self.context
            .assert_compatible_nbatch(b.context.nbatch(), "gemm_oo");
        for (batch, data) in self.data.iter_mut().enumerate() {
            *data *= faer::Scale(beta);
            matmul(
                data.as_mut(),
                Accum::Add,
                a.data[batch % a.data.len()].as_ref(),
                b.data[batch % b.data.len()].as_ref(),
                alpha,
                get_global_parallelism(),
            );
        }
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(a.context.nbatch(), "gemm_vo");
        self.context
            .assert_compatible_nbatch(b.context.nbatch(), "gemm_vo");
        for (batch, data) in self.data.iter_mut().enumerate() {
            *data *= faer::Scale(beta);
            matmul(
                data.as_mut(),
                Accum::Add,
                a.data[batch % a.data.len()],
                b.data[batch % b.data.len()].as_ref(),
                alpha,
                get_global_parallelism(),
            );
        }
    }
}

impl<T: FaerScalar> DenseMatrix for FaerMat<T> {
    type View<'a> = FaerMatRef<'a, T>;
    type ViewMut<'a> = FaerMatMut<'a, T>;

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        assert_eq!(data.len(), nrows * ncols * ctx.nbatch());
        let data = data
            .chunks(nrows * ncols)
            .map(|data| Mat::from_fn(nrows, ncols, |i, j| data[i + j * nrows]))
            .collect();
        Self { data, context: ctx }
    }

    fn resize_cols(&mut self, ncols: IndexType) {
        if ncols == self.ncols() {
            return;
        }
        let nrows = self.nrows();
        for data in &mut self.data {
            data.resize_with(nrows, ncols, |_, _| T::zero());
        }
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[0][(i, j)]
    }

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.context
            .assert_compatible_nbatch(a.context.nbatch(), "gemm");
        self.context
            .assert_compatible_nbatch(b.context.nbatch(), "gemm");
        for (batch, data) in self.data.iter_mut().enumerate() {
            *data *= faer::Scale(beta);
            matmul(
                data.as_mut(),
                Accum::Add,
                a.data[batch % a.data.len()].as_ref(),
                b.data[batch % b.data.len()].as_ref(),
                alpha,
                get_global_parallelism(),
            );
        }
    }
    fn column_mut(&mut self, i: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let data = self
            .data
            .iter_mut()
            .map(|data| data.get_mut(0..data.nrows(), i))
            .collect();
        FaerVecMut {
            data,
            context: self.context,
        }
    }

    fn columns_mut(&mut self, start: usize, end: usize) -> Self::ViewMut<'_> {
        let data = self
            .data
            .iter_mut()
            .map(|data| data.get_mut(0..data.nrows(), start..end))
            .collect();
        FaerMatMut {
            data,
            context: self.context,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        for data in &mut self.data {
            data[(i, j)] = value;
        }
    }

    fn column(&self, i: usize) -> <Self::V as Vector>::View<'_> {
        let data = self
            .data
            .iter()
            .map(|data| data.get(0..data.nrows(), i))
            .collect();
        FaerVecRef {
            data,
            context: self.context,
        }
    }
    fn columns(&self, start: usize, end: usize) -> Self::View<'_> {
        let data = self
            .data
            .iter()
            .map(|data| data.get(0..data.nrows(), start..end))
            .collect();
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
        for data in &mut self.data {
            let nrows = data.nrows();
            if i < j {
                let (left, right) = data.as_mut().split_at_col_mut(j);
                zip!(left.get_mut(0..nrows, i), right.get(0..nrows, 0))
                    .for_each(|unzip!(dst, src)| *dst += alpha * *src);
            } else {
                let (left, right) = data.as_mut().split_at_col_mut(i);
                zip!(right.get_mut(0..nrows, 0), left.get(0..nrows, j))
                    .for_each(|unzip!(dst, src)| *dst += alpha * *src);
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
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        assert_eq!(indices.len(), self.nrows() * self.ncols());
        if self.nrows() == 0 || self.ncols() == 0 {
            return;
        }
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "gather");
        let nrows = self.nrows();
        for (batch, matrix) in self.data.iter_mut().enumerate() {
            let other = &other.data[batch % other.data.len()];
            for (j, src_indices) in indices.data.chunks(nrows).enumerate() {
                for (dst, src) in matrix.col_as_slice_mut(j).iter_mut().zip(src_indices) {
                    *dst = other[(*src % other.nrows(), *src / other.nrows())];
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
        for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
            let i = dst_i % self.nrows();
            let j = dst_i / self.nrows();
            for (batch, matrix) in self.data.iter_mut().enumerate() {
                matrix[(i, j)] = data.data[batch % data.data.len()][*src_i];
            }
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        v.context
            .assert_compatible_nbatch(self.context.nbatch(), "add_column_to_vector");
        let nrows = self.nrows();
        for (batch, v) in v.data.iter_mut().enumerate() {
            zip!(
                v.as_mut(),
                self.data[batch % self.data.len()].get(0..nrows, j)
            )
            .for_each(|unzip!(v, column)| *v += *column);
        }
    }

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let indices: Vec<_> = (0..self.ncols())
            .flat_map(move |j| (0..self.nrows()).map(move |i| (i, j)))
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
                let mut m = Mat::zeros(nrows, ncols);
                for ((i, j), v) in indices.iter().zip(values) {
                    m[(*i, *j)] = *v;
                }
                m
            })
            .collect();
        Ok(Self { data, context: ctx })
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.context
            .assert_compatible_nbatch(self.context.nbatch(), "gemv");
        y.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv");
        for (batch, data) in y.data.iter_mut().enumerate() {
            *data *= faer::Scale(beta);
            matmul(
                data.as_mut(),
                Accum::Add,
                self.data[batch % self.data.len()].as_ref(),
                x.data[batch % x.data.len()].as_ref(),
                alpha,
                get_global_parallelism(),
            );
        }
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = (0..ctx.nbatch())
            .map(|_| Mat::zeros(nrows, ncols))
            .collect();
        Self { data, context: ctx }
    }
    fn copy_from(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (batch, data) in self.data.iter_mut().enumerate() {
            data.copy_from(&other.data[batch % other.data.len()]);
        }
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let dim = v.len();
        let data = v
            .data
            .iter()
            .map(|v| Mat::from_fn(dim, dim, |i, j| if i == j { v[i] } else { T::zero() }))
            .collect();
        Self {
            data,
            context: *v.context(),
        }
    }
    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let diagonal = self.data[0].diagonal().column_vector();
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
        self.context
            .assert_compatible_nbatch(v.context.nbatch(), "set_column");
        let nrows = self.nrows();
        for (batch, matrix) in self.data.iter_mut().enumerate() {
            matrix
                .get_mut(0..nrows, j)
                .copy_from(v.data[batch % v.data.len()].as_ref());
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "scale_add_and_assign");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "scale_add_and_assign");
        for (batch, data) in self.data.iter_mut().enumerate() {
            zip!(
                data.as_mut(),
                x.data[batch % x.data.len()].as_ref(),
                y.data[batch % y.data.len()].as_ref()
            )
            .for_each(|unzip!(s, x, y)| *s = *x + beta * *y);
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

#[cfg(test)]
mod tests {
    use super::super::tests::triplet_values;
    use super::*;

    #[test]
    fn test_column_axpy() {
        super::super::tests::test_column_axpy::<FaerMat<f64>>();
    }

    fn mat(values: [f64; 4], ctx: FaerContext) -> FaerMat<f64> {
        FaerMat::from_vec(2, 2, values.repeat(ctx.nbatch()).to_vec(), ctx)
    }

    #[test]
    fn test_owned_rhs() {
        super::super::tests::test_owned_rhs_m::<FaerMat<f64>>();
    }

    #[test]
    fn test_batched_owned_rhs_broadcast() {
        super::super::tests::test_batched_owned_rhs_broadcast_m::<FaerMat<f64>>(
            FaerContext::default().clone_with_nbatch(2).unwrap(),
        );
    }

    /// faer-specific: the value semantics live in the generic suite, this pins down that an
    /// owned operand really is written into instead of reallocated.
    #[test]
    fn test_owned_operands_reuse_allocation() {
        let ctx = FaerContext::default();
        let a = mat([10.0, 20.0, 30.0, 40.0], ctx);

        let lhs = mat([10.0, 20.0, 30.0, 40.0], ctx);
        let buffer = lhs.data[0].as_ptr();
        let b = mat([1.0, 2.0, 3.0, 4.0], ctx);
        let c = lhs - &b;
        assert_eq!(c.data[0].as_ptr(), buffer, "lhs buffer not reused");
        assert_eq!(triplet_values(&c), vec![9.0, 18.0, 27.0, 36.0]);

        let rhs = mat([1.0, 2.0, 3.0, 4.0], ctx);
        let buffer = rhs.data[0].as_ptr();
        let c = &a - rhs;
        assert_eq!(c.data[0].as_ptr(), buffer, "rhs buffer not reused");
        assert_eq!(triplet_values(&c), vec![9.0, 18.0, 27.0, 36.0]);

        let rhs = mat([1.0, 2.0, 3.0, 4.0], ctx);
        let buffer = rhs.data[0].as_ptr();
        let ncols = a.ncols();
        let c = a.columns(0, ncols) + rhs;
        assert_eq!(c.data[0].as_ptr(), buffer, "rhs buffer not reused");
        assert_eq!(triplet_values(&c), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// faer-specific: the owned rhs is reused when it already spans every batch, and left
    /// alone when the result needs more batches than it holds.
    #[test]
    fn test_owned_rhs_broadcast_reuse() {
        let ctx1 = FaerContext::default();
        let ctx2 = FaerContext::default().clone_with_nbatch(2).unwrap();

        let a = mat([10.0, 20.0, 30.0, 40.0], ctx1);
        let rhs = mat([1.0, 2.0, 3.0, 4.0], ctx2);
        let buffer = rhs.data[0].as_ptr();
        let c = &a - rhs;
        assert_eq!(c.context().nbatch(), 2);
        assert_eq!(c.data[0].as_ptr(), buffer, "rhs buffer not reused");
        assert_eq!(
            triplet_values(&c),
            vec![9.0, 18.0, 27.0, 36.0, 9.0, 18.0, 27.0, 36.0]
        );

        // the rhs is a single batch, so the two-batch result has to be allocated
        let a = mat([10.0, 20.0, 30.0, 40.0], ctx2);
        let rhs = mat([1.0, 2.0, 3.0, 4.0], ctx1);
        let c = &a - rhs;
        assert_eq!(c.context().nbatch(), 2);
        assert_eq!(
            triplet_values(&c),
            vec![9.0, 18.0, 27.0, 36.0, 9.0, 18.0, 27.0, 36.0]
        );
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<FaerMat<f64>>();
    }

    #[test]
    fn test_resize_cols() {
        super::super::tests::test_resize_cols::<FaerMat<f64>>();
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
