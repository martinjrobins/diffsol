use std::ops::{AddAssign, Mul, MulAssign};

use super::default_solver::DefaultSolver;
use super::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut};
use crate::op::NonLinearOp;
use crate::scalar::{IndexType, Scalar, Scale};
use crate::FaerLU;
use crate::{Dense, DenseRef, Vector};
use anyhow::Result;
use faer::{linalg::matmul::matmul, Col, ColMut, ColRef, Mat, MatMut, MatRef, Parallelism, mat::As2DMut, mat::As2D};
use faer::{unzipped, zipped};

impl<T: Scalar> DefaultSolver for Mat<T> {
    type LS<C: NonLinearOp<M = Mat<T>, V = Col<T>, T = T>> = FaerLU<T, C>;
}

macro_rules! impl_matrix_common {
    ($mat_type:ty) => {
        impl<'a, T: Scalar> MatrixCommon for $mat_type {
            type T = T;
            type V = Col<T>;

            fn nrows(&self) -> IndexType {
                self.nrows()
            }
            fn ncols(&self) -> IndexType {
                self.ncols()
            }
        }
    };
}

impl_matrix_common!(MatMut<'a, T>);
impl_matrix_common!(MatRef<'a, T>);
impl_matrix_common!(Mat<T>);

macro_rules! impl_mul_scale {
    ($mat_type:ty) => {
        impl<'a, T: Scalar> Mul<Scale<T>> for $mat_type {
            type Output = Mat<T>;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let scale: faer::Scale<T> = rhs.into();
                self * scale
            }
        }
    };
}
impl_mul_scale!(MatRef<'a, T>);
impl_mul_scale!(Mat<T>);
impl_mul_scale!(&Mat<T>);

impl<'a, T: Scalar> MulAssign<Scale<T>> for MatMut<'a, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        let scale: faer::Scale<T> = rhs.into();
        *self *= scale;
    }
}

impl<'a, T: Scalar> MatrixView<'a> for MatRef<'a, T> {
    type Owned = Mat<T>;

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        matmul(
            y.as_2d_mut(),
            self.as_2d_ref(),
            x.as_2d_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        );
    }
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        matmul(
            y.as_2d_mut(),
            self.as_2d_ref(),
            x.as_2d_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        );
    }
}

impl<'a, T: Scalar> MatrixViewMut<'a> for MatMut<'a, T> {
    type Owned = Mat<T>;
    type View = MatRef<'a, T>;

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        matmul(
            self.as_mut(),
            a.as_ref(),
            b.as_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        )
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        matmul(
            self.as_mut(),
            a.as_ref(),
            b.as_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        )
    }
}

impl<T: Scalar> DenseMatrix for Mat<T> {
    type View<'a> = MatRef<'a, T>;
    type ViewMut<'a> = MatMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        matmul(
            self.as_mut(),
            a.as_ref(),
            b.as_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        )
    }
    fn column_mut(&mut self, i: usize) -> ColMut<'_, T> {
        self.get_mut(0..self.nrows(), i)
    }

    fn columns_mut(&mut self, start: usize, ncols: usize) -> MatMut<'_, T> {
        self.get_mut(0..self.nrows(), start..ncols)
    }

    fn column(&self, i: usize) -> ColRef<'_, T> {
        self.get(0..self.nrows(), i)
    }
    fn columns(&self, start: usize, nrows: usize) -> MatRef<'_, T> {
        self.get(0..self.nrows(), start..nrows)
    }

    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, beta: Self::T, i: IndexType) {
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
                unsafe { beta * self.read_unchecked(k, i) + alpha * self.read_unchecked(k, j) };
            unsafe { self.write_unchecked(k, i, value) };
        }
    }
}

impl<T: Scalar> Matrix for Mat<T> {
    type Sparsity = Dense<Self>;
    type SparsityRef<'a> = DenseRef<'a, Self>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        None
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        for (dst_i, src_i) in dst_indices.iter().zip(src_indices.iter()) {
            let i = dst_i % self.nrows();
            let j = dst_i / self.nrows();
            self[(i, j)] = data[*src_i];
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        v.add_assign(&self.column(j));
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, &Self::T)> {
        (0..self.nrows()).flat_map(move |i| (0..self.ncols()).map(move |j| (i, j, &self[(i, j)])))
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
    ) -> Result<Self> {
        let mut m = Self::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(m)
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        matmul(
            y.as_2d_mut(),
            self.as_2d_ref(),
            x.as_2d_ref(),
            Some(beta),
            alpha,
            Parallelism::None,
        );
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self::zeros(nrows, ncols)
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
    fn from_diagonal(v: &Col<T>) -> Self {
        let dim = v.nrows();
        Self::from_fn(dim, dim, |i, j| if i == j { v[i] } else { T::zero() })
    }
    fn diagonal(&self) -> Self::V {
        self.diagonal().column_vector().to_owned()
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        self.column_mut(j).copy_from(v);
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        zipped!(self, x, y).for_each(|unzipped!(mut s, x, y)| s.write(x.read() + beta * y.read()));
    }

    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        _sparsity: Option<Self::Sparsity>,
    ) -> Self {
        Self::zeros(nrows, ncols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_axpy() {
        // M = [1 2]
        //     [3 4]
        let mut a = Mat::zeros(2, 2);
        a[(0, 0)] = 1.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 3.0;
        a[(1, 1)] = 4.0;

        // op is M(:, 1) = 2 * M(:, 0) + M(:, 1)
        a.column_axpy(2.0, 0, 1.0, 1);
        // M = [1 4]
        //     [3 10]
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(0, 1)], 4.0);
        assert_eq!(a[(1, 0)], 3.0);
        assert_eq!(a[(1, 1)], 10.0);
    }
}
