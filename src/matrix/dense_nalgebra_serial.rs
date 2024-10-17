use std::ops::{AddAssign, Mul, MulAssign};

use nalgebra::{
    DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorView, DVectorViewMut, RawStorage,
    RawStorageMut,
};

use crate::{scalar::Scale, IndexType, Scalar, Vector};

use super::default_solver::DefaultSolver;
use super::sparsity::{Dense, DenseRef};
use crate::error::DiffsolError;
use crate::{DenseMatrix, Matrix, MatrixCommon, MatrixView, MatrixViewMut, NalgebraLU};

impl<T: Scalar> DefaultSolver for DMatrix<T> {
    type LS = NalgebraLU<T>;
}

macro_rules! impl_matrix_common {
    ($matrix_type:ty) => {
        impl<'a, T: Scalar> MatrixCommon for $matrix_type {
            type V = DVector<T>;
            type T = T;

            fn nrows(&self) -> IndexType {
                self.nrows()
            }

            fn ncols(&self) -> IndexType {
                self.ncols()
            }
        }
    };
}

impl_matrix_common!(DMatrixViewMut<'a, T>);
impl_matrix_common!(DMatrixView<'a, T>);
impl_matrix_common!(DMatrix<T>);

macro_rules! impl_mul_scale {
    ($matrix_type:ty) => {
        impl<'a, T: Scalar> Mul<Scale<T>> for $matrix_type {
            type Output = DMatrix<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }

        impl<'a, T: Scalar> Mul<Scale<T>> for &$matrix_type {
            type Output = DMatrix<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }
    };
}

impl_mul_scale!(DMatrixView<'a, T>);
impl_mul_scale!(DMatrix<T>);

impl<'a, T: Scalar> MulAssign<Scale<T>> for DMatrixViewMut<'a, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        *self *= rhs.value();
    }
}

impl<'a, T: Scalar> MatrixView<'a> for DMatrixView<'a, T> {
    type Owned = DMatrix<T>;

    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        y.gemv(alpha, self, x, beta);
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.gemv(alpha, self, x, beta);
    }
}

impl<'a, T: Scalar> MatrixViewMut<'a> for DMatrixViewMut<'a, T> {
    type Owned = DMatrix<T>;
    type View = DMatrixView<'a, T>;
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }
}

impl<T: Scalar> Matrix for DMatrix<T> {
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
        let n = self.ncols();
        let m = self.nrows();
        (0..m).flat_map(move |i| (0..n).map(move |j| (i, j, &self[(i, j)])))
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
    ) -> Result<Self, DiffsolError> {
        let mut m = Self::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(m)
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self::zeros(nrows, ncols)
    }
    fn from_diagonal(v: &DVector<T>) -> Self {
        Self::from_diagonal(v)
    }
    fn diagonal(&self) -> Self::V {
        self.diagonal()
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.gemv(alpha, self, x, beta);
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        self.column_mut(j).copy_from(v);
    }
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        self.copy_from(y);
        self.mul_assign(beta);
        self.add_assign(x);
    }
    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        _sparsity: Option<Self::Sparsity>,
    ) -> Self {
        Self::zeros(nrows, ncols)
    }
}

impl<T: Scalar> DenseMatrix for DMatrix<T> {
    type View<'a> = DMatrixView<'a, T>;
    type ViewMut<'a> = DMatrixViewMut<'a, T>;

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.gemm(alpha, a, b, beta);
    }

    fn column_mut(&mut self, i: IndexType) -> DVectorViewMut<'_, T> {
        self.column_mut(i)
    }

    fn columns_mut(&mut self, start: IndexType, ncols: IndexType) -> Self::ViewMut<'_> {
        self.columns_mut(start, ncols)
    }

    fn column(&self, i: IndexType) -> DVectorView<'_, T> {
        self.column(i)
    }
    fn columns(&self, start: IndexType, ncols: IndexType) -> Self::View<'_> {
        self.columns(start, ncols)
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
            let value = unsafe {
                beta * *self.data.get_unchecked(k, i) + alpha * *self.data.get_unchecked(k, j)
            };
            unsafe {
                *self.data.get_unchecked_mut(k, i) = value;
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_axpy() {
        // M = [1 2]
        //     [3 4]
        let mut a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
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
