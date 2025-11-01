use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use super::extract_block::CscBlock;
use super::sparsity::MatrixSparsityRef;
use super::utils::*;
use super::{Matrix, MatrixCommon, MatrixSparsity};
use crate::error::{DiffsolError, MatrixError};
use crate::{DefaultSolver, FaerSparseLU, IndexType, Scalar, Scale};
use crate::{FaerContext, FaerVec, FaerVecIndex, Vector, VectorIndex};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::ops::{ternary_op_assign_into, union_symbolic};
use faer::sparse::{Pair, SparseColMat, SymbolicSparseColMat, SymbolicSparseColMatRef, Triplet};

#[derive(Clone, Debug)]
pub struct FaerSparseMat<T: Scalar> {
    pub(crate) data: SparseColMat<IndexType, T>,
    pub(crate) context: FaerContext,
}

impl<T: Scalar> DefaultSolver for FaerSparseMat<T> {
    type LS = FaerSparseLU<T>;
}

impl_matrix_common!(FaerSparseMat<T>, FaerVec<T>, FaerContext, SparseColMat<IndexType, T>);

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

impl_mul_scalar!(FaerSparseMat<T>, FaerSparseMat<T>);
impl_mul_scalar!(&FaerSparseMat<T>, FaerSparseMat<T>);

impl_add!(FaerSparseMat<T>, &FaerSparseMat<T>, FaerSparseMat<T>);

impl_sub!(FaerSparseMat<T>, &FaerSparseMat<T>, FaerSparseMat<T>);

impl<T: Scalar> MatrixSparsity<FaerSparseMat<T>> for SymbolicSparseColMat<IndexType> {
    fn union(
        self,
        other: SymbolicSparseColMatRef<IndexType>,
    ) -> Result<SymbolicSparseColMat<IndexType>, DiffsolError> {
        union_symbolic(self.rb(), other).map_err(|e| DiffsolError::Other(e.to_string()))
    }

    fn as_ref(&self) -> SymbolicSparseColMatRef<'_, IndexType> {
        self.rb()
    }

    fn nrows(&self) -> IndexType {
        self.nrows()
    }

    fn ncols(&self) -> IndexType {
        self.ncols()
    }

    fn is_sparse() -> bool {
        true
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        let mut indices = Vec::with_capacity(self.compute_nnz());
        for col_i in 0..self.ncols() {
            for row_j in self.col_range(col_i) {
                indices.push((row_j, col_i));
            }
        }
        indices
    }

    fn new_diagonal(n: IndexType) -> Self {
        let indices = (0..n).map(|i| Pair::new(i, i)).collect::<Vec<_>>();
        SymbolicSparseColMat::try_new_from_indices(n, n, indices.as_slice())
            .unwrap()
            .0
    }

    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self, DiffsolError> {
        let indices = indices
            .iter()
            .map(|(i, j)| Pair::new(*i, *j))
            .collect::<Vec<_>>();
        match Self::try_new_from_indices(nrows, ncols, indices.as_slice()) {
            Ok((sparsity, _)) => Ok(sparsity),
            Err(e) => Err(DiffsolError::Other(e.to_string())),
        }
    }

    fn get_index(
        &self,
        indices: &[(IndexType, IndexType)],
        ctx: FaerContext,
    ) -> <<FaerSparseMat<T> as MatrixCommon>::V as Vector>::Index {
        let col_ptrs = self.col_ptr();
        let row_indices = self.row_idx();
        let mut ret = Vec::with_capacity(indices.len());
        for &(i, j) in indices.iter() {
            let col_ptr = col_ptrs[j];
            let next_col_ptr = col_ptrs[j + 1];
            for (ii, &ri) in row_indices
                .iter()
                .enumerate()
                .take(next_col_ptr)
                .skip(col_ptr)
            {
                if ri == i {
                    ret.push(ii);
                    break;
                }
            }
        }
        FaerVecIndex {
            data: ret,
            context: ctx,
        }
    }
}

impl<'a, T: Scalar> MatrixSparsityRef<'a, FaerSparseMat<T>>
    for SymbolicSparseColMatRef<'a, IndexType>
{
    fn to_owned(&self) -> SymbolicSparseColMat<IndexType> {
        self.to_owned().unwrap()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }

    fn ncols(&self) -> IndexType {
        self.ncols()
    }

    fn is_sparse() -> bool {
        true
    }

    fn split(
        &self,
        indices: &<<FaerSparseMat<T> as MatrixCommon>::V as Vector>::Index,
    ) -> [(
        SymbolicSparseColMat<IndexType>,
        <<FaerSparseMat<T> as MatrixCommon>::V as Vector>::Index,
    ); 4] {
        let (_ni, _nj, col_ptrs, _col_nnz, row_idx) = self.parts();
        let ctx = indices.context();
        let (ul_blk, ur_blk, ll_blk, lr_blk) = CscBlock::split(row_idx, col_ptrs, indices);
        let ul_sym = SymbolicSparseColMat::new_checked(
            ul_blk.nrows,
            ul_blk.ncols,
            ul_blk.col_pointers,
            None,
            ul_blk.row_indices,
        );
        let ur_sym = SymbolicSparseColMat::new_checked(
            ur_blk.nrows,
            ur_blk.ncols,
            ur_blk.col_pointers,
            None,
            ur_blk.row_indices,
        );
        let ll_sym = SymbolicSparseColMat::new_checked(
            ll_blk.nrows,
            ll_blk.ncols,
            ll_blk.col_pointers,
            None,
            ll_blk.row_indices,
        );
        let lr_sym = SymbolicSparseColMat::new_checked(
            lr_blk.nrows,
            lr_blk.ncols,
            lr_blk.col_pointers,
            None,
            lr_blk.row_indices,
        );
        [
            (
                ul_sym,
                FaerVecIndex {
                    data: ul_blk.src_indices,
                    context: *ctx,
                },
            ),
            (
                ur_sym,
                FaerVecIndex {
                    data: ur_blk.src_indices,
                    context: *ctx,
                },
            ),
            (
                ll_sym,
                FaerVecIndex {
                    data: ll_blk.src_indices,
                    context: *ctx,
                },
            ),
            (
                lr_sym,
                FaerVecIndex {
                    data: lr_blk.src_indices,
                    context: *ctx,
                },
            ),
        ]
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        let mut indices = Vec::with_capacity(self.compute_nnz());
        for col_i in 0..self.ncols() {
            for row_j in self.col_range(col_i) {
                indices.push((row_j, col_i));
            }
        }
        indices
    }
}

impl<T: Scalar> Matrix for FaerSparseMat<T> {
    type Sparsity = SymbolicSparseColMat<IndexType>;
    type SparsityRef<'a> = SymbolicSparseColMatRef<'a, IndexType>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        Some(self.data.symbolic())
    }
    fn context(&self) -> &FaerContext {
        &self.context
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        let dst_data = self.data.val_mut();
        let src_data = other.data.val();
        for (dst_i, idx) in dst_data.iter_mut().zip(indices.data.iter()) {
            *dst_i = src_data[*idx];
        }
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        let values = self.data.val_mut();
        for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
            values[*dst_i] = data[*src_i];
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        for i in self.data.col_range(j) {
            let row = self.data.row_idx()[i];
            v[row] += self.data.val()[i];
        }
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, Self::T)> {
        (0..self.ncols()).flat_map(move |j| {
            self.data.col_range(j).map(move |i| {
                let row = self.data.row_idx()[i];
                (row, j, self.data.val()[i])
            })
        })
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let triplets = triplets
            .iter()
            .map(|(i, j, v)| Triplet::new(*i, *j, *v))
            .collect::<Vec<_>>();
        match faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, triplets.as_slice()) {
            Ok(mat) => Ok(Self {
                data: mat,
                context: ctx,
            }),
            Err(e) => Err(DiffsolError::from(
                MatrixError::FailedToCreateMatrixFromTriplets(e),
            )),
        }
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let tmp = Self::V {
            data: &self.data * &x.data,
            context: self.context,
        };
        y.axpy(alpha, &tmp, beta);
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        Self {
            data: SparseColMat::try_new_from_triplets(nrows, ncols, &[]).unwrap(),
            context: ctx,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data = faer::sparse::SparseColMat::new(
            other.data.symbolic().to_owned().unwrap(),
            other.data.val().to_vec(),
        )
    }
    fn from_diagonal(v: &FaerVec<T>) -> Self {
        let dim = v.len();
        let triplets = (0..dim)
            .map(|i| Triplet::new(i, i, v[i]))
            .collect::<Vec<_>>();
        Self {
            data: SparseColMat::try_new_from_triplets(dim, dim, &triplets).unwrap(),
            context: *v.context(),
        }
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let mut indices_zero_diag = vec![];
        let mut indices_non_zero_diag = vec![];
        'outer: for j in 0..self.ncols() {
            for (i, v) in self.data.row_idx_of_col(j).zip(self.data.val_of_col(j)) {
                if i == j && *v != T::zero() {
                    indices_non_zero_diag.push(j);
                    continue 'outer;
                } else if i > j {
                    break;
                }
            }
            indices_zero_diag.push(j);
        }
        (
            <Self::V as Vector>::Index::from_vec(indices_zero_diag, self.context),
            <Self::V as Vector>::Index::from_vec(indices_non_zero_diag, self.context),
        )
    }

    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        assert_eq!(v.len(), self.nrows());
        for i in self.data.col_range(j) {
            let row_i = self.data.row_idx()[i];
            self.data.val_mut()[i] = v[row_i];
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        ternary_op_assign_into(self.data.rb_mut(), x.data.rb(), y.data.rb(), |s, x, y| {
            *s = *x.unwrap_or(&T::zero()) + beta * *y.unwrap_or(&T::zero())
        });
    }

    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self {
        let sparsity = sparsity.expect("Sparsity pattern required for sparse matrix");
        assert_eq!(sparsity.nrows(), nrows);
        assert_eq!(sparsity.ncols(), ncols);
        let nnz = sparsity.row_idx().len();
        Self {
            data: SparseColMat::new(sparsity, vec![T::zero(); nnz]),
            context: ctx,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{FaerSparseMat, Matrix};
    #[test]
    fn test_triplet_iter() {
        let triplets = vec![(0, 0, 1.0), (1, 0, 2.0), (2, 2, 3.0), (3, 2, 4.0)];
        let mat =
            FaerSparseMat::<f64>::try_from_triplets(4, 3, triplets.clone(), Default::default())
                .unwrap();
        let mut iter = mat.triplet_iter();
        for triplet in triplets {
            let (i, j, val) = iter.next().unwrap();
            assert_eq!(i, triplet.0);
            assert_eq!(j, triplet.1);
            assert_eq!(val, triplet.2);
        }
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<FaerSparseMat<f64>>();
    }
}
