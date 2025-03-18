use std::fmt::Debug;
use std::ops::Mul;

use super::extract_block::CscBlock;
use super::sparsity::MatrixSparsityRef;
use super::{Matrix, MatrixCommon, MatrixSparsity};
use crate::error::{DiffsolError, MatrixError};
use crate::{DefaultSolver, FaerSparseLU, IndexType, Scalar, Scale};
use crate::{Vector, VectorIndex};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::ops::{ternary_op_assign_into, union_symbolic};
use faer::sparse::{Pair, SymbolicSparseColMat, SymbolicSparseColMatRef, Triplet};
use faer::Col;

pub struct SparseColMat<T: Scalar>(faer::sparse::SparseColMat<IndexType, T>);

impl<T: Scalar> Debug for SparseColMat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Scalar> Clone for SparseColMat<T> {
    fn clone(&self) -> Self {
        let sparsity = self.0.symbolic().to_owned().unwrap();
        let values = self.0.val().to_vec();
        Self(faer::sparse::SparseColMat::new(sparsity, values))
    }
}

impl<T: Scalar> SparseColMat<T> {
    pub fn faer(&self) -> &faer::sparse::SparseColMat<IndexType, T> {
        &self.0
    }
    pub fn faer_mut(&mut self) -> &mut faer::sparse::SparseColMat<IndexType, T> {
        &mut self.0
    }
}

impl<T: Scalar> DefaultSolver for SparseColMat<T> {
    type LS = FaerSparseLU<T>;
}

impl<T: Scalar> MatrixCommon for SparseColMat<T> {
    type T = T;
    type V = Col<T>;

    fn nrows(&self) -> IndexType {
        self.0.nrows()
    }
    fn ncols(&self) -> IndexType {
        self.0.ncols()
    }
}

impl<T: Scalar> MatrixSparsity<SparseColMat<T>> for SymbolicSparseColMat<IndexType> {
    fn union(
        self,
        other: SymbolicSparseColMatRef<IndexType>,
    ) -> Result<SymbolicSparseColMat<IndexType>, DiffsolError> {
        union_symbolic(self.rb(), other).map_err(|e| DiffsolError::Other(e.to_string()))
    }

    fn as_ref(&self) -> SymbolicSparseColMatRef<IndexType> {
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
    ) -> <<SparseColMat<T> as MatrixCommon>::V as Vector>::Index {
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
        ret
    }
}

impl<'a, T: Scalar> MatrixSparsityRef<'a, SparseColMat<T>>
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
        indices: &<<SparseColMat<T> as MatrixCommon>::V as Vector>::Index,
        transpose: bool,
    ) -> [(
        SymbolicSparseColMat<IndexType>,
        <<SparseColMat<T> as MatrixCommon>::V as Vector>::Index,
    ); 4] {
        let (_ni, _nj, col_ptrs, _col_nnz, row_idx) = self.parts();
        let (ul_blk, ur_blk, ll_blk, lr_blk) =
            CscBlock::split(row_idx, col_ptrs, indices, transpose);
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
            (ul_sym, ul_blk.src_indices),
            (ur_sym, ur_blk.src_indices),
            (ll_sym, ll_blk.src_indices),
            (lr_sym, lr_blk.src_indices),
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

impl<T: Scalar> Mul<Scale<T>> for SparseColMat<T> {
    type Output = SparseColMat<T>;

    fn mul(mut self, rhs: Scale<T>) -> Self::Output {
        for v in self.0.val_mut() {
            v.mul_assign(rhs.value());
        }
        self
    }
}

impl<T: Scalar> Mul<Scale<T>> for &SparseColMat<T> {
    type Output = SparseColMat<T>;

    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: Scalar> Matrix for SparseColMat<T> {
    type Sparsity = SymbolicSparseColMat<IndexType>;
    type SparsityRef<'a> = SymbolicSparseColMatRef<'a, IndexType>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        Some(self.0.symbolic())
    }

    fn copy_block_from(&mut self, src_indices: &<Self::V as Vector>::Index, parent: &Self) {
        CscBlock::copy_block_from(self.0.val_mut(), parent.0.val(), src_indices);
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        let values = self.0.val_mut();
        for (dst_i, src_i) in dst_indices.iter().zip(src_indices.iter()) {
            values[*dst_i] = data[*src_i];
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        for i in self.0.col_range(j) {
            let row = self.0.row_idx()[i];
            v[row] += self.0.val()[i];
        }
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, &Self::T)> {
        (0..self.ncols()).flat_map(move |j| {
            self.0.col_range(j).map(move |i| {
                let row = self.0.row_idx()[i];
                (row, j, &self.0.val()[i])
            })
        })
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
    ) -> Result<Self, DiffsolError> {
        let triplets = triplets
            .iter()
            .map(|(i, j, v)| Triplet::new(*i, *j, *v))
            .collect::<Vec<_>>();
        match faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, triplets.as_slice()) {
            Ok(mat) => Ok(Self(mat)),
            Err(e) => Err(DiffsolError::from(
                MatrixError::FailedToCreateMatrixFromTriplets(e),
            )),
        }
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let tmp = &self.0 * x;
        y.axpy(alpha, &tmp, beta);
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self(faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, &[]).unwrap())
    }
    fn copy_from(&mut self, other: &Self) {
        self.0 = faer::sparse::SparseColMat::new(
            other.0.symbolic().to_owned().unwrap(),
            other.0.val().to_vec(),
        )
    }
    fn from_diagonal(v: &Col<T>) -> Self {
        let dim = v.nrows();
        let triplets = (0..dim)
            .map(|i| Triplet::new(i, i, v[i]))
            .collect::<Vec<_>>();
        Self(faer::sparse::SparseColMat::try_new_from_triplets(dim, dim, &triplets).unwrap())
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let mut indices_zero_diag = vec![];
        let mut indices_non_zero_diag = vec![];
        'outer: for j in 0..self.ncols() {
            for (i, v) in self.0.row_idx_of_col(j).zip(self.0.val_of_col(j)) {
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
            <Self::V as Vector>::Index::from_vec(indices_zero_diag),
            <Self::V as Vector>::Index::from_vec(indices_non_zero_diag),
        )
    }

    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        assert_eq!(v.len(), self.nrows());
        for i in self.0.col_range(j) {
            let row_i = self.0.row_idx()[i];
            self.0.val_mut()[i] = v[row_i];
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        ternary_op_assign_into(self.0.rb_mut(), x.0.rb(), y.0.rb(), |s, x, y| {
            *s = *x.unwrap_or(&T::zero()) + beta * *y.unwrap_or(&T::zero())
        });
    }

    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
    ) -> Self {
        let sparsity = sparsity.expect("Sparsity pattern required for sparse matrix");
        assert_eq!(sparsity.nrows(), nrows);
        assert_eq!(sparsity.ncols(), ncols);
        let nnz = sparsity.row_idx().len();
        Self(faer::sparse::SparseColMat::new(
            sparsity,
            vec![T::zero(); nnz],
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, SparseColMat};
    #[test]
    fn test_triplet_iter() {
        let triplets = vec![(0, 0, 1.0), (1, 0, 2.0), (2, 2, 3.0), (3, 2, 4.0)];
        let mat = SparseColMat::<f64>::try_from_triplets(4, 3, triplets.clone()).unwrap();
        let mut iter = mat.triplet_iter();
        for triplet in triplets {
            let (i, j, val) = iter.next().unwrap();
            assert_eq!(i, triplet.0);
            assert_eq!(j, triplet.1);
            assert_eq!(*val, triplet.2);
        }
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<SparseColMat<f64>>();
    }
}
