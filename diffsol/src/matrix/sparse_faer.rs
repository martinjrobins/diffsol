use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use super::extract_block::CscBlock;
use super::sparsity::MatrixSparsityRef;
use super::{Matrix, MatrixCommon, MatrixSparsity};
use crate::error::{DiffsolError, MatrixError};
use crate::{Context, FaerContext, FaerVec, FaerVecIndex, Vector, VectorIndex};
use crate::{DefaultSolver, FaerScalar, FaerSparseLU, IndexType, Scalar, Scale};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::sparse::ops::{ternary_op_assign_into, union_symbolic};
use faer::sparse::{Pair, SparseColMat, SymbolicSparseColMat, SymbolicSparseColMatRef, Triplet};
use faer::Accum;

/// Sparse matrix backed by a faer [`SparseColMat`].
///
/// # Data layout with batching
///
/// When `nbatch > 1`, `data` is a `Vec` containing one independent
/// [`SparseColMat`] per batch.  All batches **must share the same sparsity
/// pattern** (symbolic structure); only the numeric values differ.
///
/// Broadcasting from `nbatch = 1` is supported for [`gemv`] but not for
/// arithmetic operations (Add, Sub, etc.).
#[derive(Clone, Debug)]
pub struct FaerSparseMat<T: FaerScalar> {
    pub(crate) data: Vec<SparseColMat<IndexType, T>>,
    pub(crate) context: FaerContext,
}

impl<T: FaerScalar> DefaultSolver for FaerSparseMat<T> {
    type LS = FaerSparseLU<T>;
}

impl<T: Scalar + FaerScalar> MatrixCommon for FaerSparseMat<T> {
    type T = T;
    type V = FaerVec<T>;
    type C = FaerContext;
    type Inner = Vec<SparseColMat<IndexType, T>>;

    fn nrows(&self) -> IndexType {
        self.data.first().map(|m| m.nrows()).unwrap_or(0)
    }
    fn ncols(&self) -> IndexType {
        self.data.first().map(|m| m.ncols()).unwrap_or(0)
    }
    fn inner(&self) -> &Self::Inner {
        &self.data
    }
}

impl<'a, T: FaerScalar> Mul<Scale<T>> for &FaerSparseMat<T> {
    type Output = FaerSparseMat<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        let scale: faer::Scale<T> = rhs.into();
        FaerSparseMat {
            data: self.data.iter().map(|m| &*m * scale).collect(),
            context: self.context,
        }
    }
}

impl<T: FaerScalar> Mul<Scale<T>> for FaerSparseMat<T> {
    type Output = FaerSparseMat<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T: Scalar + FaerScalar> Add<&FaerSparseMat<T>> for FaerSparseMat<T> {
    type Output = FaerSparseMat<T>;
    fn add(self, rhs: &FaerSparseMat<T>) -> Self::Output {
        FaerSparseMat {
            data: self
                .data
                .into_iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            context: self.context,
        }
    }
}

impl<T: Scalar + FaerScalar> Sub<&FaerSparseMat<T>> for FaerSparseMat<T> {
    type Output = FaerSparseMat<T>;
    fn sub(self, rhs: &FaerSparseMat<T>) -> Self::Output {
        FaerSparseMat {
            data: self
                .data
                .into_iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
            context: self.context,
        }
    }
}

impl<T: FaerScalar> MatrixSparsity<FaerSparseMat<T>> for SymbolicSparseColMat<IndexType> {
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

impl<'a, T: FaerScalar> MatrixSparsityRef<'a, FaerSparseMat<T>>
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

impl<T: FaerScalar> Matrix for FaerSparseMat<T> {
    type Sparsity = SymbolicSparseColMat<IndexType>;
    type SparsityRef<'a> = SymbolicSparseColMatRef<'a, IndexType>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        self.data.first().map(|m| m.symbolic())
    }
    fn context(&self) -> &FaerContext {
        &self.context
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        let nbatch = self.data.len();
        for b in 0..nbatch {
            let dst_data = self.data[b].val_mut();
            let src_data = other.data[b].val();
            for (dst_i, idx) in dst_data.iter_mut().zip(indices.data.iter()) {
                *dst_i = src_data[*idx];
            }
        }
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        let nbatch = self.data.len();
        for b in 0..nbatch {
            let values = self.data[b].val_mut();
            for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
                values[*dst_i] = data.data[(*src_i, b)];
            }
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        let nbatch = self.data.len();
        for b in 0..nbatch {
            for i in self.data[b].col_range(j) {
                let row = self.data[b].row_idx()[i];
                v.data[(row, b)] += self.data[b].val()[i];
            }
        }
    }

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let nbatch = self.data.len();
        let ncols = self.ncols();
        let indices = (0..ncols).flat_map(move |j| {
            self.data[0].col_range(j).map(move |i| {
                let row = self.data[0].row_idx()[i];
                (row, j)
            })
        });
        let values = (0..nbatch).flat_map(move |b| {
            (0..ncols).flat_map(move |j| {
                self.data[b]
                    .col_range(j)
                    .map(move |i| self.data[b].val()[i])
            })
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
        assert_eq!(
            values.len(),
            nnz * nbatch,
            "values length {} must be nnz ({}) * nbatch ({})",
            values.len(),
            nnz,
            nbatch
        );
        let mut data = Vec::with_capacity(nbatch);
        for b in 0..nbatch {
            let batch_values = &values[b * nnz..(b + 1) * nnz];
            let triplets = indices
                .iter()
                .zip(batch_values)
                .map(|(&(i, j), &v)| Triplet::new(i, j, v))
                .collect::<Vec<_>>();
            match faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, &triplets) {
                Ok(mat) => data.push(mat),
                Err(e) => {
                    return Err(DiffsolError::from(
                        MatrixError::FailedToCreateMatrixFromTriplets(e),
                    ))
                }
            }
        }
        Ok(Self { data, context: ctx })
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let x_nbatch = x.data.ncols();
        let self_nbatch = self.data.len();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv");
        let max_nbatch = self_nbatch.max(x_nbatch);
        for b in 0..max_nbatch {
            let mat_b = if self_nbatch == 1 { 0 } else { b };
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let mut y_col = y.data.col_mut(b);
            y_col *= faer::Scale(beta);
            sparse_dense_matmul(
                y_col.as_mat_mut(),
                Accum::Add,
                self.data[mat_b].rb(),
                x.data.col(x_b).as_mat(),
                alpha,
                self.context.par,
            );
        }
    }

    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let empty = SparseColMat::try_new_from_triplets(nrows, ncols, &[]).unwrap();
        let data = (0..nbatch).map(|_| empty.clone()).collect();
        Self { data, context: ctx }
    }

    fn copy_from(&mut self, other: &Self) {
        let nbatch = other.data.len();
        self.data.clear();
        for b in 0..nbatch {
            self.data.push(faer::sparse::SparseColMat::new(
                other.data[b].symbolic().to_owned().unwrap(),
                other.data[b].val().to_vec(),
            ));
        }
    }

    fn from_diagonal(v: &FaerVec<T>) -> Self {
        let nbatch = v.context().nbatch();
        let dim = v.len();
        let mut data = Vec::with_capacity(nbatch);
        for b in 0..nbatch {
            let triplets = (0..dim)
                .map(|i| Triplet::new(i, i, v.data[(i, b)]))
                .collect::<Vec<_>>();
            data.push(SparseColMat::try_new_from_triplets(dim, dim, &triplets).unwrap());
        }
        Self {
            data,
            context: *v.context(),
        }
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let mut indices_zero_diag = vec![];
        let mut indices_non_zero_diag = vec![];
        'outer: for j in 0..self.ncols() {
            for (i, v) in self.data[0]
                .row_idx_of_col(j)
                .zip(self.data[0].val_of_col(j))
            {
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
        let nbatch = self.data.len();
        for b in 0..nbatch {
            for i in self.data[b].col_range(j) {
                let row_i = self.data[b].row_idx()[i];
                self.data[b].val_mut()[i] = v.data[(row_i, b)];
            }
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        let nbatch = self.data.len();
        for b in 0..nbatch {
            ternary_op_assign_into(
                self.data[b].rb_mut(),
                x.data[b].rb(),
                y.data[b].rb(),
                |s, x, y| *s = *x.unwrap_or(&T::zero()) + beta * *y.unwrap_or(&T::zero()),
            );
        }
    }

    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self {
        let nbatch = ctx.nbatch();
        let sparsity = sparsity.expect("Sparsity pattern required for sparse matrix");
        assert_eq!(sparsity.nrows(), nrows);
        assert_eq!(sparsity.ncols(), ncols);
        let nnz = sparsity.row_idx().len();
        let data = (0..nbatch)
            .map(|_| SparseColMat::new(sparsity.clone(), vec![T::zero(); nnz]))
            .collect();
        Self { data, context: ctx }
    }
}

#[cfg(test)]
mod tests {
    use crate::{FaerContext, FaerSparseMat};

    super::super::generate_matrix_tests!(
        faer_sparse,
        FaerSparseMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );

    super::super::generate_sparse_matrix_tests!(
        faer_sparse,
        FaerSparseMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );
}
