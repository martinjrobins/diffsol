use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use super::extract_block::CscBlock;
use super::sparsity::MatrixSparsityRef;
use super::{Matrix, MatrixCommon, MatrixSparsity};
use crate::error::{LaError, MatrixError};
use crate::{Context, FaerContext, FaerVec, FaerVecIndex, Vector, VectorIndex};
use crate::{DefaultSolver, FaerScalar, FaerSparseLU, IndexType, Scale};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::ops::{ternary_op_assign_into, union_symbolic};
use faer::sparse::{Pair, SparseColMat, SymbolicSparseColMat, SymbolicSparseColMatRef, Triplet};
use faer::zip;

#[derive(Clone, Debug)]
pub struct FaerSparseMat<T: FaerScalar> {
    pub(crate) data: Vec<SparseColMat<IndexType, T>>,
    pub(crate) context: FaerContext,
}

impl<T: FaerScalar> FaerSparseMat<T> {
    /// Matrix holding `batch`, broadcasting a single-batch operand over all batches.
    ///
    /// The dense backends return a column index here; a batch of a sparse matrix is a
    /// whole `SparseColMat`, so this returns that instead.
    #[inline]
    pub(crate) fn batch(&self, batch: usize) -> &SparseColMat<IndexType, T> {
        &self.data[batch % self.data.len()]
    }
}

impl<T: FaerScalar> DefaultSolver for FaerSparseMat<T> {
    type LS = FaerSparseLU<T>;
}

impl<T: FaerScalar> MatrixCommon for FaerSparseMat<T> {
    type T = T;
    type V = FaerVec<T>;
    type C = FaerContext;
    type Inner = Vec<SparseColMat<IndexType, T>>;

    fn nrows(&self) -> IndexType {
        self.data[0].nrows()
    }
    fn ncols(&self) -> IndexType {
        self.data[0].ncols()
    }
    fn inner(&self) -> &Self::Inner {
        &self.data
    }
}

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

impl_mul_scalar!(FaerSparseMat<T>, FaerSparseMat<T>);
impl_mul_scalar!(&FaerSparseMat<T>, FaerSparseMat<T>);

macro_rules! sparse_binary {
    ($trait:ident, $method:ident, $binary:tt) => {
        impl<T: FaerScalar> $trait<&FaerSparseMat<T>> for FaerSparseMat<T> {
            type Output = FaerSparseMat<T>;

            fn $method(self, rhs: &FaerSparseMat<T>) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                // either side may broadcast, so the result carries the larger batch count
                let nbatch = self.data.len().max(rhs.data.len());
                FaerSparseMat {
                    data: (0..nbatch)
                        .map(|b| self.batch(b) $binary rhs.batch(b))
                        .collect(),
                    context: if self.data.len() == nbatch {
                        self.context
                    } else {
                        rhs.context
                    },
                }
            }
        }
    };
}
sparse_binary!(Add, add, +);
sparse_binary!(Sub, sub, -);

impl<T: FaerScalar> MatrixSparsity<FaerSparseMat<T>> for SymbolicSparseColMat<IndexType> {
    fn union(
        self,
        other: SymbolicSparseColMatRef<IndexType>,
    ) -> Result<SymbolicSparseColMat<IndexType>, LaError> {
        union_symbolic(self.rb(), other).map_err(|e| LaError::Other(e.to_string()))
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
            for pos in self.col_range(col_i) {
                indices.push((self.row_idx()[pos], col_i));
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
    ) -> Result<Self, LaError> {
        let indices = indices
            .iter()
            .map(|(i, j)| Pair::new(*i, *j))
            .collect::<Vec<_>>();
        match Self::try_new_from_indices(nrows, ncols, indices.as_slice()) {
            Ok((sparsity, _)) => Ok(sparsity),
            Err(e) => Err(LaError::Other(e.to_string())),
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
            for pos in self.col_range(col_i) {
                indices.push((self.row_idx()[pos], col_i));
            }
        }
        indices
    }
}

impl<T: FaerScalar> Matrix for FaerSparseMat<T> {
    type Sparsity = SymbolicSparseColMat<IndexType>;
    type SparsityRef<'a> = SymbolicSparseColMatRef<'a, IndexType>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        Some(self.data[0].symbolic())
    }
    fn context(&self) -> &FaerContext {
        &self.context
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "gather");
        for (batch, data) in self.data.iter_mut().enumerate() {
            let src = other.batch(batch).val();
            for (dst, index) in data.val_mut().iter_mut().zip(&indices.data) {
                *dst = src[*index];
            }
        }
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        self.context
            .assert_compatible_nbatch(data.context.nbatch(), "set_data_with_indices");
        for (batch, matrix) in self.data.iter_mut().enumerate() {
            let values = matrix.val_mut();
            let column = data.data.rb().col(data.batch(batch));
            for (dst, src) in dst_indices.data.iter().zip(&src_indices.data) {
                values[*dst] = column[*src];
            }
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        self.context
            .assert_compatible_nbatch(v.context.nbatch(), "add_column_to_vector");
        for batch in 0..v.data.ncols() {
            let matrix = self.batch(batch);
            let mut column = v.data.rb_mut().col_mut(batch);
            for i in matrix.col_range(j) {
                column[matrix.row_idx()[i]] += matrix.val()[i];
            }
        }
    }

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let indices: Vec<_> = (0..self.ncols())
            .flat_map(move |j| {
                self.data[0]
                    .col_range(j)
                    .map(move |i| (self.data[0].row_idx()[i], j))
            })
            .collect();
        let values: Vec<_> = self
            .data
            .iter()
            .flat_map(|data| data.val().iter().copied())
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
                let triplets = indices
                    .iter()
                    .zip(values)
                    .map(|((i, j), v)| Triplet::new(*i, *j, *v))
                    .collect::<Vec<_>>();
                faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
                    .map_err(|e| LaError::from(MatrixError::FailedToCreateMatrixFromTriplets(e)))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { data, context: ctx })
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "gemv");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "gemv");
        for batch in 0..y.data.ncols() {
            let tmp = self.batch(batch) * x.data.rb().col(x.batch(batch));
            zip!(y.data.rb_mut().col_mut(batch), tmp.rb())
                .for_each(|faer::unzip!(y, x)| *y = beta * *y + alpha * *x);
        }
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        Self {
            data: (0..ctx.nbatch())
                .map(|_| SparseColMat::try_new_from_triplets(nrows, ncols, &[]).unwrap())
                .collect(),
            context: ctx,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.context
            .assert_compatible_nbatch(other.context.nbatch(), "copy_from");
        for (batch, data) in self.data.iter_mut().enumerate() {
            let other = other.batch(batch);
            *data = SparseColMat::new(other.symbolic().to_owned().unwrap(), other.val().to_vec());
        }
    }
    fn from_diagonal(v: &FaerVec<T>) -> Self {
        let dim = v.len();
        let data = (0..v.data.ncols())
            .map(|b| {
                let column = v.data.rb().col(b);
                let triplets = (0..dim)
                    .map(|i| Triplet::new(i, i, column[i]))
                    .collect::<Vec<_>>();
                SparseColMat::try_new_from_triplets(dim, dim, &triplets).unwrap()
            })
            .collect();
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
        self.context
            .assert_compatible_nbatch(v.context.nbatch(), "set_column");
        for (batch, data) in self.data.iter_mut().enumerate() {
            let column = v.data.rb().col(v.batch(batch));
            for i in data.col_range(j) {
                data.val_mut()[i] = column[data.row_idx()[i]];
            }
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "scale_add_and_assign");
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "scale_add_and_assign");
        for (batch, data) in self.data.iter_mut().enumerate() {
            ternary_op_assign_into(
                data.rb_mut(),
                x.batch(batch).rb(),
                y.batch(batch).rb(),
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
        let sparsity = sparsity.expect("Sparsity pattern required for sparse matrix");
        assert_eq!(sparsity.nrows(), nrows);
        assert_eq!(sparsity.ncols(), ncols);
        let nnz = sparsity.row_idx().len();
        Self {
            data: (0..ctx.nbatch())
                .map(|_| SparseColMat::new(sparsity.clone(), vec![T::zero(); nnz]))
                .collect(),
            context: ctx,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{FaerContext, FaerSparseMat, Matrix, MatrixSparsity, MatrixSparsityRef};
    use faer::sparse::SymbolicSparseColMat;

    #[test]
    fn sparsity_indices_round_trip() {
        let indices = vec![(0, 0), (3, 0), (1, 2), (2, 2)];
        let sparsity =
            <SymbolicSparseColMat<usize> as MatrixSparsity<FaerSparseMat<f64>>>::try_from_indices(
                4,
                3,
                indices.clone(),
            )
            .unwrap();

        let mut round_tripped =
            <SymbolicSparseColMat<usize> as MatrixSparsity<FaerSparseMat<f64>>>::indices(&sparsity);
        round_tripped.sort_unstable();
        let mut expected = indices;
        expected.sort_unstable();

        assert_eq!(round_tripped, expected);

        let sparsity_ref =
            <SymbolicSparseColMat<usize> as MatrixSparsity<FaerSparseMat<f64>>>::as_ref(&sparsity);
        let mut ref_round_tripped = MatrixSparsityRef::<FaerSparseMat<f64>>::indices(&sparsity_ref);
        ref_round_tripped.sort_unstable();

        assert_eq!(ref_round_tripped, expected);
    }

    #[test]
    fn test_triplet_iter() {
        let indices = vec![(0, 0), (1, 0), (2, 2), (3, 2)];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mat = FaerSparseMat::<f64>::try_from_triplets(
            4,
            3,
            indices.clone(),
            values.clone(),
            Default::default(),
        )
        .unwrap();
        let (iter_indices, iter_values): (Vec<_>, Vec<_>) = {
            let (i, v) = mat.triplet_iter();
            (i.collect(), v.collect())
        };
        assert_eq!(iter_indices, indices);
        assert_eq!(iter_values, values);
    }

    #[test]
    fn test_set_data_with_indices() {
        use crate::{FaerVec, FaerVecIndex, Vector, VectorIndex};

        // CSC data-array order: column-major, ascending row within each column.
        let triplets_full = [
            (0, 0, 1.0),
            (2, 0, 2.0),
            (1, 1, 3.0),
            (0, 2, 4.0),
            (2, 2, 5.0),
        ];
        let (indices, init_values): (Vec<_>, Vec<_>) =
            triplets_full.iter().map(|&(i, j, v)| ((i, j), v)).unzip();
        let mut mat = FaerSparseMat::<f64>::try_from_triplets(
            3,
            3,
            indices.clone(),
            init_values,
            Default::default(),
        )
        .unwrap();

        assert_eq!(mat.inner_mut()[0].val_mut().len(), indices.len());

        let new_values = [10.0, 11.0, 12.0, 13.0, 14.0];
        mat.inner_mut()[0].val_mut().copy_from_slice(&new_values);
        let (indices_iter, values_iter) = mat.triplet_iter();
        let got: Vec<(usize, usize, f64)> = indices_iter
            .zip(values_iter)
            .map(|((i, j), v)| (i, j, v))
            .collect();
        let expected: Vec<(usize, usize, f64)> = triplets_full
            .iter()
            .zip(new_values.iter())
            .map(|(&(i, j, _), &v)| (i, j, v))
            .collect();
        assert_eq!(got, expected);

        let mut via_set_data = FaerSparseMat::<f64>::try_from_triplets(
            3,
            3,
            indices,
            new_values.iter().map(|_| 0.0).collect(),
            Default::default(),
        )
        .unwrap();
        let nnz = new_values.len();
        let identity = FaerVecIndex::from_vec((0..nnz).collect(), Default::default());
        let data = FaerVec::from_vec(new_values.to_vec(), Default::default());
        via_set_data.set_data_with_indices(&identity, &identity, &data);
        assert_eq!(
            mat.inner_mut()[0].val_mut(),
            via_set_data.inner_mut()[0].val_mut()
        );
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<FaerSparseMat<f64>>();
    }

    super::super::generate_matrix_tests_nonbatched!(faer_sparse, FaerSparseMat<f64>);
    super::super::generate_matrix_tests_batched!(
        faer_sparse,
        FaerSparseMat<f64>,
        FaerContext::default(),
        FaerContext::with_nbatch(2)
    );
}
