use crate::{vector::VectorIndex, IndexType, Matrix, Scalar};

use super::DenseMatrix;
pub(crate) struct CscBlock {
    pub(crate) nrows: IndexType,
    pub(crate) ncols: IndexType,
    pub(crate) row_indices: Vec<IndexType>,
    pub(crate) col_pointers: Vec<IndexType>,
    pub(crate) src_indices: Vec<IndexType>,
}

impl CscBlock {
    fn new_from_vec(
        nrows: IndexType,
        ncols: IndexType,
        cols: Vec<Vec<(IndexType, IndexType)>>,
    ) -> Self {
        assert_eq!(ncols, cols.len());
        let mut row_indices = Vec::new();
        let mut col_pointers = Vec::new();
        let mut src_indices = Vec::new();
        let mut acc = 0;
        for col in cols.into_iter() {
            col_pointers.push(acc);
            acc += col.len();
            for (src_i, my_i) in col.into_iter() {
                assert!(my_i < nrows);
                row_indices.push(my_i);
                src_indices.push(src_i);
            }
        }
        col_pointers.push(acc);
        assert_eq!(col_pointers.len(), ncols + 1);
        Self {
            ncols,
            nrows,
            row_indices,
            col_pointers,
            src_indices,
        }
    }

    fn new(nrows: IndexType, ncols: IndexType) -> Self {
        Self {
            ncols,
            nrows,
            row_indices: Vec::new(),
            col_pointers: Vec::new(),
            src_indices: Vec::new(),
        }
    }

    pub fn copy_block_from<I: VectorIndex, T: Scalar>(
        data: &mut [T],
        src_data: &[T],
        src_indices: &I,
    ) {
        for i in 0..data.len() {
            data[i] = src_data[src_indices[i]];
        }
    }

    /// split a square csc matrix into four blocks based on a predicate
    ///
    /// M = [UL(false, false), UR(false, true)]
    ///     [LL(true, false), LR(true, true)]
    ///     
    /// return order is (UL, UR, LL, LR)
    ///
    pub fn split<F>(
        row_indices: &[IndexType],
        col_pointers: &[IndexType],
        f: F,
        transpose: bool,
    ) -> (Self, Self, Self, Self)
    where
        F: Fn(IndexType) -> bool,
    {
        if transpose {
            Self::split_transpose(row_indices, col_pointers, f)
        } else {
            Self::split_no_tranpose(row_indices, col_pointers, f)
        }
    }

    fn split_no_tranpose<F>(
        row_indices: &[IndexType],
        col_pointers: &[IndexType],
        f: F,
    ) -> (Self, Self, Self, Self)
    where
        F: Fn(IndexType) -> bool,
    {
        let n = col_pointers.len() - 1;
        let (cat, upper_indices, lower_indices, nni, ni) = Self::setup_split(f, n);
        let mut ur = Self::new(nni, ni);
        let mut ul = Self::new(nni, nni);
        let mut lr = Self::new(ni, ni);
        let mut ll = Self::new(ni, nni);
        for j in 0..n {
            let col_ptr = col_pointers[j];
            let next_col_ptr = col_pointers[j + 1];
            if cat[j] {
                ur.col_pointers.push(ur.row_indices.len());
                lr.col_pointers.push(lr.row_indices.len());
                for (data_i, &i) in row_indices
                    .iter()
                    .enumerate()
                    .take(next_col_ptr)
                    .skip(col_ptr)
                {
                    if !cat[i] {
                        let ii = upper_indices[i];
                        ur.row_indices.push(ii);
                        ur.src_indices.push(data_i);
                    } else {
                        let ii = lower_indices[i];
                        lr.row_indices.push(ii);
                        lr.src_indices.push(data_i);
                    }
                }
            } else {
                ul.col_pointers.push(ul.row_indices.len());
                ll.col_pointers.push(ll.row_indices.len());
                for (data_i, &i) in row_indices
                    .iter()
                    .enumerate()
                    .take(next_col_ptr)
                    .skip(col_ptr)
                {
                    if !cat[i] {
                        let ii = upper_indices[i];
                        ul.row_indices.push(ii);
                        ul.src_indices.push(data_i);
                    } else {
                        let ii = lower_indices[i];
                        ll.row_indices.push(ii);
                        ll.src_indices.push(data_i);
                    }
                }
            }
        }

        ur.col_pointers.push(ur.row_indices.len());
        ul.col_pointers.push(ul.row_indices.len());
        lr.col_pointers.push(lr.row_indices.len());
        ll.col_pointers.push(ll.row_indices.len());
        (ul, ur, ll, lr)
    }

    /// split a square csc matrix into four blocks based on a predicate
    ///
    /// M = [UL(false, false), UR(false, true)]
    ///     [LL(true, false), LR(true, true)]
    ///     
    /// returns the transpose of each block, but in the same return order
    /// return order is (UL, UR, LL, LR)
    ///
    pub fn split_transpose<F>(
        row_indices: &[IndexType],
        col_pointers: &[IndexType],
        f: F,
    ) -> (Self, Self, Self, Self)
    where
        F: Fn(IndexType) -> bool,
    {
        let n = col_pointers.len() - 1;
        let (cat, upper_indices, lower_indices, n_up, n_low) = Self::setup_split(f, n);
        let mut ur_tmp = vec![Vec::new(); n_up];
        let mut ul_tmp = vec![Vec::new(); n_low];
        let mut lr_tmp = vec![Vec::new(); n_up];
        let mut ll_tmp = vec![Vec::new(); n_low];

        for j in 0..n {
            let col_ptr = col_pointers[j];
            let next_col_ptr = col_pointers[j + 1];
            if cat[j] {
                let jj = lower_indices[j];
                for (data_i, &i) in row_indices
                    .iter()
                    .enumerate()
                    .take(next_col_ptr)
                    .skip(col_ptr)
                {
                    if !cat[i] {
                        let ii = upper_indices[i];
                        ur_tmp[ii].push((data_i, jj));
                    } else {
                        let ii = lower_indices[i];
                        lr_tmp[ii].push((data_i, jj));
                    }
                }
            } else {
                let jj = upper_indices[j];
                for (data_i, &i) in row_indices
                    .iter()
                    .enumerate()
                    .take(next_col_ptr)
                    .skip(col_ptr)
                {
                    if !cat[i] {
                        let ii = upper_indices[i];
                        ul_tmp[ii].push((data_i, jj));
                    } else {
                        let ii = lower_indices[i];
                        ll_tmp[ii].push((data_i, jj));
                    }
                }
            }
        }
        let ur = Self::new_from_vec(n_low, n_up, ur_tmp);
        let ul = Self::new_from_vec(n_low, n_low, ul_tmp);
        let lr = Self::new_from_vec(n_up, n_up, lr_tmp);
        let ll = Self::new_from_vec(n_up, n_low, ll_tmp);
        (ul, ur, ll, lr)
    }

    fn setup_split<F>(
        f: F,
        n: usize,
    ) -> (
        Vec<bool>,
        Vec<IndexType>,
        Vec<IndexType>,
        IndexType,
        IndexType,
    )
    where
        F: Fn(IndexType) -> bool,
    {
        let cat = (0..n).map(f).collect::<Vec<_>>();

        let mut upper_indices = Vec::with_capacity(n);
        let mut lower_indices = Vec::with_capacity(n);
        let mut upper_acc = 0;
        let mut lower_acc = 0;
        for c in cat.iter() {
            lower_indices.push(lower_acc);
            upper_indices.push(upper_acc);
            if *c {
                lower_acc += 1;
            } else {
                upper_acc += 1;
            }
        }
        (cat, upper_indices, lower_indices, upper_acc, lower_acc)
    }
}

pub(crate) struct ColMajBlock<I: VectorIndex> {
    pub(crate) nrows: IndexType,
    pub(crate) ncols: IndexType,
    pub(crate) src_indices: I,
}

impl<I: VectorIndex> ColMajBlock<I> {
    fn new(
        nrows: IndexType,
        ncols: IndexType,
        row_indices: &[IndexType],
        col_indices: &[IndexType],
        transpose: bool,
    ) -> Self {
        let mut src_indices = if transpose {
            col_indices
                .iter()
                .chain(row_indices.iter())
                .copied()
                .collect::<Vec<_>>()
        } else {
            row_indices
                .iter()
                .chain(col_indices.iter())
                .copied()
                .collect::<Vec<_>>()
        };
        src_indices.push(if transpose { 1 } else { 0 });
        let src_indices = I::from_slice(src_indices.as_slice());
        let nrows = if transpose { ncols } else { nrows };
        let ncols = if transpose { nrows } else { ncols };
        Self {
            nrows,
            ncols,
            src_indices,
        }
    }

    /// split a square csc matrix into four blocks based on a predicate
    ///
    /// M = [UL(false, false), UR(false, true)]
    ///     [LL(true, false), LR(true, true)]
    ///     
    /// return order is (UL, UR, LL, LR)
    ///
    pub fn split<F>(
        nrows: IndexType,
        ncols: IndexType,
        f: F,
        transpose: bool,
    ) -> (Self, Self, Self, Self)
    where
        F: Fn(IndexType) -> bool,
    {
        if nrows != ncols {
            panic!("Matrix must be square");
        }
        let n = nrows;
        let cat = (0..n).map(f).collect::<Vec<_>>();

        let mut upper_indices = Vec::new();
        let mut lower_indices = Vec::new();
        for (i, &c) in cat.iter().enumerate() {
            if c {
                lower_indices.push(i);
            } else {
                upper_indices.push(i);
            }
        }
        let n_up = upper_indices.len();
        let n_low = lower_indices.len();

        let ul = Self::new(n_up, n_up, &upper_indices, &upper_indices, transpose);
        let ur = Self::new(n_up, n_low, &upper_indices, &lower_indices, transpose);
        let ll = Self::new(n_low, n_up, &lower_indices, &upper_indices, transpose);
        let lr = Self::new(n_low, n_low, &lower_indices, &lower_indices, transpose);
        (ul, ur, ll, lr)
    }

    pub fn copy_block_from<M: DenseMatrix>(data: &mut M, src_data: &M, src_indices: &I) {
        let nrows = data.nrows();
        let ncols = data.ncols();
        let transpose = src_indices[src_indices.len() - 1] == 1;
        if transpose {
            for j in 0..ncols {
                let jj = src_indices[nrows + j];
                for i in 0..nrows {
                    let ii = src_indices[i];
                    data[(i, j)] = src_data[(jj, ii)];
                }
            }
        } else {
            for j in 0..ncols {
                let jj = src_indices[nrows + j];
                for i in 0..nrows {
                    let ii = src_indices[i];
                    data[(i, j)] = src_data[(ii, jj)];
                }
            }
        }
    }
}

/// Combine four matrices into a single matrix according to a predicate
pub fn combine<F, M>(ul: &M, ur: &M, ll: &M, lr: &M, f: F) -> M
where
    F: Fn(IndexType) -> bool,
    M: Matrix,
{
    let n = ul.nrows() + ll.nrows();
    let m = ul.ncols() + ur.ncols();
    if ul.ncols() != ll.ncols()
        || ur.ncols() != lr.ncols()
        || ul.nrows() != ur.nrows()
        || ll.nrows() != lr.nrows()
    {
        panic!("Matrices must have the same shape");
    }
    let mut triplets = Vec::new();
    let cat = (0..n).map(f).collect::<Vec<_>>();
    let ni = cat.len();
    let mut upper_indices = Vec::with_capacity(n);
    let mut lower_indices = Vec::with_capacity(n);
    let mut upper_indices_short = Vec::with_capacity(n - ni);
    let mut lower_indices_short = Vec::with_capacity(ni);
    let mut upper_acc = 0;
    let mut lower_acc = 0;
    for (i, c) in cat.iter().enumerate() {
        lower_indices.push(lower_acc);
        upper_indices.push(upper_acc);
        if *c {
            lower_indices_short.push(i);
            lower_acc += 1;
        } else {
            upper_indices_short.push(i);
            upper_acc += 1;
        }
    }
    for (i, j, &v) in ul.triplet_iter() {
        let ii = upper_indices_short[i];
        let jj = upper_indices_short[j];
        triplets.push((ii, jj, v));
    }
    for (i, j, &v) in ur.triplet_iter() {
        let ii = upper_indices_short[i];
        let jj = lower_indices_short[j];
        triplets.push((ii, jj, v));
    }
    for (i, j, &v) in ll.triplet_iter() {
        let ii = lower_indices_short[i];
        let jj = upper_indices_short[j];
        triplets.push((ii, jj, v));
    }
    for (i, j, &v) in lr.triplet_iter() {
        let ii = lower_indices_short[i];
        let jj = lower_indices_short[j];
        triplets.push((ii, jj, v));
    }
    M::try_from_triplets(n, m, triplets).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseColMat;
    use faer::Mat;

    #[test]
    fn test_split_combine_faer_sparse() {
        test_split_combine::<SparseColMat<f64>>();
    }

    #[test]
    fn test_split_combine_faer_dense() {
        test_split_combine::<Mat<f64>>();
    }

    #[test]
    fn test_split_combine_nalgebra_dense() {
        test_split_combine::<nalgebra::DMatrix<f64>>();
    }

    #[test]
    fn test_split_combine_nalgebra_sparse() {
        test_split_combine::<nalgebra_sparse::CscMatrix<f64>>();
    }

    fn test_split_combine<M: Matrix>() {
        let triplets = vec![
            (0, 0, 1.0),
            (1, 0, 5.0),
            (2, 0, 9.0),
            (3, 0, 13.0),
            (0, 1, 2.0),
            (1, 1, 6.0),
            (2, 1, 10.0),
            (3, 1, 14.0),
            (0, 2, 3.0),
            (1, 2, 7.0),
            (2, 2, 11.0),
            (3, 2, 15.0),
            (0, 3, 4.0),
            (1, 3, 8.0),
            (2, 3, 12.0),
            (3, 3, 16.0),
        ];
        let triplets = triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let m = M::try_from_triplets(4, 4, triplets.clone()).unwrap();
        let indices = [0, 2];

        let [(ul, _ul_idx), (ur, _ur_idx), (ll, _ll_idx), (lr, _lr_idx)] =
            m.split(|i| indices.contains(&i), false);
        let ul_triplets = [(0, 0, 6.0), (1, 0, 14.0), (0, 1, 8.0), (1, 1, 16.0)];
        let ur_triplets = [(0, 0, 5.0), (1, 0, 13.0), (0, 1, 7.0), (1, 1, 15.0)];
        let ll_triplets = [(0, 0, 2.0), (1, 0, 10.0), (0, 1, 4.0), (1, 1, 12.0)];
        let lr_triplets = [(0, 0, 1.0), (1, 0, 9.0), (0, 1, 3.0), (1, 1, 11.0)];
        let ul_triplets = ul_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let ur_triplets = ur_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let ll_triplets = ll_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let lr_triplets = lr_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        assert_eq!(
            ul_triplets,
            ul.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ur_triplets,
            ur.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ll_triplets,
            ll.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            lr_triplets,
            lr.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );

        let mat = M::combine(&ul, &ur, &ll, &lr, |i| indices.contains(&i));
        assert_eq!(
            triplets,
            mat.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );

        let [(ul, _ul_idx), (ur, _ur_idx), (ll, _ll_idx), (lr, _lr_idx)] =
            m.split(|i| indices.contains(&i), true);
        let ul_triplets = [(0, 0, 6.0), (1, 0, 8.0), (0, 1, 14.0), (1, 1, 16.0)];
        let ur_triplets = [(0, 0, 5.0), (1, 0, 7.0), (0, 1, 13.0), (1, 1, 15.0)];
        let ll_triplets = [(0, 0, 2.0), (1, 0, 4.0), (0, 1, 10.0), (1, 1, 12.0)];
        let lr_triplets = [(0, 0, 1.0), (1, 0, 3.0), (0, 1, 9.0), (1, 1, 11.0)];
        let ul_triplets = ul_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let ur_triplets = ur_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let ll_triplets = ll_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let lr_triplets = lr_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        assert_eq!(
            ul_triplets,
            ul.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ur_triplets,
            ur.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ll_triplets,
            ll.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            lr_triplets,
            lr.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );


        let indices = [2];

        let [(ul, _ul_idx), (ur, _ur_idx), (ll, _ll_idx), (lr, _lr_idx)] =
            m.split(|i| indices.contains(&i), false);
        let ul_triplets = [(0, 0, 1.0), (1, 0, 5.0), (2, 0, 13.0), (0, 1, 2.0), (1, 1, 6.0), (2, 1, 14.0), (0, 2, 4.0), (1, 2, 8.0), (2, 2, 16.0)];
        let ur_triplets = [(0, 0, 3.0), (1, 0, 7.0), (2, 0, 15.0)];
        let ll_triplets = [(0, 0, 9.0), (0, 1, 10.0), (0, 2, 12.0)];
        let lr_triplets = [(0, 0, 11.0)];
        let ul_triplets = ul_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let ur_triplets = ur_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let ll_triplets = ll_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        let lr_triplets = lr_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from(*v)))
            .collect::<Vec<_>>();
        assert_eq!(
            ul_triplets,
            ul.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ur_triplets,
            ur.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ll_triplets,
            ll.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            lr_triplets,
            lr.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );

        let mat = M::combine(&ul, &ur, &ll, &lr, |i| indices.contains(&i));
        assert_eq!(
            triplets,
            mat.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
    }
}
