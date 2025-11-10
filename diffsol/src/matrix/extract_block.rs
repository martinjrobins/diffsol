use crate::{vector::VectorIndex, IndexType, Matrix};

pub(crate) struct CscBlock {
    pub(crate) nrows: IndexType,
    pub(crate) ncols: IndexType,
    pub(crate) row_indices: Vec<IndexType>,
    pub(crate) col_pointers: Vec<IndexType>,
    pub(crate) src_indices: Vec<IndexType>,
}

impl CscBlock {
    fn new(nrows: IndexType, ncols: IndexType) -> Self {
        Self {
            ncols,
            nrows,
            row_indices: Vec::new(),
            col_pointers: Vec::new(),
            src_indices: Vec::new(),
        }
    }

    /// split a square csc matrix into four blocks based on a predicate
    ///
    /// M = [UL(false, false), UR(false, true)]
    ///     [LL(true, false), LR(true, true)]
    ///     
    /// return order is (UL, UR, LL, LR)
    ///
    pub fn split<I>(
        row_indices: &[IndexType],
        col_pointers: &[IndexType],
        indices: &I,
    ) -> (Self, Self, Self, Self)
    where
        I: VectorIndex,
    {
        let n = col_pointers.len() - 1;
        let (cat, upper_indices, lower_indices, nni, ni) = Self::setup_split(indices, n);
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

    fn setup_split<I>(
        indices: &I,
        n: usize,
    ) -> (
        Vec<bool>,
        Vec<IndexType>,
        Vec<IndexType>,
        IndexType,
        IndexType,
    )
    where
        I: VectorIndex,
    {
        let indices = indices.clone_as_vec();
        let mut cat = vec![false; n];
        indices.iter().for_each(|&i| cat[i] = true);

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
    fn new(nrows: IndexType, ncols: IndexType, src_indices: I) -> Self {
        Self {
            nrows,
            ncols,
            src_indices,
        }
    }

    fn src_indices(
        nrows: IndexType,
        row_indices: &[IndexType],
        col_indices: &[IndexType],
        ctx: I::C,
    ) -> I {
        let mut src_indices = Vec::new();
        for &j in col_indices {
            for &i in row_indices {
                src_indices.push(j * nrows + i);
            }
        }
        I::from_vec(src_indices, ctx)
    }

    /// split a square csc matrix into four blocks based on a predicate
    ///
    /// M = [UL(false, false), UR(false, true)]
    ///     [LL(true, false), LR(true, true)]
    ///     
    /// return order is (UL, UR, LL, LR)
    ///
    pub fn split(nrows: IndexType, ncols: IndexType, indices: &I) -> (Self, Self, Self, Self) {
        if nrows != ncols {
            panic!("Matrix must be square");
        }
        let n = nrows;
        let all_indices = indices.clone_as_vec();
        let mut cat = vec![false; n];
        all_indices.iter().for_each(|&i| cat[i] = true);

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

        let ul = Self::new(
            n_up,
            n_up,
            Self::src_indices(
                nrows,
                &upper_indices,
                &upper_indices,
                indices.context().clone(),
            ),
        );
        let ur = Self::new(
            n_up,
            n_low,
            Self::src_indices(
                nrows,
                &upper_indices,
                &lower_indices,
                indices.context().clone(),
            ),
        );
        let ll = Self::new(
            n_low,
            n_up,
            Self::src_indices(
                nrows,
                &lower_indices,
                &upper_indices,
                indices.context().clone(),
            ),
        );
        let lr = Self::new(
            n_low,
            n_low,
            Self::src_indices(
                nrows,
                &lower_indices,
                &lower_indices,
                indices.context().clone(),
            ),
        );
        (ul, ur, ll, lr)
    }
}

/// Combine four matrices into a single matrix according to a predicate
pub fn combine<I, M>(ul: &M, ur: &M, ll: &M, lr: &M, indices: &I) -> M
where
    I: VectorIndex,
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
    let indices = indices.clone_as_vec();
    let mut cat = vec![false; n];
    indices.iter().for_each(|&i| cat[i] = true);
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
    for (i, j, v) in ul.triplet_iter() {
        let ii = upper_indices_short[i];
        let jj = upper_indices_short[j];
        triplets.push((ii, jj, v));
    }
    for (i, j, v) in ur.triplet_iter() {
        let ii = upper_indices_short[i];
        let jj = lower_indices_short[j];
        triplets.push((ii, jj, v));
    }
    for (i, j, v) in ll.triplet_iter() {
        let ii = lower_indices_short[i];
        let jj = upper_indices_short[j];
        triplets.push((ii, jj, v));
    }
    for (i, j, v) in lr.triplet_iter() {
        let ii = lower_indices_short[i];
        let jj = lower_indices_short[j];
        triplets.push((ii, jj, v));
    }
    M::try_from_triplets(n, m, triplets, ul.context().clone()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FaerMat, FaerSparseMat, NalgebraMat, Vector};
    use num_traits::FromPrimitive;

    #[test]
    fn test_split_combine_faer_sparse() {
        test_split_combine::<FaerSparseMat<f64>>();
    }

    #[test]
    fn test_split_combine_faer_dense() {
        test_split_combine::<FaerMat<f64>>();
    }

    #[test]
    fn test_split_combine_nalgebra_dense() {
        test_split_combine::<NalgebraMat<f64>>();
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
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let m = M::try_from_triplets(4, 4, triplets.clone(), Default::default()).unwrap();
        let indices = <M::V as Vector>::Index::from_vec(vec![0, 2], Default::default());

        let [(ul, _ul_idx), (ur, _ur_idx), (ll, _ll_idx), (lr, _lr_idx)] = m.split(&indices);
        let ul_triplets = [(0, 0, 6.0), (1, 0, 14.0), (0, 1, 8.0), (1, 1, 16.0)];
        let ur_triplets = [(0, 0, 5.0), (1, 0, 13.0), (0, 1, 7.0), (1, 1, 15.0)];
        let ll_triplets = [(0, 0, 2.0), (1, 0, 10.0), (0, 1, 4.0), (1, 1, 12.0)];
        let lr_triplets = [(0, 0, 1.0), (1, 0, 9.0), (0, 1, 3.0), (1, 1, 11.0)];
        let ul_triplets = ul_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let ur_triplets = ur_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let ll_triplets = ll_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let lr_triplets = lr_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(ul_triplets, ul.triplet_iter().collect::<Vec<_>>());
        assert_eq!(ur_triplets, ur.triplet_iter().collect::<Vec<_>>());
        assert_eq!(ll_triplets, ll.triplet_iter().collect::<Vec<_>>());
        assert_eq!(lr_triplets, lr.triplet_iter().collect::<Vec<_>>());

        let mat = M::combine(&ul, &ur, &ll, &lr, &indices);
        assert_eq!(triplets, mat.triplet_iter().collect::<Vec<_>>());

        let indices = <M::V as Vector>::Index::from_vec(vec![2], Default::default());

        let [(ul, _ul_idx), (ur, _ur_idx), (ll, _ll_idx), (lr, _lr_idx)] = m.split(&indices);
        let ul_triplets = [
            (0, 0, 1.0),
            (1, 0, 5.0),
            (2, 0, 13.0),
            (0, 1, 2.0),
            (1, 1, 6.0),
            (2, 1, 14.0),
            (0, 2, 4.0),
            (1, 2, 8.0),
            (2, 2, 16.0),
        ];
        let ur_triplets = [(0, 0, 3.0), (1, 0, 7.0), (2, 0, 15.0)];
        let ll_triplets = [(0, 0, 9.0), (0, 1, 10.0), (0, 2, 12.0)];
        let lr_triplets = [(0, 0, 11.0)];
        assert_eq!(ul.nrows(), 3);
        assert_eq!(ul.ncols(), 3);
        assert_eq!(ur.nrows(), 3);
        assert_eq!(ur.ncols(), 1);
        assert_eq!(ll.nrows(), 1);
        assert_eq!(ll.ncols(), 3);
        assert_eq!(lr.nrows(), 1);
        assert_eq!(lr.ncols(), 1);

        let ul_triplets = ul_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let ur_triplets = ur_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let ll_triplets = ll_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        let lr_triplets = lr_triplets
            .iter()
            .map(|(i, j, v)| (*i, *j, M::T::from_f64(*v).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(ul_triplets, ul.triplet_iter().collect::<Vec<_>>());
        assert_eq!(ur_triplets, ur.triplet_iter().collect::<Vec<_>>());
        assert_eq!(ll_triplets, ll.triplet_iter().collect::<Vec<_>>());
        assert_eq!(lr_triplets, lr.triplet_iter().collect::<Vec<_>>());

        let mat = M::combine(&ul, &ur, &ll, &lr, &indices);
        assert_eq!(triplets, mat.triplet_iter().collect::<Vec<_>>());
    }
}
