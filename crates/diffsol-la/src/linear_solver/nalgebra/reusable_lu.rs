use nalgebra::linalg::{gauss_step, gauss_step_swap};
use nalgebra::{Dim, Dyn, Matrix as NaMatrix, StorageMut};

use crate::{
    error::LaError, linear_solver_error, matrix::dense_nalgebra_serial::NalgebraMat, Context,
    LinearOp, LinearSolver, Matrix, NalgebraContext, NalgebraScalar, NalgebraVec,
};

// `gauss_step`/`gauss_step_swap` are `#[doc(hidden)]` in nalgebra (not officially documented,
// though `pub fn` and re-exported at the crate root) -- they are the exact elimination steps
// `nalgebra::linalg::LU::new` calls internally.

/// LU factorization state for a (possibly batched) [NalgebraMat], factorized and solved
/// entirely in place: `a` is written to directly by `LinearOp::matrix_inplace` in
/// [ReusableLU::set_linearisation] (no separate scratch matrix + clone, unlike
/// [NalgebraNativeLU](crate::NalgebraNativeLU)), then decomposed in place, batch block by
/// batch block.
struct BatchLu<T: NalgebraScalar> {
    a: NalgebraMat<T>,
    /// Row-swap history per batch (replaces nalgebra's `PermutationSequence`), reused across
    /// `factor` calls.
    piv: Vec<Vec<(usize, usize)>>,
}

impl<T: NalgebraScalar> BatchLu<T> {
    fn new(a: NalgebraMat<T>) -> Self {
        let nbatch = a.context.nbatch();
        Self {
            a,
            piv: vec![Vec::new(); nbatch],
        }
    }

    fn nbatch(&self) -> usize {
        self.piv.len()
    }

    fn logical_ncols(&self) -> usize {
        self.a.data.ncols() / self.nbatch()
    }

    /// Factorize `self.a` in place, one square batch block at a time. Body copied from
    /// `nalgebra::linalg::LU::new`, run against a mutable view of each block instead of an
    /// owned matrix.
    fn factor(&mut self) {
        let ncols = self.logical_ncols();
        for batch in 0..self.nbatch() {
            let piv = &mut self.piv[batch];
            piv.clear();
            let mut block = self.a.data.columns_mut(batch * ncols, ncols);
            for i in 0..ncols {
                let p = block.view_range(i.., i).icamax() + i;
                let diag = block[(p, i)].clone();
                if diag.is_zero() {
                    continue;
                }
                if p != i {
                    piv.push((i, p));
                    block.columns_range_mut(..i).swap_rows(i, p);
                    gauss_step_swap(&mut block, diag, i, p);
                } else {
                    gauss_step(&mut block, diag, i);
                }
            }
        }
    }

    /// Solve `block(batch) * x = b` in place, `b` holding the input and receiving the
    /// solution. Body copied from `nalgebra::linalg::LU::solve_mut`.
    fn solve_block<C2: Dim, S2: StorageMut<T, Dyn, C2>>(
        &self,
        batch: usize,
        b: &mut NaMatrix<T, Dyn, C2, S2>,
    ) -> bool {
        let ncols = self.logical_ncols();
        let block = self.a.data.columns(batch * ncols, ncols);
        for &(i, p) in &self.piv[batch] {
            b.swap_rows(i, p);
        }
        let _ = block.solve_lower_triangular_with_diag_mut(b, T::one());
        block.solve_upper_triangular_mut(b)
    }
}

/// The default nalgebra [LinearSolver], exported as [NalgebraLU](crate::NalgebraLU). Factorizes
/// its matrix entirely in place (no per-call matrix clone + fresh permutation sequence, both of
/// which `nalgebra::linalg::LU::new` allocates every time). See
/// [NalgebraNativeLU](crate::NalgebraNativeLU) for the allocating equivalent, kept for
/// comparison/benchmarking.
pub struct ReusableLU<T>
where
    T: NalgebraScalar,
{
    lu: Option<BatchLu<T>>,
}

impl<T: NalgebraScalar> Default for ReusableLU<T> {
    fn default() -> Self {
        Self { lu: None }
    }
}

impl<T: NalgebraScalar> LinearSolver<NalgebraMat<T>> for ReusableLU<T> {
    fn solve_in_place(&self, state: &mut NalgebraVec<T>) -> Result<(), LaError> {
        let Some(lu) = self.lu.as_ref() else {
            return Err(linear_solver_error!(LuNotInitialized));
        };
        state
            .context
            .assert_compatible_nbatch(lu.nbatch(), "lu_solve");
        if state.context.nbatch() == 1 {
            if lu.solve_block(0, &mut state.data) {
                return Ok(());
            }
            return Err(linear_solver_error!(LuSolveFailed));
        }
        for batch in 0..state.context.nbatch() {
            let mut state_batch = state.data.column_mut(batch);
            if !lu.solve_block(batch % lu.nbatch(), &mut state_batch) {
                return Err(linear_solver_error!(LuSolveFailed));
            }
        }
        Ok(())
    }

    fn set_linearisation<
        C: LinearOp<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>, C = NalgebraContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let lu = self.lu.as_mut().expect("Matrix not set");
        op.matrix_inplace(&mut lu.a);
        lu.factor();
    }

    fn set_sparsity<
        C: LinearOp<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>, C = NalgebraContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let ncols = op.ncols();
        let nrows = op.nrows();
        let a = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), *op.context());
        self.lu = Some(BatchLu::new(a));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear_solver::tests::diagonal_op, Vector};

    #[test]
    fn test_lu() {
        let mut s = ReusableLU::<f64>::default();
        let op = diagonal_op::<NalgebraMat<f64>>(2.0);
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = NalgebraVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &NalgebraVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }

    /// A square matrix stored as a [NalgebraMat], built by direct indexing.
    fn make_mat(n: usize, data: &[f64]) -> NalgebraMat<f64> {
        let mut m: NalgebraMat<f64> = Matrix::new_from_sparsity(n, n, None, Default::default());
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = data[i * n + j];
            }
        }
        m
    }

    /// `ReusableLU` should agree with `nalgebra::linalg::LU` on the same matrices, including
    /// across repeated `factor` calls that exercise the in-place refactor path, and including
    /// cases that require row pivoting (first case has a zero (0,0) entry).
    #[test]
    fn matches_nalgebra_lu() {
        let cases: Vec<(usize, Vec<f64>)> = vec![
            (1, vec![2.0]),
            (2, vec![0.0, 1.0, 1.0, 1.0]),
            (3, vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0]),
            (
                5,
                vec![
                    2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0,
                    0.0, 2.0, 5.0, 1.0, 1.0, 0.0, 0.0, 1.0, 6.0,
                ],
            ),
        ];
        for (n, data) in cases {
            let nm = make_mat(n, &data);
            let b = nalgebra::DMatrix::<f64>::from_fn(n, 1, |i, _| (i + 1) as f64);

            let mut lu = BatchLu::new(nm.clone());
            lu.factor();
            let mut got = b.clone();
            assert!(lu.solve_block(0, &mut got));

            let m = nalgebra::DMatrix::from_row_slice(n, n, &data);
            let expected_lu = m.lu();
            let mut expected = b.clone();
            assert!(expected_lu.solve_mut(&mut expected));

            for i in 0..n {
                assert!(
                    (got[(i, 0)] - expected[(i, 0)]).abs() < 1e-9,
                    "n={n} row={i}: got {} expected {}",
                    got[(i, 0)],
                    expected[(i, 0)]
                );
            }

            // refactoring in place (buffer reuse path) should give the same answer again --
            // `a` must be repopulated first, mirroring what `set_linearisation` does via
            // `matrix_inplace` before each `factor()` call (factoring an already-factored
            // matrix again is not a supported operation).
            lu.a = make_mat(n, &data);
            lu.factor();
            let mut got2 = b.clone();
            assert!(lu.solve_block(0, &mut got2));
            for i in 0..n {
                assert!((got2[(i, 0)] - expected[(i, 0)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn singular_matrix_fails_like_nalgebra_lu() {
        let nm = make_mat(2, &[1.0, 2.0, 2.0, 4.0]);
        let mut lu = BatchLu::new(nm);
        lu.factor();
        let mut b = nalgebra::DMatrix::<f64>::from_row_slice(2, 1, &[1.0, 2.0]);
        assert!(!lu.solve_block(0, &mut b));
    }
}
