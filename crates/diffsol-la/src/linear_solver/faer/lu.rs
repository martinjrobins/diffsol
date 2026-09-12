use crate::context::broadcast_batch;
use crate::{error::LaError, linear_solver_error};
use crate::{Context, FaerContext};

use crate::{
    linear_solver::LinearSolver, FaerMat, FaerScalar, FaerVec, LinearOp, Matrix, MatrixCommon,
};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::full_pivoting::{factor, solve};
use faer::perm::Perm;
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Conj, Mat, Par};

/// The $LU$ factors of one batch, stored packed: the strict lower triangle of `lu` is $L$ (unit
/// diagonal), the upper triangle is $U$.  faer's triangular solves read only their own triangle
/// and substitute ones on $L$'s diagonal, so the same matrix serves as both operands and no
/// splitting copy is needed.
struct PackedLu<T: FaerScalar> {
    lu: Mat<T>,
    row: Perm<usize>,
    col: Perm<usize>,
}

impl<T: FaerScalar> PackedLu<T> {
    fn new(a: faer::MatRef<'_, T>, par: Par) -> Self {
        let (m, n) = a.shape();
        let mut lu = a.to_owned();
        let mut row_fwd = vec![0usize; m];
        let mut row_bwd = vec![0usize; m];
        let mut col_fwd = vec![0usize; n];
        let mut col_bwd = vec![0usize; n];
        factor::lu_in_place(
            lu.as_mut(),
            &mut row_fwd,
            &mut row_bwd,
            &mut col_fwd,
            &mut col_bwd,
            par,
            MemStack::new(&mut MemBuffer::new(
                factor::lu_in_place_scratch::<usize, T>(m, n, par, Default::default()),
            )),
            Default::default(),
        );
        Self {
            lu,
            row: Perm::new_checked(row_fwd.into_boxed_slice(), row_bwd.into_boxed_slice(), m),
            col: Perm::new_checked(col_fwd.into_boxed_slice(), col_bwd.into_boxed_slice(), n),
        }
    }

    fn solve_in_place(&self, rhs: faer::MatMut<'_, T>, par: Par) {
        let n = self.lu.nrows();
        let k = rhs.ncols();
        solve::solve_in_place_with_conj(
            self.lu.rb(),
            self.lu.rb(),
            self.row.rb(),
            self.col.rb(),
            Conj::No,
            rhs,
            par,
            MemStack::new(&mut MemBuffer::new(
                solve::solve_in_place_scratch::<usize, T>(n, k, par),
            )),
        );
    }
}

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
///
/// The low-level `faer::linalg::lu` API is used rather than `faer::linalg::solvers` so that the
/// parallelism from [`FaerContext::par`] is honoured; the wrapper types hardcode faer's
/// process-global setting instead.
pub struct LU<T>
where
    T: FaerScalar,
{
    lu: Vec<PackedLu<T>>,
    matrix: Option<FaerMat<T>>,
}

impl<T> Default for LU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: None,
        }
    }
}

impl<T: FaerScalar> LinearSolver<FaerMat<T>> for LU<T> {
    fn set_linearisation<C: LinearOp<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.matrix_inplace(matrix);
        let nc = matrix.ncols();
        let par = matrix.context.par;
        self.lu = (0..matrix.context.nbatch())
            .map(|b| PackedLu::new(matrix.data.rb().subcols(b * nc, nc), par))
            .collect();
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        x.context
            .assert_broadcastable_into(self.lu.len(), "lu_solve");
        let par = x.context.par;
        let nlu = self.lu.len();
        let nb = x.data.ncols();
        for batch in 0..nb {
            self.lu[broadcast_batch(batch, nlu, nb)]
                .solve_in_place(x.data.rb_mut().col_mut(batch).as_mat_mut(), par);
        }
        Ok(())
    }

    fn set_sparsity<C: LinearOp<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) {
        let ncols = op.ncols();
        let nrows = op.nrows();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), *op.context());
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linear_solver::tests::{diagonal_op, test_grouped_lu_solve, test_narrow_state_lu_solve},
        Vector,
    };

    #[test]
    fn test_lu() {
        let mut s = LU::<f64>::default();
        let op = diagonal_op::<FaerMat<f64>>(2.0);
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = FaerVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &FaerVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }

    #[test]
    fn test_grouped_lu() {
        test_grouped_lu_solve::<FaerMat<f64>, LU<f64>>(FaerContext::with_nbatch(2));
    }

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_narrow_state_lu() {
        test_narrow_state_lu_solve::<FaerMat<f64>, LU<f64>>(FaerContext::with_nbatch(2));
    }
}
