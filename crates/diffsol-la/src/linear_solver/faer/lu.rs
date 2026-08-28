use crate::context::broadcast_batch;
use crate::{error::LaError, linear_solver_error};
use crate::{Context, FaerContext};

use crate::{
    linear_solver::LinearSolver, FaerMat, FaerScalar, FaerVec, LinearOp, Matrix, MatrixCommon,
};

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{linalg::solvers::FullPivLu, linalg::solvers::Solve};
/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T>
where
    T: FaerScalar,
{
    lu: Vec<FullPivLu<T>>,
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
        self.lu = (0..matrix.context.nbatch())
            .map(|b| matrix.data.rb().subcols(b * nc, nc).full_piv_lu())
            .collect();
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        x.context
            .assert_broadcastable_into(self.lu.len(), "lu_solve");
        let nlu = self.lu.len();
        let nb = x.data.ncols();
        for batch in 0..nb {
            self.lu[broadcast_batch(batch, nlu, nb)].solve_in_place(x.data.rb_mut().col_mut(batch));
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
