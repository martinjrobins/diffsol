use nalgebra::Dyn;

use crate::context::broadcast_batch;
use crate::{
    error::LaError, linear_solver_error, matrix::dense_nalgebra_serial::NalgebraMat, Context,
    LinearOp, LinearSolver, Matrix, NalgebraContext, NalgebraScalar, NalgebraVec,
};

/// A [LinearSolver] that uses the LU decomposition in the [`nalgebra` library](https://nalgebra.org/) to solve the linear system.
///
/// Exported as [NalgebraNativeLU](crate::NalgebraNativeLU): re-allocates its factorization
/// (a matrix clone plus a fresh permutation sequence) on every
/// [`set_linearisation`](LinearSolver::set_linearisation) call, unlike the default
/// [NalgebraLU](crate::NalgebraLU), which reuses its workspace in place. Kept around for
/// comparison/benchmarking.
#[derive(Clone)]
pub struct LU<T>
where
    T: NalgebraScalar,
{
    matrix: Option<NalgebraMat<T>>,
    lu: Vec<nalgebra::LU<T, Dyn, Dyn>>,
}

impl<T> Default for LU<T>
where
    T: NalgebraScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: None,
        }
    }
}

impl<T: NalgebraScalar> LinearSolver<NalgebraMat<T>> for LU<T> {
    fn solve_in_place(&self, state: &mut NalgebraVec<T>) -> Result<(), LaError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        state
            .context
            .assert_broadcastable_into(self.lu.len(), "lu_solve");
        if state.context.nbatch() == 1 {
            if self.lu[0].solve_mut(&mut state.data) {
                return Ok(());
            }
            return Err(linear_solver_error!(LuSolveFailed));
        }
        let nb = state.context.nbatch();
        for batch in 0..nb {
            let mut state_batch = state.data.column_mut(batch);
            if !self.lu[broadcast_batch(batch, self.lu.len(), nb)].solve_mut(&mut state_batch) {
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
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.matrix_inplace(matrix);
        if matrix.context.nbatch() == 1 {
            self.lu = vec![matrix.data.clone().lu()];
            return;
        }
        let ncols = matrix.data.ncols() / matrix.context.nbatch();
        self.lu = (0..matrix.context.nbatch())
            .map(|batch| matrix.data.columns(batch * ncols, ncols).into_owned().lu())
            .collect();
    }

    fn set_sparsity<
        C: LinearOp<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>, C = NalgebraContext>,
    >(
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
    #[test]
    fn test_grouped_lu() {
        test_grouped_lu_solve::<NalgebraMat<f64>, LU<f64>>(NalgebraContext::with_nbatch(2));
    }

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_narrow_state_lu() {
        test_narrow_state_lu_solve::<NalgebraMat<f64>, LU<f64>>(NalgebraContext::with_nbatch(2));
    }
}
