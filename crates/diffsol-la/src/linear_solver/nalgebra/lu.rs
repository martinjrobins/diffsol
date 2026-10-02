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

    crate::linear_solver::generate_lu_tests!(
        f64,
        NalgebraMat<f64>,
        LU<f64>,
        NalgebraContext::with_nbatch(2)
    );
    crate::linear_solver::generate_lu_tests!(
        f32,
        NalgebraMat<f32>,
        LU<f32>,
        NalgebraContext::with_nbatch(2)
    );
}
