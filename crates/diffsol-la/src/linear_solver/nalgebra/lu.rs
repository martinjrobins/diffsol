use nalgebra::Dyn;

use crate::{
    error::LaError, linear_solver_error, matrix::dense_nalgebra_serial::NalgebraMat, Context,
    LinearOp, LinearSolver, Matrix, NalgebraContext, NalgebraScalar, NalgebraVec,
};

/// A [LinearSolver] that uses the LU decomposition in the [`nalgebra` library](https://nalgebra.org/) to solve the linear system.
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
            .assert_compatible_nbatch(self.lu.len(), "lu_solve");
        for (batch, state) in state.data.iter_mut().enumerate() {
            if !self.lu[batch % self.lu.len()].solve_mut(state) {
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
        self.lu = matrix
            .data
            .iter()
            .cloned()
            .map(|matrix| matrix.lu())
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
    use crate::{linear_solver::tests::diagonal_op, Vector};

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
}
