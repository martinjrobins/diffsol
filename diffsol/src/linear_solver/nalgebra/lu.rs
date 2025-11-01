use nalgebra::Dyn;

use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver_error,
    matrix::dense_nalgebra_serial::NalgebraMat,
    LinearSolver, Matrix, NalgebraContext, NalgebraVec, NonLinearOpJacobian, Scalar,
};

/// A [LinearSolver] that uses the LU decomposition in the [`nalgebra` library](https://nalgebra.org/) to solve the linear system.
#[derive(Clone)]
pub struct LU<T>
where
    T: Scalar,
{
    matrix: Option<NalgebraMat<T>>,
    lu: Option<nalgebra::LU<T, Dyn, Dyn>>,
}

impl<T> Default for LU<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            lu: None,
            matrix: None,
        }
    }
}

impl<T: Scalar> LinearSolver<NalgebraMat<T>> for LU<T> {
    fn solve_in_place(&self, state: &mut NalgebraVec<T>) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        match lu.solve_mut(&mut state.data) {
            true => Ok(()),
            false => Err(linear_solver_error!(LuSolveFailed))?,
        }
    }

    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>>>(
        &mut self,
        op: &C,
        x: &NalgebraVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(matrix.data.clone().lu());
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>, C = NalgebraContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let ncols = op.nstates();
        let nrows = op.nout();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity(), *op.context());
        self.matrix = Some(matrix);
    }
}
