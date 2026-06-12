use nalgebra::Dyn;

use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver_error,
    matrix::{dense_nalgebra_serial::NalgebraMat, MatrixCommon},
    Context, LinearSolver, Matrix, NalgebraContext, NalgebraScalar, NalgebraVec,
    NonLinearOpJacobian, Vector,
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
    fn solve_in_place(&self, state: &mut NalgebraVec<T>) -> Result<(), DiffsolError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let nbatch = state.context().nbatch();
        for b in 0..nbatch {
            let lu = &self.lu[b];
            let mut col = state.data.column_mut(b);
            match lu.solve_mut(&mut col) {
                true => {}
                false => return Err(linear_solver_error!(LuSolveFailed))?,
            }
        }
        Ok(())
    }

    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>>>(
        &mut self,
        op: &C,
        x: &NalgebraVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        let nbatch = matrix.context.nbatch();
        let ncols = matrix.ncols();
        self.lu.clear();
        for b in 0..nbatch {
            let sub = matrix.data.columns(b * ncols, ncols).into_owned();
            self.lu.push(sub.lu());
        }
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
