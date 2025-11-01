use crate::FaerContext;
use crate::{error::LinearSolverError, linear_solver_error};

use crate::{
    error::DiffsolError, linear_solver::LinearSolver, FaerMat, FaerVec, Matrix,
    NonLinearOpJacobian, Scalar,
};

use faer::{linalg::solvers::FullPivLu, linalg::solvers::Solve};
/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T>
where
    T: Scalar,
{
    lu: Option<FullPivLu<T>>,
    matrix: Option<FaerMat<T>>,
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

impl<T: Scalar> LinearSolver<FaerMat<T>> for LU<T> {
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerMat<T>>>(
        &mut self,
        op: &C,
        x: &FaerVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(matrix.data.full_piv_lu());
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(x.data.as_mut());
        Ok(())
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>,
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
