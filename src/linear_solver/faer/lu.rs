use std::rc::Rc;

use crate::{error::LinearSolverError, linear_solver_error};

use crate::{
    error::DiffsolError, linear_solver::LinearSolver, Matrix,
    NonLinearOpJacobian, Scalar,
};

use faer::{linalg::solvers::FullPivLu, solvers::SpSolver, Col, Mat};
/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T>
where
    T: Scalar,
{
    lu: Option<FullPivLu<T>>,
    matrix: Option<Mat<T>>,
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

impl<T: Scalar> LinearSolver<Mat<T>> for LU<T> {
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = Col<T>, M = Mat<T>>>(
        &mut self,
        op: &C,
        x: &Col<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(matrix.full_piv_lu());
    }

    fn solve_in_place(&self, x: &mut Col<T>) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(x);
        Ok(())
    }

    fn set_problem<C: NonLinearOpJacobian<T = T, V = Col<T>, M = Mat<T>>>(
        &mut self,
        op: &C,
        _rtol: T,
        _atol: Rc<Col<T>>,
    ) {
        let ncols = op.nstates();
        let nrows = op.nout();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity());
        self.matrix = Some(matrix);
    }
}
