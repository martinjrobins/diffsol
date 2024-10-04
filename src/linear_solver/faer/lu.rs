use crate::{error::LinearSolverError, linear_solver_error};
use std::rc::Rc;

use crate::{
    error::DiffsolError, linear_solver::LinearSolver, op::linearise::LinearisedOp,
    solver::SolverProblem, LinearOp, Matrix, MatrixSparsityRef, NonLinearOp, Op, Scalar,
};

use faer::{linalg::solvers::FullPivLu, solvers::SpSolver, Col, Mat};
/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T, C>
where
    T: Scalar,
    C: NonLinearOp<M = Mat<T>, V = Col<T>, T = T>,
{
    lu: Option<FullPivLu<T>>,
    problem: Option<SolverProblem<LinearisedOp<C>>>,
    matrix: Option<Mat<T>>,
}

impl<T, C> Default for LU<T, C>
where
    T: Scalar,
    C: NonLinearOp<M = Mat<T>, V = Col<T>, T = T>,
{
    fn default() -> Self {
        Self {
            lu: None,
            problem: None,
            matrix: None,
        }
    }
}

impl<T: Scalar, C: NonLinearOp<M = Mat<T>, V = Col<T>, T = T>> LinearSolver<C> for LU<T, C> {
    type SelfNewOp<C2: NonLinearOp<T = C::T, V = C::V, M = C::M>> = LU<T, C2>;

    fn set_linearisation(&mut self, x: &C::V, t: C::T) {
        Rc::<LinearisedOp<C>>::get_mut(&mut self.problem.as_mut().expect("Problem not set").f)
            .unwrap()
            .set_x(x);
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        self.problem.as_ref().unwrap().f.matrix_inplace(t, matrix);
        self.lu = Some(matrix.full_piv_lu());
    }

    fn solve_in_place(&self, x: &mut C::V) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(x);
        Ok(())
    }

    fn set_problem(&mut self, problem: &SolverProblem<C>) {
        let linearised_problem = problem.linearise();
        let ncols = linearised_problem.f.nstates();
        let nrows = linearised_problem.f.nout();
        let matrix = C::M::new_from_sparsity(
            nrows,
            ncols,
            linearised_problem.f.sparsity().map(|s| s.to_owned()),
        );
        self.problem = Some(linearised_problem);
        self.matrix = Some(matrix);
    }

    fn clear_problem(&mut self) {
        self.problem = None;
        self.matrix = None;
        self.lu = None;
    }
}
