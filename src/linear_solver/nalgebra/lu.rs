use nalgebra::{DMatrix, DVector, Dyn};
use std::rc::Rc;

use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver_error,
    matrix::sparsity::MatrixSparsityRef,
    op::{linearise::LinearisedOp, NonLinearOp},
    LinearOp, LinearSolver, Matrix, Op, Scalar, SolverProblem,
};

/// A [LinearSolver] that uses the LU decomposition in the [`nalgebra` library](https://nalgebra.org/) to solve the linear system.
#[derive(Clone)]
pub struct LU<T, C>
where
    T: Scalar,
    C: NonLinearOp<M = DMatrix<T>, V = DVector<T>, T = T>,
{
    matrix: Option<DMatrix<T>>,
    lu: Option<nalgebra::LU<T, Dyn, Dyn>>,
    problem: Option<SolverProblem<LinearisedOp<C>>>,
}

impl<T, C> Default for LU<T, C>
where
    T: Scalar,
    C: NonLinearOp<M = DMatrix<T>, V = DVector<T>, T = T>,
{
    fn default() -> Self {
        Self {
            lu: None,
            problem: None,
            matrix: None,
        }
    }
}

impl<T: Scalar, C: NonLinearOp<M = DMatrix<T>, V = DVector<T>, T = T>> LinearSolver<C>
    for LU<T, C>
{
    type SelfNewOp<C2: NonLinearOp<T = C::T, V = C::V, M = C::M>> = LU<T, C2>;

    fn solve_in_place(&self, state: &mut C::V) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        match lu.solve_mut(state) {
            true => Ok(()),
            false => Err(linear_solver_error!(LuSolveFailed))?,
        }
    }

    fn set_linearisation(&mut self, x: &<C as Op>::V, t: <C as Op>::T) {
        Rc::<LinearisedOp<C>>::get_mut(&mut self.problem.as_mut().expect("Problem not set").f)
            .unwrap()
            .set_x(x);
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        self.problem.as_ref().unwrap().f.matrix_inplace(t, matrix);
        self.lu = Some(matrix.clone().lu());
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
