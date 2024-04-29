use crate::{
    linear_solver::LinearSolver, op::linearise::LinearisedOp, solver::SolverProblem, NonLinearOp,
    Op, Scalar, Matrix, LinearOp,
};
use anyhow::Result;
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
    fn set_linearisation(&mut self, x: &C::V, t: C::T) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        let problem = self.problem.as_ref().expect("Problem not set");
        problem.f.set_x(x);
        problem.f.matrix_inplace(t, matrix);
        self.lu = Some(matrix.full_piv_lu());
    }

    fn solve_in_place(&self, x: &mut C::V) -> Result<()> {
        if self.lu.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(x);
        Ok(())
    }

    fn set_problem(&mut self, problem: &SolverProblem<C>) {
        let linearised_problem = problem.linearise();
        let ncols = linearised_problem.f.nstates();
        let nrows = linearised_problem.f.nout();
        let matrix = C::M::new_from_sparsity(nrows, ncols, linearised_problem.f.sparsity());
        self.problem = Some(linearised_problem);
        self.matrix = Some(matrix);
    }
}
