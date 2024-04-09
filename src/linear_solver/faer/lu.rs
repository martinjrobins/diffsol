use crate::{linear_solver::LinearSolver, op::LinearOp, solver::SolverProblem, Scalar};
use anyhow::Result;
use faer::{linalg::solvers::FullPivLu, solvers::SpSolver, Col, Mat};
/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T, C>
where
    T: Scalar,
    C: LinearOp<M = Mat<T>, V = Col<T>, T = T>,
{
    lu: Option<FullPivLu<T>>,
    problem: Option<SolverProblem<C>>,
}

impl<T, C> Default for LU<T, C>
where
    T: Scalar,
    C: LinearOp<M = Mat<T>, V = Col<T>, T = T>,
{
    fn default() -> Self {
        Self {
            lu: None,
            problem: None,
        }
    }
}

impl<T: Scalar, C: LinearOp<M = Mat<T>, V = Col<T>, T = T>> LinearSolver<C> for LU<T, C> {
    fn problem(&self) -> Option<&SolverProblem<C>> {
        self.problem.as_ref()
    }
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>> {
        self.problem.as_mut()
    }
    fn take_problem(&mut self) -> Option<SolverProblem<C>> {
        self.lu = None;
        Option::take(&mut self.problem)
    }

    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()> {
        if self.lu.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(state);
        Ok(())
    }

    fn set_problem(&mut self, problem: SolverProblem<C>) {
        self.lu = Some(problem.f.jacobian(problem.t).full_piv_lu());
        self.problem = Some(problem);
    }
}
