use anyhow::{Ok, Result};
use nalgebra::{DMatrix, DVector, Dyn};

use crate::{op::LinearOp, LinearSolver, Scalar, SolverProblem};

pub struct SVD<T, C>
where
    T: Scalar,
    C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>,
{
    svd: Option<nalgebra::SVD<T, Dyn, Dyn>>,
    problem: Option<SolverProblem<C>>,
}

impl<T, C> Default for SVD<T, C>
where
    T: Scalar,
    C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>,
{
    fn default() -> Self {
        Self {
            svd: None,
            problem: None,
        }
    }
}

impl<T: Scalar, C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>> LinearSolver<C> for SVD<T, C> {
    fn problem(&self) -> Option<&SolverProblem<C>> {
        self.problem.as_ref()
    }
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>> {
        self.problem.as_mut()
    }
    fn take_problem(&mut self) -> Option<SolverProblem<C>> {
        self.svd = None;
        self.problem.take()
    }

    fn solve_in_place(&self, _state: &mut C::V) -> Result<()> {
        Err(anyhow::anyhow!("Not implemented"))
    }

    fn solve(&self, b: &<C as crate::op::Op>::V) -> Result<<C as crate::op::Op>::V> {
        if self.svd.is_none() {
            return Err(anyhow::anyhow!("SVD not initialized"));
        }
        let svd = self.svd.as_ref().unwrap();
        let eps = self.problem.as_ref().unwrap().atol.as_ref();
        svd.solve(b, eps[0].abs()).map_err(anyhow::Error::msg)
    }

    fn set_problem(&mut self, problem: SolverProblem<C>) {
        self.svd = Some(nalgebra::SVD::new(
            problem.f.jacobian(problem.t),
            true,
            true,
        ));
        self.problem = Some(problem);
    }
}
