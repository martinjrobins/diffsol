use nalgebra::{DVector, Dyn, DMatrix};
use anyhow::Result;

use crate::{callable::LinearOp, solver::LinearSolver, Scalar, SolverProblem};

pub struct LU<T, C> 
where
    T: Scalar,
    C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>,
{
    lu: Option<nalgebra::LU<T, Dyn, Dyn>>,
    problem: Option<SolverProblem<C>>,
}

impl<T, C> Default for LU<T, C> 
where
    T: Scalar,
    C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>,
{
    fn default() -> Self {
        Self {
            lu: None,
            problem: None,
        }
    }
}

impl<T: Scalar, C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>> LinearSolver<C> for LU<T, C> {
    fn problem(&self) -> Option<&SolverProblem<C>> {
        self.problem.as_ref()
    }
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>> {
        self.problem.as_mut()
    }
    fn take_problem(&mut self) -> Option<SolverProblem<C>> {
        self.lu = None;
        self.problem.take()
    }

    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()> {
        if self.lu.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let lu = self.lu.as_ref().unwrap();
        match lu.solve_mut(state) {
            true => Ok(()),
            false => Err(anyhow::anyhow!("LU solve failed")),
        }
    }

    fn set_problem(&mut self, problem: SolverProblem<C>) {
        self.lu = Some(nalgebra::LU::new(problem.f.jacobian(&problem.p, problem.t)));
        self.problem = Some(problem);
    }
}
    
