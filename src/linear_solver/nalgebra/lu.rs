use anyhow::Result;
use nalgebra::{DMatrix, DVector, Dyn};

use crate::{
    op::{linearise::LinearisedOp, NonLinearOp},
    LinearSolver, Op, Scalar, SolverProblem,
};

/// A [LinearSolver] that uses the LU decomposition in the [`nalgebra` library](https://nalgebra.org/) to solve the linear system.
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
    fn solve_in_place(&self, state: &mut C::V) -> Result<()> {
        if self.lu.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let lu = self.lu.as_ref().unwrap();
        match lu.solve_mut(state) {
            true => Ok(()),
            false => Err(anyhow::anyhow!("LU solve failed")),
        }
    }

    fn set_linearisation(&mut self, x: &<C as Op>::V, t: <C as Op>::T) {
        let problem = self.problem.as_ref().expect("Problem not set");
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        problem.f.jacobian_inplace(x, t, matrix);
        self.lu = Some(matrix.lu());
    }

    fn set_problem(&mut self, problem: &SolverProblem<C>) {
        let linearised_problem = problem.linearise();
        let matrix = DMatrix::zeros(
            linearised_problem.f.nstates(),
            linearised_problem.f.nstates(),
        );
        self.problem = Some(linearised_problem);
        self.matrix = Some(matrix);
    }
}
