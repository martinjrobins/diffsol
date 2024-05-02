use std::rc::Rc;

use anyhow::Result;
use nalgebra::{DMatrix, DVector, Dyn};

use crate::{
    op::{linearise::LinearisedOp, NonLinearOp},
    LinearOp, LinearSolver, Matrix, Op, Scalar, SolverProblem,
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
        let matrix = C::M::new_from_sparsity(nrows, ncols, linearised_problem.f.sparsity());
        self.problem = Some(linearised_problem);
        self.matrix = Some(matrix);
    }
}
