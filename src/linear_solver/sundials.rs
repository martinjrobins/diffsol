use anyhow::Result;
use sundials_sys::{
    realtype, SUNLinSolFree, SUNLinSolSetup, SUNLinSolSolve, SUNLinSol_Dense, SUNLinearSolver,
};

use crate::{
    matrix::sundials::SundialsMatrix,
    ode_solver::sundials::sundials_check,
    op::LinearOp,
    solver::SolverProblem,
    vector::sundials::{get_suncontext, SundialsVector},
};

use super::LinearSolver;

pub struct SundialsLinearSolver<Op>
where
    Op: LinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    linear_solver: SUNLinearSolver,
    problem: Option<SolverProblem<Op>>,
    is_setup: bool,
    matrix: SundialsMatrix,
}

impl<Op> SundialsLinearSolver<Op>
where
    Op: LinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    pub fn new_dense() -> Self {
        let y0 = SundialsVector::new_serial(1);
        let matrix = SundialsMatrix::new_dense(1, 1);
        let ctx = *get_suncontext();
        let linear_solver =
            unsafe { SUNLinSol_Dense(y0.sundials_vector(), matrix.sundials_matrix(), ctx) };
        Self {
            linear_solver,
            problem: None,
            is_setup: false,
            matrix,
        }
    }
}

impl<Op> Drop for SundialsLinearSolver<Op>
where
    Op: LinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    fn drop(&mut self) {
        unsafe { SUNLinSolFree(self.linear_solver) };
    }
}

impl<Op> LinearSolver<Op> for SundialsLinearSolver<Op>
where
    Op: LinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    fn set_problem(&mut self, problem: SolverProblem<Op>) {
        self.matrix = problem.f.jacobian(problem.t);
        sundials_check(unsafe {
            SUNLinSolSetup(self.linear_solver, self.matrix.sundials_matrix())
        })
        .unwrap();
        self.is_setup = true;
    }

    fn problem(&self) -> Option<&SolverProblem<Op>> {
        self.problem.as_ref()
    }

    fn problem_mut(&mut self) -> Option<&mut SolverProblem<Op>> {
        self.problem.as_mut()
    }

    fn take_problem(&mut self) -> Option<SolverProblem<Op>> {
        self.is_setup = false;
        self.problem.take()
    }

    fn solve_in_place(&mut self, b: &mut Op::V) -> Result<()> {
        if !self.is_setup {
            return Err(anyhow::anyhow!("Linear solver not setup"));
        }
        let tol = 1e-6;
        sundials_check(unsafe {
            SUNLinSolSolve(
                self.linear_solver,
                self.matrix.sundials_matrix(),
                b.sundials_vector(),
                b.sundials_vector(),
                tol,
            )
        })
    }
}
