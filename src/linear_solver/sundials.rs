use std::rc::Rc;

use crate::sundials_sys::{
    realtype, SUNLinSolFree, SUNLinSolSetup, SUNLinSolSolve, SUNLinSol_Dense, SUNLinearSolver,
};

use crate::{
    error::*, linear_solver_error, ode_solver::sundials::sundials_check, ode_solver_error,
    op::linearise::LinearisedOp, vector::sundials::SundialsVector, LinearOp, Matrix, NonLinearOp,
    Op, SolverProblem, SundialsMatrix,
};

#[cfg(not(sundials_version_major = "5"))]
use crate::vector::sundials::get_suncontext;

use super::LinearSolver;

pub struct SundialsLinearSolver<Op>
where
    Op: NonLinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    linear_solver: Option<SUNLinearSolver>,
    problem: Option<SolverProblem<LinearisedOp<Op>>>,
    is_setup: bool,
    matrix: Option<SundialsMatrix>,
}

impl<Op> Default for SundialsLinearSolver<Op>
where
    Op: NonLinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    fn default() -> Self {
        Self::new_dense()
    }
}

impl<Op> SundialsLinearSolver<Op>
where
    Op: NonLinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    pub fn new_dense() -> Self {
        Self {
            linear_solver: None,
            problem: None,
            is_setup: false,
            matrix: None,
        }
    }
}

impl<Op> Drop for SundialsLinearSolver<Op>
where
    Op: NonLinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    fn drop(&mut self) {
        if let Some(linear_solver) = self.linear_solver {
            unsafe { SUNLinSolFree(linear_solver) };
        }
    }
}

impl<Op> LinearSolver<Op> for SundialsLinearSolver<Op>
where
    Op: NonLinearOp<M = SundialsMatrix, V = SundialsVector, T = realtype>,
{
    fn set_problem(&mut self, problem: &SolverProblem<Op>) {
        let linearised_problem = problem.linearise();
        let matrix = SundialsMatrix::zeros(
            linearised_problem.f.nstates(),
            linearised_problem.f.nstates(),
        );
        let y0 = SundialsVector::new_serial(linearised_problem.f.nstates());

        #[cfg(not(sundials_version_major = "5"))]
        let linear_solver = {
            let ctx = *get_suncontext();
            unsafe { SUNLinSol_Dense(y0.sundials_vector(), matrix.sundials_matrix(), ctx) }
        };

        #[cfg(sundials_version_major = "5")]
        let linear_solver =
            unsafe { SUNLinSol_Dense(y0.sundials_vector(), matrix.sundials_matrix()) };

        self.matrix = Some(matrix);
        self.problem = Some(linearised_problem);
        self.linear_solver = Some(linear_solver);
    }

    fn set_linearisation(&mut self, x: &Op::V, t: Op::T) {
        Rc::<LinearisedOp<Op>>::get_mut(&mut self.problem.as_mut().expect("Problem not set").f)
            .unwrap()
            .set_x(x);
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        let linear_solver = self.linear_solver.expect("Linear solver not set");
        self.problem.as_ref().unwrap().f.matrix_inplace(t, matrix);
        sundials_check(unsafe { SUNLinSolSetup(linear_solver, matrix.sundials_matrix()) }).unwrap();
        self.is_setup = true;
    }

    fn solve_in_place(&self, b: &mut Op::V) -> Result<(), DiffsolError> {
        if !self.is_setup {
            return Err(linear_solver_error!(LinearSolverNotSetup));
        }
        let linear_solver = self.linear_solver.expect("Linear solver not set");
        let matrix = self.matrix.as_ref().expect("Matrix not set");
        let tol = 1e-6;
        sundials_check(unsafe {
            SUNLinSolSolve(
                linear_solver,
                matrix.sundials_matrix(),
                b.sundials_vector(),
                b.sundials_vector(),
                tol,
            )
        })
    }
}
