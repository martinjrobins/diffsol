use std::rc::Rc;

use crate::sundials_sys::{
    realtype, SUNLinSolFree, SUNLinSolSetup, SUNLinSolSolve, SUNLinSol_Dense, SUNLinearSolver,
};

use crate::{
    error::*, linear_solver_error, ode_solver::sundials::sundials_check,
    vector::sundials::SundialsVector, Matrix, NonLinearOpJacobian, SundialsMatrix,
};

#[cfg(not(sundials_version_major = "5"))]
use crate::vector::sundials::get_suncontext;

use super::LinearSolver;

pub struct SundialsLinearSolver {
    linear_solver: Option<SUNLinearSolver>,
    is_setup: bool,
    matrix: Option<SundialsMatrix>,
}

impl Default for SundialsLinearSolver {
    fn default() -> Self {
        Self::new_dense()
    }
}

impl SundialsLinearSolver {
    pub fn new_dense() -> Self {
        Self {
            linear_solver: None,
            is_setup: false,
            matrix: None,
        }
    }
}

impl Drop for SundialsLinearSolver {
    fn drop(&mut self) {
        if let Some(linear_solver) = self.linear_solver {
            unsafe { SUNLinSolFree(linear_solver) };
        }
    }
}

impl LinearSolver<SundialsMatrix> for SundialsLinearSolver {
    fn set_problem<C: NonLinearOpJacobian<T = realtype, V = SundialsVector, M = SundialsMatrix>>(
        &mut self,
        op: &C,
        _rtol: realtype,
        _atol: Rc<SundialsVector>,
    ) {
        let matrix = SundialsMatrix::zeros(op.nstates(), op.nstates());
        let y0 = SundialsVector::new_serial(op.nstates());

        #[cfg(not(sundials_version_major = "5"))]
        let linear_solver = {
            let ctx = *get_suncontext();
            unsafe { SUNLinSol_Dense(y0.sundials_vector(), matrix.sundials_matrix(), ctx) }
        };

        #[cfg(sundials_version_major = "5")]
        let linear_solver =
            unsafe { SUNLinSol_Dense(y0.sundials_vector(), matrix.sundials_matrix()) };

        self.matrix = Some(matrix);
        self.linear_solver = Some(linear_solver);
    }

    fn set_linearisation<
        C: NonLinearOpJacobian<T = realtype, V = SundialsVector, M = SundialsMatrix>,
    >(
        &mut self,
        op: &C,
        x: &SundialsVector,
        t: realtype,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        let linear_solver = self.linear_solver.expect("Linear solver not set");
        op.jacobian_inplace(x, t, matrix);
        sundials_check(unsafe { SUNLinSolSetup(linear_solver, matrix.sundials_matrix()) }).unwrap();
        self.is_setup = true;
    }

    fn solve_in_place(&self, b: &mut SundialsVector) -> Result<(), DiffsolError> {
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
