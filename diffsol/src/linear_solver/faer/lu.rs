use crate::FaerContext;
use crate::{error::LinearSolverError, linear_solver_error};

use crate::{
    error::DiffsolError, linear_solver::LinearSolver, matrix::MatrixCommon, Context, FaerMat,
    FaerScalar, FaerVec, Matrix, NonLinearOpJacobian, Vector,
};

use faer::{linalg::solvers::FullPivLu, linalg::solvers::Solve};

pub struct LU<T>
where
    T: FaerScalar,
{
    lu: Vec<FullPivLu<T>>,
    matrix: Option<FaerMat<T>>,
}

impl<T> Default for LU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: None,
        }
    }
}

impl<T: FaerScalar> LinearSolver<FaerMat<T>> for LU<T> {
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerMat<T>>>(
        &mut self,
        op: &C,
        x: &FaerVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        let nbatch = matrix.context.nbatch();
        let ncols = matrix.ncols();
        self.lu.clear();
        for b in 0..nbatch {
            let sub = matrix
                .data
                .get(0..matrix.nrows(), b * ncols..(b + 1) * ncols)
                .to_owned();
            self.lu.push(sub.full_piv_lu());
        }
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let nbatch = x.context().nbatch();
        for b in 0..nbatch {
            let lu = &self.lu[b];
            lu.solve_in_place(x.data.col_mut(b));
        }
        Ok(())
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let ncols = op.nstates();
        let nrows = op.nout();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity(), *op.context());
        self.matrix = Some(matrix);
    }
}
