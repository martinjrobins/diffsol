use crate::FaerContext;
use crate::{error::LinearSolverError, linear_solver_error};

use crate::{
    error::DiffsolError, linear_solver::LinearSolver, FaerMat, FaerScalar, FaerVec, Matrix,
    NonLinearOpJacobian, Vector, VectorCommon, VectorHost,
};

use faer::{linalg::solvers::FullPivLu, linalg::solvers::Solve, Col};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T>
where
    T: FaerScalar,
{
    lu: Vec<Option<FullPivLu<T>>>,
    matrix: Vec<Option<FaerMat<T>>>,
}

impl<T> Default for LU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: Vec::new(),
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
        let nbatch = x.context().nbatch;
        let nstates = x.len();
        for b in 0..nbatch {
            let matrix = self.matrix[b].as_mut().expect("Matrix not set");
            let x_batch = Col::from_fn(nstates, |i| x.as_slice()[i * nbatch + b]);
            let tmp_x = FaerVec::from(x_batch);
            op.jacobian_inplace(&tmp_x, t, matrix);
            self.lu[b] = Some(matrix.data.full_piv_lu());
        }
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        let nbatch = x.context().nbatch;
        let nstates = x.len();
        for b in 0..nbatch {
            let lu = self.lu[b].as_ref().ok_or(linear_solver_error!(LuNotInitialized))?;
            let mut x_batch = Col::from_fn(nstates, |i| x.as_slice()[i * nbatch + b]);
            lu.solve_in_place(x_batch.as_mut());
            for i in 0..nstates {
                x.as_mut_slice()[i * nbatch + b] = x_batch[i];
            }
        }
        Ok(())
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let nbatch = op.context().nbatch;
        let ncols = op.nstates();
        let nrows = op.nout();
        self.matrix.resize(nbatch, None);
        self.lu.resize(nbatch, None);
        for b in 0..nbatch {
            let matrix =
                C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity(), *op.context());
            self.matrix[b] = Some(matrix);
        }
    }
}
