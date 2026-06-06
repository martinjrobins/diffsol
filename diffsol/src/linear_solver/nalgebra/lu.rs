use nalgebra::{DVector, Dyn};

use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver_error,
    matrix::dense_nalgebra_serial::NalgebraMat,
    LinearSolver, Matrix, NalgebraContext, NalgebraScalar, NalgebraVec, NonLinearOpJacobian,
    Vector, VectorHost,
};

/// A [LinearSolver] that uses the LU decomposition in the [`nalgebra` library](https://nalgebra.org/) to solve the linear system.
#[derive(Clone)]
pub struct LU<T>
where
    T: NalgebraScalar,
{
    matrix: Vec<Option<NalgebraMat<T>>>,
    lu: Vec<Option<nalgebra::LU<T, Dyn, Dyn>>>,
}

impl<T> Default for LU<T>
where
    T: NalgebraScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: Vec::new(),
        }
    }
}

impl<T: NalgebraScalar> LinearSolver<NalgebraMat<T>> for LU<T> {
    fn solve_in_place(&self, state: &mut NalgebraVec<T>) -> Result<(), DiffsolError> {
        let nbatch = state.context().nbatch;
        let nstates = state.len();
        for b in 0..nbatch {
            let lu = self.lu[b]
                .as_ref()
                .ok_or(linear_solver_error!(LuNotInitialized))?;
            let mut x_batch =
                DVector::from_fn(nstates, |i, _| state.as_slice()[i * nbatch + b]);
            match lu.solve_mut(&mut x_batch) {
                true => {
                    for i in 0..nstates {
                        state.as_mut_slice()[i * nbatch + b] = x_batch[i];
                    }
                }
                false => return Err(linear_solver_error!(LuSolveFailed))?,
            }
        }
        Ok(())
    }

    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>>>(
        &mut self,
        op: &C,
        x: &NalgebraVec<T>,
        t: T,
    ) {
        let nbatch = x.context().nbatch;
        let nstates = x.len();
        for b in 0..nbatch {
            let matrix = self.matrix[b].as_mut().expect("Matrix not set");
            let x_batch =
                DVector::from_fn(nstates, |i, _| x.as_slice()[i * nbatch + b]);
            let tmp_x = NalgebraVec {
                data: x_batch,
                context: *x.context(),
            };
            op.jacobian_inplace(&tmp_x, t, matrix);
            self.lu[b] = Some(matrix.data.clone().lu());
        }
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = NalgebraVec<T>, M = NalgebraMat<T>, C = NalgebraContext>,
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
