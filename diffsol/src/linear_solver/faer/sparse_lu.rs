use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver::LinearSolver,
    linear_solver_error,
    scalar::IndexType,
    FaerContext, FaerScalar, FaerSparseMat, FaerVec, Matrix, NonLinearOpJacobian,
    Vector, VectorHost,
};

use faer::{
    linalg::solvers::Solve,
    reborrow::Reborrow,
    sparse::linalg::{solvers::Lu, solvers::SymbolicLu},
    Col,
};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct FaerSparseLU<T>
where
    T: FaerScalar,
{
    lu: Vec<Option<Lu<IndexType, T>>>,
    lu_symbolic: Option<SymbolicLu<IndexType>>,
    matrix: Vec<Option<FaerSparseMat<T>>>,
}

impl<T> Default for FaerSparseLU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
            matrix: Vec::new(),
            lu_symbolic: None,
        }
    }
}

impl<T: FaerScalar> LinearSolver<FaerSparseMat<T>> for FaerSparseLU<T> {
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerSparseMat<T>>>(
        &mut self,
        op: &C,
        x: &FaerVec<T>,
        t: T,
    ) {
        let nbatch = x.context().nbatch;
        let nstates = x.len();
        let lu_sym = self.lu_symbolic.as_ref().expect("Symbolic LU not initialised");
        for b in 0..nbatch {
            let matrix = self.matrix[b].as_mut().expect("Matrix not set");
            let x_batch = Col::from_fn(nstates, |i| x.as_slice()[i * nbatch + b]);
            let tmp_x = FaerVec::from(x_batch);
            op.jacobian_inplace(&tmp_x, t, matrix);
            self.lu[b] = Some(
                Lu::try_new_with_symbolic(lu_sym.clone(), matrix.data.rb())
                    .expect("Failed to factorise matrix"),
            );
        }
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        let nbatch = x.context().nbatch;
        let nstates = x.len();
        for b in 0..nbatch {
            let lu = self.lu[b].as_ref().ok_or(linear_solver_error!(LuNotInitialized))?;
            let mut x_batch = Col::from_fn(nstates, |i| x.as_slice()[i * nbatch + b]);
            lu.solve_in_place(&mut x_batch);
            for i in 0..nstates {
                x.as_mut_slice()[i * nbatch + b] = x_batch[i];
            }
        }
        Ok(())
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>,
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
        self.lu_symbolic = Some(
            SymbolicLu::try_new(self.matrix[0].as_ref().unwrap().data.symbolic())
                .expect("Failed to create symbolic LU"),
        );
    }
}
