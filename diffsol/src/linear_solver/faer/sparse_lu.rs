use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver::LinearSolver,
    linear_solver_error,
    scalar::IndexType,
    FaerContext, FaerSparseMat, FaerVec, Matrix, NonLinearOpJacobian, Scalar,
};

use faer::{
    linalg::solvers::Solve,
    reborrow::Reborrow,
    sparse::linalg::{solvers::Lu, solvers::SymbolicLu},
};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct FaerSparseLU<T>
where
    T: Scalar,
{
    lu: Option<Lu<IndexType, T>>,
    lu_symbolic: Option<SymbolicLu<IndexType>>,
    matrix: Option<FaerSparseMat<T>>,
}

impl<T> Default for FaerSparseLU<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            lu: None,
            matrix: None,
            lu_symbolic: None,
        }
    }
}

impl<T: Scalar> LinearSolver<FaerSparseMat<T>> for FaerSparseLU<T> {
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerSparseMat<T>>>(
        &mut self,
        op: &C,
        x: &FaerVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(
            Lu::try_new_with_symbolic(self.lu_symbolic.as_ref().unwrap().clone(), matrix.data.rb())
                .expect("Failed to factorise matrix"),
        )
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(&mut x.data);
        Ok(())
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let ncols = op.nstates();
        let nrows = op.nout();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity(), *op.context());
        self.matrix = Some(matrix);
        self.lu_symbolic = Some(
            SymbolicLu::try_new(self.matrix.as_ref().unwrap().data.symbolic())
                .expect("Failed to create symbolic LU"),
        );
    }
}
