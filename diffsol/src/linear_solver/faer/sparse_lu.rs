use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver::LinearSolver,
    linear_solver_error,
    scalar::IndexType,
    FaerContext, FaerScalar, FaerSparseMat, FaerVec, Matrix, NonLinearOpJacobian,
};

use faer::{
    linalg::solvers::Solve,
    reborrow::Reborrow,
    sparse::linalg::{solvers::Lu, solvers::SymbolicLu},
};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct FaerSparseLU<T>
where
    T: FaerScalar,
{
    lu: Option<Lu<IndexType, T>>,
    lu_symbolic: Option<SymbolicLu<IndexType>>,
    matrix: Option<FaerSparseMat<T>>,
}

impl<T> Default for FaerSparseLU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: None,
            matrix: None,
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
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(
            Lu::try_new_with_symbolic(self.lu_symbolic.as_ref().unwrap().clone(), matrix.data.rb())
                .expect("Failed to factorise matrix"),
        );
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        let lu = self
            .lu
            .as_ref()
            .ok_or_else(|| linear_solver_error!(LuNotInitialized))?;
        lu.solve_in_place(x.data.as_mut());
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
        self.lu_symbolic = Some(
            SymbolicLu::try_new(matrix.data.symbolic()).expect("Failed to create symbolic LU"),
        );
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        linear_solver::tests::{linear_problem, test_linear_solver},
        op::ParameterisedOp,
        FaerSparseMat, Op, Vector,
    };

    use super::*;

    #[test]
    fn test_sparse_lu() {
        let (op, rtol, atol, solns) = linear_problem::<FaerSparseMat<f64>>();
        let p = FaerVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let s = FaerSparseLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }
}
