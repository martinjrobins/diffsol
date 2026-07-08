use crate::{
    error::LaError, linear_solver::LinearSolver, linear_solver_error, scalar::IndexType,
    FaerContext, FaerScalar, FaerSparseMat, FaerVec, LinearOp, Matrix,
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
    fn set_linearisation<
        C: LinearOp<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.matrix_inplace(matrix);
        self.lu = Some(
            Lu::try_new_with_symbolic(self.lu_symbolic.as_ref().unwrap().clone(), matrix.data.rb())
                .expect("Failed to factorise matrix"),
        );
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        let lu = self
            .lu
            .as_ref()
            .ok_or_else(|| linear_solver_error!(LuNotInitialized))?;
        lu.solve_in_place(&mut x.data);
        Ok(())
    }

    fn set_sparsity<C: LinearOp<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) {
        let ncols = op.ncols();
        let nrows = op.nrows();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), *op.context());
        self.lu_symbolic = Some(
            SymbolicLu::try_new(matrix.data.symbolic()).expect("Failed to create symbolic LU"),
        );
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear_solver::tests::diagonal_op, Vector};

    #[test]
    fn test_sparse_lu() {
        let mut s = FaerSparseLU::<f64>::default();
        let op = diagonal_op::<FaerSparseMat<f64>>(2.0);
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = FaerVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &FaerVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }
}
