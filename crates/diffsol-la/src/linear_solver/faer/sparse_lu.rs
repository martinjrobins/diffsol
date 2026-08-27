use crate::context::broadcast_batch;
use crate::{
    error::LaError, linear_solver::LinearSolver, linear_solver_error, scalar::IndexType, Context,
    FaerContext, FaerScalar, FaerSparseMat, FaerVec, LinearOp, Matrix,
};

use faer::{
    linalg::solvers::Solve,
    reborrow::{Reborrow, ReborrowMut},
    sparse::linalg::{solvers::Lu, solvers::SymbolicLu},
};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct FaerSparseLU<T>
where
    T: FaerScalar,
{
    lu: Vec<Lu<IndexType, T>>,
    lu_symbolic: Option<SymbolicLu<IndexType>>,
    matrix: Option<FaerSparseMat<T>>,
}

impl<T> Default for FaerSparseLU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: Vec::new(),
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
        self.lu = matrix
            .data
            .iter()
            .map(|matrix| {
                Lu::try_new_with_symbolic(self.lu_symbolic.as_ref().unwrap().clone(), matrix.rb())
                    .expect("Failed to factorise matrix")
            })
            .collect();
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        if self.lu.is_empty() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        x.context
            .assert_broadcastable_into(self.lu.len(), "sparse_lu_solve");
        let nlu = self.lu.len();
        let nb = x.data.ncols();
        for batch in 0..nb {
            self.lu[broadcast_batch(batch, nlu, nb)].solve_in_place(x.data.rb_mut().col_mut(batch));
        }
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
            SymbolicLu::try_new(matrix.data[0].symbolic()).expect("Failed to create symbolic LU"),
        );
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linear_solver::tests::{diagonal_op, test_grouped_lu_solve, test_narrow_state_lu_solve},
        Vector,
    };

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

    #[test]
    fn test_grouped_sparse_lu() {
        test_grouped_lu_solve::<FaerSparseMat<f64>, FaerSparseLU<f64>>(FaerContext::with_nbatch(2));
    }

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_narrow_state_sparse_lu() {
        test_narrow_state_lu_solve::<FaerSparseMat<f64>, FaerSparseLU<f64>>(
            FaerContext::with_nbatch(2),
        );
    }
}
