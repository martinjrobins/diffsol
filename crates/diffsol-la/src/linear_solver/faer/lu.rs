use crate::FaerContext;
use crate::{error::LaError, linear_solver_error};

use crate::{linear_solver::LinearSolver, FaerMat, FaerScalar, FaerVec, LinearOp, Matrix};

use faer::{linalg::solvers::FullPivLu, linalg::solvers::Solve};
/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct LU<T>
where
    T: FaerScalar,
{
    lu: Option<FullPivLu<T>>,
    matrix: Option<FaerMat<T>>,
}

impl<T> Default for LU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self {
            lu: None,
            matrix: None,
        }
    }
}

impl<T: FaerScalar> LinearSolver<FaerMat<T>> for LU<T> {
    fn set_linearisation<C: LinearOp<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.matrix_inplace(matrix);
        self.lu = Some(matrix.data.full_piv_lu());
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(&mut x.data);
        Ok(())
    }

    fn set_sparsity<C: LinearOp<T = T, V = FaerVec<T>, M = FaerMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) {
        let ncols = op.ncols();
        let nrows = op.nrows();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), *op.context());
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear_solver::tests::diagonal_op, Vector};

    #[test]
    fn test_lu() {
        let mut s = LU::<f64>::default();
        let op = diagonal_op::<FaerMat<f64>>(2.0);
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
