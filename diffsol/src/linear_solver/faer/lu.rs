use crate::FaerContext;
use crate::{error::LinearSolverError, linear_solver_error};

use crate::{
    error::DiffsolError, linear_solver::LinearSolver, FaerMat, FaerScalar, FaerVec, Matrix,
    NonLinearOpJacobian,
};

use faer::{linalg::solvers::FullPivLu, linalg::solvers::Solve};

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
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = FaerVec<T>, M = FaerMat<T>>>(
        &mut self,
        op: &C,
        x: &FaerVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(matrix.data.to_owned().full_piv_lu());
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

#[cfg(test)]
mod tests {
    use crate::{
        linear_solver::tests::{linear_problem, test_linear_solver},
        op::ParameterisedOp,
        FaerMat, Op, Vector,
    };

    use super::*;

    #[test]
    fn test_lu() {
        let (op, rtol, atol, solns) = linear_problem::<FaerMat<f64>>();
        let p = FaerVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let s = LU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }
}
