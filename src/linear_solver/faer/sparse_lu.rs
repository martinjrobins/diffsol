use std::rc::Rc;

use crate::{
    error::{DiffsolError, LinearSolverError},
    linear_solver::LinearSolver,
    linear_solver_error,
    matrix::sparsity::MatrixSparsityRef,
    op::linearise::LinearisedOp,
    scalar::IndexType,
    solver::SolverProblem,
    LinearOp, Matrix, NonLinearOp, Op, Scalar, SparseColMat,
};

use faer::{
    solvers::SpSolver,
    sparse::linalg::{solvers::Lu, solvers::SymbolicLu},
    Col,
};

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
pub struct FaerSparseLU<T, C>
where
    T: Scalar,
    C: NonLinearOp<M = SparseColMat<T>, V = Col<T>, T = T>,
{
    lu: Option<Lu<IndexType, T>>,
    lu_symbolic: Option<SymbolicLu<IndexType>>,
    problem: Option<SolverProblem<LinearisedOp<C>>>,
    matrix: Option<SparseColMat<T>>,
}

impl<T, C> Default for FaerSparseLU<T, C>
where
    T: Scalar,
    C: NonLinearOp<M = SparseColMat<T>, V = Col<T>, T = T>,
{
    fn default() -> Self {
        Self {
            lu: None,
            problem: None,
            matrix: None,
            lu_symbolic: None,
        }
    }
}

impl<T: Scalar, C: NonLinearOp<M = SparseColMat<T>, V = Col<T>, T = T>> LinearSolver<C>
    for FaerSparseLU<T, C>
{
    fn set_linearisation(&mut self, x: &C::V, t: C::T) {
        Rc::<LinearisedOp<C>>::get_mut(&mut self.problem.as_mut().expect("Problem not set").f)
            .unwrap()
            .set_x(x);
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        self.problem.as_ref().unwrap().f.matrix_inplace(t, matrix);
        self.lu = Some(
            Lu::try_new_with_symbolic(
                self.lu_symbolic.as_ref().unwrap().clone(),
                matrix.faer().as_ref(),
            )
            .expect("Failed to factorise matrix"),
        )
    }

    fn solve_in_place(&self, x: &mut C::V) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(x);
        Ok(())
    }

    fn set_problem(&mut self, problem: &SolverProblem<C>) {
        let linearised_problem = problem.linearise();
        let ncols = linearised_problem.f.nstates();
        let nrows = linearised_problem.f.nout();
        let matrix = C::M::new_from_sparsity(
            nrows,
            ncols,
            linearised_problem
                .f
                .sparsity()
                .map(|s| MatrixSparsityRef::<SparseColMat<T>>::to_owned(&s)),
        );
        self.problem = Some(linearised_problem);
        self.matrix = Some(matrix);
        self.lu_symbolic = Some(
            SymbolicLu::try_new(self.matrix.as_ref().unwrap().faer().symbolic())
                .expect("Failed to create symbolic LU"),
        );
    }
}
