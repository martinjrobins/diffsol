use diffsol_la::{IndexType, LinearOp as LaLinearOp, LinearSolver, Matrix, Vector};

use crate::{
    convergence::{Convergence, ConvergenceStatus},
    error::{NlError, NonLinearSolverError},
    line_search::LineSearch,
    non_linear_solver_error,
    nonlinear_op::{NonLinearOp, NonLinearOpJacobian},
    nonlinear_solver::NonLinearSolver,
};

#[allow(clippy::too_many_arguments)]
pub fn newton_iteration<V: Vector>(
    xn: &mut V,
    tmp: &mut V,
    error_y: &V,
    fun: impl Fn(&V, &mut V),
    linear_solver: impl Fn(&mut V) -> Result<(), NlError>,
    convergence: &mut Convergence<V>,
    line_search: &mut impl LineSearch<V>,
) -> Result<(), NlError> {
    convergence.reset();
    line_search.reset();
    for _ in 0..convergence.max_iter() {
        let res =
            line_search.take_optimal_step(xn, tmp, error_y, &fun, &linear_solver, convergence)?;
        // xn = xn + alpha * delta_n, where alpha is determined by line search, return status

        match res {
            ConvergenceStatus::Continue => continue,
            ConvergenceStatus::Converged => return Ok(()),
            ConvergenceStatus::Diverged => return Err(non_linear_solver_error!(NewtonDiverged)),
        }
    }
    Err(non_linear_solver_error!(NewtonMaxIterations))
}

/// A borrowing adapter that presents a [NonLinearOpJacobian] (evaluated at a fixed
/// state `x`) as a [`diffsol_la::LinearOp`].
///
/// This is the bridge that allows the [NewtonNonlinearSolver] to be implemented on
/// top of the time-unaware [`diffsol_la::LinearSolver`] backends.
struct JacobianRef<'a, C: NonLinearOpJacobian> {
    op: &'a C,
    x: Option<&'a C::V>,
}

impl<'a, C: NonLinearOpJacobian> JacobianRef<'a, C> {
    /// Create an adapter used only to query the sparsity pattern (no state set).
    fn sparsity_only(op: &'a C) -> Self {
        Self { op, x: None }
    }

    /// Create an adapter that evaluates the Jacobian at `x`.
    fn at(op: &'a C, x: &'a C::V) -> Self {
        Self { op, x: Some(x) }
    }
}

impl<C: NonLinearOpJacobian> LaLinearOp for JacobianRef<'_, C> {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    type C = C::C;

    fn nrows(&self) -> IndexType {
        self.op.nout()
    }

    fn ncols(&self) -> IndexType {
        self.op.nstates()
    }

    fn context(&self) -> &Self::C {
        self.op.context()
    }

    fn matrix_inplace(&self, y: &mut Self::M) {
        let x = self.x.expect("JacobianRef: state x not set");
        self.op.jacobian_inplace(x, y);
    }

    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.jacobian_sparsity()
    }
}

pub struct NewtonNonlinearSolver<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>> {
    linear_solver: Ls,
    line_search: Lsearch,
    is_jacobian_set: bool,
    tmp: M::V,
}

impl<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>>
    NewtonNonlinearSolver<M, Ls, Lsearch>
{
    pub fn new(linear_solver: Ls, line_search: Lsearch) -> Self {
        Self {
            linear_solver,
            line_search,
            is_jacobian_set: false,
            tmp: M::V::zeros(0, Default::default()),
        }
    }
    pub fn linear_solver(&self) -> &Ls {
        &self.linear_solver
    }
}

impl<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>> Default
    for NewtonNonlinearSolver<M, Ls, Lsearch>
{
    fn default() -> Self {
        Self::new(Ls::default(), Lsearch::default())
    }
}

impl<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>> NonLinearSolver<M>
    for NewtonNonlinearSolver<M, Ls, Lsearch>
{
    fn clear_jacobian(&mut self) {
        self.is_jacobian_set = false;
    }

    fn is_jacobian_set(&self) -> bool {
        self.is_jacobian_set
    }

    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C) {
        self.linear_solver
            .set_sparsity(&JacobianRef::sparsity_only(op));
        self.is_jacobian_set = false;
        self.tmp = C::V::zeros(op.nstates(), op.context().clone());
    }

    fn reset_jacobian<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &C::V,
    ) {
        self.linear_solver
            .set_linearisation(&JacobianRef::at(op, x));
        self.is_jacobian_set = true;
    }

    fn solve_linearised_in_place(&self, x: &mut M::V) -> Result<(), NlError> {
        self.linear_solver.solve_in_place(x).map_err(Into::into)
    }

    fn solve_in_place<C: NonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        xn: &mut M::V,
        error_y: &M::V,
        convergence: &mut Convergence<M::V>,
    ) -> Result<(), NlError> {
        if !self.is_jacobian_set {
            return Err(non_linear_solver_error!(JacobianNotReset));
        }
        if xn.len() != op.nstates() {
            let error = NonLinearSolverError::WrongStateLength {
                expected: op.nstates(),
                found: xn.len(),
            };
            return Err(NlError::from(error));
        }
        let linear_solver = |x: &mut C::V| self.linear_solver.solve_in_place(x).map_err(Into::into);
        let fun = |x: &C::V, y: &mut C::V| op.call_inplace(x, y);
        newton_iteration(
            xn,
            &mut self.tmp,
            error_y,
            fun,
            linear_solver,
            convergence,
            &mut self.line_search,
        )
    }
}
