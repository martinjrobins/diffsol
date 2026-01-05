use crate::{
    error::{DiffsolError, NonLinearSolverError},
    non_linear_solver_error, Convergence, ConvergenceStatus, LineSearch, LinearSolver, Matrix,
    NonLinearOp, NonLinearOpJacobian, NonLinearSolver, Vector,
};

#[allow(clippy::too_many_arguments)]
pub fn newton_iteration<V: Vector>(
    xn: &mut V,
    tmp: &mut V,
    error_y: &V,
    fun: impl Fn(&V, &mut V),
    linear_solver: impl Fn(&mut V) -> Result<(), DiffsolError>,
    convergence: &mut Convergence<V>,
    line_search: &mut impl LineSearch<V>,
) -> Result<(), DiffsolError> {
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
        self.linear_solver.set_problem(op);
        self.is_jacobian_set = false;
        self.tmp = C::V::zeros(op.nstates(), op.context().clone());
    }

    fn reset_jacobian<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &C::V,
        t: C::T,
    ) {
        self.linear_solver.set_linearisation(op, x, t);
        self.is_jacobian_set = true;
    }

    fn solve_linearised_in_place(&self, x: &mut M::V) -> Result<(), DiffsolError> {
        self.linear_solver.solve_in_place(x)
    }

    fn solve_in_place<C: NonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        xn: &mut M::V,
        t: M::T,
        error_y: &M::V,
        convergence: &mut Convergence<M::V>,
    ) -> Result<(), DiffsolError> {
        if !self.is_jacobian_set {
            return Err(non_linear_solver_error!(JacobianNotReset));
        }
        if xn.len() != op.nstates() {
            let error = NonLinearSolverError::WrongStateLength {
                expected: op.nstates(),
                found: xn.len(),
            };
            return Err(DiffsolError::from(error));
        }
        let linear_solver = |x: &mut C::V| self.linear_solver.solve_in_place(x);
        let fun = |x: &C::V, y: &mut C::V| op.call_inplace(x, t, y);
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
