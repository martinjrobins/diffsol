use crate::{
    error::{DiffsolError, NonLinearSolverError},
    non_linear_solver_error, Convergence, ConvergenceStatus, LinearSolver, Matrix, NonLinearOp,
    NonLinearOpJacobian, NonLinearSolver, Vector,
};

pub fn newton_iteration<V: Vector>(
    xn: &mut V,
    tmp: &mut V,
    error_y: &V,
    fun: impl Fn(&V, &mut V),
    linear_solver: impl Fn(&mut V) -> Result<(), DiffsolError>,
    convergence: &mut Convergence<V>,
) -> Result<(), DiffsolError> {
    convergence.reset();
    loop {
        fun(xn, tmp);
        //tmp = f_at_n

        linear_solver(tmp)?;
        //tmp = -delta_n

        xn.sub_assign(&*tmp);
        // xn = xn + delta_n

        let res = convergence.check_new_iteration(tmp, error_y);
        match res {
            ConvergenceStatus::Continue => continue,
            ConvergenceStatus::Converged => return Ok(()),
            ConvergenceStatus::Diverged => break,
            ConvergenceStatus::MaximumIterations => break,
        }
    }
    Err(non_linear_solver_error!(NewtonDidNotConverge))
}

pub struct NewtonNonlinearSolver<M: Matrix, Ls: LinearSolver<M>> {
    linear_solver: Ls,
    is_jacobian_set: bool,
    tmp: M::V,
}

impl<M: Matrix, Ls: LinearSolver<M>> NewtonNonlinearSolver<M, Ls> {
    pub fn new(linear_solver: Ls) -> Self {
        Self {
            linear_solver,
            is_jacobian_set: false,
            tmp: M::V::zeros(0, Default::default()),
        }
    }
    pub fn linear_solver(&self) -> &Ls {
        &self.linear_solver
    }
}

impl<M: Matrix, Ls: LinearSolver<M>> Default for NewtonNonlinearSolver<M, Ls> {
    fn default() -> Self {
        Self::new(Ls::default())
    }
}

impl<M: Matrix, Ls: LinearSolver<M>> NonLinearSolver<M> for NewtonNonlinearSolver<M, Ls> {
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
        newton_iteration(xn, &mut self.tmp, error_y, fun, linear_solver, convergence)
    }
}
