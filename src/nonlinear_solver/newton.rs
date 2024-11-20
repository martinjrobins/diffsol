use std::rc::Rc;

use crate::{
    error::{DiffsolError, NonLinearSolverError}, non_linear_solver_error, Convergence, ConvergenceStatus, DefaultSolver, LinearSolver, Matrix, NonLinearOp, NonLinearOpJacobian, NonLinearSolver, Vector
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

pub struct NewtonNonlinearSolver<M: Matrix + DefaultSolver, Ls: LinearSolver<M> = <M as DefaultSolver>::LS> {
    convergence: Option<Convergence<M::V>>,
    linear_solver: Ls,
    is_jacobian_set: bool,
    tmp: M::V,
}

impl<M: Matrix + DefaultSolver, Ls: LinearSolver<M>> NewtonNonlinearSolver<M, Ls> {
    pub fn new(linear_solver: Ls) -> Self {
        Self {
            convergence: None,
            linear_solver,
            is_jacobian_set: false,
            tmp: M::V::zeros(0),
        }
    }
    pub fn linear_solver(&self) -> &Ls {
        &self.linear_solver
    }
}

impl<M: Matrix + DefaultSolver, Ls: LinearSolver<M>> Default for NewtonNonlinearSolver<M, Ls> {
    fn default() -> Self {
        Self::new(Ls::default())
    }
}

impl<M: Matrix + DefaultSolver, Ls: LinearSolver<M>> NonLinearSolver<M> for NewtonNonlinearSolver<M, Ls> {
    fn convergence(&self) -> &Convergence<M::V> {
        self.convergence
            .as_ref()
            .expect("NewtonNonlinearSolver::convergence() called before set_problem")
    }

    fn convergence_mut(&mut self) -> &mut Convergence<M::V> {
        self.convergence
            .as_mut()
            .expect("NewtonNonlinearSolver::convergence_mut() called before set_problem")
    }

    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        rtol: M::T,
        atol: Rc<M::V>,
    ) {
        self.linear_solver.set_problem(op, rtol, atol.clone());
        self.convergence = Some(Convergence::new(rtol, atol));
        self.is_jacobian_set = false;
        self.tmp = C::V::zeros(op.nstates());
    }

    fn reset_jacobian<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M>>(
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
    ) -> Result<(), DiffsolError> {
        if self.convergence.is_none() {
            panic!("NewtonNonlinearSolver::solve() called before set_problem");
        }
        if !self.is_jacobian_set {
            panic!("NewtonNonlinearSolver::solve_in_place() called before reset_jacobian");
        }
        if xn.len() != op.nstates() {
            panic!("NewtonNonlinearSolver::solve() called with state of wrong size, expected {}, got {}", op.nstates(), xn.len());
        }
        let linear_solver = |x: &mut C::V| self.linear_solver.solve_in_place(x);
        let fun = |x: &C::V, y: &mut C::V| op.call_inplace(x, t, y);
        let convergence = self.convergence.as_mut().unwrap();
        newton_iteration(xn, &mut self.tmp, error_y, fun, linear_solver, convergence)
    }
}
