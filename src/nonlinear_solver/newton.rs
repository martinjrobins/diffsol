use crate::{
    op::NonLinearOp, Convergence, ConvergenceStatus, LinearSolver, NonLinearSolver, SolverProblem,
    Vector,
};
use anyhow::{anyhow, Result};

pub fn newton_iteration<V: Vector>(
    xn: &mut V,
    fun: impl Fn(&V, &mut V),
    linear_solver: impl Fn(&mut V) -> Result<()>,
    convergence: &mut Convergence<V>,
) -> Result<usize> {
    convergence.reset();
    let mut tmp = xn.clone();
    let mut niter = 0;
    loop {
        niter += 1;
        fun(xn, &mut tmp);
        //tmp = f_at_n

        linear_solver(&mut tmp)?;
        //tmp = -delta_n

        xn.sub_assign(&tmp);
        // xn = xn + delta_n

        let res = convergence.check_new_iteration(&mut tmp, xn);
        match res {
            ConvergenceStatus::Continue => continue,
            ConvergenceStatus::Converged => return Ok(niter),
            ConvergenceStatus::Diverged => break,
            ConvergenceStatus::MaximumIterations => break,
        }
    }
    Err(anyhow!("Newton iteration did not converge"))
}

pub struct NewtonNonlinearSolver<C: NonLinearOp, Ls: LinearSolver<C>> {
    convergence: Option<Convergence<C::V>>,
    linear_solver: Ls,
    problem: Option<SolverProblem<C>>,
    max_iter: usize,
    niter: usize,
    is_jacobian_set: bool,
}

impl<C: NonLinearOp, Ls: LinearSolver<C>> NewtonNonlinearSolver<C, Ls> {
    pub fn new(linear_solver: Ls) -> Self {
        Self {
            problem: None,
            convergence: None,
            linear_solver,
            max_iter: 100,
            niter: 0,
            is_jacobian_set: false,
        }
    }
}

impl<C: NonLinearOp, Ls: LinearSolver<C>> NonLinearSolver<C> for NewtonNonlinearSolver<C, Ls> {
    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }
    fn max_iter(&self) -> usize {
        self.max_iter
    }
    fn niter(&self) -> usize {
        self.niter
    }
    fn problem(&self) -> &SolverProblem<C> {
        self.problem
            .as_ref()
            .expect("NewtonNonlinearSolver::problem() called before set_problem")
    }
    fn set_problem(&mut self, problem: &SolverProblem<C>) {
        self.problem = Some(problem.clone());
        self.linear_solver.set_problem(problem);
        let problem = self.problem.as_ref().unwrap();
        self.convergence = Some(Convergence::new_from_problem(problem, self.max_iter));
        self.is_jacobian_set = false;
    }

    fn reset_jacobian(&mut self, x: &C::V, t: C::T) {
        self.linear_solver.set_linearisation(x, t);
        self.is_jacobian_set = true;
    }

    fn solve_linearised_in_place(&self, x: &mut C::V) -> Result<()> {
        self.linear_solver.solve_in_place(x)
    }

    fn solve_in_place(&mut self, xn: &mut C::V, t: C::T) -> Result<()> {
        if self.convergence.is_none() || self.problem.is_none() {
            panic!("NewtonNonlinearSolver::solve() called before set_problem");
        }
        if !self.is_jacobian_set {
            self.reset_jacobian(xn, t);
        }
        if xn.len() != self.problem.as_ref().unwrap().f.nstates() {
            panic!("NewtonNonlinearSolver::solve() called with state of wrong size, expected {}, got {}", self.problem.as_ref().unwrap().f.nstates(), xn.len());
        }
        let linear_solver = |x: &mut C::V| self.linear_solver.solve_in_place(x);
        let problem = self.problem.as_ref().unwrap();
        let fun = |x: &C::V, y: &mut C::V| problem.f.call_inplace(x, t, y);
        let convergence = self.convergence.as_mut().unwrap();
        self.niter = newton_iteration(xn, fun, linear_solver, convergence)?;
        Ok(())
    }
}
