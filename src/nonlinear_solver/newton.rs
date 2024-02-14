use std::rc::Rc;

use crate::{callable::{linearise::LinearisedOp, Jacobian, NonLinearOp}, solver::NonLinearSolver, vector::Vector, IterativeSolver, Scalar, Solver, SolverProblem, LU};
use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector};
use std::ops::SubAssign;

use super::{Convergence, ConvergenceStatus};

pub struct NewtonNonlinearSolver<C: NonLinearOp> 
{
    convergence: Option<Convergence<C>>,
    linear_solver: Box<dyn Solver<LinearisedOp<C>>>,
    problem: Option<Rc<SolverProblem<C>>>,
    max_iter: usize,
    niter: usize,
}

impl <T: Scalar, C: Jacobian<M = DMatrix<T>, V = DVector<T>, T = T>> Default for NewtonNonlinearSolver<C> 
{
    fn default() -> Self {
        let linear_solver = Box::<LU<T>>::default();
        Self {
            problem: None,
            convergence: None,
            linear_solver,
            max_iter: 100,
            niter: 0,
        }
    }
}


impl <C: NonLinearOp> NewtonNonlinearSolver<C> 
{
    pub fn new<S: Solver<LinearisedOp<C>> + 'static>(linear_solver: S) -> Self {
        let linear_solver = Box::new(linear_solver);
        Self {
            problem: None,
            convergence: None,
            linear_solver,
            max_iter: 100,
            niter: 0,
        }
    }
}

impl<C: NonLinearOp> IterativeSolver<C> for NewtonNonlinearSolver<C> 
{
    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }
    fn max_iter(&self) -> usize {
        self.max_iter
    }
    fn niter(&self) -> usize {
        self.niter 
    }
}

impl<C: NonLinearOp> NonLinearSolver<C> for NewtonNonlinearSolver<C> 
{
    fn update_problem(&mut self, problem: Rc<SolverProblem<C>>) {
        self.problem = Some(problem);
        let problem = self.problem.as_ref().unwrap();
        if self.convergence.is_none() {
            self.convergence = Some(Convergence::new(
                problem.clone(), self.max_iter
            ));
        } else {
            self.convergence.as_mut().unwrap().problem = problem.clone();
        }
    }
}

impl<C: NonLinearOp> Solver<C> for NewtonNonlinearSolver<C> {
    fn set_problem(&mut self, problem: Rc<SolverProblem<C>>) {
        self.clear_problem();
        self.update_problem(problem);
    }
    fn is_problem_set(&self) -> bool {
        self.problem.is_some()
    }
    fn clear_problem(&mut self) {
        self.problem = None;
        self.linear_solver.clear_problem();
    }
    fn solve_in_place(&mut self, xn: & mut C::V) -> Result<()> {
        if self.convergence.is_none() || self.problem.is_none() {
            return Err(anyhow!("NewtonNonlinearSolver::solve() called before set_problem"));
        }
        if xn.len() != self.problem.as_ref().unwrap().f.nstates() {
            return Err(anyhow!("NewtonNonlinearSolver::solve() called with state of wrong size, expected {}, got {}", self.problem.as_ref().unwrap().f.nstates(), xn.len()));
        }
        let convergence = self.convergence.as_mut().unwrap();
        let problem = self.problem.as_ref().unwrap();
        let x0 = xn.clone();
        convergence.reset(&x0);
        let mut tmp = x0.clone();
        if !self.linear_solver.is_problem_set() {
            self.linear_solver.set_problem(Rc::new(problem.linearise(&x0)));
        };
        self.niter = 0;
        loop {
            self.niter += 1;
            problem.f.call_inplace(xn, &problem.p, problem.t, &mut tmp);
            //tmp = f_at_n

            self.linear_solver.solve_in_place(&mut tmp)?;
            //tmp = -delta_n

            xn.sub_assign(&tmp);
            // xn = xn + delta_n

            let res = convergence.check_new_iteration(&mut tmp);
            match res  {
                ConvergenceStatus::Continue => continue,
                ConvergenceStatus::Converged => return Ok(()),
                ConvergenceStatus::Diverged => break,
                ConvergenceStatus::MaximumIterations => break,
            }
        }
        Err(anyhow!("Newton iteration did not converge"))
    }

    
}

// tests
#[cfg(test)]
mod tests {

    use crate::LU;
    use crate::callable::closure::Closure;

    use super::*;
    use super::super::tests::test_nonlinear_solver;

    #[test]
    fn test_newton_nalgebra() {
        type T = f64;
        type M = nalgebra::DMatrix<T>;
        type C = Closure<M, M>;
        type S = NewtonNonlinearSolver<C>;
        let lu = LU::<T>::default();
        let s = S::new(lu);
        test_nonlinear_solver(s);
    }
}
