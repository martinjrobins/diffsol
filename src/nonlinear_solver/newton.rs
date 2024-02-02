use std::rc::Rc;

use crate::{solver::IterativeSolver, vector::Vector, Callable, Solver, SolverProblem};
use anyhow::{anyhow, Result};
use std::ops::SubAssign;

use super::{Convergence, ConvergenceStatus};

pub struct NewtonNonlinearSolver<C: Callable> 
{
    convergence: Option<Convergence<C>>,
    linear_solver: Box<dyn Solver<C>>,
    problem: Option<Rc<SolverProblem<C>>>,
    max_iter: usize,
    niter: usize,
}

impl <C: Callable> NewtonNonlinearSolver<C> 
{
    pub fn new<S: Solver<C> + 'static>(linear_solver: S) -> Self {
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

impl<C: Callable> IterativeSolver<C> for NewtonNonlinearSolver<C> 
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

impl<C: Callable> Solver<C> for NewtonNonlinearSolver<C> 
{

    fn problem(&self) -> Option<&SolverProblem<C>> {
        self.problem.as_deref()
    }

    fn set_problem(&mut self, state: &C::V, problem: Rc<SolverProblem<C>>) {
        self.problem = Some(problem);
        let problem = self.problem.as_ref().unwrap();
        if self.convergence.is_none() {
            self.convergence = Some(Convergence::new(
                problem.clone(), self.max_iter
            ));
        } else {
            self.convergence.as_mut().unwrap().problem = problem.clone();
        }
        self.linear_solver.set_problem(state, problem.clone());
    }
    
    
    fn solve_in_place(&mut self, xn: & mut C::V) -> Result<()> {
        if self.convergence.is_none() || self.problem.is_none() {
            return Err(anyhow!("NewtonNonlinearSolver::solve() called before set_problem"));
        }
        let convergence = self.convergence.as_mut().unwrap();
        let problem = self.problem.as_ref().unwrap();
        let x0 = xn.clone();
        convergence.reset(&x0);
        let mut tmp = x0.clone();
        let mut updated_jacobian = false;
        self.niter = 0;
        loop {
            loop {
                self.niter += 1;
                problem.f.call(&xn, &problem.p, &mut tmp);

                //tmp = f_at_n
                self.linear_solver.solve_in_place(&mut tmp)?;

                //tmp = delta_n
                xn.sub_assign(&tmp);

                let res = convergence.check_new_iteration(&mut tmp);
                match res  {
                    ConvergenceStatus::Continue => continue,
                    ConvergenceStatus::Converged => return Ok(()),
                    ConvergenceStatus::Diverged => break,
                    ConvergenceStatus::MaximumIterations => break,
                }
            }
            // only get here if we've diverged or hit max iterations
            // if we havn't updated the jacobian, we can update it and try again
            if !updated_jacobian {
                self.linear_solver.set_problem(&x0, problem.clone());
                xn.copy_from(&x0);
                updated_jacobian = true;
                continue;
            } else {
                break;
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
        type V = nalgebra::DVector<T>;
        type M = nalgebra::DMatrix<T>;
        type C = Closure<M, fn(&V, &V, &mut V, &M), fn(&V, &V, &V, &mut V, &M), M>;
        type S = NewtonNonlinearSolver<C>;
        let lu = LU::<T>::default();
        let s = S::new(lu);
        test_nonlinear_solver(s);
    }
}
