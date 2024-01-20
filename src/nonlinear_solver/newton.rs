use crate::{Scalar, Vector, Callable, IndexType, Solver, SolverStatistics, Jacobian, Matrix, SolverOptions, LU, SolverProblem};
use anyhow::{anyhow, Result};

use super::{Convergence, ConvergenceStatus};

pub struct NewtonNonlinearSolver<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> {
    convergence: Option<Convergence<'a, T, V>>,
    linear_solver: Box<dyn Solver<'a, T, V, C>>,
    atol: Option<V>,
    statistics: SolverStatistics,
    options: SolverOptions<T>,
    problem: Option<SolverProblem<'a, T, V, C>>,
}

impl <'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> NewtonNonlinearSolver<'a, T, V, C> {
    pub fn new(linear_solver: impl Solver<'a, T, V, C>) -> Self {
        let options = SolverOptions::<T, V>::default();
        let statistics = SolverStatistics {
            niter: 0,
            nmaxiter: 0,
        };
        let linear_solver = Box::new(linear_solver);
        Self {
            problem: None,
            convergence: None,
            atol: None,
            linear_solver,
            statistics,
            options,
        }
    }
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> Solver<'a, T, V, C> for NewtonNonlinearSolver<'a, T, V, C> {

    fn options(&self) -> Option<&SolverOptions<T>> {
        Some(&self.options)
    }

    fn set_options(&mut self, options: SolverOptions<T>) {
        self.options = options;
    }

    fn problem(&self) -> Option<&SolverProblem<'a, T, V, C>> {
        self.problem.as_ref()
    }

    fn set_problem(&mut self, problem: SolverProblem<'a, T, V, C>) {
        let nstates = problem.f.nstates();
        let atol = if problem.atol.is_some() {
            problem.atol.unwrap()
        } else {
            if self.atol.is_none() {
                self.atol = Some(V::from_element(nstates, self.options.atol));
            } else if self.atol.unwrap().len() != nstates {
                self.atol = Some(V::from_element(nstates, self.options.atol));
            }
            &self.atol.unwrap()
        };
        self.problem = Some(problem);
        if self.convergence.is_none() {
            self.convergence = Some(Convergence::new(
                self.options.rtol,
                atol,
                self.options.max_iter,
            ));
        } else {
            self.convergence.unwrap().rtol = self.options.rtol;
            self.convergence.unwrap().atol = atol;
            self.convergence.unwrap().max_iter = self.options.max_iter;
        }
    }
    
    
    fn get_statistics(&self) -> &SolverStatistics {
        &self.statistics
    }

    fn solve(&mut self, x0: &V) -> Result<V> {
        if self.convergence.is_none() || self.problem.is_none() {
            return Err(anyhow!("NewtonNonlinearSolver::solve() called before set_problem"));
        }
        let convergence = self.convergence.as_mut().unwrap();
        let problem = self.problem.as_ref().unwrap();
        let mut xn = x0.clone();
        self.convergence.reset(&xn);
        let mut f_at_n = xn.clone();
        let mut updated_jacobian = false;
        self.statistics.niter = 0;
        loop {
            if self.linear_solver.problem().is_none() {
                self.linear_solver.set_problem(problem.clone());
                updated_jacobian = true;
            }
            loop {
                self.statistics.niter += 1;
                problem.f.call(&xn, problem.p, &mut f_at_n);
                
                let mut delta_xn = self.linear_solver.solve(&f_at_n)?;

                xn -= delta_xn;

                let res = self.convergence.check_new_iteration(delta_xn);
                match res  {
                    ConvergenceStatus::Continue => continue,
                    ConvergenceStatus::Converged => return Ok(xn),
                    ConvergenceStatus::Diverged => break,
                    ConvergenceStatus::MaximumIterations => break,
                }
            }
            // only get here if we've diverged or hit max iterations
            // if we havn't updated the jacobian, we can update it and try again
            if !updated_jacobian {
                self.linear_solver.clear_callable();
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
    use crate::nonlinear_solver::tests::get_square_problem;

    use super::*;
    use super::super::tests::test_nonlinear_solver;

    #[test]
    fn test_newton_nalgebra() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        type M = nalgebra::DMatrix<T>;
        type C = Closure<fn(&V, &V, &mut V, &M), fn(&V, &V, &V, &mut V, &M), M>;
        type S = NewtonNonlinearSolver<'static, T, V, C>;
        let lu = LU::<T>::default();
        let op = get_square_problem::<T, V, M>();
        let s = S::new(lu);
        test_nonlinear_solver::<T, V, M, C, S>(s, op);
    }
}
