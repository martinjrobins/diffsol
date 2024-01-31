use crate::{solver::atol::Atol, vector::VectorRef, Callable, Solver, SolverOptions, SolverProblem, SolverStatistics, Vector};
use anyhow::{anyhow, Result};

use super::{Convergence, ConvergenceStatus};

pub struct NewtonNonlinearSolver<'a, V: Vector, C: Callable<V>> 
where
    for <'b> &'b V: VectorRef<V>,
{
    convergence: Option<Convergence<'a, V>>,
    linear_solver: Box<dyn Solver<'a, V, C>>,
    atol: Atol<V>,
    statistics: SolverStatistics,
    options: SolverOptions<V::T>,
    problem: Option<SolverProblem<'a, V, C>>,
}

impl <'a, V: Vector, C: Callable<V>> NewtonNonlinearSolver<'a, V, C> 
where
    for <'b> &'b V: VectorRef<V>,
{
    pub fn new(linear_solver: impl Solver<'a, V, C>) -> Self {
        let options = SolverOptions::<V::T>::default();
        let statistics = SolverStatistics {
            niter: 0,
            nmaxiter: 0,
        };
        let linear_solver = Box::new(linear_solver);
        Self {
            problem: None,
            convergence: None,
            atol: Atol::default(),
            linear_solver,
            statistics,
            options,
        }
    }
}

impl<'a, V: Vector, C: Callable<V>> Solver<'a, V, C> for NewtonNonlinearSolver<'a, V, C> 
where
    for <'b> &'b V: VectorRef<V>,
{

    fn options(&self) -> Option<&SolverOptions<V::T>> {
        Some(&self.options)
    }

    fn set_options(&mut self, options: SolverOptions<V::T>) {
        self.options = options;
    }

    fn problem(&self) -> Option<&SolverProblem<'a, V, C>> {
        self.problem.as_ref()
    }

    fn set_problem(&mut self, state: &V, problem: SolverProblem<'a, V, C>) {
        let nstates = problem.f.nstates();
        let atol = self.atol.value(&problem, &self.options);
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
        self.linear_solver.set_problem(state, problem.clone());
    }
    
    
    fn get_statistics(&self) -> &SolverStatistics {
        &self.statistics
    }

    fn solve(&mut self, x0: V) -> Result<V> {
        if self.convergence.is_none() || self.problem.is_none() {
            return Err(anyhow!("NewtonNonlinearSolver::solve() called before set_problem"));
        }
        let convergence = self.convergence.as_mut().unwrap();
        let problem = self.problem.as_ref().unwrap();
        let mut xn = x0.clone();
        convergence.reset(&xn);
        let mut f_at_n = xn.clone();
        let mut updated_jacobian = false;
        self.statistics.niter = 0;
        loop {
            loop {
                self.statistics.niter += 1;
                problem.f.call(&xn, problem.p, &mut f_at_n);
                
                let mut delta_xn = self.linear_solver.solve(f_at_n)?;

                xn -= delta_xn;

                let res = convergence.check_new_iteration(delta_xn);
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
                self.linear_solver.set_problem(&x0, problem.clone());
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
        type S = NewtonNonlinearSolver<'static, V, C>;
        let lu = LU::<T>::default();
        let op = get_square_problem::<M>();
        let s = S::new(lu);
        test_nonlinear_solver::<M, C, S>(s, op);
    }
}
