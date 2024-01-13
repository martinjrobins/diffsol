use crate::{Scalar, Vector, Callable, IndexType, Solver, SolverStatistics};
use anyhow::{anyhow, Result};

use super::{Convergence, ConvergenceStatus};

pub struct NewtonNonlinearSolver<'a, T: Scalar, V: Vector<T>> {
    callable: Option<&'a dyn Callable<T, V>>,
    params: Option<&'a V>,
    convergence: Convergence<'a, T, V>,
    zero_by_mask: Option<V>,
    linear_solver: Box<dyn Solver<'a, T, V>>,
    statistics: SolverStatistics,
}

impl <'a, T: Scalar, V: Vector<T>> NewtonNonlinearSolver<'a, T, V> {
    pub fn new(rtol: T, atol: &'a V, max_iter: IndexType, linear_solver: impl Solver<'a, T, V>, mask: Option<Vec<IndexType>>) -> Self {
        let convergence = Convergence::new(rtol, atol, max_iter);
        let zero_by_mask= match mask {
            Some(mask) => {
                let diag = V::from_vec(mask.iter().map(|i| if *i == 0 { T::zero() } else { T::one() }).collect());
                Some(diag)
            },
            None => None,
        };
        let statistics = SolverStatistics {
            niter: 0,
            nmaxiter: max_iter,
        };
        Self {
            callable: None,
            params: None,
            linear_solver: Box::new(linear_solver),
            convergence,
            zero_by_mask,
            statistics,
        }
    }
}


impl<'a, T: Scalar, V: Vector<T>> Solver<'a, T, V> for NewtonNonlinearSolver<'a, T, V> {
    fn set_callable(&mut self, c: &'a impl Callable<T, V>, p: &'a V) {
        if self.convergence.atol.len() != c.nstates() {
            panic!("NewtonNonlinearSolver::set_callable() called with callable with different number of states");
        }
        self.callable = Some(c);
        self.params = Some(p);
        self.linear_solver.clear_callable()
    }
    fn is_callable_set(&self) -> bool {
        self.callable.is_some()
    }

    fn clear_callable(&mut self) {
        self.callable = None;
        self.params = None;
        self.linear_solver.clear_callable()
    }
    
    fn get_statistics(&self) -> &SolverStatistics {
        &self.statistics
    }

    fn solve(&mut self, x0: &V) -> Result<V> {
        if self.callable.is_none() {
            return Err(anyhow!("NewtonNonlinearSolver::solve() called before callable was set"));
        }
        let op = self.callable.unwrap();
        let p = self.params.unwrap();
        let mut xn = x0.clone();
        self.convergence.reset(&xn);
        let mut f_at_n = xn.clone();
        let mut updated_jacobian = false;
        self.statistics.niter = 0;
        loop {
            if !self.linear_solver.is_callable_set() {
                self.linear_solver.set_callable(op, p);
                updated_jacobian = true;
            }
            loop {
                self.statistics.niter += 1;
                op.call(&xn, p, &mut f_at_n);
                
                let delta_xn = {
                    let mut delta_xn = self.linear_solver.solve(&f_at_n)?;
                    if let Some(zero_indices) = &self.zero_by_mask {
                        delta_xn.component_mul_assign(zero_indices);
                    }
                    delta_xn
                };

                xn -= &delta_xn;

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
    use nalgebra::Dyn;

    use crate::LU;

    use super::*;
    use super::super::tests::test_nonlinear_solver;

    #[test]
    fn test_newton_nalgebra() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        type M = nalgebra::DMatrix<T>;
        type LS = nalgebra::LU<T, Dyn, Dyn>;
        let s = NewtonNonlinearSolver::new(
            T::from(1e-6),
            &V::from_vec(vec![T::from(1e-6), T::from(1e-6)]),
            100,
            LU::<T>::default(),
            None,
        );
        test_nonlinear_solver::<T, V, M>(s);
    }
}
