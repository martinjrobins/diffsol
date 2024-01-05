use std::marker::PhantomData;

use crate::{Scalar, Vector, Matrix, Callable, LinearSolver, IndexType};
use anyhow::{anyhow, Result};
use nalgebra_sparse::ops::Op;

use super::{NonLinearSolver, Convergence, ConvergenceStatus};

pub struct NewtonNonlinearSolver<T: Scalar, V: Vector<T>, M: Matrix<T, V>, LS: LinearSolver<T, V, M>, C: Callable<T, V>> {
    callable: C,
    jacobian_linear_solver: Option<LS>,
    convergence: Convergence<T, V>,
    zero_by_mask: Option<M>,
    _phantom: PhantomData<(T, M)>
}

impl <T: Scalar, V: Vector<T>, M: Matrix<T, V>, LS: LinearSolver<T, V, M>, C: Callable<T, V>> NewtonNonlinearSolver<T, V, M, LS, C> {
    fn new(callable: C, mask: Option<Vec<IndexType>>) -> Self {
        let nstates = callable.nstates();
        let convergence = Convergence::new(T::from(1e-6), V::from_element(nstates, 1e-6.into()), 10);
        let zero_by_mask= match mask {
            Some(mask) => {
                Some(M::from_diagonal(mask))
            },
            None => None,
        };
        Self {
            callable,
            jacobian_linear_solver: None,
            convergence,
            zero_by_mask,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, V: Vector<T>, M: Matrix<T, V>, LS: LinearSolver<T, V, M>, C: Callable<T, V>> NonLinearSolver<T, V, C> for NewtonNonlinearSolver<T, V, M, LS, C> {
    fn solve(&mut self, x0: &V, p: &V) -> Result<V> {
        let mut xn = x0.clone();
        self.convergence.reset(&xn);
        let mut f_at_n = xn.clone();
        let mut updated_jacobian = false;
        loop {
            if self.jacobian_linear_solver.is_none() {
                let jac = self.callable.jacobian::<M>(&xn, p);
                self.jacobian_linear_solver = Some(LS::new(jac));
                updated_jacobian = true;
            }
            let jac = self.jacobian_linear_solver.as_ref().unwrap();
            loop {
                self.callable.call(&xn, p, &mut f_at_n);
                
                let delta_xn = jac.solve(&f_at_n)?;
                if let Some(zero_indices) = &self.zero_by_mask {
                    delta_xn *= self.zero_by_mask;
                }

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
                self.jacobian_linear_solver = None;
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

    use crate::nonlinear_solver::tests::SquareClosure;
    use super::*;
    use super::super::tests::test_nonlinear_solver;

    #[test]
    fn test_newton_nalgebra() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        type M = nalgebra::DMatrix<T>;
        type LS = nalgebra::LU<T, Dyn, Dyn>;
        type S = NewtonNonlinearSolver<T, V, M, LS, SquareClosure<V, M>>;
        test_nonlinear_solver::<T, V, M, S>();
    }
}
