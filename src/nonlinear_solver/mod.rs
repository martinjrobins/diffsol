use core::panic;
use std::rc::Rc;
use num_traits::{Pow, One};

use crate::{callable::Op, solver::SolverProblem, IndexType, Scalar, Vector};

struct Convergence<C: Op> {
    problem: Rc<SolverProblem<C>>,
    tol: C::T,
    max_iter: IndexType,
    iter: IndexType,
    scale: Option<C::V>,
    old_norm: Option<C::T>,
}

enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
    MaximumIterations
}

impl <C: Op> Convergence<C> {
    fn new(problem: Rc<SolverProblem<C>>, max_iter: IndexType) -> Self {
        let rtol = problem.rtol;
        let minimum_tol = C::T::from(10.0) * C::T::EPSILON / rtol;
        let maximum_tol = C::T::from(0.03);
        let mut tol = rtol.pow(C::T::from(0.5));
        if tol > maximum_tol {
            tol = maximum_tol;
        }
        if tol < minimum_tol {
            tol = minimum_tol;
        }
        Self {
            problem,
            tol,
            max_iter,
            scale: None,
            old_norm: None,
            iter: 0,
        }
    }
    fn reset(&mut self, y: &C::V) {
        let mut scale = y.abs() * self.problem.rtol;
        scale += &self.problem.as_ref().atol;
        self.scale = Some(scale);
        self.iter = 0;
        self.old_norm = None;
    }
    fn check_new_iteration(&mut self, dy: &mut C::V) -> ConvergenceStatus {
        if self.scale.is_none() {
            panic!("Convergence::check_new_iteration() called before Convergence::reset()");
        }
        dy.component_div_assign(self.scale.as_ref().unwrap());
        let norm = dy.norm();
        // if norm is zero then we are done
        if norm <= C::T::EPSILON {
            return ConvergenceStatus::Converged;
        }
        if let Some(old_norm) = self.old_norm {
            let rate = norm / old_norm;
            
            // if converged then break out of iteration successfully
            if rate / (C::T::one() - rate) * norm < self.tol {
                return ConvergenceStatus::Converged;
            }
            
            // if iteration is not going to converge in NEWTON_MAXITER
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.iter).unwrap()) / (C::T::from(1.0) - rate) * norm > self.tol {
                return ConvergenceStatus::Diverged;
            }
        }
        self.iter += 1;
        self.old_norm = Some(norm);
        if self.iter >= self.max_iter {
            ConvergenceStatus::MaximumIterations
        } else {
            ConvergenceStatus::Continue
        }
    }
}


pub mod newton;

//tests
#[cfg(test)]
pub mod tests {
    use crate::{callable::closure::Closure, Matrix, Solver, SolverProblem};
    use super::*;
    use num_traits::{One, Zero};
    
    // 0 = J * x * x - 8
    fn square<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
        y.component_mul_assign(x);
        y.add_scalar_mut(M::T::from(-8.0));
    }

    // J = 2 * J * x * dx
    fn square_jacobian<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::from(2.0), x, M::T::zero(), y); // y = 2 * J * x
        y.component_mul_assign(v);
    }
    
    
    pub fn get_square_problem<M: Matrix + 'static>() -> Closure<M, M>{
        let jac = Matrix::from_diagonal(&M::V::from_vec(vec![2.0.into(), 2.0.into()]));
        Closure::new(
            square,
            square_jacobian,
            jac, 
            2,
        )
    }
    
    pub fn test_nonlinear_solver<M: Matrix + 'static, S: Solver<Closure<M, M>>> (mut solver: S) 
    {
        let op = Rc::new(get_square_problem::<M>());
        let problem = Rc::new(SolverProblem::new(op, <M::V as Vector>::zeros(0), M::T::zero()));
        let x0 = M::V::from_vec(vec![2.1.into(), 2.1.into()]);
        solver.set_problem(problem);
        let x = solver.solve(&x0).unwrap();
        let expect = M::V::from_vec(vec![2.0.into(), 2.0.into()]);
        x.assert_eq(&expect, 1e-6.into());
    }
    
    
    
}