use core::panic;
use std::rc::Rc;
use num_traits::{Pow, One};
use anyhow::{Result, anyhow};

use crate::{op::Op, solver::SolverProblem, IndexType, Scalar, Vector};


pub struct NonLinearSolveSolution<V> {
    pub x0: V,
    pub x: V,
}

impl <V> NonLinearSolveSolution<V> {
    pub fn new(x0: V, x: V) -> Self {
        Self { x0, x }
    }
}

pub trait NonLinearSolver<C: Op> {
    fn set_problem(&mut self, problem: SolverProblem<C>);
    fn problem(&self) -> Option<&SolverProblem<C>>;
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>>;
    fn take_problem(&mut self) -> Option<SolverProblem<C>>;
    fn reset(&mut self) {
        if let Some(problem) = self.take_problem() {
            self.set_problem(problem);
        }
    }
    fn set_time(&mut self, t: C::T) -> Result<()> {
        self.problem_mut().ok_or_else(|| anyhow!("No problem set"))?.t = t;
        Ok(())
    }
    fn solve(&mut self, state: &C::V) -> Result<C::V> {
        let mut state = state.clone();
        self.solve_in_place(&mut state)?;
        Ok(state)
    }
    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()>;
    fn set_max_iter(&mut self, max_iter: usize);
    fn max_iter(&self) -> usize;
    fn niter(&self) -> usize;
}

struct Convergence<C: Op> {
    rtol: C::T,
    atol: Rc<C::V>,
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
    fn new(problem: &SolverProblem<C>, max_iter: IndexType) -> Self {
        let rtol = problem.rtol;
        let atol = problem.atol.clone();
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
            rtol,
            atol,
            tol,
            max_iter,
            scale: None,
            old_norm: None,
            iter: 0,
        }
    }
    fn reset(&mut self, y: &C::V) {
        let mut scale = y.abs() * self.rtol;
        scale += self.atol.as_ref();
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
    use crate::{op::{closure::Closure, NonLinearOp}, linear_solver::lu::LU, matrix::MatrixCommon, DenseMatrix};
    use self::newton::NewtonNonlinearSolver;

    use super::*;
    use num_traits::{One, Zero};
    
    
    pub fn get_square_problem<M>() -> (SolverProblem<impl NonLinearOp<M = M, V = M::V, T = M::T>>, Vec<NonLinearSolveSolution<M::V>>)
    where
        M: DenseMatrix + 'static,
    {
        let jac1 = M::from_diagonal(&M::V::from_vec(vec![2.0.into(), 2.0.into()]));
        let jac2 = jac1.clone();
        let p = Rc::new(M::V::zeros(0));
        let op = Closure::new(
            // 0 = J * x * x - 8
            move |x: &<M as MatrixCommon>::V, _p: &<M as MatrixCommon>::V, _t, y| {
                jac1.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
                y.component_mul_assign(x);
                y.add_scalar_mut(M::T::from(-8.0));
            },
            // J = 2 * J * x * dx
            move |x: &<M as MatrixCommon>::V, _p: &<M as MatrixCommon>::V, _t, v, y | {
                jac2.gemv(M::T::from(2.0), x, M::T::zero(), y); // y = 2 * J * x
                y.component_mul_assign(v);
            },
            2, 2, p,
        );
        let rtol = M::T::from(1e-6);
        let atol = M::V::from_vec(vec![1e-6.into(), 1e-6.into()]);
        let t = M::T::zero();
        let problem = SolverProblem::new(Rc::new(op), t, Rc::new(atol), rtol);
        let solns = vec![
            NonLinearSolveSolution::new(M::V::from_vec(vec![2.1.into(), 2.1.into()]), M::V::from_vec(vec![2.0.into(), 2.0.into()]))
        ];
        (problem, solns)
    }
    
    pub fn test_nonlinear_solver<C>(mut solver: impl NonLinearSolver<C>, problem: SolverProblem<C>, solns: Vec<NonLinearSolveSolution<C::V>>) 
    where
        C: NonLinearOp,
    {
        solver.set_problem(problem);
        for soln in solns {
            let x = solver.solve(&soln.x0).unwrap();
            let tol = {
                let problem = solver.problem().unwrap();
                soln.x.abs() * problem.rtol + problem.atol.as_ref()
            };
            x.assert_eq(&soln.x, tol[0]);
        }
    }


    type MCpu = nalgebra::DMatrix<f64>;

    #[test]
    fn test_newton_cpu_square() {
        let lu = LU::default();
        let (prob, soln) = get_square_problem::<MCpu>();
        let s = NewtonNonlinearSolver::new(lu);
        test_nonlinear_solver(s, prob, soln);
    }
    
    
    
}