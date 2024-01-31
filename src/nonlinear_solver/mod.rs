use core::panic;
use num_traits::Pow;

use crate::{Scalar, Vector, IndexType};

struct Convergence<'a, V: Vector> {
    rtol: V::T,
    atol: &'a V,
    tol: V::T,
    max_iter: IndexType,
    iter: IndexType,
    scale: Option<V>,
    old_norm: Option<V::T>,
}

enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
    MaximumIterations
}

impl <'a, V: Vector> Convergence<'a, V> {
    fn new(rtol: V::T, atol: &'a V, max_iter: IndexType) -> Self {
        let minimum_tol = V::T::from(10.0) * V::T::EPSILON / rtol;
        let maximum_tol = V::T::from(0.03);
        let mut tol = rtol.pow(V::T::from(0.5));
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
    fn reset(&mut self, y: &V) {
        let mut scale = y.abs() * self.rtol;
        scale += self.atol;
        self.scale = Some(scale);
        self.iter = 0;
        self.old_norm = None;
    }
    fn check_new_iteration(&mut self, mut dy: V) -> ConvergenceStatus {
        if self.scale.is_none() {
            panic!("Convergence::check_new_iteration() called before Convergence::reset()");
        }
        dy.component_div_assign(self.scale.as_ref().unwrap());
        let norm = dy.norm();
        if let Some(old_norm) = self.old_norm {
            let rate = norm / old_norm;
            
            // if converged then break out of iteration successfully
            if rate / (V::T::from(1.0) - rate) * norm < self.tol {
                return ConvergenceStatus::Converged;
            }
            
            // if iteration is not going to converge in NEWTON_MAXITER
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.iter).unwrap()) / (V::T::from(1.0) - rate) * norm > self.tol {
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
    use crate::{callable::{closure::Closure, Callable}, vector::VectorRef, Matrix, Solver, SolverProblem};
    use super::*;
    use num_traits::{One, Zero};
    
    // 0 = J * x * x - 8
    fn square<M: Matrix>(x: &M::V, p: &M::V, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
        y.component_mul_assign(x);
        y.add_scalar_mut(M::T::from(-8.0));
    }

    // J = 2 * J * x * dx
    fn square_jacobian<M: Matrix>(x: &M::V, p: &M::V, v: &M::V, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::from(2.0), x, M::T::zero(), y); // y = 2 * J * x
        y.component_mul_assign(v);
    }
    
    pub type SquareClosure<V, M> = Closure<fn(&V, &mut V, &M), fn(&V, &V, &mut V, &M), M>;
    
    pub fn get_square_problem<M: Matrix>() -> Closure<fn(&M::V, &M::V, &mut M::V, &M), fn(&M::V, &M::V, &M::V, &mut M::V, &M), M> {
        let jac = Matrix::from_diagonal(&M::V::from_vec(vec![2.0.into(), 2.0.into()]));
        Closure::<fn(&M::V, &M::V, &mut M::V, &M), fn(&M::V, &M::V, &M::V, &mut M::V, &M), M>::new(
            square,
            square_jacobian,
            jac, 
            2,
        )
    }
    
    pub fn test_nonlinear_solver<'a, M: Matrix, C: Callable<M::V>, S: Solver<'a, M::V, C>> (mut solver: S, op: C) 
    where
        for <'b> &'b M::V: VectorRef<M::V>,
    {
        let problem = SolverProblem::new(&op, &<M::V as Vector>::zeros(0));
        let x0 = M::V::from_vec(vec![2.1.into(), 2.1.into()]);
        solver.set_problem(&x0, problem);
        let x = solver.solve(x0).unwrap();
        let expect = M::V::from_vec(vec![2.0.into(), 2.0.into()]);
        x.assert_eq(&expect, 1e-6.into());
    }
    
    
    
}