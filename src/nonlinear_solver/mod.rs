use core::panic;

use crate::{Scalar, Vector, IndexType};

struct Convergence<'a, T: Scalar, V: Vector<T>> {
    rtol: T,
    atol: &'a V,
    tol: T,
    max_iter: IndexType,
    iter: IndexType,
    scale: Option<V>,
    old_norm: Option<T>,
}

enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
    MaximumIterations
}

impl <'a, T: Scalar, V: Vector<T>> Convergence<'a, T, V> {
    fn new(rtol: T, atol: &'a V, max_iter: IndexType) -> Self {
        let minimum_tol = T::from(10.0) * T::EPSILON / rtol;
        let maximum_tol = T::from(0.03);
        let mut tol = rtol.pow(T::from(0.5));
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
        scale += &self.atol;
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
            if rate / (T::from(1.0) - rate) * norm < self.tol {
                return ConvergenceStatus::Converged;
            }
            
            // if iteration is not going to converge in NEWTON_MAXITER
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.iter).unwrap()) / (T::from(1.0) - rate) * norm > self.tol {
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
    use crate::{callable::closure::Closure, Matrix, Solver};
    use super::*;
    
    // 0 = J * x * x - 8
    fn square<T: Scalar, V: Vector<T>, M: Matrix<T, V>>(x: &V, y: &mut V, jac: &M) {
        jac.mul_to(x, y);
        y.component_mul_assign(x);
        y.add_scalar_mut(T::from(-8.0));
    }

    // J = 2 * J * x * dx
    fn square_jacobian<T: Scalar, V: Vector<T>, M: Matrix<T, V>>(x: &V, v: &V, y: &mut V, jac: &M) {
        jac.mul_to(x, y);
        y.component_mul_assign(v);
        *y *= T::from(2.0);
    }
    
    pub type SquareClosure<V, M> = Closure<fn(&V, &mut V, &M), fn(&V, &V, &mut V, &M), M>;
    
    pub fn get_square_problem<T: Scalar, V: Vector<T>, M: Matrix<T, V>>() -> SquareClosure<V, M> {
        let jac = Matrix::from_diagonal(&V::from_vec(vec![2.0.into(), 2.0.into()]));
        Closure::<fn(&V, &mut V, &M), fn(&V, &V, &mut V, &M), M>::new(
            square,
            square_jacobian,
            jac, 
            2,
        )
    }
    
    pub fn test_nonlinear_solver<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, S: Solver<'a, T, V>> (solver: S) {
        solver.set_callable(&get_square_problem::<T, V, M>(), &V::zeros(0));
        let x0 = V::from_vec(vec![2.1.into(), 2.1.into()]);
        let x = solver.solve(&x0).unwrap();
        let expect = V::from_vec(vec![2.0.into(), 2.0.into()]);
        x.assert_eq(&expect, 1e-6.into());
        
        solver.set_callable(&get_square_problem::<T, V, M>(), vec![0, 1]);
        let x0 = V::from_vec(vec![2.1.into(), 2.1.into()]);
        let x = solver.solve(&x0).unwrap();
        let expect = V::from_vec(vec![2.1.into(), 2.0.into()]);
        x.assert_eq(&expect, 1e-6.into());

    }
    
    
    
}