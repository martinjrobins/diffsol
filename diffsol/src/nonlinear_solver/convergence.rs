use nalgebra::ComplexField;
use num_traits::{FromPrimitive, One, Pow};

use crate::{scalar::IndexType, Scalar, Vector};

#[derive(Clone)]
pub struct Convergence<'a, V: Vector> {
    pub rtol: V::T,
    pub atol: &'a V,
    tol: V::T,
    max_iter: IndexType,
    niter: IndexType,
    old_norm: Option<V::T>,
}

pub enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
    MaximumIterations,
}

impl<'a, V: Vector> Convergence<'a, V> {
    pub fn max_iter(&self) -> IndexType {
        self.max_iter
    }
    pub fn set_max_iter(&mut self, value: IndexType) {
        self.max_iter = value;
    }
    pub fn niter(&self) -> IndexType {
        self.niter
    }
    pub fn new(rtol: V::T, atol: &'a V) -> Self {
        let minimum_tol = V::T::from_f64(10.0).unwrap() * V::T::EPSILON / rtol;
        let maximum_tol = V::T::from_f64(0.03).unwrap();
        let mut tol = V::T::from_f64(0.33).unwrap();
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
            max_iter: 10,
            old_norm: None,
            niter: 0,
        }
    }
    pub fn reset(&mut self) {
        self.niter = 0;
        self.old_norm = None;
    }

    pub fn check_new_iteration(&mut self, dy: &mut V, y: &V) -> ConvergenceStatus {
        self.niter += 1;
        let norm = dy.squared_norm(y, self.atol, self.rtol).sqrt();
        // if norm is zero then we are done
        if norm <= V::T::EPSILON {
            return ConvergenceStatus::Converged;
        }
        if let Some(old_norm) = self.old_norm {
            let rate = norm / old_norm;

            // check if iteration is diverging
            if rate > V::T::from_f64(0.9).unwrap() {
                return ConvergenceStatus::Diverged;
            }

            let eta = rate / (V::T::one() - rate);

            // check if iteration is converging
            if eta * norm < self.tol {
                return ConvergenceStatus::Converged;
            }

            // if iteration is not going to converge in max_iter
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.niter).unwrap()) / (V::T::one() - rate)
                * norm
                > self.tol
            {
                return ConvergenceStatus::Diverged;
            }
        } else {
            // no rate, just test with a large eta
            if V::T::from_f64(1000.0).unwrap() * norm < self.tol {
                return ConvergenceStatus::Converged;
            }
        };

        // we havn't converged, so store norm for next iteration
        self.old_norm = Some(norm);

        // check if we have reached the maximum
        if self.niter >= self.max_iter {
            ConvergenceStatus::MaximumIterations
        } else {
            ConvergenceStatus::Continue
        }
    }
}
