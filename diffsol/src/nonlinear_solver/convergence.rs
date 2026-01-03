use nalgebra::ComplexField;
use num_traits::{FromPrimitive, One, Pow, ToPrimitive};

use crate::{scalar::IndexType, Scalar, Vector};

#[derive(Clone)]
pub struct Convergence<'a, V: Vector> {
    pub rtol: V::T,
    pub atol: &'a V,
    tol: V::T,
    max_iter: IndexType,
    niter: IndexType,
    old_norm: Option<V::T>,
    eta: V::T,
}

pub enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
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
    pub fn eta(&self) -> V::T {
        self.eta
    }
    pub fn reset_eta(&mut self) {
        // timestep change case takes precedence
        if self.eta != V::T::from_f64(100.).unwrap() {
            self.eta = V::T::from_f64(20.).unwrap();
        }
    }

    pub fn reset_eta_timestep_change(&mut self) {
        self.eta = V::T::from_f64(100.).unwrap();
    }
    
    pub fn new(rtol: V::T, atol: &'a V) -> Self {
        let tol = V::T::from_f64(0.33).unwrap();
        Self {
            rtol,
            atol,
            tol,
            max_iter: 10,
            old_norm: None,
            eta: V::T::from_f64(20.0).unwrap(),
            niter: 0,
        }
    }
    pub fn reset(&mut self) {
        self.niter = 0;
        self.old_norm = None;
    }

    pub fn norm(&self, dy: &V, y: &V) -> V::T {
        dy.squared_norm(y, self.atol, self.rtol).sqrt()
    }

    pub fn check_norm(&mut self, norm: V::T) -> ConvergenceStatus {
        println!("  Iteration {}, norm = {:.3e}", self.niter + 1, norm.to_f64().unwrap());
        self.niter += 1;
        // if norm is zero then we are done
        if norm <= V::T::EPSILON {
            return ConvergenceStatus::Converged;
        }
        let eta = if let Some(old_norm) = self.old_norm {
            let rate = (norm / old_norm).pow(V::T::one() / (V::T::from_usize(self.niter - 1).unwrap()));

            // check if iteration is diverging
            if rate > V::T::from_f64(0.9).unwrap() {
                println!("Diverged with rate {}", rate);
                return ConvergenceStatus::Diverged;
            }
            
            // if iteration is not going to converge in max_iter
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.niter).unwrap()) / (V::T::one() - rate)
                * norm
                > self.tol
            {
                return ConvergenceStatus::Diverged;
            }

            let eta = rate / (V::T::one() - rate);
            self.eta = eta;
            eta
        } else {
            // todo: should be able to use previous eta, but this is failing for sdirk
            // self.eta
            V::T::from_f64(20.).unwrap()
        };
        // check if iteration is converged
        println!("  Estimated eta = {:.3e} eta * norm = {:.3e}", eta.to_f64().unwrap(), (eta * norm).to_f64().unwrap());
        if eta * norm < self.tol {
            return ConvergenceStatus::Converged;
        }
        ConvergenceStatus::Continue
    }

    pub fn check_new_iteration(&mut self, norm: V::T) -> ConvergenceStatus {
        let status = self.check_norm(norm);
        if self.niter == 1 {
            self.old_norm = Some(norm);
        }
        status
    }
}
