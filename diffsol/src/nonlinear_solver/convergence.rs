use log::trace;
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
        self.eta = V::T::from_f64(20.0.pow(1.25)).unwrap();
    }

    pub fn reset_eta_timestep_change(&mut self) {
        self.eta = V::T::from_f64(100.0.pow(1.25)).unwrap();
    }

    pub fn new(rtol: V::T, atol: &'a V) -> Self {
        let tol = V::T::from_f64(0.2).unwrap();
        Self {
            rtol,
            atol,
            tol,
            max_iter: 10,
            old_norm: None,
            eta: V::T::from_f64(20.0.pow(1.25)).unwrap(),
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
        trace!(
            "  Iteration {}, check non-linear solver norm = {:.3e}",
            self.niter + 1,
            norm.to_f64().unwrap()
        );
        self.niter += 1;
        if let Some(old_norm) = self.old_norm {
            let rate =
                (norm / old_norm).pow(V::T::one() / (V::T::from_usize(self.niter - 1).unwrap()));

            // check if iteration is diverging
            if rate > V::T::from_f64(0.9).unwrap() {
                trace!("  Diverged with rate {}", rate);
                return ConvergenceStatus::Diverged;
            }

            // if iteration is not going to converge in max_iter
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.niter).unwrap()) / (V::T::one() - rate)
                * norm
                > self.tol
            {
                trace!(
                    "  Diverged as will not converge in max iterations with rate {}",
                    rate
                );
                return ConvergenceStatus::Diverged;
            }

            let eta = rate / (V::T::one() - rate);
            trace!(
                "  Updated mean convergence rate = {:.3e}, eta = {:.3e}",
                rate.to_f64().unwrap(),
                eta.to_f64().unwrap()
            );
            self.eta = eta;
        } else {
            let min_eta = V::T::from_f64(1e4).unwrap() * V::T::EPSILON;
            if self.eta < min_eta {
                self.eta = min_eta;
            }
            self.eta = self.eta.pow(V::T::from_f64(0.8).unwrap());
            trace!(
                "  First iteration, set eta = {:.3e}",
                self.eta.to_f64().unwrap()
            );
        };
        // check if iteration is converged
        if self.eta * norm < self.tol {
            trace!(
                "  Converged with eta * norm = {:.3e} < tol = {:.3e}",
                (self.eta * norm).to_f64().unwrap(),
                self.tol.to_f64().unwrap()
            );
            return ConvergenceStatus::Converged;
        }
        trace!(
            "  Not yet converged: eta * norm = {:.3e} >= tol = {:.3e}",
            (self.eta * norm).to_f64().unwrap(),
            self.tol.to_f64().unwrap()
        );
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
