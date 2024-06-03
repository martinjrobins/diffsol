use nalgebra::ComplexField;
use num_traits::{One, Pow};
use std::rc::Rc;

use crate::{scalar::IndexType, solver::SolverProblem, NonLinearOp, Scalar, Vector};

pub struct Convergence<V: Vector> {
    rtol: V::T,
    atol: Rc<V>,
    tol: V::T,
    max_iter: IndexType,
    niter: IndexType,
    old_norm: Option<V::T>,
    old_eta: Option<V::T>,
    save_eta: bool,
}

pub enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
    MaximumIterations,
}

impl<V: Vector> Convergence<V> {
    pub fn max_iter(&self) -> IndexType {
        self.max_iter
    }
    pub fn set_max_iter(&mut self, value: IndexType) {
        self.max_iter = value;
    }
    pub fn niter(&self) -> IndexType {
        self.niter
    }
    pub fn new_from_problem<C: NonLinearOp<V = V, T = V::T>>(
        problem: &SolverProblem<C>,
        max_iter: IndexType,
    ) -> Self {
        let rtol = problem.rtol;
        let atol = problem.atol.clone();
        Self::new(rtol, atol, max_iter)
    }
    pub fn new(rtol: V::T, atol: Rc<V>, max_iter: usize) -> Self {
        let minimum_tol = V::T::from(10.0) * V::T::EPSILON / rtol;
        let maximum_tol = V::T::from(0.03);
        let mut tol = V::T::from(0.5) * rtol.pow(V::T::from(0.5));
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
            old_norm: None,
            old_eta: None,
            niter: 0,
            save_eta: false,
        }
    }
    pub fn reset(&mut self) {
        self.niter = 0;
        self.old_norm = None;
    }

    pub fn reset_saved_eta(&mut self) {
        self.save_eta = true;
        self.old_eta = None;
    }
    pub fn check_new_iteration(&mut self, dy: &mut V, y: &V) -> ConvergenceStatus {
        let norm = dy.squared_norm(y, &self.atol, self.rtol).sqrt();
        // if norm is zero then we are done
        if norm <= V::T::EPSILON {
            return ConvergenceStatus::Converged;
        }
        if let Some(old_norm) = self.old_norm {
            let rate = norm / old_norm;

            // check if iteration is diverging
            if rate > V::T::from(1.0) {
                return ConvergenceStatus::Diverged;
            }

            let eta = rate / (V::T::one() - rate);

            // check if iteration is converging
            if eta * norm < self.tol {
                // store eta for next step is required
                if self.save_eta {
                    self.old_eta = Some(eta);
                }
                return ConvergenceStatus::Converged;
            }

            // if iteration is not going to converge in max_iter
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.niter).unwrap())
                / (V::T::from(1.0) - rate)
                * norm
                > self.tol
            {
                return ConvergenceStatus::Diverged;
            }

        } else if let Some(mut eta) = self.old_eta {
            if eta < V::T::EPSILON {
                eta = V::T::EPSILON;
            }
            let eta = eta.pow(V::T::from(0.8));

            // check if iteration is converging
            if eta * norm < self.tol {
                // store eta for next step is required
                if self.save_eta {
                    self.old_eta = Some(eta);
                }
                return ConvergenceStatus::Converged;
            }
        } else {
            // no rate or eta, just test directly
            if norm <= self.tol {
                return ConvergenceStatus::Converged;
            }
        };

        // we havn't converged, so store norm for next iteration
        self.old_norm = Some(norm);

        // increment iteration counter and check if we have reached the maximum
        self.niter += 1;
        if self.niter >= self.max_iter {
            ConvergenceStatus::MaximumIterations
        } else {
            ConvergenceStatus::Continue
        }
    }
}
