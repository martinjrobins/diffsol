use log::trace;
use num_traits::{FromPrimitive, One, Pow, ToPrimitive};

use diffsol_la::{IndexType, Scalar, Vector};

#[derive(Clone)]
pub struct Convergence<'a, V: Vector> {
    pub rtol: V::T,
    pub atol: &'a V,
    tol: V::T,
    max_iter: IndexType,
    niter: IndexType,
    old_norm: Option<V::T>,
    eta: V::T,
    /// The number of updates at the start of each iteration left out of its measured rate
    transient_updates: IndexType,
}

/// Computes `x.pow(1 / n)`, taking a cheap exact shortcut via sqrt/cbrt chains
/// for the small integer `n` values that dominate in practice (Newton usually
/// converges within a handful of iterations).
fn root_fast<T: Scalar>(x: T, n: IndexType) -> T {
    match n {
        1 => x,
        2 => x.sqrt(),
        3 => x.cbrt(),
        4 => x.sqrt().sqrt(),
        _ => x.pow(T::one() / T::from_usize(n).unwrap()),
    }
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
    pub fn tolerance(&self) -> V::T {
        self.tol
    }
    pub fn set_tolerance(&mut self, value: V::T) {
        self.tol = value;
    }
    pub fn set_transient_updates(&mut self, value: IndexType) {
        self.transient_updates = value;
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
        Self::with_tolerance(rtol, atol, V::T::from_f64(0.2).unwrap())
    }

    pub fn with_tolerance(rtol: V::T, atol: &'a V, tol: V::T) -> Self {
        Self {
            rtol,
            atol,
            tol,
            max_iter: 10,
            old_norm: None,
            eta: V::T::from_f64(20.0.pow(1.25)).unwrap(),
            niter: 0,
            transient_updates: 0,
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
        let one = V::T::one();
        let rate = self
            .old_norm
            .map(|old_norm| root_fast(norm / old_norm, self.niter - 1 - self.transient_updates));

        // update the estimate of the error to update ratio, eta, unless the iteration is
        // not contracting, when it has none
        match rate {
            Some(rate) if rate < one => {
                self.eta = rate / (one - rate);
                trace!(
                    "  Updated mean convergence rate = {:.3e}, eta = {:.3e}",
                    rate.to_f64().unwrap(),
                    self.eta.to_f64().unwrap()
                );
            }
            Some(_) => {}
            None => {
                let min_eta = V::T::from_f64(1e4).unwrap() * V::T::EPSILON;
                if self.eta < min_eta {
                    self.eta = min_eta;
                }
                self.eta = self.eta.pow(V::T::from_f64(0.8).unwrap());
                trace!(
                    "  No rate measured yet, set eta = {:.3e}",
                    self.eta.to_f64().unwrap()
                );
            }
        }

        // check if iteration is converged, on its estimated error, before checking a rate that
        // has not converged it for divergence
        if rate.is_none_or(|rate| rate < one) && self.eta * norm < self.tol {
            trace!(
                "  Converged with norm = {:.3e}, eta = {:.3e}, tol = {:.3e}",
                norm.to_f64().unwrap(),
                self.eta.to_f64().unwrap(),
                self.tol.to_f64().unwrap()
            );
            return ConvergenceStatus::Converged;
        }

        if let Some(rate) = rate {
            // check if iteration is diverging
            if rate > V::T::from_f64(0.9).unwrap() {
                trace!("  Diverged with rate {}", rate);
                return ConvergenceStatus::Diverged;
            }

            // if iteration is not going to converge in max_iter
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.niter).unwrap()) / (one - rate) * norm
                > self.tol
            {
                trace!(
                    "  Diverged as will not converge in max iterations with rate {}",
                    rate
                );
                return ConvergenceStatus::Diverged;
            }
        }
        trace!(
            "  Not yet converged: norm = {:.3e}, eta = {:.3e}, tol = {:.3e}",
            norm.to_f64().unwrap(),
            self.eta.to_f64().unwrap(),
            self.tol.to_f64().unwrap()
        );
        ConvergenceStatus::Continue
    }

    pub fn check_new_iteration(&mut self, norm: V::T) -> ConvergenceStatus {
        let status = self.check_norm(norm);
        if self.niter == 1 + self.transient_updates {
            self.old_norm = Some(norm);
        }
        status
    }
}

#[cfg(test)]
mod tests {
    use super::{Convergence, ConvergenceStatus};
    use diffsol_la::{NalgebraVec, Vector};

    type V = NalgebraVec<f64>;

    fn status(c: &mut Convergence<V>, norm: f64) -> &'static str {
        match c.check_new_iteration(norm) {
            ConvergenceStatus::Converged => "converged",
            ConvergenceStatus::Diverged => "diverged",
            ConvergenceStatus::Continue => "continue",
        }
    }

    /// With the first update left out of the rate, the rate is measured from the second, so a
    /// first update far larger than the rest does not make the rate look fast.
    #[test]
    fn transient_update_left_out_of_rate() {
        let atol = V::from_element(1, 1e-6, Default::default());
        let mut c = Convergence::new(1e-6, &atol);
        c.set_transient_updates(1);
        c.reset_eta();
        c.reset();
        assert_eq!(status(&mut c, 1e4), "continue");
        // no rate yet: eta, relaxed from its reset value, is far too large to accept
        assert_eq!(status(&mut c, 2.0), "continue");
        // the rate 0.5 from the second update gives eta = 1, so 0.5 is not converged
        assert_eq!(status(&mut c, 1.0), "continue");
        assert_eq!(c.eta(), 1.0);
        assert_eq!(status(&mut c, 0.1), "converged");

        // measured from the first update, the same updates converge after the second
        let mut c = Convergence::new(1e-6, &atol);
        c.reset_eta();
        c.reset();
        assert_eq!(status(&mut c, 1e4), "continue");
        assert_eq!(status(&mut c, 2.0), "converged");
    }

    /// The rate detects divergence, and a first update whose error estimated from the carried
    /// `eta` is below the tolerance converges at once.
    #[test]
    fn detects_divergence() {
        let atol = V::from_element(1, 1e-6, Default::default());
        let mut c = Convergence::new(1e-6, &atol);
        c.reset();
        assert_eq!(status(&mut c, 1.0), "continue");
        assert_eq!(status(&mut c, 0.95), "diverged");

        c.reset_eta();
        c.reset();
        assert_eq!(status(&mut c, 1e-3), "converged");
    }

    /// An update that converges is accepted before the divergence checks: on its last allowed
    /// update, the projection of a slowly converging iteration would call it divergent.
    #[test]
    fn converges_before_checking_divergence() {
        let atol = V::from_element(1, 1e-6, Default::default());
        let mut c = Convergence::new(1e-6, &atol);
        c.set_max_iter(2);
        c.reset();
        assert_eq!(status(&mut c, 0.3), "continue");
        // the rate 0.5 gives eta = 1, so the estimated error 0.15 converges
        assert_eq!(status(&mut c, 0.15), "converged");
    }
}
