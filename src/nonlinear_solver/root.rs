use crate::{scalar::Scalar, OdeEquations, OdeSolverMethod};

pub struct RootFinder<Eqn: OdeEquations> {
    t0: Eqn::T,
    g0: Eqn::V,
    g1: Eqn::V,
    gmid: Eqn::V,
}

impl<Eqn: OdeEquations> RootFinder<Eqn> {
    pub fn new() -> Self {
        Self {
            t0: Eqn::T::zero(),
            g0: Eqn::V::zeros(0),
            g1: Eqn::V::zeros(0),
            gmid: Eqn::V::zeros(0),
        }
    }

    /// Set the lower boundary of the root search.
    /// This function should be called first after [Self::new]
    pub fn set_g0(&mut self, method: &impl OdeSolverMethod<Eqn>, &y: Eqn::V, t: Eqn::T) {
        let root_fn = method
            .problem()
            .eqn()
            .root()
            .expect("Root function not defined");
        if self.g0.len() != y.len() {
            self.g0 = Eqn::V::zeros(y.len());
            self.g1 = Eqn::V::zeros(y.len());
            self.gmid = Eqn::V::zeros(y.len());
        }
        root_fn.call_inplace(&y, t, &mut self.g0);
        self.t0 = t;
    }

    /// Set the upper boundary of the root search and checks for a zero crossing.
    /// If a zero crossing is found, the index of the crossing is returned
    /// This function should be called after [Self::set_g0]
    ///
    /// We find the root of a function using the method proposed by Sundials docs (https://sundials.readthedocs.io/en/latest/cvode/Mathematics_link.html#rootfinding)
    pub fn set_g1(
        &mut self,
        method: &impl OdeSolverMethod<Eqn>,
        &y: Eqn::V,
        t: Eqn::T,
    ) -> Option<Eqn::T> {
        let root_fn = method
            .problem()
            .eqn()
            .root()
            .expect("Root function not defined");

        root_fn.call_inplace(&y, t, &mut self.g1);

        let sign_change = |acc, g0, g1, i| {
            if g0 == Eqn::T::zero() {
                acc.0 = true;
            } else if g0 * g1 < Eqn::T::zero() {
                let gfrac = (g1 / (g1 - g0)).abs();
                if gfrac > acc.1 {
                    acc.1 = gfrac;
                    acc.2 = i;
                }
            }
        };
        let (rootfnd, gfracmax, imax) =
            self.g0
                .binary_fold(&self.g1, (false, Eqn::T::zero(), -1), sign_change);

        // if no sign change we don't need to find the root
        if imax < 0 {
            return if rootfnd {
                // found a root at the upper boundary and no other sign change, return the root
                Some(t)
            } else {
                // no root found or sign change, return None
                None
            };
        }

        // otherwise we need to do the modified secant method to find the root
        let err = 1.0;
        let alpha = 1.0;
        let mut sign_change = [false, true];
        let mut i = 0;
        let mut t1 = t;
        let mut t0 = self.t0;
        let tol = 100 * Eqn::T::EPSILON * (t1.abs() + (t1 - t0).abs());
        while (t1 - t0).abs() <= tol {
            let mut t_mid =
                t1 - (t1 - t0) * self.g1[imax] / (self.g1[imax] - alpha * self.g0[imax]);

            // adjust t_mid away from the boundaries
            if (t_mid - t0).abs() < 0.5 * tol {
                let fracint = (t1 - t0).abs() / tol;
                let fracsub = if fracint > 5.0 { 0.1 } else { 0.5 / fracint };
                t_mid = t0 + fracsub * (t1 - t0);
            }
            if (t1 - t_mid).abs() < 0.5 * tol {
                let fracint = (t1 - t0).abs() / tol;
                let fracsub = if fracint > 5.0 { 0.1 } else { 0.5 / fracint };
                t_mid = t1 - fracsub * (t1 - t0);
            }

            let ymid = method.interpolate(t_mid).unwrap();
            root_fn.call_inplace(&ymid, t_mid, &mut self.gmid);

            let (rootfnd, gfracmax, imax) =
                self.g0
                    .binary_fold(&self.gmid, (false, Eqn::T::zero(), -1), sign_change);

            if imax >= 0 {
                // Sign change found in (tlo,tmid); replace thi with tmid.
                t1 = t_mid;
                std::mem::swap(&mut self.g1, &mut self.gmid);
            } else if rootfnd {
                // No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid.
                t0 = t_mid;
                self.g0 = self.gmid;
            } else {
                // No sign change in (tlo,tmid), and no zero at tmid. Sign change must be in (tmid,thi).  Replace tlo with tmid.
                t0 = t_mid;
                std::mem::swap(&mut self.g0, &mut self.gmid);
            }
        }
        Some(t1)
    }
}
