use crate::{
    scalar::{IndexType, Scalar},
    NonLinearOp, OdeEquations, Vector,
};
use anyhow::Result;
use num_traits::{abs, One, Zero};

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
    pub fn set_g0(
        &mut self,
        root_fn: &impl NonLinearOp<V = Eqn::V, T = Eqn::T>,
        y: &Eqn::V,
        t: Eqn::T,
    ) {
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
        interpolate: &impl Fn(Eqn::T) -> Result<Eqn::V>,
        root_fn: &impl NonLinearOp<V = Eqn::V, T = Eqn::T>,
        y: &Eqn::V,
        t: Eqn::T,
    ) -> Option<Eqn::T> {
        root_fn.call_inplace(&y, t, &mut self.g1);

        let sign_change_fn =
            |mut acc: (bool, Eqn::T, i32), g0: Eqn::T, g1: Eqn::T, i: IndexType| {
                if g0 == Eqn::T::zero() {
                    acc.0 = true;
                } else if g0 * g1 < Eqn::T::zero() {
                    let gfrac = abs(g1 / (g1 - g0));
                    if gfrac > acc.1 {
                        acc.1 = gfrac;
                        acc.2 = i32::try_from(i).unwrap();
                    }
                }
                acc
            };
        let (rootfnd, _gfracmax, imax) =
            self.g0
                .binary_fold(&self.g1, (false, Eqn::T::zero(), -1), sign_change_fn);

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
        let mut imax = IndexType::try_from(imax).unwrap();
        let mut alpha = Eqn::T::one();
        let mut sign_change = [false, true];
        let mut i = 0;
        let mut t1 = t;
        let mut t0 = self.t0;
        let tol = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(t1) + abs(t1 - t0));
        let half = Eqn::T::from(0.5);
        let double = Eqn::T::from(2.0);
        let five = Eqn::T::from(5.0);
        let pntone = Eqn::T::from(0.1);
        while abs(t1 - t0) <= tol {
            let mut t_mid =
                t1 - (t1 - t0) * self.g1[imax] / (self.g1[imax] - alpha * self.g0[imax]);

            // adjust t_mid away from the boundaries
            if abs(t_mid - t0) < half * tol {
                let fracint = abs(t1 - t0) / tol;
                let fracsub = if fracint > five {
                    pntone
                } else {
                    half / fracint
                };
                t_mid = t0 + fracsub * (t1 - t0);
            }
            if abs(t1 - t_mid) < half * tol {
                let fracint = abs(t1 - t0) / tol;
                let fracsub = if fracint > five {
                    pntone
                } else {
                    half / fracint
                };
                t_mid = t1 - fracsub * (t1 - t0);
            }

            let ymid = interpolate(t_mid).unwrap();
            root_fn.call_inplace(&ymid, t_mid, &mut self.gmid);

            let (rootfnd, _gfracmax, imax_i32) =
                self.g0
                    .binary_fold(&self.gmid, (false, Eqn::T::zero(), -1), sign_change_fn);
            imax = IndexType::try_from(imax_i32).unwrap();

            let lower = imax >= 0;
            if lower {
                // Sign change found in (tlo,tmid); replace thi with tmid.
                t1 = t_mid;
                std::mem::swap(&mut self.g1, &mut self.gmid);
            } else if rootfnd {
                // No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid.
                return Some(t_mid);
            } else {
                // No sign change in (tlo,tmid), and no zero at tmid. Sign change must be in (tmid,thi).  Replace tlo with tmid.
                t0 = t_mid;
                std::mem::swap(&mut self.g0, &mut self.gmid);
            }

            sign_change[i % 2] = lower;
            if i >= 2 {
                alpha = if sign_change[0] != sign_change[1] {
                    Eqn::T::one()
                } else if sign_change[0] {
                    half * alpha
                } else {
                    double * alpha
                };
            }
            i += 1;
        }
        Some(t1)
    }
}
