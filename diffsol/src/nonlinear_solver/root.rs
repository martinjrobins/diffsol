use std::cell::RefCell;

use crate::{
    error::DiffsolError,
    scalar::{IndexType, Scalar},
    NonLinearOp, Vector,
};

use num_traits::{abs, One, Zero};

#[derive(Clone)]
pub struct RootFinder<V: Vector> {
    t0: RefCell<V::T>,
    g0: RefCell<V>,
    g1: RefCell<V>,
    gmid: RefCell<V>,
}

impl<V: Vector> RootFinder<V> {
    pub fn new(n: usize) -> Self {
        Self {
            t0: RefCell::new(V::T::zero()),
            g0: RefCell::new(V::zeros(n)),
            g1: RefCell::new(V::zeros(n)),
            gmid: RefCell::new(V::zeros(n)),
        }
    }

    /// Set the lower boundary of the root search.
    /// This function should be called first after [Self::new]
    pub fn init(&self, root_fn: &impl NonLinearOp<V = V, T = V::T>, y: &V, t: V::T) {
        root_fn.call_inplace(y, t, &mut self.g0.borrow_mut());
        self.t0.replace(t);
    }

    /// Set the upper boundary of the root search and checks for a zero crossing.
    /// If a zero crossing is found, the index of the crossing is returned
    ///
    /// This function assumes that g0 and t0 have already beeen set via [Self::init]
    /// or previous iterations of [Self::check_root]
    ///
    /// We find the root of a function using the method proposed by Sundials [docs](https://sundials.readthedocs.io/en/latest/cvode/Mathematics_link.html#rootfinding)
    pub fn check_root(
        &self,
        interpolate: &impl Fn(V::T) -> Result<V, DiffsolError>,
        root_fn: &impl NonLinearOp<V = V, T = V::T>,
        y: &V,
        t: V::T,
    ) -> Option<V::T> {
        let g1 = &mut *self.g1.borrow_mut();
        let g0 = &mut *self.g0.borrow_mut();
        let gmid = &mut *self.gmid.borrow_mut();
        root_fn.call_inplace(y, t, g1);

        let sign_change_fn = |mut acc: (bool, V::T, i32), g0: V::T, g1: V::T, i: IndexType| {
            if g1 == V::T::zero() {
                acc.0 = true;
            } else if g0 * g1 < V::T::zero() {
                let gfrac = abs(g1 / (g1 - g0));
                if gfrac > acc.1 {
                    acc.1 = gfrac;
                    acc.2 = i32::try_from(i).unwrap();
                }
            }
            acc
        };
        let (rootfnd, _gfracmax, imax) =
            (*g0).binary_fold(g1, (false, V::T::zero(), -1), sign_change_fn);

        // if no sign change we don't need to find the root
        if imax < 0 {
            // setup g0 for next iteration
            std::mem::swap(g0, g1);
            self.t0.replace(t);
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
        let mut alpha = V::T::one();
        let mut sign_change = [false, true];
        let mut i = 0;
        let mut t1 = t;
        let mut t0 = *self.t0.borrow();
        let tol = V::T::from(100.0) * V::T::EPSILON * (abs(t1) + abs(t1 - t0));
        let half = V::T::from(0.5);
        let double = V::T::from(2.0);
        let five = V::T::from(5.0);
        let pntone = V::T::from(0.1);
        while abs(t1 - t0) > tol {
            let mut t_mid = t1 - (t1 - t0) * g1[imax] / (g1[imax] - alpha * g0[imax]);

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
            root_fn.call_inplace(&ymid, t_mid, gmid);

            let (rootfnd, _gfracmax, imax_i32) =
                (*g0).binary_fold(gmid, (false, V::T::zero(), -1), sign_change_fn);
            let lower = imax_i32 >= 0;

            if lower {
                // Sign change found in (tlo,tmid); replace thi with tmid.
                t1 = t_mid;
                imax = IndexType::try_from(imax_i32).unwrap();
                std::mem::swap(g1, gmid);
            } else if rootfnd {
                // we are returning so make sure g0 is set for next iteration
                root_fn.call_inplace(y, t, g0);

                // No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid.
                return Some(t_mid);
            } else {
                // No sign change in (tlo,tmid), and no zero at tmid. Sign change must be in (tmid,thi).  Replace tlo with tmid.
                t0 = t_mid;
                std::mem::swap(g0, gmid);
            }

            sign_change[i % 2] = lower;
            if i >= 2 {
                alpha = if sign_change[0] != sign_change[1] {
                    V::T::one()
                } else if sign_change[0] {
                    half * alpha
                } else {
                    double * alpha
                };
            }
            i += 1;
        }
        // we are returning so make sure g0 is set for next iteration
        root_fn.call_inplace(y, t, g0);
        Some(t1)
    }
}

#[cfg(test)]
mod tests {
    use crate::{error::DiffsolError, op::ParameterisedOp, ClosureNoJac, RootFinder, Vector};

    #[test]
    fn test_root() {
        type V = nalgebra::DVector<f64>;
        type M = nalgebra::DMatrix<f64>;
        let interpolate = |t: f64| -> Result<V, DiffsolError> { Ok(Vector::from_vec(vec![t])) };
        let p = V::zeros(0);
        let root_fn = ClosureNoJac::<M, _>::new(
            |y: &V, _p: &V, _t: f64, g: &mut V| {
                g[0] = y[0] - 0.4;
            },
            1,
            1,
            p.len(),
        );
        let root_fn = ParameterisedOp::new(&root_fn, &p);

        // check no root
        let root_finder = RootFinder::new(1);
        root_finder.init(&root_fn, &Vector::from_vec(vec![0.0]), 0.0);
        let root =
            root_finder.check_root(&interpolate, &root_fn, &Vector::from_vec(vec![0.3]), 0.3);
        assert_eq!(root, None);

        // check root
        let root_finder = RootFinder::new(1);
        root_finder.init(&root_fn, &Vector::from_vec(vec![0.0]), 0.0);
        let root =
            root_finder.check_root(&interpolate, &root_fn, &Vector::from_vec(vec![1.3]), 1.3);
        if let Some(root) = root {
            assert!((root - 0.4).abs() < 1e-10);
        } else {
            unreachable!();
        }
    }
}
