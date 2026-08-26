//! Step-size factor from a PI controller.
//!
//! Shared by the Runge-Kutta and BDF step-size logic, both of which pick a new `h` from the
//! current error norm and, optionally, the previous one.

use crate::Scalar;

/// Computes `x.pow(exponent)`, but takes a cheap exact shortcut via chained
/// square roots when `|exponent|` is a small negative power of two (as is the
/// case for the default PI-controller gains at low solver orders) instead of
/// a full transcendental `pow()` call. Falls back to `x.pow(exponent)` for any
/// other exponent, so non-default gains are unaffected.
fn pow_neg_pow2_fast<T: Scalar>(x: T, exponent: T) -> T {
    let neg = exponent < T::zero();
    let ax = if neg { -exponent } else { exponent };
    let mut candidate = T::one();
    for k in 0..=3 {
        if ax == candidate {
            let mut r = x;
            for _ in 0..k {
                r = r.sqrt();
            }
            return if neg { T::one() / r } else { r };
        }
        candidate /= T::one() + T::one();
    }
    x.pow(exponent)
}

/// PI controller raw factor computation (before safety multiplier and clamping).
pub(crate) fn pi_controller_raw<T: Scalar>(
    error_norm: T,
    prev_error_norm: Option<T>,
    pi_integral: T,
    pi_proportional: T,
    eff_order: usize,
) -> T {
    let order_f = T::from_usize(eff_order).unwrap();
    let ki = pi_integral / order_f;
    if pi_proportional == T::zero() {
        pow_neg_pow2_fast(error_norm, -ki)
    } else {
        match &prev_error_norm {
            Some(prev) => {
                let kp = pi_proportional / order_f;
                let e_iexp = pow_neg_pow2_fast(error_norm, -(ki + kp));
                let e_pexp = pow_neg_pow2_fast(*prev, kp);
                e_iexp * e_pexp
            }
            None => pow_neg_pow2_fast(error_norm, -ki),
        }
    }
}
