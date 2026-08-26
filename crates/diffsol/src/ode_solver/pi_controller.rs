//! Step-size factor from a PI controller.
//!
//! Shared by the Runge-Kutta and BDF step-size logic, both of which pick a new `h` from the
//! current error norm and, optionally, the previous one.

use crate::Scalar;

/// PI controller raw factor computation (before safety multiplier and clamping).
///
/// The exponentiations go through `pow` rather than a chained-`sqrt` shortcut for exponents that
/// happen to be negative powers of two. The shortcut is not value-preserving — it changes which
/// steps the solver takes — and it only covers `|exponent|` in `{1, 1/2, 1/4, 1/8}`, so the
/// remaining orders still reach the platform's `pow`. Mixing the two made the step counts of
/// stiff problems diverge between libm implementations, which broke the snapshot tests on
/// Windows while passing on Linux.
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
        error_norm.pow(-ki)
    } else {
        match &prev_error_norm {
            Some(prev) => {
                let kp = pi_proportional / order_f;
                let e_iexp = error_norm.pow(-(ki + kp));
                let e_pexp = prev.pow(kp);
                e_iexp * e_pexp
            }
            None => error_norm.pow(-ki),
        }
    }
}
