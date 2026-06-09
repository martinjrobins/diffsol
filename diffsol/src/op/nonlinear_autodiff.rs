use crate::{Matrix, Scalar};

#[cfg(feature = "autodiff")]
use std::autodiff::{autodiff_forward, autodiff_reverse};

/// Trait for defining the right-hand side of an ODE with automatic differentiation support.
///
/// With the `autodiff` feature enabled (requires nightly), users only need to implement
/// [`rhs_inplace`](Self::rhs_inplace). The derivative methods `rhs_jvp_inplace`,
/// `rhs_vjp_inplace`, and `rhs_sens_vjp_inplace` are automatically generated via
/// `std::autodiff`.
///
/// # Example (with `autodiff` feature)
/// ```ignore
/// struct Logistic;
///
/// impl NonLinearAutodiff<M> for Logistic {
///     type T = f64;
///     fn rhs_inplace(x: &[f64], p: &[f64], y: &mut [f64]) {
///         y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
///     }
/// }
/// // rhs_jvp_inplace, rhs_vjp_inplace, rhs_sens_vjp_inplace are automatically provided
/// ```
pub trait NonLinearAutodiff<M: Matrix> {
    type T: Scalar;

    /// Compute the right-hand side `y = f(x, p)`.
    #[cfg_attr(feature = "autodiff", autodiff_forward(rhs_jvp_inplace, Dual, Const, Dual))]
    #[cfg_attr(feature = "autodiff", autodiff_reverse(rhs_vjp_inplace, Duplicated, Const, Duplicated))]
    #[cfg_attr(
        feature = "autodiff",
        autodiff_reverse(rhs_sens_vjp_inplace, Const, Duplicated, Duplicated)
    )]
    fn rhs_inplace(x: &[Self::T], p: &[Self::T], y: &mut [Self::T]);
}
