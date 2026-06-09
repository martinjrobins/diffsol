use crate::{Matrix, Scalar};

#[cfg(feature = "autodiff")]
use std::autodiff::autodiff_reverse;

/// Trait for defining the initial condition of an ODE with automatic differentiation support.
///
/// With the `autodiff` feature enabled (requires nightly), users only need to implement
/// [`init_inplace`](Self::init_inplace). The derivative method `init_sens_vjp_inplace` is
/// automatically generated via `std::autodiff`.
///
/// # Example (with `autodiff` feature)
/// ```ignore
/// struct MyInit;
///
/// impl ConstantAutodiff<M> for MyInit {
///     type T = f64;
///     fn init_inplace(p: &[f64], y: &mut [f64]) {
///         y[0] = p[2];
///     }
/// }
/// // init_sens_vjp_inplace is automatically provided
/// ```
pub trait ConstantAutodiff<M: Matrix> {
    type T: Scalar;

    /// Compute the initial state `y = y0(p)`.
    #[cfg_attr(
        feature = "autodiff",
        autodiff_reverse(init_sens_vjp_inplace, Duplicated, Duplicated)
    )]
    fn init_inplace(p: &[Self::T], y: &mut [Self::T]);
}
