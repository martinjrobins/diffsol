use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

#[cfg(any(feature = "cuda", feature = "cuda-oxide"))]
pub mod cuda;

use crate::vector::VectorView;

/// A scalar type suitable for numerical computations in ODE solvers.
///
/// This trait aggregates the crate-local numeric requirements shared across diffsol.
///
/// # Implementations
/// diffsol provides implementations for `f64` and `f32`.
///
/// # Examples
/// ```
/// use diffsol_la::Scalar;
///
/// fn compute<T: Scalar>(x: T, y: T) -> T {
///     x * x + y
/// }
/// ```
pub trait Scalar:
    num_traits::Signed
    + num_traits::FromPrimitive
    + num_traits::ToPrimitive
    + Display
    + std::fmt::Debug
    + Copy
    + PartialEq
    + PartialOrd
    + AddAssign
    + SubAssign
    + MulAssign
    + std::ops::DivAssign
    + Send
    + Sync
    + 'static
{
    /// Machine epsilon for this scalar type (smallest representable positive value such that 1.0 + EPSILON != 1.0).
    const EPSILON: Self;
    /// Positive infinity value for this scalar type.
    const INFINITY: Self;
    /// Not-a-Number (NaN) value for this scalar type.
    const NAN: Self;
    /// Check if this value is NaN.
    fn is_nan(self) -> bool;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Cube root.
    fn cbrt(self) -> Self;
    /// Exponential.
    fn exp(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Raise to a floating-point power.
    fn powf(self, n: Self) -> Self;
    /// Raise to an integer power.
    fn powi(self, n: i32) -> Self;
    /// Maximum of two values, NaN if either is NaN.
    fn max(self, other: Self) -> Self;

    /// Addition, potentially reassociated/relaxed for speed under the `algebraic-ops` feature.
    fn algebraic_add(self, other: Self) -> Self {
        self + other
    }
    /// Subtraction, potentially reassociated/relaxed for speed under the `algebraic-ops` feature.
    fn algebraic_sub(self, other: Self) -> Self {
        self - other
    }
    /// Multiplication, potentially reassociated/relaxed for speed under the `algebraic-ops` feature.
    fn algebraic_mul(self, other: Self) -> Self {
        self * other
    }
    /// Division, potentially reassociated/relaxed for speed under the `algebraic-ops` feature.
    fn algebraic_div(self, other: Self) -> Self {
        self / other
    }
}

/// A [`Scalar`] that also satisfies nalgebra's numeric field requirements.
pub trait NalgebraScalar:
    Scalar + nalgebra::Scalar + nalgebra::SimdRealField + nalgebra::ComplexField<RealField = Self>
{
}

impl<T> NalgebraScalar for T where
    T: Scalar
        + nalgebra::Scalar
        + nalgebra::SimdRealField
        + nalgebra::ComplexField<RealField = Self>
{
}

/// A [`Scalar`] that also satisfies faer's numeric field requirements.
pub trait FaerScalar: Scalar + faer::traits::ComplexField + faer::traits::RealField {}

impl<T> FaerScalar for T where T: Scalar + faer::traits::ComplexField + faer::traits::RealField {}

/// Integer power by exponentiation by squaring.
///
/// This uses a fixed sequence of IEEE multiplications, so the result is the same on all
/// platforms.
fn powi_by_squaring<T>(x: T, n: i32) -> T
where
    T: Copy + num_traits::One + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
{
    let mut base = x;
    let mut e = n.unsigned_abs();
    let mut acc = T::one();
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * base;
        }
        e >>= 1;
        if e > 0 {
            base = base * base;
        }
    }
    if n < 0 {
        T::one() / acc
    } else {
        acc
    }
}

/// The index type used throughout diffsol for indexing vectors and matrices.
pub type IndexType = usize;

impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const NAN: Self = f64::NAN;
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    fn cbrt(self) -> Self {
        libm::cbrt(self)
    }
    fn exp(self) -> Self {
        libm::exp(self)
    }
    fn sin(self) -> Self {
        libm::sin(self)
    }
    fn cos(self) -> Self {
        libm::cos(self)
    }
    fn ln(self) -> Self {
        libm::log(self)
    }
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    fn powi(self, n: i32) -> Self {
        powi_by_squaring(self, n)
    }
    fn max(self, other: Self) -> Self {
        // NaN must win so a failed batch lane fails the error test rather than vanishing
        // TODO: swap to f64::maximum once stabilized
        if self.is_nan() || other.is_nan() {
            Self::NAN
        } else {
            self.max(other)
        }
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_add(self, other: Self) -> Self {
        self.algebraic_add(other)
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_sub(self, other: Self) -> Self {
        self.algebraic_sub(other)
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_mul(self, other: Self) -> Self {
        self.algebraic_mul(other)
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_div(self, other: Self) -> Self {
        self.algebraic_div(other)
    }
}

impl Scalar for f32 {
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NAN: Self = f32::NAN;
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    fn cbrt(self) -> Self {
        libm::cbrtf(self)
    }
    fn exp(self) -> Self {
        libm::expf(self)
    }
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    fn ln(self) -> Self {
        libm::logf(self)
    }
    fn powf(self, n: Self) -> Self {
        libm::powf(self, n)
    }
    fn powi(self, n: i32) -> Self {
        powi_by_squaring(self, n)
    }
    fn max(self, other: Self) -> Self {
        // NaN must win so a failed batch lane fails the error test rather than vanishing
        // TODO: swap to f32::maximum once stabilized
        if self.is_nan() || other.is_nan() {
            Self::NAN
        } else {
            self.max(other)
        }
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_add(self, other: Self) -> Self {
        self.algebraic_add(other)
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_sub(self, other: Self) -> Self {
        self.algebraic_sub(other)
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_mul(self, other: Self) -> Self {
        self.algebraic_mul(other)
    }
    #[cfg(feature = "algebraic-ops")]
    fn algebraic_div(self, other: Self) -> Self {
        self.algebraic_div(other)
    }
}

impl<T: FaerScalar> From<faer::Scale<T>> for Scale<T> {
    fn from(s: faer::Scale<T>) -> Self {
        Scale(s.0)
    }
}
impl<T: FaerScalar> From<Scale<T>> for faer::Scale<T> {
    fn from(s: Scale<T>) -> Self {
        faer::Scale(s.value())
    }
}
impl<T: Scalar> From<T> for Scale<T> {
    fn from(s: T) -> Self {
        Scale(s)
    }
}

/// A wrapper for scalar values used when scaling vectors and matrices.
#[derive(Copy, Clone, Debug)]
pub struct Scale<E: Scalar>(pub E);

impl<E: Scalar> Scale<E> {
    /// Get the underlying scalar value.
    #[inline]
    pub fn value(self) -> E {
        self.0
    }
}

/// Create a `Scale` wrapper from a scalar value.
///
/// This is a convenience function equivalent to `Scale(value)`.
#[inline]
pub fn scale<E: Scalar>(value: E) -> Scale<E> {
    Scale(value)
}

macro_rules! impl_bin_op {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<E: Scalar> $trait<Scale<E>> for Scale<E> {
            type Output = Scale<E>;

            #[inline]
            fn $method(self, rhs: Scale<E>) -> Self::Output {
                Scale(self.0 $operator rhs.0)
            }
        }
    };
}

macro_rules! impl_assign_bin_op {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<E: Scalar> $trait<Scale<E>> for Scale<E> {
            #[inline]
            fn $method(&mut self, rhs: Scale<E>) {
                self.0 = self.0 $operator rhs.0
            }
        }
    };
}

impl_bin_op!(Mul, mul, *);
impl_bin_op!(Add, add, +);
impl_bin_op!(Sub, sub, -);

impl_assign_bin_op!(MulAssign, mul_assign, *);
impl_assign_bin_op!(AddAssign, add_assign, +);
impl_assign_bin_op!(SubAssign, sub_assign, -);

impl<E: Scalar> PartialEq for Scale<E> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl<V: VectorView<'static, T = E>, E: Scalar> Mul<V> for Scale<E> {
    type Output = V::Owned;
    #[inline]
    fn mul(self, rhs: V) -> Self::Output {
        rhs * scale(self.0)
    }
}

#[test]
fn test_scale() {
    assert_eq!(scale(2.0) * scale(3.0), scale(6.0));
}

#[cfg(test)]
mod tests {
    use super::Scalar;

    #[test]
    fn powi_by_squaring_matches_exact_values() {
        assert_eq!(Scalar::powi(2.0_f64, 0), 1.0);
        assert_eq!(Scalar::powi(2.0_f64, 1), 2.0);
        assert_eq!(Scalar::powi(2.0_f64, 10), 1024.0);
        assert_eq!(Scalar::powi(2.0_f64, -3), 0.125);
        assert_eq!(Scalar::powi(-3.0_f64, 3), -27.0);
        assert_eq!(Scalar::powi(1.5_f32, 2), 2.25);
        assert_eq!(Scalar::powi(2.0_f64, i32::MIN), 0.0);
        let x = 1.1_f64;
        assert!((Scalar::powi(x, 9) - x.powi(9)).abs() <= 4.0 * f64::EPSILON * x.powi(9));
    }
}
