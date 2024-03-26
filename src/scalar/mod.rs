use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use nalgebra::{ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, ComplexField, SimdRealField};
use num_traits::{Pow, Signed};

use crate::vector::{Vector, VectorCommon, VectorRef, VectorView};

pub trait Scalar:
    nalgebra::Scalar
    + From<f64>
    + Display
    + SimdRealField
    + ComplexField
    + Copy
    + ClosedSub
    + From<f64>
    + ClosedMul
    + ClosedDiv
    + ClosedAdd
    + Signed
    + PartialOrd
    + Pow<Self, Output = Self>
    + Pow<i32, Output = Self>
{
    const EPSILON: Self;
    const INFINITY: Self;
    const NAN: Self;
    fn is_nan(self) -> bool;
}

pub type IndexType = usize;

impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const NAN: Self = f64::NAN;
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Scale<E: Scalar>(pub E);

impl<E: Scalar> Scale<E> {
    #[inline]
    pub fn value(self) -> E {
        self.0
    }
}

#[inline]
pub fn scale<E: Scalar>(value: E) -> Scale<E> {
    Scale(value)
}

//TODO: Is it possible for us to need RhsE != LhsE?
impl<E: Scalar> Mul<Scale<E>> for Scale<E> {
    type Output = Scale<E>;

    #[inline]
    fn mul(self, rhs: Scale<E>) -> Self::Output {
        Scale(self.0 * rhs.0)
    }
}

impl<E: Scalar> Add<Scale<E>> for Scale<E> {
    type Output = Scale<E>;

    #[inline]
    fn add(self, rhs: Scale<E>) -> Self::Output {
        Scale(self.0 + rhs.0)
    }
}

impl<E: Scalar> Sub<Scale<E>> for Scale<E> {
    type Output = Scale<E>;

    #[inline]
    fn sub(self, rhs: Scale<E>) -> Self::Output {
        Scale(self.0 - rhs.0)
    }
}

impl<E: Scalar> MulAssign<Scale<E>> for Scale<E> {
    #[inline]
    fn mul_assign(&mut self, rhs: Scale<E>) {
        self.0 = self.0 * rhs.0
    }
}

impl<E: Scalar> AddAssign<Scale<E>> for Scale<E> {
    #[inline]
    fn add_assign(&mut self, rhs: Scale<E>) {
        self.0 = self.0 + rhs.0
    }
}

impl<E: Scalar> SubAssign<Scale<E>> for Scale<E> {
    #[inline]
    fn sub_assign(&mut self, rhs: Scale<E>) {
        self.0 = self.0 - rhs.0
    }
}

impl<E: Scalar> PartialEq for Scale<E> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

// I have to implement Mul between Scale and VectorView
impl<'a, V: VectorView<'static, T = E>, E: Scalar> Mul<V> for Scale<E> {
    type Output = V::Owned;
    #[inline]
    fn mul(self, rhs: V) -> Self::Output {
        rhs.scalar_mul(self.0)
    }
}

#[test]
fn test_scale() {
    assert_eq!(scale(2.0) * scale(3.0), scale(6.0));
}
