use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

#[cfg(feature = "cuda")]
pub mod cuda;

use crate::vector::VectorView;
pub trait Scalar:
    nalgebra::Scalar
    + faer::traits::ComplexField
    + faer::traits::RealField
    + nalgebra::SimdRealField
    + nalgebra::ComplexField<RealField = Self>
    + num_traits::Signed
    + num_traits::Pow<Self, Output = Self>
    + num_traits::Pow<i32, Output = Self>
    + num_traits::FromPrimitive
    + num_traits::ToPrimitive
    + Display
    + Copy
    + PartialOrd
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

impl Scalar for f32 {
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NAN: Self = f32::NAN;
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl<T: Scalar> From<faer::Scale<T>> for Scale<T> {
    fn from(s: faer::Scale<T>) -> Self {
        Scale(s.0)
    }
}
impl<T: Scalar> From<Scale<T>> for faer::Scale<T> {
    fn from(s: Scale<T>) -> Self {
        faer::Scale(s.value())
    }
}
impl<T: Scalar> From<T> for Scale<T> {
    fn from(s: T) -> Self {
        Scale(s)
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
