use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use nalgebra::{ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, ComplexField, SimdRealField};
use num_traits::{Pow, Signed};

use crate::vector::VectorView;

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
    + Into<f64>
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

//TODO: Is it possible for us to need RhsE != LhsE?

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

impl<'a, V: VectorView<'static, T = E>, E: Scalar> Mul<V> for Scale<E> {
    type Output = V::Owned;
    #[inline]
    fn mul(self, rhs: V) -> Self::Output {
        rhs.scalar_mul(self.0)
    }
}

//TODO: Not sure why it is now working
// impl<'a, E, V> Mul<Scale<E>> for V
// where
//     V: VectorView<'static, T = E>,
//     E: Scalar,
// {
//     type Output = V::Owned;
//     #[inline]
//     fn mul(self, rhs: Scale<E>) -> Self::Output {
//         self.scalar_mul(rhs.0)
//     }
// }

#[test]
fn test_scale() {
    assert_eq!(scale(2.0) * scale(3.0), scale(6.0));
}
