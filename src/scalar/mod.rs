use std::fmt::Display;

use nalgebra::{ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, ComplexField, SimdRealField};
use num_traits::{Pow, Signed};

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
