// Data type Python enum

use diffsl::execution::external::ExternSymbols;
use diffsol::DiffSlScalar;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarType {
    F32,
    F64,
}

pub trait Scalar: DiffSlScalar + ToScalarType + ExternSymbols {}

impl<T: DiffSlScalar + ToScalarType + ExternSymbols> Scalar for T {}

pub trait ToScalarType {
    fn scalar_type() -> ScalarType;
}

impl ToScalarType for f32 {
    fn scalar_type() -> ScalarType {
        ScalarType::F32
    }
}

impl ToScalarType for f64 {
    fn scalar_type() -> ScalarType {
        ScalarType::F64
    }
}
