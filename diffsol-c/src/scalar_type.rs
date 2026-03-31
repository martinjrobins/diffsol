// Data type Python enum

#[cfg(feature = "external")]
use diffsl::execution::external::ExternSymbols;
use diffsol::DiffSlScalar;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarType {
    F32,
    F64,
}

pub trait Scalar: DiffSlScalar + ToScalarType {}

impl<T: DiffSlScalar + ToScalarType> Scalar for T {}

#[cfg(feature = "external")]
pub trait ExternalScalar: Scalar + ExternSymbols {}

#[cfg(feature = "external")]
impl<T: Scalar + ExternSymbols> ExternalScalar for T {}

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
