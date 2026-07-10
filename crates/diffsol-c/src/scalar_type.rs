// Data type Python enum

#[cfg(feature = "external")]
use diffsl::execution::external::ExternSymbols;
use diffsol::DiffSlScalar;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
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
