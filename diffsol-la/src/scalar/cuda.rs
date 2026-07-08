use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use super::Scalar;

pub enum CudaType {
    F64,
}

pub trait ScalarCuda: Scalar + ValidAsZeroBits + DeviceRepr {
    fn as_enum() -> CudaType;
    fn as_f64(self) -> f64 {
        panic!("Unsupported type for as_f64");
    }
    fn as_str() -> &'static str {
        match Self::as_enum() {
            CudaType::F64 => "f64",
        }
    }
}

impl ScalarCuda for f64 {
    fn as_enum() -> CudaType {
        CudaType::F64
    }
    fn as_f64(self) -> f64 {
        self
    }
}
