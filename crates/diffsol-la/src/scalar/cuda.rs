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

    /// Device-only: `warp::shuffle_xor_*` for this type.
    #[cfg(feature = "cuda-oxide")]
    fn shuffle_xor(self, lane_mask: u32) -> Self;
    /// Device-only: full-warp sum, `warp::reduce_sum_*` for this type.
    #[cfg(feature = "cuda-oxide")]
    fn warp_reduce_sum(self) -> Self;
    /// Bit pattern whose unsigned order matches the value's order for `self >= 0`,
    /// read back on the host with `f64::from_bits`.
    #[cfg(feature = "cuda-oxide")]
    fn to_max_bits(self) -> u64;
}

impl ScalarCuda for f64 {
    fn as_enum() -> CudaType {
        CudaType::F64
    }
    fn as_f64(self) -> f64 {
        self
    }

    #[cfg(feature = "cuda-oxide")]
    #[inline(always)]
    fn shuffle_xor(self, lane_mask: u32) -> Self {
        cuda_device::warp::shuffle_xor_f64(self, lane_mask)
    }
    #[cfg(feature = "cuda-oxide")]
    #[inline(always)]
    fn warp_reduce_sum(self) -> Self {
        cuda_device::warp::reduce_sum_f64(self)
    }
    #[cfg(feature = "cuda-oxide")]
    #[inline(always)]
    fn to_max_bits(self) -> u64 {
        self.to_bits()
    }
}
