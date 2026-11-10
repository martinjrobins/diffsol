use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::scalar_type::ScalarType;

const SCALAR_TYPE_F32: &[u8] = b"f32\0";
const SCALAR_TYPE_F64: &[u8] = b"f64\0";

/// The C value of `ScalarType::F32`.
pub const DIFFSOL_SCALAR_TYPE_F32: i32 = 0;
/// The C value of `ScalarType::F64`.
pub const DIFFSOL_SCALAR_TYPE_F64: i32 = 1;

// the `dtype` field of a `TensorView` is a `ScalarType`, so its values must be the C values
const _: () = assert!(
    ScalarType::F32 as i32 == DIFFSOL_SCALAR_TYPE_F32
        && ScalarType::F64 as i32 == DIFFSOL_SCALAR_TYPE_F64
);

pub(crate) fn scalar_type_from_i32(value: i32) -> Option<ScalarType> {
    match value {
        DIFFSOL_SCALAR_TYPE_F32 => Some(ScalarType::F32),
        DIFFSOL_SCALAR_TYPE_F64 => Some(ScalarType::F64),
        _ => None,
    }
}

pub(crate) fn scalar_type_to_i32(value: ScalarType) -> i32 {
    match value {
        ScalarType::F32 => DIFFSOL_SCALAR_TYPE_F32,
        ScalarType::F64 => DIFFSOL_SCALAR_TYPE_F64,
    }
}

/// Return the number of scalar type enum values.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_scalar_type_count() -> usize {
    2
}

/// Return whether a scalar type enum value is valid.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_scalar_type_is_valid(value: i32) -> i32 {
    if scalar_type_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid scalar_type");
        0
    }
}

/// Return the name of a scalar type enum value.
///
/// # Safety
/// The returned pointer is borrowed from static storage owned by this library
/// and must not be freed or mutated by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_scalar_type_name(value: i32) -> *const c_char {
    match scalar_type_from_i32(value) {
        Some(ScalarType::F32) => SCALAR_TYPE_F32.as_ptr() as *const c_char,
        Some(ScalarType::F64) => SCALAR_TYPE_F64.as_ptr() as *const c_char,
        None => {
            c_invalid_arg!("invalid scalar_type");
            ptr::null()
        }
    }
}
