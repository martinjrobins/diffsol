use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::scalar_type::ScalarType;

const SCALAR_TYPE_F32: &[u8] = b"f32\0";
const SCALAR_TYPE_F64: &[u8] = b"f64\0";

pub(crate) fn scalar_type_from_i32(value: i32) -> Option<ScalarType> {
    match value {
        0 => Some(ScalarType::F32),
        1 => Some(ScalarType::F64),
        _ => None,
    }
}

pub(crate) fn scalar_type_to_i32(value: ScalarType) -> i32 {
    match value {
        ScalarType::F32 => 0,
        ScalarType::F64 => 1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_scalar_type_count() -> usize {
    2
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_scalar_type_is_valid(value: i32) -> i32 {
    if scalar_type_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid scalar_type");
        0
    }
}

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
