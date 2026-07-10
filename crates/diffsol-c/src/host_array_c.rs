use std::ptr;

use crate::c_invalid_arg;
use crate::host_array::HostArray;
use crate::scalar_type_c::{scalar_type_from_i32, scalar_type_to_i32};

/// Allocate a one-dimensional host array of the requested length and scalar type.
///
/// # Safety
/// The returned pointer must be freed with `diffsol_host_array_free`. `dtype`
/// must be a valid scalar type enum defined by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_alloc_vector(len: usize, dtype: i32) -> *mut HostArray {
    let dtype = match scalar_type_from_i32(dtype) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid dtype");
            return ptr::null_mut();
        }
    };
    let array = HostArray::alloc_vector(len, dtype);
    Box::into_raw(Box::new(array))
}

/// Free a host array previously returned by this library.
///
/// # Safety
/// `array` must be either null or a pointer returned by this library that has
/// not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_free(array: *mut HostArray) {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(array));
    }
}

/// Return the raw data pointer stored by a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library.
/// The returned pointer is borrowed and remains owned by the array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_ptr(array: *const HostArray) -> *const u8 {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return ptr::null();
    }
    let array = unsafe { &*array };
    array.data_ptr()
}

/// Return the number of dimensions in a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_ndim(array: *const HostArray) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    array.ndim()
}

/// Return the size of a single dimension in a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library.
/// `index` must be in bounds for the array shape.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_dim(array: *const HostArray, index: usize) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    array.dim(index)
}

/// Return the stride, in bytes, for a single dimension in a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library.
/// `index` must be in bounds for the array shape.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_stride(array: *const HostArray, index: usize) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    array.stride(index)
}

/// Return the scalar type enum stored by a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_dtype(array: *const HostArray) -> i32 {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return -1;
    }
    let array = unsafe { &*array };
    scalar_type_to_i32(array.dtype())
}
