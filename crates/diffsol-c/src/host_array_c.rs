use std::ptr;

use crate::c_invalid_arg;
use crate::host_array::HostArray;
use crate::scalar_type_c::scalar_type_to_i32;

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
    array.shape().len()
}

/// Return the size of a single dimension in a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library. If `index` is not
/// less than the number of axes, the result is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_dim(array: *const HostArray, index: usize) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    // an index out of range gives 0, because a panic must not cross the C boundary
    array.shape().get(index).copied().unwrap_or(0)
}

/// Return the stride, in bytes, for a single dimension in a host array.
///
/// # Safety
/// `array` must be a valid pointer to a `HostArray` created by this library. If `index` is not
/// less than the number of axes, the result is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_stride(array: *const HostArray, index: usize) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    // an index out of range gives 0, because a panic must not cross the C boundary
    array.strides().get(index).copied().unwrap_or(0)
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
