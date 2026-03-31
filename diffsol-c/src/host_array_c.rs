use std::ptr;

use crate::c_invalid_arg;
use crate::host_array::HostArray;
use crate::scalar_type_c::{scalar_type_from_i32, scalar_type_to_i32};

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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_ptr(array: *const HostArray) -> *const u8 {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return ptr::null();
    }
    let array = unsafe { &*array };
    array.data_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_ndim(array: *const HostArray) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    array.ndim()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_dim(array: *const HostArray, index: usize) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    array.dim(index)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_stride(array: *const HostArray, index: usize) -> usize {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return 0;
    }
    let array = unsafe { &*array };
    array.stride(index)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_dtype(array: *const HostArray) -> i32 {
    if array.is_null() {
        c_invalid_arg!("host array is null");
        return -1;
    }
    let array = unsafe { &*array };
    scalar_type_to_i32(array.dtype())
}
