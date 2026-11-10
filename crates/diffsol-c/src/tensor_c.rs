use std::ptr;

use crate::c_invalid_arg;
use crate::scalar_type_c::scalar_type_to_i32;
use crate::tensor::Tensor;

/// Free a array previously returned by this library.
///
/// # Safety
/// `tensor` must be either null or a pointer returned by this library that has
/// not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_tensor_free(tensor: *mut Tensor) {
    if tensor.is_null() {
        c_invalid_arg!("tensor is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(tensor));
    }
}

/// Return the raw data pointer stored by a array.
///
/// # Safety
/// `tensor` must be a valid pointer to an array created by this library.
/// The returned pointer is borrowed and remains owned by the array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_tensor_ptr(tensor: *const Tensor) -> *const u8 {
    if tensor.is_null() {
        c_invalid_arg!("tensor is null");
        return ptr::null();
    }
    let tensor = unsafe { &*tensor };
    tensor.data_ptr()
}

/// Return the number of dimensions in a array.
///
/// # Safety
/// `tensor` must be a valid pointer to an array created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_tensor_ndim(tensor: *const Tensor) -> usize {
    if tensor.is_null() {
        c_invalid_arg!("tensor is null");
        return 0;
    }
    let tensor = unsafe { &*tensor };
    tensor.shape().len()
}

/// Return the size of a single dimension in a array.
///
/// # Safety
/// `tensor` must be a valid pointer to an array created by this library. If `index` is not
/// less than the number of axes, the result is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_tensor_dim(tensor: *const Tensor, index: usize) -> usize {
    if tensor.is_null() {
        c_invalid_arg!("tensor is null");
        return 0;
    }
    let tensor = unsafe { &*tensor };
    // an index out of range gives 0, because a panic must not cross the C boundary
    tensor.shape().get(index).copied().unwrap_or(0)
}

/// Return the stride, in bytes, for a single dimension in a array.
///
/// # Safety
/// `tensor` must be a valid pointer to an array created by this library. If `index` is not
/// less than the number of axes, the result is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_tensor_stride(tensor: *const Tensor, index: usize) -> usize {
    if tensor.is_null() {
        c_invalid_arg!("tensor is null");
        return 0;
    }
    let tensor = unsafe { &*tensor };
    // an index out of range gives 0, because a panic must not cross the C boundary
    tensor.strides().get(index).copied().unwrap_or(0)
}

/// Return the scalar type enum stored by a array.
///
/// # Safety
/// `tensor` must be a valid pointer to an array created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_tensor_dtype(tensor: *const Tensor) -> i32 {
    if tensor.is_null() {
        c_invalid_arg!("tensor is null");
        return -1;
    }
    let tensor = unsafe { &*tensor };
    scalar_type_to_i32(tensor.dtype())
}
