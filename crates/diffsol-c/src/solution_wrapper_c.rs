use crate::c_api_utils::{DIFFSOL_BAD_ARG, DIFFSOL_ERR, DIFFSOL_OK};
use crate::solution_wrapper::SolutionWrapper;
use crate::tensor::Tensor;
use crate::{c_error, c_invalid_arg};

fn boxed_tensor(tensor: Tensor) -> *mut Tensor {
    Box::into_raw(Box::new(tensor))
}

fn boxed_tensor_list(tensors: Vec<Tensor>) -> (*mut *mut Tensor, usize) {
    let boxed = tensors
        .into_iter()
        .map(boxed_tensor)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut *mut Tensor;
    (ptr, len)
}

/// Free a solution wrapper previously returned by this library.
///
/// # Safety
/// `solution` must be either null or a pointer returned by this library that
/// has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_free(solution: *mut SolutionWrapper) {
    if solution.is_null() {
        c_invalid_arg!("solution wrapper is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(solution));
    }
}

/// Return the recorded solution values as a array.
///
/// # Safety
/// `solution` must be a valid pointer created by this library. `out_tensor`
/// must be a valid, writable pointer to receive ownership of the returned array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_ys(
    solution: *const SolutionWrapper,
    out_tensor: *mut *mut Tensor,
) -> i32 {
    if solution.is_null() || out_tensor.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_ys");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_ys() {
        Ok(tensor) => {
            unsafe {
                *out_tensor = boxed_tensor(tensor);
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Return the recorded solution times as a array.
///
/// # Safety
/// `solution` must be a valid pointer created by this library. `out_tensor`
/// must be a valid, writable pointer to receive ownership of the returned array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_ts(
    solution: *const SolutionWrapper,
    out_tensor: *mut *mut Tensor,
) -> i32 {
    if solution.is_null() || out_tensor.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_ts");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_ts() {
        Ok(tensor) => {
            unsafe {
                *out_tensor = boxed_tensor(tensor);
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Return the recorded forward sensitivities as a list of arrays.
///
/// # Safety
/// `solution` must be a valid pointer created by this library. `out_sens` and
/// `out_sens_len` must be valid, writable pointers to receive the result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_sens(
    solution: *const SolutionWrapper,
    out_sens: *mut *mut *mut Tensor,
    out_sens_len: *mut usize,
) -> i32 {
    if solution.is_null() || out_sens.is_null() || out_sens_len.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_sens");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_sens() {
        Ok(sens) => {
            let (sens_ptr, sens_len) = boxed_tensor_list(sens);
            unsafe {
                *out_sens = sens_ptr;
                *out_sens_len = sens_len;
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}
