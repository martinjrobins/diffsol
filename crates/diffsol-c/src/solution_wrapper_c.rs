use crate::c_api_utils::{DIFFSOL_BAD_ARG, DIFFSOL_ERR, DIFFSOL_OK};
use crate::host_array::HostArray;
use crate::solution_wrapper::SolutionWrapper;
use crate::{c_error, c_invalid_arg};

fn boxed_host_array(array: HostArray) -> *mut HostArray {
    Box::into_raw(Box::new(array))
}

fn boxed_host_array_list(arrays: Vec<HostArray>) -> (*mut *mut HostArray, usize) {
    let boxed = arrays
        .into_iter()
        .map(boxed_host_array)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut *mut HostArray;
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

/// Return the recorded solution values as a host array.
///
/// # Safety
/// `solution` must be a valid pointer created by this library. `out_array`
/// must be a valid, writable pointer to receive ownership of the returned array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_ys(
    solution: *const SolutionWrapper,
    out_array: *mut *mut HostArray,
) -> i32 {
    if solution.is_null() || out_array.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_ys");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_ys() {
        Ok(array) => {
            unsafe {
                *out_array = boxed_host_array(array);
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Return the recorded solution times as a host array.
///
/// # Safety
/// `solution` must be a valid pointer created by this library. `out_array`
/// must be a valid, writable pointer to receive ownership of the returned array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_ts(
    solution: *const SolutionWrapper,
    out_array: *mut *mut HostArray,
) -> i32 {
    if solution.is_null() || out_array.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_ts");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_ts() {
        Ok(array) => {
            unsafe {
                *out_array = boxed_host_array(array);
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Return the recorded forward sensitivities as a list of host arrays.
///
/// # Safety
/// `solution` must be a valid pointer created by this library. `out_sens` and
/// `out_sens_len` must be valid, writable pointers to receive the result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_sens(
    solution: *const SolutionWrapper,
    out_sens: *mut *mut *mut HostArray,
    out_sens_len: *mut usize,
) -> i32 {
    if solution.is_null() || out_sens.is_null() || out_sens_len.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_sens");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_sens() {
        Ok(sens) => {
            let (sens_ptr, sens_len) = boxed_host_array_list(sens);
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
