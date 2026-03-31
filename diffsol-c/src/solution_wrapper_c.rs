use crate::c_api_utils::{DIFFSOL_BAD_ARG, DIFFSOL_ERR, DIFFSOL_OK, valid_f64_ptr};
use crate::host_array::HostArray;
use crate::solution_wrapper::SolutionWrapper;
use crate::{c_error, c_invalid_arg};

fn boxed_host_array(array: HostArray) -> *mut HostArray {
    Box::into_raw(Box::new(array))
}

fn boxed_host_array_list(arrays: Vec<HostArray>) -> (*mut *mut HostArray, usize) {
    let mut boxed: Vec<*mut HostArray> = arrays.into_iter().map(boxed_host_array).collect();
    let len = boxed.len();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    (ptr, len)
}

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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_set_current_state(
    solution: *mut SolutionWrapper,
    y_ptr: *const f64,
    y_len: usize,
) -> i32 {
    if solution.is_null() || !valid_f64_ptr(y_ptr, y_len) {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_set_current_state");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &mut *solution };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, y_len) };
    match solution.set_current_state(y) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_solution_wrapper_get_current_state(
    solution: *const SolutionWrapper,
    out_array: *mut *mut HostArray,
) -> i32 {
    if solution.is_null() || out_array.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_solution_wrapper_get_current_state");
        return DIFFSOL_BAD_ARG;
    }
    let solution = unsafe { &*solution };
    match solution.get_current_state() {
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
