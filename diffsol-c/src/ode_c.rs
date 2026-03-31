use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

use crate::c_api_utils::{DIFFSOL_BAD_ARG, DIFFSOL_ERR, DIFFSOL_OK, valid_f64_ptr};
use crate::host_array::HostArray;
use crate::linear_solver_type_c::{linear_solver_from_i32, linear_solver_to_i32};
use crate::matrix_type_c::{matrix_type_from_i32, matrix_type_to_i32};
use crate::ode::OdeWrapper;
use crate::ode_solver_type_c::{ode_solver_from_i32, ode_solver_to_i32};
use crate::scalar_type::ScalarType;
use crate::solution_wrapper::SolutionWrapper;
use crate::{c_error, c_invalid_arg};

fn boxed_host_array(array: HostArray) -> *mut HostArray {
    Box::into_raw(Box::new(array))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_list_free(list: *mut *mut HostArray, len: usize) {
    if list.is_null() {
        c_invalid_arg!("host array list is null");
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(list, len, len));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_new(
    code: *const c_char,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
    rhs_state_deps_ptr: *const usize,
    rhs_state_deps_len: usize,
    rhs_input_deps_ptr: *const usize,
    rhs_input_deps_len: usize,
    mass_state_deps_ptr: *const usize,
    mass_state_deps_len: usize,
) -> *mut OdeWrapper {
    if code.is_null() {
        c_invalid_arg!("code is null");
        return ptr::null_mut();
    }
    let code = unsafe { CStr::from_ptr(code) };
    let _code = match code.to_str() {
        Ok(value) => value,
        Err(_) => {
            c_error!("code is not valid UTF-8");
            return ptr::null_mut();
        }
    };
    let matrix_type = match matrix_type_from_i32(matrix_type) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid matrix_type");
            return ptr::null_mut();
        }
    };
    let linear_solver = match linear_solver_from_i32(linear_solver) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid linear_solver");
            return ptr::null_mut();
        }
    };
    let ode_solver = match ode_solver_from_i32(ode_solver) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid ode_solver");
            return ptr::null_mut();
        }
    };

    // Convert dependency pointers to vectors of tuples
    #[cfg(feature = "external")]
    let rhs_state_deps = if !rhs_state_deps_ptr.is_null() && rhs_state_deps_len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts(
                rhs_state_deps_ptr as *const (usize, usize),
                rhs_state_deps_len,
            );
            slice.to_vec()
        }
    } else {
        Vec::new()
    };

    #[cfg(feature = "external")]
    let rhs_input_deps = if !rhs_input_deps_ptr.is_null() && rhs_input_deps_len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts(
                rhs_input_deps_ptr as *const (usize, usize),
                rhs_input_deps_len,
            );
            slice.to_vec()
        }
    } else {
        Vec::new()
    };

    #[cfg(feature = "external")]
    let mass_state_deps = if !mass_state_deps_ptr.is_null() && mass_state_deps_len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts(
                mass_state_deps_ptr as *const (usize, usize),
                mass_state_deps_len,
            );
            slice.to_vec()
        }
    } else {
        Vec::new()
    };

    let scalar_type = ScalarType::F64;

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    let _ = (
        rhs_state_deps_ptr,
        rhs_state_deps_len,
        rhs_input_deps_ptr,
        rhs_input_deps_len,
        mass_state_deps_ptr,
        mass_state_deps_len,
    );

    #[cfg(all(
        feature = "external",
        not(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))
    ))]
    {
        return match OdeWrapper::new(
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            scalar_type,
            matrix_type,
            linear_solver,
            ode_solver,
        ) {
            Ok(ode) => Box::into_raw(Box::new(ode)),
            Err(err) => {
                c_error!(&format!("{}", err));
                ptr::null_mut()
            }
        };
    }

    #[cfg(all(
        any(feature = "diffsl-cranelift", feature = "diffsl-llvm"),
        not(feature = "external")
    ))]
    {
        return match OdeWrapper::new(_code, scalar_type, matrix_type, linear_solver, ode_solver) {
            Ok(ode) => Box::into_raw(Box::new(ode)),
            Err(err) => {
                c_error!(&format!("{}", err));
                ptr::null_mut()
            }
        };
    }

    #[cfg(not(any(
        all(
            feature = "external",
            not(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))
        ),
        all(
            any(feature = "diffsl-cranelift", feature = "diffsl-llvm"),
            not(feature = "external")
        )
    )))]
    {
        ptr::null_mut()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_free(ode: *mut OdeWrapper) {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(ode));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_ic_options(
    ode: *const OdeWrapper,
    out_options: *mut *mut crate::initial_condition_options::InitialConditionSolverOptions,
) -> i32 {
    if ode.is_null() || out_options.is_null() {
        return c_invalid_arg!("invalid arguments to diffsol_ode_get_ic_options");
    }
    let ode = unsafe { &*ode };
    let options = ode.get_ic_options();
    let boxed = Box::new(options);
    unsafe {
        *out_options = Box::into_raw(boxed);
    }
    DIFFSOL_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_options(
    ode: *const OdeWrapper,
    out_options: *mut *mut crate::ode_options::OdeSolverOptions,
) -> i32 {
    if ode.is_null() || out_options.is_null() {
        return c_invalid_arg!("invalid arguments to diffsol_ode_get_options");
    }
    let ode = unsafe { &*ode };
    let options = ode.get_options();
    let boxed = Box::new(options);
    unsafe {
        *out_options = Box::into_raw(boxed);
    }
    DIFFSOL_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_y0(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    out_array: *mut *mut HostArray,
) -> i32 {
    if ode.is_null() || out_array.is_null() || !valid_f64_ptr(params_ptr, params_len) {
        c_invalid_arg!("invalid arguments to diffsol_ode_y0");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.y0(params) {
        Ok(array) => {
            let boxed = boxed_host_array(array);
            unsafe {
                *out_array = boxed;
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
pub unsafe extern "C" fn diffsol_ode_rhs(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t: f64,
    y_ptr: *const f64,
    y_len: usize,
    out_array: *mut *mut HostArray,
) -> i32 {
    if ode.is_null()
        || out_array.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(y_ptr, y_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_rhs");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let y = HostArray::new_vector(y_ptr as *mut u8, y_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.rhs(params, t, y) {
        Ok(array) => {
            let boxed = boxed_host_array(array);
            unsafe {
                *out_array = boxed;
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
pub unsafe extern "C" fn diffsol_ode_rhs_jac_mul(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t: f64,
    y_ptr: *const f64,
    y_len: usize,
    v_ptr: *const f64,
    v_len: usize,
    out_array: *mut *mut HostArray,
) -> i32 {
    if ode.is_null()
        || out_array.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(y_ptr, y_len)
        || !valid_f64_ptr(v_ptr, v_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_rhs_jac_mul");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let y = HostArray::new_vector(y_ptr as *mut u8, y_len, ScalarType::F64);
    let v = HostArray::new_vector(v_ptr as *mut u8, v_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.rhs_jac_mul(params, t, y, v) {
        Ok(array) => {
            let boxed = boxed_host_array(array);
            unsafe {
                *out_array = boxed;
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
pub unsafe extern "C" fn diffsol_ode_solve(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    final_time: f64,
    solution: *mut SolutionWrapper,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null() || out_solution.is_null() || !valid_f64_ptr(params_ptr, params_len) {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let solution_in = if solution.is_null() {
        None
    } else {
        Some(unsafe { (&*solution).clone() })
    };
    let ode = unsafe { &mut *ode };
    match ode.solve(params, final_time, solution_in) {
        Ok(new_solution) => {
            unsafe {
                if solution.is_null() {
                    *out_solution = Box::into_raw(Box::new(new_solution));
                } else {
                    *solution = new_solution;
                    *out_solution = solution;
                }
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
pub unsafe extern "C" fn diffsol_ode_solve_dense(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
    solution: *mut SolutionWrapper,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null()
        || out_solution.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(t_eval_ptr, t_eval_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve_dense");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let t_eval = HostArray::new_vector(t_eval_ptr as *mut u8, t_eval_len, ScalarType::F64);
    let solution_in = if solution.is_null() {
        None
    } else {
        Some(unsafe { (&*solution).clone() })
    };
    let ode = unsafe { &mut *ode };
    match ode.solve_dense(params, t_eval, solution_in) {
        Ok(new_solution) => {
            unsafe {
                if solution.is_null() {
                    *out_solution = Box::into_raw(Box::new(new_solution));
                } else {
                    *solution = new_solution;
                    *out_solution = solution;
                }
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
pub unsafe extern "C" fn diffsol_ode_solve_fwd_sens(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
    solution: *mut SolutionWrapper,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null()
        || out_solution.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(t_eval_ptr, t_eval_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve_fwd_sens");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let t_eval = HostArray::new_vector(t_eval_ptr as *mut u8, t_eval_len, ScalarType::F64);
    let solution_in = if solution.is_null() {
        None
    } else {
        Some(unsafe { (&*solution).clone() })
    };
    let ode = unsafe { &mut *ode };
    match ode.solve_fwd_sens(params, t_eval, solution_in) {
        Ok(new_solution) => {
            unsafe {
                if solution.is_null() {
                    *out_solution = Box::into_raw(Box::new(new_solution));
                } else {
                    *solution = new_solution;
                    *out_solution = solution;
                }
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
pub unsafe extern "C" fn diffsol_ode_solve_sum_squares_adj(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    data_ptr: *const f64,
    data_rows: usize,
    data_cols: usize,
    data_row_stride: usize,
    data_col_stride: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
    out_value: *mut f64,
    out_sens: *mut *mut HostArray,
) -> i32 {
    if ode.is_null()
        || out_value.is_null()
        || out_sens.is_null()
        || data_ptr.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(t_eval_ptr, t_eval_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve_sum_squares_adj");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let t_eval = HostArray::new_vector(t_eval_ptr as *mut u8, t_eval_len, ScalarType::F64);
    let data = HostArray::new_col_major(
        data_ptr as *mut u8,
        data_rows,
        data_cols,
        data_row_stride as isize,
        data_col_stride as isize,
        ScalarType::F64,
    );
    let ode = unsafe { &mut *ode };
    match ode.solve_sum_squares_adj(params, data, t_eval) {
        Ok((value, sens)) => {
            let sens_boxed = boxed_host_array(sens);
            unsafe {
                *out_value = value;
                *out_sens = sens_boxed;
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
pub unsafe extern "C" fn diffsol_ode_get_matrix_type(ode: *const OdeWrapper) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return -1;
    }
    let ode = unsafe { &*ode };
    match ode.get_matrix_type() {
        Ok(value) => matrix_type_to_i32(value),
        Err(err) => {
            c_error!(&format!("{}", err));
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_ode_solver(ode: *const OdeWrapper) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return -1;
    }
    let ode = unsafe { &*ode };
    match ode.get_ode_solver() {
        Ok(value) => ode_solver_to_i32(value),
        Err(err) => {
            c_error!(&format!("{}", err));
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_ode_solver(ode: *mut OdeWrapper, value: i32) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let value = match ode_solver_from_i32(value) {
        Some(v) => v,
        None => {
            c_invalid_arg!("invalid ode_solver");
            return DIFFSOL_BAD_ARG;
        }
    };
    let ode = unsafe { &mut *ode };
    match ode.set_ode_solver(value) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_linear_solver(ode: *const OdeWrapper) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return -1;
    }
    let ode = unsafe { &*ode };
    match ode.get_linear_solver() {
        Ok(value) => linear_solver_to_i32(value),
        Err(err) => {
            c_error!(&format!("{}", err));
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_linear_solver(ode: *mut OdeWrapper, value: i32) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let value = match linear_solver_from_i32(value) {
        Some(v) => v,
        None => {
            c_invalid_arg!("invalid linear_solver");
            return DIFFSOL_BAD_ARG;
        }
    };
    let ode = unsafe { &mut *ode };
    match ode.set_linear_solver(value) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_rtol(ode: *const OdeWrapper, out_value: *mut f64) -> i32 {
    if ode.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_rtol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_rtol() {
        Ok(value) => {
            unsafe {
                *out_value = value;
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_rtol(ode: *mut OdeWrapper, value: f64) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_rtol(value) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_atol(ode: *const OdeWrapper, out_value: *mut f64) -> i32 {
    if ode.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_atol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_atol() {
        Ok(value) => {
            unsafe {
                *out_value = value;
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_atol(ode: *mut OdeWrapper, value: f64) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_atol(value) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[cfg(all(test, feature = "external-f64"))]
mod tests {
    use std::ptr;

    use crate::initial_condition_options::InitialConditionSolverOptions;
    use crate::linear_solver_type::LinearSolverType;
    use crate::linear_solver_type_c::{
        diffsol_linear_solver_type_count, diffsol_linear_solver_type_is_valid,
        diffsol_linear_solver_type_name, linear_solver_to_i32,
    };
    use crate::matrix_type::MatrixType;
    use crate::ode_options::OdeSolverOptions;
    use crate::ode_options_c::{
        diffsol_ode_options_free, diffsol_ode_options_get_max_nonlinear_solver_iterations,
        diffsol_ode_options_get_min_timestep,
        diffsol_ode_options_set_max_nonlinear_solver_iterations,
        diffsol_ode_options_set_min_timestep,
    };
    use crate::ode_solver_type::OdeSolverType;
    use crate::ode_solver_type_c::{
        diffsol_ode_solver_type_count, diffsol_ode_solver_type_is_valid,
        diffsol_ode_solver_type_name, ode_solver_to_i32,
    };
    use crate::scalar_type::ScalarType;
    use crate::scalar_type_c::{
        diffsol_scalar_type_count, diffsol_scalar_type_is_valid, diffsol_scalar_type_name,
        scalar_type_to_i32,
    };
    use crate::solution_wrapper_c::{
        diffsol_solution_wrapper_get_current_state, diffsol_solution_wrapper_get_sens,
        diffsol_solution_wrapper_get_ts, diffsol_solution_wrapper_get_ys,
        diffsol_solution_wrapper_set_current_state,
    };
    use crate::test_support::{
        ASSERT_TOL, assert_close, assert_last_error_contains, c_string, clear_last_error,
        dummy_code, ffi_free_solution, ffi_read_host_array_list_matrices,
        ffi_read_host_array_matrix, ffi_read_host_array_vector, find_time_window, logistic_state,
        mass_state_deps, rhs_input_deps, rhs_state_deps,
    };
    use crate::{
        initial_condition_options_c::{
            diffsol_ic_options_free, diffsol_ic_options_get_max_linesearch_iterations,
            diffsol_ic_options_get_use_linesearch,
            diffsol_ic_options_set_max_linesearch_iterations,
            diffsol_ic_options_set_use_linesearch,
        },
        matrix_type_c::{
            diffsol_matrix_type_count, diffsol_matrix_type_is_valid, diffsol_matrix_type_name,
            matrix_type_to_i32,
        },
    };

    use super::*;

    unsafe fn make_ode_ptr(
        matrix_type: i32,
        linear_solver: i32,
        ode_solver: i32,
    ) -> *mut OdeWrapper {
        let code = dummy_code();
        let rhs_state_deps = rhs_state_deps();
        let rhs_input_deps = rhs_input_deps();
        let mass_state_deps = mass_state_deps();
        unsafe {
            diffsol_ode_new(
                code.as_ptr(),
                matrix_type,
                linear_solver,
                ode_solver,
                rhs_state_deps.as_ptr() as *const usize,
                rhs_state_deps.len(),
                rhs_input_deps.as_ptr() as *const usize,
                rhs_input_deps.len(),
                mass_state_deps.as_ptr() as *const usize,
                mass_state_deps.len(),
            )
        }
    }

    #[test]
    fn c_api_reports_enum_metadata() {
        clear_last_error();
        unsafe {
            assert_eq!(diffsol_matrix_type_count(), 3);
            assert_eq!(diffsol_ode_solver_type_count(), 4);
            assert_eq!(diffsol_linear_solver_type_count(), 3);
            assert_eq!(diffsol_scalar_type_count(), 2);

            assert_eq!(
                c_string(diffsol_matrix_type_name(matrix_type_to_i32(
                    MatrixType::NalgebraDense
                ))),
                "nalgebra_dense"
            );
            assert_eq!(
                c_string(diffsol_ode_solver_type_name(ode_solver_to_i32(
                    OdeSolverType::Bdf
                ))),
                "bdf"
            );
            assert_eq!(
                c_string(diffsol_linear_solver_type_name(linear_solver_to_i32(
                    LinearSolverType::Default
                ))),
                "default"
            );
            assert_eq!(
                c_string(diffsol_scalar_type_name(scalar_type_to_i32(
                    ScalarType::F64
                ))),
                "f64"
            );
        }
    }

    #[test]
    fn c_api_invalid_enums_set_last_error() {
        clear_last_error();
        unsafe {
            assert_eq!(diffsol_matrix_type_is_valid(99), 0);
            assert_last_error_contains("invalid matrix_type");
            clear_last_error();

            assert_eq!(diffsol_ode_solver_type_is_valid(99), 0);
            assert_last_error_contains("invalid ode_solver_type");
            clear_last_error();

            assert_eq!(diffsol_linear_solver_type_is_valid(99), 0);
            assert_last_error_contains("invalid linear_solver_type");
            clear_last_error();

            assert_eq!(diffsol_scalar_type_is_valid(99), 0);
            assert_last_error_contains("invalid scalar_type");
        }
    }

    #[test]
    fn c_api_rejects_invalid_ode_arguments() {
        clear_last_error();
        unsafe {
            let mut out_array = ptr::null_mut();
            let status = diffsol_ode_y0(ptr::null_mut(), ptr::null(), 0, &mut out_array);
            assert_eq!(status, DIFFSOL_BAD_ARG);
            assert!(out_array.is_null());
            assert_last_error_contains("invalid arguments to diffsol_ode_y0");
            clear_last_error();

            let ode = make_ode_ptr(
                99,
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            );
            assert!(ode.is_null());
            assert_last_error_contains("invalid matrix_type");
        }
    }

    #[test]
    fn c_api_full_lifecycle_matches_external_logistic_model() {
        clear_last_error();
        unsafe {
            let ode = make_ode_ptr(
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            );
            assert!(!ode.is_null());

            assert_eq!(
                diffsol_ode_get_matrix_type(ode),
                matrix_type_to_i32(MatrixType::NalgebraDense)
            );
            assert_eq!(
                diffsol_ode_get_ode_solver(ode),
                ode_solver_to_i32(OdeSolverType::Bdf)
            );
            assert_eq!(
                diffsol_ode_get_linear_solver(ode),
                linear_solver_to_i32(LinearSolverType::Default)
            );

            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Tsit45)),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ode_get_ode_solver(ode),
                ode_solver_to_i32(OdeSolverType::Tsit45)
            );
            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Bdf)),
                DIFFSOL_OK
            );

            assert_eq!(diffsol_ode_set_rtol(ode, 1e-8), DIFFSOL_OK);
            assert_eq!(diffsol_ode_set_atol(ode, 1e-8), DIFFSOL_OK);
            let mut rtol = 0.0;
            let mut atol = 0.0;
            assert_eq!(diffsol_ode_get_rtol(ode, &mut rtol), DIFFSOL_OK);
            assert_eq!(diffsol_ode_get_atol(ode, &mut atol), DIFFSOL_OK);
            assert_close(rtol, 1e-8, ASSERT_TOL, "rtol roundtrip");
            assert_close(atol, 1e-8, ASSERT_TOL, "atol roundtrip");

            let mut ic_options: *mut InitialConditionSolverOptions = ptr::null_mut();
            assert_eq!(diffsol_ode_get_ic_options(ode, &mut ic_options), DIFFSOL_OK);
            assert!(!ic_options.is_null());
            let mut use_linesearch = 0;
            let mut max_linesearch_iterations = 0usize;
            assert_eq!(
                diffsol_ic_options_get_use_linesearch(ic_options, &mut use_linesearch),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ic_options_set_use_linesearch(ic_options, 1),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ic_options_get_use_linesearch(ic_options, &mut use_linesearch),
                DIFFSOL_OK
            );
            assert_eq!(use_linesearch, 1);
            assert_eq!(
                diffsol_ic_options_set_max_linesearch_iterations(ic_options, 23),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ic_options_get_max_linesearch_iterations(
                    ic_options,
                    &mut max_linesearch_iterations
                ),
                DIFFSOL_OK
            );
            assert_eq!(max_linesearch_iterations, 23);
            diffsol_ic_options_free(ic_options);

            let mut ode_options: *mut OdeSolverOptions = ptr::null_mut();
            assert_eq!(diffsol_ode_get_options(ode, &mut ode_options), DIFFSOL_OK);
            assert!(!ode_options.is_null());
            let mut max_nonlinear_iterations = 0usize;
            let mut min_timestep = 0.0;
            assert_eq!(
                diffsol_ode_options_set_max_nonlinear_solver_iterations(ode_options, 17),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ode_options_get_max_nonlinear_solver_iterations(
                    ode_options,
                    &mut max_nonlinear_iterations
                ),
                DIFFSOL_OK
            );
            assert_eq!(max_nonlinear_iterations, 17);
            assert_eq!(
                diffsol_ode_options_set_min_timestep(ode_options, 1e-4),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ode_options_get_min_timestep(ode_options, &mut min_timestep),
                DIFFSOL_OK
            );
            assert_close(min_timestep, 1e-4, ASSERT_TOL, "min_timestep roundtrip");
            diffsol_ode_options_free(ode_options);

            let params = [2.0f64];
            let y = [0.25f64];
            let v = [3.0f64];

            let mut y0_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_y0(ode, params.as_ptr(), params.len(), &mut y0_ptr),
                DIFFSOL_OK
            );
            assert_eq!(ffi_read_host_array_vector(y0_ptr), vec![1.0]);

            let mut rhs_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_rhs(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    0.0,
                    y.as_ptr(),
                    y.len(),
                    &mut rhs_ptr,
                ),
                DIFFSOL_OK
            );
            assert_close(
                ffi_read_host_array_vector(rhs_ptr)[0],
                0.375,
                ASSERT_TOL,
                "ffi rhs",
            );

            let mut rhs_jac_mul_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_rhs_jac_mul(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    0.0,
                    y.as_ptr(),
                    y.len(),
                    v.as_ptr(),
                    v.len(),
                    &mut rhs_jac_mul_ptr,
                ),
                DIFFSOL_OK
            );
            assert_close(
                ffi_read_host_array_vector(rhs_jac_mul_ptr)[0],
                3.0,
                ASSERT_TOL,
                "ffi rhs_jac_mul",
            );

            let mut solve_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
            assert_eq!(
                diffsol_ode_solve(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    1e-9,
                    ptr::null_mut(),
                    &mut solve_solution_ptr
                ),
                DIFFSOL_OK
            );
            assert!(!solve_solution_ptr.is_null());

            let mut current_state_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_solution_wrapper_get_current_state(
                    solve_solution_ptr,
                    &mut current_state_ptr
                ),
                DIFFSOL_OK
            );
            assert_eq!(ffi_read_host_array_vector(current_state_ptr), vec![1.0]);
            ffi_free_solution(solve_solution_ptr);

            let mut solution_ptr: *mut SolutionWrapper = ptr::null_mut();
            let seed_t_eval = [1e-9f64];
            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Tsit45)),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ode_solve_dense(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    seed_t_eval.as_ptr(),
                    seed_t_eval.len(),
                    ptr::null_mut(),
                    &mut solution_ptr,
                ),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_solution_wrapper_set_current_state(solution_ptr, [0.1].as_ptr(), 1),
                DIFFSOL_OK
            );
            let mut updated_state_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_solution_wrapper_get_current_state(solution_ptr, &mut updated_state_ptr),
                DIFFSOL_OK
            );
            assert_close(
                ffi_read_host_array_vector(updated_state_ptr)[0],
                0.1,
                ASSERT_TOL,
                "ffi current state",
            );

            let t_eval = [0.25f64, 0.5f64, 1.0f64];
            assert_eq!(
                diffsol_ode_solve_dense(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    t_eval.as_ptr(),
                    t_eval.len(),
                    solution_ptr,
                    &mut solution_ptr,
                ),
                DIFFSOL_OK
            );
            let mut ys_ptr = ptr::null_mut();
            let mut ts_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_solution_wrapper_get_ys(solution_ptr, &mut ys_ptr),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_solution_wrapper_get_ts(solution_ptr, &mut ts_ptr),
                DIFFSOL_OK
            );
            let (rows, cols, ys) = ffi_read_host_array_matrix(ys_ptr);
            let ts = ffi_read_host_array_vector(ts_ptr);
            assert_eq!(rows, 1);
            assert_eq!(cols, ts.len());
            let start = find_time_window(&ts, &t_eval, ASSERT_TOL);
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(ts[start + i], t, ASSERT_TOL, "ffi solution time");
                assert_close(
                    ys[start + i],
                    logistic_state(0.1, 2.0, t),
                    5e-4,
                    "ffi solution value",
                );
            }
            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Bdf)),
                DIFFSOL_OK
            );

            let analysis_ode = make_ode_ptr(
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            );
            assert!(!analysis_ode.is_null());

            let mut sens_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
            assert_eq!(
                diffsol_ode_solve_fwd_sens(
                    analysis_ode,
                    params.as_ptr(),
                    params.len(),
                    t_eval.as_ptr(),
                    t_eval.len(),
                    ptr::null_mut(),
                    &mut sens_solution_ptr,
                ),
                DIFFSOL_OK
            );
            let mut sens_list = ptr::null_mut();
            let mut sens_len = 0usize;
            assert_eq!(
                diffsol_solution_wrapper_get_sens(sens_solution_ptr, &mut sens_list, &mut sens_len),
                DIFFSOL_OK
            );
            let sens_values = ffi_read_host_array_list_matrices(sens_list, sens_len);
            assert_eq!(sens_values.len(), 1);
            assert_eq!(sens_values[0].0, 1);
            assert_eq!(sens_values[0].1, t_eval.len());
            for (i, value) in sens_values[0].2.iter().enumerate() {
                assert_close(*value, 0.0, ASSERT_TOL, &format!("ffi sensitivity[{i}]"));
            }

            let adjoint_data = [0.0f64, 0.25f64, 0.5f64, 1.0f64];
            let adjoint_t_eval = [0.0f64, 0.25f64, 0.5f64, 1.0f64];
            let mut objective = 0.0;
            let mut adjoint_grad_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_solve_sum_squares_adj(
                    analysis_ode,
                    params.as_ptr(),
                    params.len(),
                    adjoint_data.as_ptr(),
                    1,
                    adjoint_t_eval.len(),
                    1,
                    1,
                    adjoint_t_eval.as_ptr(),
                    adjoint_t_eval.len(),
                    &mut objective,
                    &mut adjoint_grad_ptr,
                ),
                DIFFSOL_OK
            );
            assert_close(objective, 0.0, ASSERT_TOL, "ffi adjoint objective");
            let grad = ffi_read_host_array_vector(adjoint_grad_ptr);
            assert_eq!(grad.len(), 1);
            assert_close(grad[0], 0.0, ASSERT_TOL, "ffi adjoint gradient");

            ffi_free_solution(sens_solution_ptr);
            diffsol_ode_free(analysis_ode);
            ffi_free_solution(solution_ptr);
            diffsol_ode_free(ode);
        }
    }
}

#[cfg(all(
    test,
    any(feature = "diffsl-cranelift", feature = "diffsl-llvm"),
    not(feature = "external")
))]
mod jit_tests {
    use std::ptr;

    use crate::linear_solver_type::LinearSolverType;
    use crate::linear_solver_type_c::linear_solver_to_i32;
    use crate::matrix_type::MatrixType;
    use crate::matrix_type_c::matrix_type_to_i32;
    use crate::ode_solver_type::OdeSolverType;
    use crate::ode_solver_type_c::ode_solver_to_i32;
    #[cfg(feature = "diffsl-llvm")]
    use crate::solution_wrapper_c::diffsol_solution_wrapper_get_sens;
    use crate::solution_wrapper_c::{
        diffsol_solution_wrapper_get_ts, diffsol_solution_wrapper_get_ys,
        diffsol_solution_wrapper_set_current_state,
    };
    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::ffi_read_host_array_list_matrices;
    use crate::test_support::{
        ASSERT_TOL, assert_close, clear_last_error, ffi_free_solution, ffi_read_host_array_matrix,
        ffi_read_host_array_vector, find_time_window, logistic_diffsl_code_cstring, logistic_state,
    };

    use super::*;

    unsafe fn make_ode_ptr(
        matrix_type: i32,
        linear_solver: i32,
        ode_solver: i32,
    ) -> *mut OdeWrapper {
        let code = logistic_diffsl_code_cstring();
        unsafe {
            diffsol_ode_new(
                code.as_ptr(),
                matrix_type,
                linear_solver,
                ode_solver,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
            )
        }
    }

    #[test]
    fn c_api_full_lifecycle_matches_jit_logistic_model() {
        clear_last_error();
        unsafe {
            let ode = make_ode_ptr(
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            );
            assert!(!ode.is_null());

            assert_eq!(
                diffsol_ode_get_matrix_type(ode),
                matrix_type_to_i32(MatrixType::NalgebraDense)
            );
            assert_eq!(
                diffsol_ode_get_ode_solver(ode),
                ode_solver_to_i32(OdeSolverType::Bdf)
            );
            assert_eq!(
                diffsol_ode_get_linear_solver(ode),
                linear_solver_to_i32(LinearSolverType::Default)
            );

            let params = [2.0f64];
            let y = [0.25f64];
            let v = [3.0f64];

            let mut y0_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_y0(ode, params.as_ptr(), params.len(), &mut y0_ptr),
                DIFFSOL_OK
            );
            assert_eq!(ffi_read_host_array_vector(y0_ptr), vec![1.0]);

            let mut rhs_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_rhs(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    0.0,
                    y.as_ptr(),
                    y.len(),
                    &mut rhs_ptr,
                ),
                DIFFSOL_OK
            );
            assert_close(
                ffi_read_host_array_vector(rhs_ptr)[0],
                0.375,
                ASSERT_TOL,
                "jit ffi rhs",
            );

            let mut rhs_jac_mul_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_rhs_jac_mul(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    0.0,
                    y.as_ptr(),
                    y.len(),
                    v.as_ptr(),
                    v.len(),
                    &mut rhs_jac_mul_ptr,
                ),
                DIFFSOL_OK
            );
            assert_close(
                ffi_read_host_array_vector(rhs_jac_mul_ptr)[0],
                3.0,
                ASSERT_TOL,
                "jit ffi rhs_jac_mul",
            );

            let mut solution_ptr: *mut SolutionWrapper = ptr::null_mut();
            let seed_t_eval = [1e-9f64];
            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Tsit45)),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_ode_solve_dense(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    seed_t_eval.as_ptr(),
                    seed_t_eval.len(),
                    ptr::null_mut(),
                    &mut solution_ptr,
                ),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_solution_wrapper_set_current_state(solution_ptr, [0.1].as_ptr(), 1),
                DIFFSOL_OK
            );

            let t_eval = [0.25f64, 0.5f64, 1.0f64];
            assert_eq!(
                diffsol_ode_solve_dense(
                    ode,
                    params.as_ptr(),
                    params.len(),
                    t_eval.as_ptr(),
                    t_eval.len(),
                    solution_ptr,
                    &mut solution_ptr,
                ),
                DIFFSOL_OK
            );
            let mut ys_ptr = ptr::null_mut();
            let mut ts_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_solution_wrapper_get_ys(solution_ptr, &mut ys_ptr),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_solution_wrapper_get_ts(solution_ptr, &mut ts_ptr),
                DIFFSOL_OK
            );
            let (rows, cols, ys) = ffi_read_host_array_matrix(ys_ptr);
            let ts = ffi_read_host_array_vector(ts_ptr);
            assert_eq!(rows, 1);
            assert_eq!(cols, ts.len());
            let start = find_time_window(&ts, &t_eval, ASSERT_TOL);
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(ts[start + i], t, ASSERT_TOL, "jit ffi solution time");
                assert_close(
                    ys[start + i],
                    logistic_state(0.1, 2.0, t),
                    5e-4,
                    "jit ffi solution value",
                );
            }
            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Bdf)),
                DIFFSOL_OK
            );

            #[cfg(feature = "diffsl-llvm")]
            {
                let analysis_ode = make_ode_ptr(
                    matrix_type_to_i32(MatrixType::NalgebraDense),
                    linear_solver_to_i32(LinearSolverType::Default),
                    ode_solver_to_i32(OdeSolverType::Bdf),
                );
                assert!(!analysis_ode.is_null());

                let mut sens_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_solve_fwd_sens(
                        analysis_ode,
                        params.as_ptr(),
                        params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
                        ptr::null_mut(),
                        &mut sens_solution_ptr,
                    ),
                    DIFFSOL_OK
                );
                let mut sens_list = ptr::null_mut();
                let mut sens_len = 0usize;
                assert_eq!(
                    diffsol_solution_wrapper_get_sens(
                        sens_solution_ptr,
                        &mut sens_list,
                        &mut sens_len
                    ),
                    DIFFSOL_OK
                );
                let sens_values = ffi_read_host_array_list_matrices(sens_list, sens_len);
                assert_eq!(sens_values.len(), 1);
                assert_eq!(sens_values[0].0, 1);
                assert_eq!(sens_values[0].1, t_eval.len());
                for (i, value) in sens_values[0].2.iter().enumerate() {
                    assert_close(
                        *value,
                        0.0,
                        ASSERT_TOL,
                        &format!("jit ffi sensitivity[{i}]"),
                    );
                }

                let adjoint_data = [0.0f64, 0.25f64, 0.5f64, 1.0f64];
                let adjoint_t_eval = [0.0f64, 0.25f64, 0.5f64, 1.0f64];
                let mut objective = 0.0;
                let mut adjoint_grad_ptr = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_solve_sum_squares_adj(
                        analysis_ode,
                        params.as_ptr(),
                        params.len(),
                        adjoint_data.as_ptr(),
                        1,
                        adjoint_t_eval.len(),
                        1,
                        1,
                        adjoint_t_eval.as_ptr(),
                        adjoint_t_eval.len(),
                        &mut objective,
                        &mut adjoint_grad_ptr,
                    ),
                    DIFFSOL_OK
                );
                assert_close(objective, 0.0, ASSERT_TOL, "jit ffi adjoint objective");
                let grad = ffi_read_host_array_vector(adjoint_grad_ptr);
                assert_eq!(grad.len(), 1);
                assert_close(grad[0], 0.0, ASSERT_TOL, "jit ffi adjoint gradient");

                ffi_free_solution(sens_solution_ptr);
                diffsol_ode_free(analysis_ode);
            }
            ffi_free_solution(solution_ptr);
            diffsol_ode_free(ode);
        }
    }
}
