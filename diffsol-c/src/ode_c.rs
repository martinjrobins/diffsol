#[cfg(any(
    feature = "diffsl-external-dynamic",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm",
))]
use std::ffi::CStr;
#[cfg(any(
    feature = "diffsl-external-dynamic",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm",
))]
use std::os::raw::c_char;
#[cfg(feature = "diffsl-external-dynamic")]
use std::path::PathBuf;
use std::ptr;

use crate::c_api_utils::{valid_f64_ptr, DIFFSOL_BAD_ARG, DIFFSOL_ERR, DIFFSOL_OK};
use crate::host_array::HostArray;
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use crate::jit_c::jit_backend_from_i32;
use crate::linear_solver_type_c::{linear_solver_from_i32, linear_solver_to_i32};
use crate::matrix_type_c::matrix_type_from_i32;
use crate::matrix_type_c::matrix_type_to_i32;
use crate::ode::OdeWrapper;
use crate::ode_solver_type_c::{ode_solver_from_i32, ode_solver_to_i32};
use crate::scalar_type::ScalarType;
use crate::solution_wrapper::SolutionWrapper;
use crate::{c_error, c_invalid_arg};

fn boxed_host_array(array: HostArray) -> *mut HostArray {
    Box::into_raw(Box::new(array))
}

#[cfg(any(feature = "external", feature = "diffsl-external-dynamic"))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// C-compatible dependency pair used by the FFI constructors.
///
/// Memory layout guarantees:
/// - `#[repr(C)]` preserves field order exactly as declared.
/// - The first machine word is `row`, followed by `col`.
/// - Each field is a `usize`, so size/alignment are target-dependent
///   (32-bit targets: 4 bytes, 64-bit targets: 8 bytes).
/// - Arrays of `DiffsolDepPair` are contiguous in memory and can be passed as
///   pointer/length (`*const DiffsolDepPair`, `usize`) to this API.
pub struct DiffsolDepPair {
    pub row: usize,
    pub col: usize,
}

fn parse_ode_new_common_args(
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
) -> Option<(
    crate::matrix_type::MatrixType,
    crate::linear_solver_type::LinearSolverType,
    crate::ode_solver_type::OdeSolverType,
)> {
    let matrix_type = match matrix_type_from_i32(matrix_type) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid matrix_type");
            return None;
        }
    };
    let linear_solver = match linear_solver_from_i32(linear_solver) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid linear_solver");
            return None;
        }
    };
    let ode_solver = match ode_solver_from_i32(ode_solver) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid ode_solver");
            return None;
        }
    };
    Some((matrix_type, linear_solver, ode_solver))
}

#[cfg(any(feature = "external", feature = "diffsl-external-dynamic"))]
unsafe fn dependency_pairs_from_raw_parts(
    deps_ptr: *const DiffsolDepPair,
    deps_len: usize,
) -> Vec<(usize, usize)> {
    if deps_ptr.is_null() || deps_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(deps_ptr, deps_len) }
            .iter()
            .map(|pair| (pair.row, pair.col))
            .collect()
    }
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
fn parse_ode_new_jit_args(
    code: *const c_char,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
) -> Option<(
    String,
    crate::matrix_type::MatrixType,
    crate::linear_solver_type::LinearSolverType,
    crate::ode_solver_type::OdeSolverType,
)> {
    if code.is_null() {
        c_invalid_arg!("code is null");
        return None;
    }
    let code = unsafe { CStr::from_ptr(code) };
    let code = match code.to_str() {
        Ok(value) => value.to_owned(),
        Err(_) => {
            c_error!("code is not valid UTF-8");
            return None;
        }
    };
    let (matrix_type, linear_solver, ode_solver) =
        parse_ode_new_common_args(matrix_type, linear_solver, ode_solver)?;
    Some((code, matrix_type, linear_solver, ode_solver))
}

#[cfg(feature = "diffsl-external-dynamic")]
fn parse_ode_new_external_dynamic_args(
    path: *const c_char,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
) -> Option<(
    PathBuf,
    crate::matrix_type::MatrixType,
    crate::linear_solver_type::LinearSolverType,
    crate::ode_solver_type::OdeSolverType,
)> {
    if path.is_null() {
        c_invalid_arg!("path is null");
        return None;
    }
    let path = unsafe { CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(value) => PathBuf::from(value),
        Err(_) => {
            c_error!("path is not valid UTF-8");
            return None;
        }
    };
    let (matrix_type, linear_solver, ode_solver) =
        parse_ode_new_common_args(matrix_type, linear_solver, ode_solver)?;
    Some((path, matrix_type, linear_solver, ode_solver))
}

/// Free a list of host arrays previously returned by this library.
///
/// # Safety
/// `list` must be either null or a pointer returned by this library for a list
/// of length `len`. Each pointed-to element remains owned separately.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_host_array_list_free(list: *mut *mut HostArray, len: usize) {
    if list.is_null() {
        c_invalid_arg!("host array list is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(list, len)));
    }
}

#[cfg(feature = "external")]
/// Construct an external-backed ODE wrapper.
///
/// # Safety
/// Dependency pointers must be either null with length `0` or point to valid
/// memory containing [`DiffsolDepPair`] values for the specified lengths for the
/// duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_new_external(
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
    rhs_state_deps_ptr: *const DiffsolDepPair,
    rhs_state_deps_len: usize,
    rhs_input_deps_ptr: *const DiffsolDepPair,
    rhs_input_deps_len: usize,
    mass_state_deps_ptr: *const DiffsolDepPair,
    mass_state_deps_len: usize,
) -> *mut OdeWrapper {
    let Some((matrix_type, linear_solver, ode_solver)) =
        parse_ode_new_common_args(matrix_type, linear_solver, ode_solver)
    else {
        return ptr::null_mut();
    };

    let rhs_state_deps =
        unsafe { dependency_pairs_from_raw_parts(rhs_state_deps_ptr, rhs_state_deps_len) };
    let rhs_input_deps =
        unsafe { dependency_pairs_from_raw_parts(rhs_input_deps_ptr, rhs_input_deps_len) };
    let mass_state_deps =
        unsafe { dependency_pairs_from_raw_parts(mass_state_deps_ptr, mass_state_deps_len) };

    let scalar_type = ScalarType::F64;
    match OdeWrapper::new_external(
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
    }
}

#[cfg(feature = "diffsl-external-dynamic")]
/// Construct a dynamic-library-backed ODE wrapper.
///
/// # Safety
/// `path` must be a valid, null-terminated UTF-8 string for the duration of
/// this call. Dependency pointers must be either null with length `0` or point
/// to valid memory containing [`DiffsolDepPair`] values for the specified lengths
/// for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_new_external_dynamic(
    path: *const c_char,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
    rhs_state_deps_ptr: *const DiffsolDepPair,
    rhs_state_deps_len: usize,
    rhs_input_deps_ptr: *const DiffsolDepPair,
    rhs_input_deps_len: usize,
    mass_state_deps_ptr: *const DiffsolDepPair,
    mass_state_deps_len: usize,
) -> *mut OdeWrapper {
    let Some((path, matrix_type, linear_solver, ode_solver)) =
        parse_ode_new_external_dynamic_args(path, matrix_type, linear_solver, ode_solver)
    else {
        return ptr::null_mut();
    };

    let rhs_state_deps =
        unsafe { dependency_pairs_from_raw_parts(rhs_state_deps_ptr, rhs_state_deps_len) };
    let rhs_input_deps =
        unsafe { dependency_pairs_from_raw_parts(rhs_input_deps_ptr, rhs_input_deps_len) };
    let mass_state_deps =
        unsafe { dependency_pairs_from_raw_parts(mass_state_deps_ptr, mass_state_deps_len) };

    let scalar_type = ScalarType::F64;
    match OdeWrapper::new_external_dynamic(
        path,
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
    }
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
/// Construct a JIT-backed ODE wrapper from DiffSL source code.
///
/// # Safety
/// `code` must be a valid, null-terminated UTF-8 string for the duration of
/// this call. The backend and solver enum values must be valid values defined by
/// this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_new_jit(
    code: *const c_char,
    jit_backend: i32,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
) -> *mut OdeWrapper {
    let Some((code, matrix_type, linear_solver, ode_solver)) =
        parse_ode_new_jit_args(code, matrix_type, linear_solver, ode_solver)
    else {
        return ptr::null_mut();
    };
    let jit_backend = match jit_backend_from_i32(jit_backend) {
        Some(value) => value,
        None => {
            c_invalid_arg!("invalid jit_backend_type");
            return ptr::null_mut();
        }
    };
    let scalar_type = ScalarType::F64;
    match OdeWrapper::new_jit(
        &code,
        jit_backend,
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
    }
}

/// Free an ODE wrapper previously returned by this library.
///
/// # Safety
/// `ode` must be either null or a pointer returned by this library that has not
/// already been freed.
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

/// Return a handle to the initial-condition solver options for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_options` must be
/// a valid, writable pointer to receive ownership of the returned options
/// object.
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

/// Return a handle to the ODE solver options for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_options` must be
/// a valid, writable pointer to receive ownership of the returned options
/// object.
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

/// Evaluate the initial condition vector for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// must be either null with `params_len == 0` or point to `params_len`
/// readable `f64` values. `out_array` must be a valid, writable pointer.
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

/// Evaluate the ODE right-hand side at a given time and state.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// and `y_ptr` must point to readable `f64` buffers of the specified lengths,
/// unless the corresponding length is zero. `out_array` must be writable.
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

/// Evaluate the ODE Jacobian-vector product at a given time and state.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`,
/// `y_ptr`, and `v_ptr` must point to readable `f64` buffers of the specified
/// lengths, unless the corresponding length is zero. `out_array` must be
/// writable.
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

/// Solve an ODE up to a final time.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// must point to `params_len` readable `f64` values unless `params_len == 0`.
/// `out_solution` must be a valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solve(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    final_time: f64,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null() || out_solution.is_null() || !valid_f64_ptr(params_ptr, params_len) {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.solve(params, final_time) {
        Ok(new_solution) => {
            unsafe {
                *out_solution = Box::into_raw(Box::new(new_solution));
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Solve a hybrid ODE up to a final time, automatically applying resets after roots.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// must point to `params_len` readable `f64` values unless `params_len == 0`.
/// `out_solution` must be a valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solve_hybrid(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    final_time: f64,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null() || out_solution.is_null() || !valid_f64_ptr(params_ptr, params_len) {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve_hybrid");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.solve_hybrid(params, final_time) {
        Ok(new_solution) => {
            unsafe {
                *out_solution = Box::into_raw(Box::new(new_solution));
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Solve an ODE and sample the solution at requested times.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// and `t_eval_ptr` must point to readable `f64` buffers of the specified
/// lengths, unless the corresponding length is zero. `out_solution` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solve_dense(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
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
    let ode = unsafe { &mut *ode };
    match ode.solve_dense(params, t_eval) {
        Ok(new_solution) => {
            unsafe {
                *out_solution = Box::into_raw(Box::new(new_solution));
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Solve a hybrid ODE and sample the solution at requested times.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// and `t_eval_ptr` must point to readable `f64` buffers of the specified
/// lengths, unless the corresponding length is zero. `out_solution` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solve_hybrid_dense(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null()
        || out_solution.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(t_eval_ptr, t_eval_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve_hybrid_dense");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let t_eval = HostArray::new_vector(t_eval_ptr as *mut u8, t_eval_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.solve_hybrid_dense(params, t_eval) {
        Ok(new_solution) => {
            unsafe {
                *out_solution = Box::into_raw(Box::new(new_solution));
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Solve an ODE and sample forward sensitivities at requested times.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// and `t_eval_ptr` must point to readable `f64` buffers of the specified
/// lengths, unless the corresponding length is zero. `out_solution` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solve_fwd_sens(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
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
    let ode = unsafe { &mut *ode };
    match ode.solve_fwd_sens(params, t_eval) {
        Ok(new_solution) => {
            unsafe {
                *out_solution = Box::into_raw(Box::new(new_solution));
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Solve a hybrid ODE with forward sensitivities at requested times.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`
/// and `t_eval_ptr` must point to readable `f64` buffers of the specified
/// lengths, unless the corresponding length is zero. `out_solution` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solve_hybrid_fwd_sens(
    ode: *mut OdeWrapper,
    params_ptr: *const f64,
    params_len: usize,
    t_eval_ptr: *const f64,
    t_eval_len: usize,
    out_solution: *mut *mut SolutionWrapper,
) -> i32 {
    if ode.is_null()
        || out_solution.is_null()
        || !valid_f64_ptr(params_ptr, params_len)
        || !valid_f64_ptr(t_eval_ptr, t_eval_len)
    {
        c_invalid_arg!("invalid arguments to diffsol_ode_solve_hybrid_fwd_sens");
        return DIFFSOL_BAD_ARG;
    }
    let params = HostArray::new_vector(params_ptr as *mut u8, params_len, ScalarType::F64);
    let t_eval = HostArray::new_vector(t_eval_ptr as *mut u8, t_eval_len, ScalarType::F64);
    let ode = unsafe { &mut *ode };
    match ode.solve_hybrid_fwd_sens(params, t_eval) {
        Ok(new_solution) => {
            unsafe {
                *out_solution = Box::into_raw(Box::new(new_solution));
            }
            DIFFSOL_OK
        }
        Err(err) => {
            c_error!(&format!("{}", err));
            DIFFSOL_ERR
        }
    }
}

/// Solve the sum-of-squares adjoint problem for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. `params_ptr`,
/// `data_ptr`, and `t_eval_ptr` must point to readable buffers matching the
/// provided dimensions. `out_value` and `out_sens` must be valid, writable
/// pointers.
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

/// Return the matrix type configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library.
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

/// Return the ODE solver enum configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library.
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

/// Set the ODE solver enum for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
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

/// Return the linear solver enum configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library.
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

/// Set the linear solver enum for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
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

/// Return the relative tolerance configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_value` must be a
/// valid, writable pointer.
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

/// Set the relative tolerance for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
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

/// Return the absolute tolerance configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_value` must be a
/// valid, writable pointer.
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

/// Set the absolute tolerance for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
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

/// Return the initial time configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_value` must be a
/// valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_t0(ode: *const OdeWrapper, out_value: *mut f64) -> i32 {
    if ode.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_t0");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_t0() {
        Ok(value) => {
            unsafe {
                *out_value = value;
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the initial time for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_t0(ode: *mut OdeWrapper, value: f64) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_t0(value) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return the initial step size configured for an ODE.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_value` must be a
/// valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_h0(ode: *const OdeWrapper, out_value: *mut f64) -> i32 {
    if ode.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_h0");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_h0() {
        Ok(value) => {
            unsafe {
                *out_value = value;
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the initial step size for an ODE.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_h0(ode: *mut OdeWrapper, value: f64) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_h0(value) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return whether output equations are integrated alongside state equations.
///
/// # Safety
/// `ode` must be a valid pointer created by this library. `out_value` must be a
/// valid, writable pointer. The value is written as 0 for false and 1 for true.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_integrate_out(
    ode: *const OdeWrapper,
    out_value: *mut i32,
) -> i32 {
    if ode.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_integrate_out");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_integrate_out() {
        Ok(value) => {
            unsafe {
                *out_value = i32::from(value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set whether output equations are integrated alongside state equations.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library. Any non-zero
/// `value` is treated as true.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_integrate_out(ode: *mut OdeWrapper, value: i32) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_integrate_out(value != 0) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

unsafe fn write_optional_f64(value: Option<f64>, out_is_some: *mut i32, out_value: *mut f64) {
    match value {
        Some(value) => unsafe {
            *out_is_some = 1;
            *out_value = value;
        },
        None => unsafe {
            *out_is_some = 0;
            *out_value = 0.0;
        },
    }
}

/// Return the optional sensitivity relative tolerance.
///
/// # Safety
/// All pointers must be valid and writable where applicable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_sens_rtol(
    ode: *const OdeWrapper,
    out_is_some: *mut i32,
    out_value: *mut f64,
) -> i32 {
    if ode.is_null() || out_is_some.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_sens_rtol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_sens_rtol() {
        Ok(value) => {
            unsafe {
                write_optional_f64(value, out_is_some, out_value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the optional sensitivity relative tolerance.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_sens_rtol(
    ode: *mut OdeWrapper,
    value_is_some: i32,
    value: f64,
) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_sens_rtol((value_is_some != 0).then_some(value)) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return the optional sensitivity absolute tolerance.
///
/// # Safety
/// All pointers must be valid and writable where applicable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_sens_atol(
    ode: *const OdeWrapper,
    out_is_some: *mut i32,
    out_value: *mut f64,
) -> i32 {
    if ode.is_null() || out_is_some.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_sens_atol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_sens_atol() {
        Ok(value) => {
            unsafe {
                write_optional_f64(value, out_is_some, out_value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the optional sensitivity absolute tolerance.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_sens_atol(
    ode: *mut OdeWrapper,
    value_is_some: i32,
    value: f64,
) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_sens_atol((value_is_some != 0).then_some(value)) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return the optional output relative tolerance.
///
/// # Safety
/// All pointers must be valid and writable where applicable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_out_rtol(
    ode: *const OdeWrapper,
    out_is_some: *mut i32,
    out_value: *mut f64,
) -> i32 {
    if ode.is_null() || out_is_some.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_out_rtol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_out_rtol() {
        Ok(value) => {
            unsafe {
                write_optional_f64(value, out_is_some, out_value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the optional output relative tolerance.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_out_rtol(
    ode: *mut OdeWrapper,
    value_is_some: i32,
    value: f64,
) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_out_rtol((value_is_some != 0).then_some(value)) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return the optional output absolute tolerance.
///
/// # Safety
/// All pointers must be valid and writable where applicable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_out_atol(
    ode: *const OdeWrapper,
    out_is_some: *mut i32,
    out_value: *mut f64,
) -> i32 {
    if ode.is_null() || out_is_some.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_out_atol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_out_atol() {
        Ok(value) => {
            unsafe {
                write_optional_f64(value, out_is_some, out_value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the optional output absolute tolerance.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_out_atol(
    ode: *mut OdeWrapper,
    value_is_some: i32,
    value: f64,
) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_out_atol((value_is_some != 0).then_some(value)) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return the optional adjoint parameter relative tolerance.
///
/// # Safety
/// All pointers must be valid and writable where applicable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_param_rtol(
    ode: *const OdeWrapper,
    out_is_some: *mut i32,
    out_value: *mut f64,
) -> i32 {
    if ode.is_null() || out_is_some.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_param_rtol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_param_rtol() {
        Ok(value) => {
            unsafe {
                write_optional_f64(value, out_is_some, out_value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the optional adjoint parameter relative tolerance.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_param_rtol(
    ode: *mut OdeWrapper,
    value_is_some: i32,
    value: f64,
) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_param_rtol((value_is_some != 0).then_some(value)) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Return the optional adjoint parameter absolute tolerance.
///
/// # Safety
/// All pointers must be valid and writable where applicable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_get_param_atol(
    ode: *const OdeWrapper,
    out_is_some: *mut i32,
    out_value: *mut f64,
) -> i32 {
    if ode.is_null() || out_is_some.is_null() || out_value.is_null() {
        c_invalid_arg!("invalid arguments to diffsol_ode_get_param_atol");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &*ode };
    match ode.get_param_atol() {
        Ok(value) => {
            unsafe {
                write_optional_f64(value, out_is_some, out_value);
            }
            DIFFSOL_OK
        }
        Err(err) => c_error!(&format!("{}", err)),
    }
}

/// Set the optional adjoint parameter absolute tolerance.
///
/// # Safety
/// `ode` must be a valid mutable pointer created by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_set_param_atol(
    ode: *mut OdeWrapper,
    value_is_some: i32,
    value: f64,
) -> i32 {
    if ode.is_null() {
        c_invalid_arg!("ode is null");
        return DIFFSOL_BAD_ARG;
    }
    let ode = unsafe { &mut *ode };
    match ode.set_param_atol((value_is_some != 0).then_some(value)) {
        Ok(()) => DIFFSOL_OK,
        Err(err) => c_error!(&format!("{}", err)),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ptr;

    use crate::error_c::{
        diffsol_error_code, diffsol_last_error_file, diffsol_last_error_line,
        diffsol_last_error_message,
    };
    #[cfg(feature = "diffsl-external-f64")]
    use crate::initial_condition_options::InitialConditionSolverOptions;
    #[cfg(feature = "diffsl-external-f64")]
    use crate::initial_condition_options_c::{
        diffsol_ic_options_free, diffsol_ic_options_get_max_linesearch_iterations,
        diffsol_ic_options_get_use_linesearch, diffsol_ic_options_set_max_linesearch_iterations,
        diffsol_ic_options_set_use_linesearch,
    };
    use crate::linear_solver_type::LinearSolverType;
    use crate::linear_solver_type_c::{
        diffsol_linear_solver_type_count, diffsol_linear_solver_type_is_valid,
        diffsol_linear_solver_type_name, linear_solver_to_i32,
    };
    use crate::matrix_type::MatrixType;
    use crate::matrix_type_c::{
        diffsol_matrix_type_count, diffsol_matrix_type_is_valid, diffsol_matrix_type_name,
        matrix_type_to_i32,
    };
    #[cfg(feature = "diffsl-external-f64")]
    use crate::ode_options::OdeSolverOptions;
    #[cfg(feature = "diffsl-external-f64")]
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
    #[cfg(feature = "diffsl-external-f64")]
    use crate::solution_wrapper_c::{
        diffsol_solution_wrapper_get_sens, diffsol_solution_wrapper_get_ts,
        diffsol_solution_wrapper_get_ys,
    };
    use crate::test_support::clear_last_error;
    #[cfg(feature = "diffsl-external-f64")]
    use crate::test_support::{
        assert_close, ffi_free_solution, ffi_read_host_array_list_matrices,
        ffi_read_host_array_matrix, ffi_read_host_array_vector, find_time_window, logistic_state,
        logistic_state_dr, mass_state_deps, rhs_input_deps, rhs_state_deps, ASSERT_TOL,
        LOGISTIC_X0,
    };

    use super::*;

    unsafe fn c_string(ptr: *const std::os::raw::c_char) -> String {
        assert!(!ptr.is_null(), "expected non-null C string");
        unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned()
    }

    unsafe fn assert_last_error_set() {
        assert_eq!(
            unsafe { diffsol_error_code() },
            1,
            "expected last error to be set"
        );
        let message_ptr = unsafe { diffsol_last_error_message() };
        assert!(
            !message_ptr.is_null(),
            "expected last error message to be set"
        );
        let message = unsafe { c_string(message_ptr) };
        assert!(
            !message.is_empty(),
            "expected last error message to be non-empty"
        );
        let file_ptr = unsafe { diffsol_last_error_file() };
        assert!(!file_ptr.is_null(), "expected last error file to be set");
        assert!(
            unsafe { diffsol_last_error_line() } > 0,
            "expected last error line to be > 0"
        );
    }

    unsafe fn assert_last_error_contains(expected_substring: &str) {
        unsafe { assert_last_error_set() };
        let message_ptr = unsafe { diffsol_last_error_message() };
        let message = unsafe { c_string(message_ptr) };
        assert!(
            message.contains(expected_substring),
            "expected last error message to contain {expected_substring:?}, got {message:?}"
        );
        let file_ptr = unsafe { diffsol_last_error_file() };
        assert!(!file_ptr.is_null(), "expected last error file to be set");
        assert!(
            unsafe { diffsol_last_error_line() } > 0,
            "expected last error line to be > 0"
        );
    }

    #[cfg(feature = "diffsl-external-f64")]
    fn to_dep_pairs(values: &[(usize, usize)]) -> Vec<DiffsolDepPair> {
        values
            .iter()
            .map(|&(row, col)| DiffsolDepPair { row, col })
            .collect()
    }

    #[cfg(feature = "diffsl-external-f64")]
    unsafe fn make_ode_ptr(
        matrix_type: i32,
        linear_solver: i32,
        ode_solver: i32,
    ) -> *mut OdeWrapper {
        let rhs_state_deps = to_dep_pairs(&rhs_state_deps());
        let rhs_input_deps = to_dep_pairs(&rhs_input_deps());
        let mass_state_deps = to_dep_pairs(&mass_state_deps());
        unsafe {
            diffsol_ode_new_external(
                matrix_type,
                linear_solver,
                ode_solver,
                rhs_state_deps.as_ptr(),
                rhs_state_deps.len(),
                rhs_input_deps.as_ptr(),
                rhs_input_deps.len(),
                mass_state_deps.as_ptr(),
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
        }
    }

    #[cfg(feature = "diffsl-external-f64")]
    #[test]
    fn c_api_rejects_invalid_external_ode_arguments() {
        clear_last_error();
        unsafe {
            let ode = make_ode_ptr(
                99,
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            );
            assert!(ode.is_null());
            assert_last_error_contains("invalid matrix_type");
        }
    }

    #[cfg(feature = "diffsl-external-f64")]
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
            assert_eq!(ffi_read_host_array_vector(y0_ptr), vec![LOGISTIC_X0]);

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
                    &mut solve_solution_ptr
                ),
                DIFFSOL_OK
            );
            assert!(!solve_solution_ptr.is_null());

            let mut solve_ys_ptr = ptr::null_mut();
            let mut solve_ts_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_solution_wrapper_get_ys(solve_solution_ptr, &mut solve_ys_ptr),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_solution_wrapper_get_ts(solve_solution_ptr, &mut solve_ts_ptr),
                DIFFSOL_OK
            );
            let (solve_rows, solve_cols, solve_ys) = ffi_read_host_array_matrix(solve_ys_ptr);
            let solve_ts = ffi_read_host_array_vector(solve_ts_ptr);
            assert_eq!(solve_rows, 1);
            assert_eq!(solve_cols, solve_ts.len());
            assert!(!solve_ts.is_empty());
            assert_close(
                *solve_ts.last().unwrap(),
                1e-9,
                ASSERT_TOL,
                "ffi solve final time",
            );
            assert_close(
                *solve_ys.last().unwrap(),
                logistic_state(LOGISTIC_X0, 2.0, 1e-9),
                ASSERT_TOL,
                "ffi solve final value",
            );
            ffi_free_solution(solve_solution_ptr);

            let mut solution_ptr: *mut SolutionWrapper = ptr::null_mut();
            assert_eq!(
                diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Tsit45)),
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

            let hybrid_t_eval = [0.5f64, 1.0, 1.25, 1.5, 2.0];
            let hybrid_ode = make_ode_ptr(
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            );
            assert!(!hybrid_ode.is_null());
            let mut hybrid_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
            assert_eq!(
                diffsol_ode_solve_hybrid_dense(
                    hybrid_ode,
                    params.as_ptr(),
                    params.len(),
                    hybrid_t_eval.as_ptr(),
                    hybrid_t_eval.len(),
                    &mut hybrid_solution_ptr,
                ),
                DIFFSOL_OK
            );
            let mut hybrid_ys_ptr = ptr::null_mut();
            let mut hybrid_ts_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_solution_wrapper_get_ys(hybrid_solution_ptr, &mut hybrid_ys_ptr),
                DIFFSOL_OK
            );
            assert_eq!(
                diffsol_solution_wrapper_get_ts(hybrid_solution_ptr, &mut hybrid_ts_ptr),
                DIFFSOL_OK
            );
            let (hybrid_rows, hybrid_cols, hybrid_ys) = ffi_read_host_array_matrix(hybrid_ys_ptr);
            let hybrid_ts = ffi_read_host_array_vector(hybrid_ts_ptr);
            assert_eq!(hybrid_rows, 1);
            assert_eq!(hybrid_cols, hybrid_t_eval.len());
            assert_eq!(hybrid_ts, hybrid_t_eval);
            assert_close(
                hybrid_ys[0],
                logistic_state(LOGISTIC_X0, 2.0, hybrid_t_eval[0]),
                5e-4,
                "ffi hybrid dense pre-root value",
            );
            assert_close(
                hybrid_ys[1],
                logistic_state(LOGISTIC_X0, 2.0, hybrid_t_eval[1]),
                5e-4,
                "ffi hybrid dense near-root value",
            );
            for (i, value) in hybrid_ys.iter().enumerate().skip(2) {
                assert_close(
                    *value,
                    1.0,
                    5e-4,
                    &format!("ffi hybrid dense post-root value[{i}]"),
                );
            }
            ffi_free_solution(hybrid_solution_ptr);
            diffsol_ode_free(hybrid_ode);

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
            for (i, (&value, &t)) in sens_values[0].2.iter().zip(t_eval.iter()).enumerate() {
                assert_close(
                    value,
                    logistic_state_dr(LOGISTIC_X0, 2.0, t),
                    ASSERT_TOL,
                    &format!("ffi sensitivity[{i}]"),
                );
            }

            let adjoint_t_eval = [0.0f64, 0.25f64, 0.5f64, 1.0f64];
            let adjoint_data: Vec<f64> = adjoint_t_eval
                .iter()
                .map(|&t| logistic_state(LOGISTIC_X0, 2.0, t))
                .collect();
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

#[cfg(all(test, feature = "diffsl-external-dynamic"))]
mod external_dynamic_tests {
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr;

    use crate::linear_solver_type::LinearSolverType;
    use crate::linear_solver_type_c::linear_solver_to_i32;
    use crate::matrix_type::MatrixType;
    use crate::matrix_type_c::matrix_type_to_i32;
    use crate::ode_solver_type::OdeSolverType;
    use crate::ode_solver_type_c::ode_solver_to_i32;
    use crate::test_support::{
        assert_last_error_contains, assert_last_error_set, clear_last_error,
        external_dynamic_fixture_path, ffi_read_host_array_vector, mass_state_deps, rhs_input_deps,
        rhs_state_deps, LOGISTIC_X0,
    };

    use super::*;

    fn to_dep_pairs(values: &[(usize, usize)]) -> Vec<DiffsolDepPair> {
        values
            .iter()
            .map(|&(row, col)| DiffsolDepPair { row, col })
            .collect()
    }

    unsafe fn make_ode_ptr(
        matrix_type: i32,
        linear_solver: i32,
        ode_solver: i32,
    ) -> *mut OdeWrapper {
        let path = CString::new(
            external_dynamic_fixture_path()
                .to_string_lossy()
                .into_owned(),
        )
        .unwrap();
        let rhs_state_deps = to_dep_pairs(&rhs_state_deps());
        let rhs_input_deps = to_dep_pairs(&rhs_input_deps());
        let mass_state_deps = to_dep_pairs(&mass_state_deps());
        unsafe {
            diffsol_ode_new_external_dynamic(
                path.as_ptr(),
                matrix_type,
                linear_solver,
                ode_solver,
                rhs_state_deps.as_ptr(),
                rhs_state_deps.len(),
                rhs_input_deps.as_ptr(),
                rhs_input_deps.len(),
                mass_state_deps.as_ptr(),
                mass_state_deps.len(),
            )
        }
    }

    #[test]
    fn c_api_constructs_dynamic_external_ode() {
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
            let params = [2.0_f64];
            let mut y0_ptr = ptr::null_mut();
            assert_eq!(
                diffsol_ode_y0(ode, params.as_ptr(), params.len(), &mut y0_ptr),
                DIFFSOL_OK
            );
            assert_eq!(ffi_read_host_array_vector(y0_ptr), vec![LOGISTIC_X0]);
            diffsol_ode_free(ode);
        }
    }

    #[test]
    fn c_api_rejects_null_dynamic_library_path() {
        clear_last_error();
        unsafe {
            let ode = diffsol_ode_new_external_dynamic(
                ptr::null(),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
            );
            assert!(ode.is_null());
            assert_last_error_contains("path is null");
        }
    }

    #[test]
    fn c_api_rejects_non_utf8_dynamic_library_path() {
        clear_last_error();
        unsafe {
            let invalid_utf8 = [0xff_u8, 0];
            let ode = diffsol_ode_new_external_dynamic(
                invalid_utf8.as_ptr() as *const c_char,
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
            );
            assert!(ode.is_null());
            assert_last_error_contains("path is not valid UTF-8");
        }
    }

    #[test]
    fn c_api_reports_missing_dynamic_library_path() {
        clear_last_error();
        unsafe {
            let missing_path = external_dynamic_fixture_path().with_file_name("does-not-exist");
            let missing_path = CString::new(missing_path.to_string_lossy().into_owned()).unwrap();
            let ode = diffsol_ode_new_external_dynamic(
                missing_path.as_ptr(),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
            );
            assert!(ode.is_null());
            assert_last_error_set();
        }
    }
}

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod jit_tests {
    use std::ffi::{CStr, CString};
    use std::ptr;

    use crate::error_c::{diffsol_error_code, diffsol_last_error_message};
    use crate::initial_condition_options_c::diffsol_ic_options_free;
    use crate::jit::JitBackendType;
    use crate::jit_c::jit_backend_to_i32;
    use crate::linear_solver_type::LinearSolverType;
    use crate::linear_solver_type_c::linear_solver_to_i32;
    use crate::matrix_type::MatrixType;
    use crate::matrix_type_c::matrix_type_to_i32;
    use crate::ode_options_c::diffsol_ode_options_free;
    use crate::ode_solver_type::OdeSolverType;
    use crate::ode_solver_type_c::ode_solver_to_i32;
    #[cfg(feature = "diffsl-llvm")]
    use crate::solution_wrapper_c::diffsol_solution_wrapper_get_sens;
    use crate::solution_wrapper_c::{
        diffsol_solution_wrapper_get_ts, diffsol_solution_wrapper_get_ys,
    };
    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::ffi_read_host_array_list_matrices;
    use crate::test_support::{
        assert_close, available_jit_backends, clear_last_error, ffi_free_solution,
        ffi_read_host_array_matrix, ffi_read_host_array_vector, find_time_window,
        hybrid_logistic_diffsl_code, hybrid_logistic_state, logistic_diffsl_code_cstring,
        logistic_state, ASSERT_TOL, LOGISTIC_X0,
    };
    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::{hybrid_logistic_state_dr, logistic_state_dr};

    use super::*;

    unsafe fn make_ode_ptr(
        jit_backend: JitBackendType,
        matrix_type: i32,
        linear_solver: i32,
        ode_solver: i32,
    ) -> *mut OdeWrapper {
        let code = logistic_diffsl_code_cstring();
        unsafe {
            make_ode_ptr_with_code(
                jit_backend,
                code.as_ptr(),
                matrix_type,
                linear_solver,
                ode_solver,
            )
        }
    }

    unsafe fn make_ode_ptr_with_code(
        jit_backend: JitBackendType,
        code: *const std::os::raw::c_char,
        matrix_type: i32,
        linear_solver: i32,
        ode_solver: i32,
    ) -> *mut OdeWrapper {
        unsafe {
            diffsol_ode_new_jit(
                code,
                jit_backend_to_i32(jit_backend),
                matrix_type,
                linear_solver,
                ode_solver,
            )
        }
    }

    unsafe fn last_error_message() -> String {
        let ptr = unsafe { diffsol_last_error_message() };
        assert_eq!(unsafe { diffsol_error_code() }, 1);
        assert!(!ptr.is_null());
        unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned()
    }

    unsafe fn assert_optional_f64(
        getter: unsafe extern "C" fn(*const OdeWrapper, *mut i32, *mut f64) -> i32,
        ode: *const OdeWrapper,
        expected: Option<f64>,
        name: &str,
    ) {
        let mut is_some = -1;
        let mut value = -1.0;
        assert_eq!(unsafe { getter(ode, &mut is_some, &mut value) }, DIFFSOL_OK);
        match expected {
            Some(expected) => {
                assert_eq!(is_some, 1, "{name} is_some");
                assert_close(value, expected, ASSERT_TOL, name);
            }
            None => {
                assert_eq!(is_some, 0, "{name} is_some");
                assert_close(value, 0.0, ASSERT_TOL, name);
            }
        }
    }

    #[test]
    fn c_api_full_lifecycle_matches_jit_logistic_model() {
        clear_last_error();
        for jit_backend in available_jit_backends() {
            unsafe {
                let ode = make_ode_ptr(
                    jit_backend,
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

                assert_eq!(diffsol_ode_set_rtol(ode, 1e-8), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_atol(ode, 1e-8), DIFFSOL_OK);
                let mut rtol = 0.0;
                let mut atol = 0.0;
                assert_eq!(diffsol_ode_get_rtol(ode, &mut rtol), DIFFSOL_OK);
                assert_eq!(diffsol_ode_get_atol(ode, &mut atol), DIFFSOL_OK);
                assert_close(rtol, 1e-8, ASSERT_TOL, "jit rtol roundtrip");
                assert_close(atol, 1e-8, ASSERT_TOL, "jit atol roundtrip");

                assert_eq!(diffsol_ode_set_t0(ode, 0.125), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_h0(ode, 0.25), DIFFSOL_OK);
                let mut t0 = 0.0;
                let mut h0 = 0.0;
                assert_eq!(diffsol_ode_get_t0(ode, &mut t0), DIFFSOL_OK);
                assert_eq!(diffsol_ode_get_h0(ode, &mut h0), DIFFSOL_OK);
                assert_close(t0, 0.125, ASSERT_TOL, "jit t0 roundtrip");
                assert_close(h0, 0.25, ASSERT_TOL, "jit h0 roundtrip");

                assert_eq!(diffsol_ode_set_integrate_out(ode, 1), DIFFSOL_OK);
                let mut integrate_out = 0;
                assert_eq!(
                    diffsol_ode_get_integrate_out(ode, &mut integrate_out),
                    DIFFSOL_OK
                );
                assert_eq!(integrate_out, 1);
                assert_eq!(diffsol_ode_set_integrate_out(ode, 0), DIFFSOL_OK);
                assert_eq!(
                    diffsol_ode_get_integrate_out(ode, &mut integrate_out),
                    DIFFSOL_OK
                );
                assert_eq!(integrate_out, 0);

                assert_eq!(diffsol_ode_set_sens_rtol(ode, 1, 1e-3), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_sens_atol(ode, 1, 1e-4), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_out_rtol(ode, 1, 2e-3), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_out_atol(ode, 1, 2e-4), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_param_rtol(ode, 1, 3e-3), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_param_atol(ode, 1, 3e-4), DIFFSOL_OK);
                assert_optional_f64(diffsol_ode_get_sens_rtol, ode, Some(1e-3), "jit sens rtol");
                assert_optional_f64(diffsol_ode_get_sens_atol, ode, Some(1e-4), "jit sens atol");
                assert_optional_f64(diffsol_ode_get_out_rtol, ode, Some(2e-3), "jit out rtol");
                assert_optional_f64(diffsol_ode_get_out_atol, ode, Some(2e-4), "jit out atol");
                assert_optional_f64(
                    diffsol_ode_get_param_rtol,
                    ode,
                    Some(3e-3),
                    "jit param rtol",
                );
                assert_optional_f64(
                    diffsol_ode_get_param_atol,
                    ode,
                    Some(3e-4),
                    "jit param atol",
                );

                assert_eq!(diffsol_ode_set_sens_rtol(ode, 0, 1e-3), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_sens_atol(ode, 0, 1e-4), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_out_rtol(ode, 0, 2e-3), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_out_atol(ode, 0, 2e-4), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_param_rtol(ode, 0, 3e-3), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_param_atol(ode, 0, 3e-4), DIFFSOL_OK);
                assert_optional_f64(diffsol_ode_get_sens_rtol, ode, None, "jit sens rtol none");
                assert_optional_f64(diffsol_ode_get_sens_atol, ode, None, "jit sens atol none");
                assert_optional_f64(diffsol_ode_get_out_rtol, ode, None, "jit out rtol none");
                assert_optional_f64(diffsol_ode_get_out_atol, ode, None, "jit out atol none");
                assert_optional_f64(diffsol_ode_get_param_rtol, ode, None, "jit param rtol none");
                assert_optional_f64(diffsol_ode_get_param_atol, ode, None, "jit param atol none");

                assert_eq!(diffsol_ode_set_t0(ode, 0.0), DIFFSOL_OK);
                assert_eq!(diffsol_ode_set_h0(ode, 1.0), DIFFSOL_OK);

                let params = [2.0f64];
                let y = [0.25f64];
                let v = [3.0f64];

                let mut y0_ptr = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_y0(ode, params.as_ptr(), params.len(), &mut y0_ptr),
                    DIFFSOL_OK
                );
                assert_eq!(ffi_read_host_array_vector(y0_ptr), vec![LOGISTIC_X0]);

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
                let t_eval = [0.25f64, 0.5f64, 1.0f64];
                assert_eq!(
                    diffsol_ode_set_ode_solver(ode, ode_solver_to_i32(OdeSolverType::Tsit45)),
                    DIFFSOL_OK
                );
                assert_eq!(
                    diffsol_ode_solve_dense(
                        ode,
                        params.as_ptr(),
                        params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
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
                        logistic_state(LOGISTIC_X0, 2.0, t),
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
                    let analysis_code = logistic_diffsl_code_cstring();
                    let analysis_ode = make_ode_ptr_with_code(
                        JitBackendType::Llvm,
                        analysis_code.as_ptr(),
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
                    for (i, (&value, &t)) in sens_values[0].2.iter().zip(t_eval.iter()).enumerate()
                    {
                        assert_close(
                            value,
                            logistic_state_dr(LOGISTIC_X0, 2.0, t),
                            ASSERT_TOL,
                            &format!("jit ffi sensitivity[{i}]"),
                        );
                    }

                    let adjoint_t_eval = [0.0f64, 0.25f64, 0.5f64, 1.0f64];
                    let adjoint_data: Vec<f64> = adjoint_t_eval
                        .iter()
                        .map(|&t| logistic_state(LOGISTIC_X0, 2.0, t))
                        .collect();
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
                    assert!(
                        grad[0].is_finite(),
                        "jit ffi adjoint gradient should be finite"
                    );

                    ffi_free_solution(sens_solution_ptr);
                    diffsol_ode_free(analysis_ode);
                }
                ffi_free_solution(solution_ptr);
                diffsol_ode_free(ode);
            }
        }
    }

    #[test]
    fn c_api_rejects_invalid_jit_arguments() {
        unsafe {
            clear_last_error();
            assert!(diffsol_ode_new_jit(
                ptr::null(),
                jit_backend_to_i32(available_jit_backends()[0]),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            )
            .is_null());
            assert!(last_error_message().contains("code is null"));

            clear_last_error();
            let invalid_utf8 = CString::from_vec_with_nul(vec![0xff, 0]).unwrap();
            assert!(diffsol_ode_new_jit(
                invalid_utf8.as_ptr(),
                jit_backend_to_i32(available_jit_backends()[0]),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            )
            .is_null());
            assert!(last_error_message().contains("valid UTF-8"));

            clear_last_error();
            let code = logistic_diffsl_code_cstring();
            assert!(diffsol_ode_new_jit(
                code.as_ptr(),
                99,
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            )
            .is_null());
            assert!(last_error_message().contains("invalid jit_backend_type"));

            clear_last_error();
            assert!(diffsol_ode_new_jit(
                code.as_ptr(),
                jit_backend_to_i32(available_jit_backends()[0]),
                99,
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            )
            .is_null());
            assert!(last_error_message().contains("invalid matrix_type"));

            clear_last_error();
            assert!(diffsol_ode_new_jit(
                code.as_ptr(),
                jit_backend_to_i32(available_jit_backends()[0]),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                99,
                ode_solver_to_i32(OdeSolverType::Bdf),
            )
            .is_null());
            assert!(last_error_message().contains("invalid linear_solver"));

            clear_last_error();
            assert!(diffsol_ode_new_jit(
                code.as_ptr(),
                jit_backend_to_i32(available_jit_backends()[0]),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                99,
            )
            .is_null());
            assert!(last_error_message().contains("invalid ode_solver"));

            clear_last_error();
            let invalid_code = CString::new("not valid diffsl").unwrap();
            assert!(diffsol_ode_new_jit(
                invalid_code.as_ptr(),
                jit_backend_to_i32(available_jit_backends()[0]),
                matrix_type_to_i32(MatrixType::NalgebraDense),
                linear_solver_to_i32(LinearSolverType::Default),
                ode_solver_to_i32(OdeSolverType::Bdf),
            )
            .is_null());
            assert!(diffsol_error_code() != 0);

            let mut ic_options = ptr::null_mut();
            assert_eq!(
                diffsol_ode_get_ic_options(ptr::null_mut(), &mut ic_options),
                DIFFSOL_BAD_ARG
            );
            let mut ode_options = ptr::null_mut();
            assert_eq!(
                diffsol_ode_get_options(ptr::null_mut(), &mut ode_options),
                DIFFSOL_BAD_ARG
            );

            let mut out_array = ptr::null_mut();
            assert_eq!(
                diffsol_ode_y0(ptr::null_mut(), ptr::null(), 0, &mut out_array),
                DIFFSOL_BAD_ARG
            );
            assert_eq!(
                diffsol_ode_rhs(
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    0.0,
                    ptr::null(),
                    0,
                    &mut out_array,
                ),
                DIFFSOL_BAD_ARG
            );
            assert_eq!(
                diffsol_ode_rhs_jac_mul(
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    0.0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    &mut out_array,
                ),
                DIFFSOL_BAD_ARG
            );

            clear_last_error();
            diffsol_ode_free(ptr::null_mut());
            assert!(last_error_message().contains("ode is null"));

            clear_last_error();
            diffsol_host_array_list_free(ptr::null_mut(), 0);
            assert!(last_error_message().contains("host array list is null"));
        }
    }

    #[test]
    fn c_api_jit_wrapper_branches_cover_runtime_success_and_errors() {
        for jit_backend in available_jit_backends() {
            unsafe {
                let ode = make_ode_ptr(
                    jit_backend,
                    matrix_type_to_i32(MatrixType::NalgebraDense),
                    linear_solver_to_i32(LinearSolverType::Default),
                    ode_solver_to_i32(OdeSolverType::Bdf),
                );
                assert!(!ode.is_null());

                let mut ic_options = ptr::null_mut();
                let mut ode_options = ptr::null_mut();
                assert_eq!(diffsol_ode_get_ic_options(ode, &mut ic_options), DIFFSOL_OK);
                assert_eq!(diffsol_ode_get_options(ode, &mut ode_options), DIFFSOL_OK);
                diffsol_ic_options_free(ic_options);
                diffsol_ode_options_free(ode_options);

                let mut out_value = 0.0;
                assert_eq!(diffsol_ode_get_rtol(ode, &mut out_value), DIFFSOL_OK);
                assert_close(out_value, 1e-6, ASSERT_TOL, "jit ffi default rtol");
                assert_eq!(diffsol_ode_set_rtol(ode, 1e-4), DIFFSOL_OK);
                assert_eq!(diffsol_ode_get_rtol(ode, &mut out_value), DIFFSOL_OK);
                assert_close(out_value, 1e-4, ASSERT_TOL, "jit ffi updated rtol");

                assert_eq!(diffsol_ode_get_atol(ode, &mut out_value), DIFFSOL_OK);
                assert_close(out_value, 1e-6, ASSERT_TOL, "jit ffi default atol");
                assert_eq!(diffsol_ode_set_atol(ode, 1e-5), DIFFSOL_OK);
                assert_eq!(diffsol_ode_get_atol(ode, &mut out_value), DIFFSOL_OK);
                assert_close(out_value, 1e-5, ASSERT_TOL, "jit ffi updated atol");

                assert_eq!(
                    diffsol_ode_set_linear_solver(ode, linear_solver_to_i32(LinearSolverType::Lu)),
                    DIFFSOL_OK
                );
                assert_eq!(
                    diffsol_ode_get_linear_solver(ode),
                    linear_solver_to_i32(LinearSolverType::Lu)
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
                    diffsol_ode_get_matrix_type(ode),
                    matrix_type_to_i32(MatrixType::NalgebraDense)
                );

                let params = [2.0f64];
                let mut solution_ptr: *mut SolutionWrapper = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_solve(ode, params.as_ptr(), params.len(), 1.0, &mut solution_ptr),
                    DIFFSOL_OK
                );
                ffi_free_solution(solution_ptr);

                let t_eval = [0.25f64, 0.5f64, 1.0f64];
                let mut dense_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_solve_dense(
                        ode,
                        params.as_ptr(),
                        params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
                        &mut dense_solution_ptr,
                    ),
                    DIFFSOL_OK
                );
                ffi_free_solution(dense_solution_ptr);

                let no_params: [f64; 0] = [];
                let y = [0.25f64];
                let v = [3.0f64];
                let mut out_array = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_y0(ode, no_params.as_ptr(), no_params.len(), &mut out_array),
                    DIFFSOL_ERR
                );
                assert_eq!(
                    diffsol_ode_rhs(
                        ode,
                        no_params.as_ptr(),
                        no_params.len(),
                        0.0,
                        y.as_ptr(),
                        y.len(),
                        &mut out_array,
                    ),
                    DIFFSOL_ERR
                );
                assert_eq!(
                    diffsol_ode_rhs_jac_mul(
                        ode,
                        no_params.as_ptr(),
                        no_params.len(),
                        0.0,
                        y.as_ptr(),
                        y.len(),
                        v.as_ptr(),
                        v.len(),
                        &mut out_array,
                    ),
                    DIFFSOL_ERR
                );

                let mut err_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_solve(
                        ode,
                        no_params.as_ptr(),
                        no_params.len(),
                        1.0,
                        &mut err_solution_ptr,
                    ),
                    DIFFSOL_ERR
                );
                assert_eq!(
                    diffsol_ode_solve_hybrid(
                        ode,
                        no_params.as_ptr(),
                        no_params.len(),
                        1.0,
                        &mut err_solution_ptr,
                    ),
                    DIFFSOL_ERR
                );
                assert_eq!(
                    diffsol_ode_solve_dense(
                        ode,
                        no_params.as_ptr(),
                        no_params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
                        &mut err_solution_ptr,
                    ),
                    DIFFSOL_ERR
                );
                assert_eq!(
                    diffsol_ode_solve_hybrid_dense(
                        ode,
                        no_params.as_ptr(),
                        no_params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
                        &mut err_solution_ptr,
                    ),
                    DIFFSOL_ERR
                );

                #[cfg(feature = "diffsl-llvm")]
                if matches!(jit_backend, JitBackendType::Llvm) {
                    assert_eq!(
                        diffsol_ode_solve_fwd_sens(
                            ode,
                            no_params.as_ptr(),
                            no_params.len(),
                            t_eval.as_ptr(),
                            t_eval.len(),
                            &mut err_solution_ptr,
                        ),
                        DIFFSOL_ERR
                    );
                    assert_eq!(
                        diffsol_ode_solve_hybrid_fwd_sens(
                            ode,
                            no_params.as_ptr(),
                            no_params.len(),
                            t_eval.as_ptr(),
                            t_eval.len(),
                            &mut err_solution_ptr,
                        ),
                        DIFFSOL_ERR
                    );

                    let adjoint_data: Vec<f64> = t_eval
                        .iter()
                        .map(|&t| logistic_state(LOGISTIC_X0, 2.0, t))
                        .collect();
                    let mut objective = 0.0;
                    let mut sens_ptr = ptr::null_mut();
                    assert_eq!(
                        diffsol_ode_solve_sum_squares_adj(
                            ode,
                            no_params.as_ptr(),
                            no_params.len(),
                            adjoint_data.as_ptr(),
                            1,
                            t_eval.len(),
                            1,
                            1,
                            t_eval.as_ptr(),
                            t_eval.len(),
                            &mut objective,
                            &mut sens_ptr,
                        ),
                        DIFFSOL_ERR
                    );
                }

                assert_eq!(diffsol_ode_get_matrix_type(ptr::null()), -1);
                assert_eq!(diffsol_ode_get_ode_solver(ptr::null()), -1);
                assert_eq!(diffsol_ode_get_linear_solver(ptr::null()), -1);
                assert_eq!(
                    diffsol_ode_set_ode_solver(ptr::null_mut(), 0),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_linear_solver(ptr::null_mut(), 0),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(diffsol_ode_set_ode_solver(ode, 99), DIFFSOL_BAD_ARG);
                assert_eq!(diffsol_ode_set_linear_solver(ode, 99), DIFFSOL_BAD_ARG);
                assert_eq!(
                    diffsol_ode_get_rtol(ptr::null(), &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(diffsol_ode_get_rtol(ode, ptr::null_mut()), DIFFSOL_BAD_ARG);
                assert_eq!(diffsol_ode_set_rtol(ptr::null_mut(), 1e-3), DIFFSOL_BAD_ARG);
                assert_eq!(
                    diffsol_ode_get_atol(ptr::null(), &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(diffsol_ode_get_atol(ode, ptr::null_mut()), DIFFSOL_BAD_ARG);
                assert_eq!(diffsol_ode_set_atol(ptr::null_mut(), 1e-3), DIFFSOL_BAD_ARG);
                assert_eq!(
                    diffsol_ode_get_t0(ptr::null(), &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(diffsol_ode_get_t0(ode, ptr::null_mut()), DIFFSOL_BAD_ARG);
                assert_eq!(diffsol_ode_set_t0(ptr::null_mut(), 0.0), DIFFSOL_BAD_ARG);
                assert_eq!(
                    diffsol_ode_get_h0(ptr::null(), &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(diffsol_ode_get_h0(ode, ptr::null_mut()), DIFFSOL_BAD_ARG);
                assert_eq!(diffsol_ode_set_h0(ptr::null_mut(), 1.0), DIFFSOL_BAD_ARG);
                let mut out_bool = 0;
                assert_eq!(
                    diffsol_ode_get_integrate_out(ptr::null(), &mut out_bool),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_integrate_out(ode, ptr::null_mut()),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_integrate_out(ptr::null_mut(), 1),
                    DIFFSOL_BAD_ARG
                );
                let mut out_is_some = 0;
                assert_eq!(
                    diffsol_ode_get_sens_rtol(ptr::null(), &mut out_is_some, &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_sens_rtol(ode, ptr::null_mut(), &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_sens_rtol(ode, &mut out_is_some, ptr::null_mut()),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_sens_rtol(ptr::null_mut(), 1, 1e-3),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_sens_atol(ptr::null(), &mut out_is_some, &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_sens_atol(ptr::null_mut(), 1, 1e-3),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_out_rtol(ptr::null(), &mut out_is_some, &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_out_rtol(ptr::null_mut(), 1, 1e-3),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_out_atol(ptr::null(), &mut out_is_some, &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_out_atol(ptr::null_mut(), 1, 1e-3),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_param_rtol(ptr::null(), &mut out_is_some, &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_param_rtol(ptr::null_mut(), 1, 1e-3),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_get_param_atol(ptr::null(), &mut out_is_some, &mut out_value),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_set_param_atol(ptr::null_mut(), 1, 1e-3),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_solve(ode, params.as_ptr(), params.len(), 1.0, ptr::null_mut()),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_solve_hybrid(
                        ode,
                        params.as_ptr(),
                        params.len(),
                        1.0,
                        ptr::null_mut(),
                    ),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_solve_dense(
                        ode,
                        params.as_ptr(),
                        params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
                        ptr::null_mut(),
                    ),
                    DIFFSOL_BAD_ARG
                );
                assert_eq!(
                    diffsol_ode_solve_hybrid_dense(
                        ode,
                        params.as_ptr(),
                        params.len(),
                        t_eval.as_ptr(),
                        t_eval.len(),
                        ptr::null_mut(),
                    ),
                    DIFFSOL_BAD_ARG
                );
                #[cfg(feature = "diffsl-llvm")]
                if matches!(jit_backend, JitBackendType::Llvm) {
                    assert_eq!(
                        diffsol_ode_solve_fwd_sens(
                            ode,
                            params.as_ptr(),
                            params.len(),
                            t_eval.as_ptr(),
                            t_eval.len(),
                            ptr::null_mut(),
                        ),
                        DIFFSOL_BAD_ARG
                    );
                    assert_eq!(
                        diffsol_ode_solve_hybrid_fwd_sens(
                            ode,
                            params.as_ptr(),
                            params.len(),
                            t_eval.as_ptr(),
                            t_eval.len(),
                            ptr::null_mut(),
                        ),
                        DIFFSOL_BAD_ARG
                    );
                    let mut objective = 0.0;
                    let mut sens_ptr = ptr::null_mut();
                    assert_eq!(
                        diffsol_ode_solve_sum_squares_adj(
                            ode,
                            params.as_ptr(),
                            params.len(),
                            t_eval.as_ptr(),
                            1,
                            t_eval.len(),
                            1,
                            1,
                            t_eval.as_ptr(),
                            t_eval.len(),
                            ptr::null_mut(),
                            &mut sens_ptr,
                        ),
                        DIFFSOL_BAD_ARG
                    );
                    assert_eq!(
                        diffsol_ode_solve_sum_squares_adj(
                            ode,
                            params.as_ptr(),
                            params.len(),
                            t_eval.as_ptr(),
                            1,
                            t_eval.len(),
                            1,
                            1,
                            t_eval.as_ptr(),
                            t_eval.len(),
                            &mut objective,
                            ptr::null_mut(),
                        ),
                        DIFFSOL_BAD_ARG
                    );
                }

                diffsol_ode_free(ode);
            }
        }
    }

    #[test]
    fn c_api_hybrid_jit_solver_paths_match_expected_values() {
        for jit_backend in available_jit_backends() {
            unsafe {
                let code = CString::new(hybrid_logistic_diffsl_code()).unwrap();
                let ode = make_ode_ptr_with_code(
                    jit_backend,
                    code.as_ptr(),
                    matrix_type_to_i32(MatrixType::NalgebraDense),
                    linear_solver_to_i32(LinearSolverType::Default),
                    ode_solver_to_i32(OdeSolverType::Bdf),
                );
                assert!(!ode.is_null());

                let params = [2.0f64];
                let mut solution_ptr: *mut SolutionWrapper = ptr::null_mut();
                assert_eq!(
                    diffsol_ode_solve_hybrid(
                        ode,
                        params.as_ptr(),
                        params.len(),
                        2.0,
                        &mut solution_ptr
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
                let (_rows, cols, ys) = ffi_read_host_array_matrix(ys_ptr);
                let ts = ffi_read_host_array_vector(ts_ptr);
                assert!(cols >= 1);
                assert_close(*ts.last().unwrap(), 2.0, 5e-4, "jit hybrid solve time");
                assert_close(
                    *ys.last().unwrap(),
                    hybrid_logistic_state(2.0, 2.0),
                    5e-4,
                    "jit hybrid solve value",
                );
                ffi_free_solution(solution_ptr);

                #[cfg(feature = "diffsl-llvm")]
                if matches!(jit_backend, JitBackendType::Llvm) {
                    let t_eval = [0.25f64, 0.5f64, 1.0f64];
                    let mut sens_solution_ptr: *mut SolutionWrapper = ptr::null_mut();
                    assert_eq!(
                        diffsol_ode_solve_hybrid_fwd_sens(
                            ode,
                            params.as_ptr(),
                            params.len(),
                            t_eval.as_ptr(),
                            t_eval.len(),
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
                    for (i, (&value, &t)) in sens_values[0].2.iter().zip(t_eval.iter()).enumerate()
                    {
                        assert_close(
                            value,
                            hybrid_logistic_state_dr(2.0, t),
                            5e-4,
                            &format!("jit hybrid sensitivity[{i}]"),
                        );
                    }
                    ffi_free_solution(sens_solution_ptr);
                }

                diffsol_ode_free(ode);
            }
        }
    }
}
