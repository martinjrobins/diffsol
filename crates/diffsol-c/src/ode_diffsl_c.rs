//! The C API constructors for DiffSL equations: JIT-compiled code, and external (static or
//! dynamic library) symbols.

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

use crate::c_error;
#[cfg(any(
    feature = "diffsl-external-dynamic",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm",
))]
use crate::c_invalid_arg;
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use crate::jit_c::jit_backend_from_i32;
use crate::ode::OdeWrapper;
use crate::ode_c::parse_ode_new_common_args;
use crate::scalar_type::ScalarType;

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
