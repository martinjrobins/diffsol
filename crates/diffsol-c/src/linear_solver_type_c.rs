use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::linear_solver_type::LinearSolverType;

const LINEAR_SOLVER_DEFAULT: &[u8] = b"default\0";
const LINEAR_SOLVER_LU: &[u8] = b"lu\0";
const LINEAR_SOLVER_KLU: &[u8] = b"klu\0";

/// The C value of `LinearSolverType::Default`.
pub const DIFFSOL_LINEAR_SOLVER_TYPE_DEFAULT: i32 = 0;
/// The C value of `LinearSolverType::Lu`.
pub const DIFFSOL_LINEAR_SOLVER_TYPE_LU: i32 = 1;
/// The C value of `LinearSolverType::Klu`.
pub const DIFFSOL_LINEAR_SOLVER_TYPE_KLU: i32 = 2;

pub(crate) fn linear_solver_from_i32(value: i32) -> Option<LinearSolverType> {
    match value {
        DIFFSOL_LINEAR_SOLVER_TYPE_DEFAULT => Some(LinearSolverType::Default),
        DIFFSOL_LINEAR_SOLVER_TYPE_LU => Some(LinearSolverType::Lu),
        DIFFSOL_LINEAR_SOLVER_TYPE_KLU => Some(LinearSolverType::Klu),
        _ => None,
    }
}

pub(crate) fn linear_solver_to_i32(value: LinearSolverType) -> i32 {
    match value {
        LinearSolverType::Default => DIFFSOL_LINEAR_SOLVER_TYPE_DEFAULT,
        LinearSolverType::Lu => DIFFSOL_LINEAR_SOLVER_TYPE_LU,
        LinearSolverType::Klu => DIFFSOL_LINEAR_SOLVER_TYPE_KLU,
    }
}

/// Return the number of linear solver enum values.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_linear_solver_type_count() -> usize {
    3
}

/// Return whether a linear solver enum value is valid.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_linear_solver_type_is_valid(value: i32) -> i32 {
    if linear_solver_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid linear_solver_type");
        0
    }
}

/// Return the name of a linear solver enum value.
///
/// # Safety
/// The returned pointer is borrowed from static storage owned by this library
/// and must not be freed or mutated by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_linear_solver_type_name(value: i32) -> *const c_char {
    match linear_solver_from_i32(value) {
        Some(LinearSolverType::Default) => LINEAR_SOLVER_DEFAULT.as_ptr() as *const c_char,
        Some(LinearSolverType::Lu) => LINEAR_SOLVER_LU.as_ptr() as *const c_char,
        Some(LinearSolverType::Klu) => LINEAR_SOLVER_KLU.as_ptr() as *const c_char,
        None => {
            c_invalid_arg!("invalid linear_solver_type");
            ptr::null()
        }
    }
}
