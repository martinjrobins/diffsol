use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::linear_solver_type::LinearSolverType;

const LINEAR_SOLVER_DEFAULT: &[u8] = b"default\0";
const LINEAR_SOLVER_LU: &[u8] = b"lu\0";
const LINEAR_SOLVER_KLU: &[u8] = b"klu\0";

pub(crate) fn linear_solver_from_i32(value: i32) -> Option<LinearSolverType> {
    match value {
        0 => Some(LinearSolverType::Default),
        1 => Some(LinearSolverType::Lu),
        2 => Some(LinearSolverType::Klu),
        _ => None,
    }
}

pub(crate) fn linear_solver_to_i32(value: LinearSolverType) -> i32 {
    match value {
        LinearSolverType::Default => 0,
        LinearSolverType::Lu => 1,
        LinearSolverType::Klu => 2,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_linear_solver_type_count() -> usize {
    3
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_linear_solver_type_is_valid(value: i32) -> i32 {
    if linear_solver_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid linear_solver_type");
        0
    }
}

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
