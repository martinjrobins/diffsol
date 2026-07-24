use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::ode_solver_type::OdeSolverType;

const ODE_SOLVER_BDF: &[u8] = b"bdf\0";
const ODE_SOLVER_ESDIRK34: &[u8] = b"esdirk34\0";
const ODE_SOLVER_TR_BDF2: &[u8] = b"tr_bdf2\0";
const ODE_SOLVER_TSIT45: &[u8] = b"tsit45\0";

pub(crate) fn ode_solver_from_i32(value: i32) -> Option<OdeSolverType> {
    match value {
        0 => Some(OdeSolverType::Bdf),
        1 => Some(OdeSolverType::Esdirk34),
        2 => Some(OdeSolverType::TrBdf2),
        3 => Some(OdeSolverType::Tsit45),
        _ => None,
    }
}

pub(crate) fn ode_solver_to_i32(value: OdeSolverType) -> i32 {
    match value {
        OdeSolverType::Bdf => 0,
        OdeSolverType::Esdirk34 => 1,
        OdeSolverType::TrBdf2 => 2,
        OdeSolverType::Tsit45 => 3,
    }
}

/// Return the number of ODE solver enum values.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solver_type_count() -> usize {
    4
}

/// Return whether an ODE solver enum value is valid.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solver_type_is_valid(value: i32) -> i32 {
    if ode_solver_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid ode_solver_type");
        0
    }
}

/// Return the name of an ODE solver enum value.
///
/// # Safety
/// The returned pointer is borrowed from static storage owned by this library
/// and must not be freed or mutated by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_solver_type_name(value: i32) -> *const c_char {
    match ode_solver_from_i32(value) {
        Some(OdeSolverType::Bdf) => ODE_SOLVER_BDF.as_ptr() as *const c_char,
        Some(OdeSolverType::Esdirk34) => ODE_SOLVER_ESDIRK34.as_ptr() as *const c_char,
        Some(OdeSolverType::TrBdf2) => ODE_SOLVER_TR_BDF2.as_ptr() as *const c_char,
        Some(OdeSolverType::Tsit45) => ODE_SOLVER_TSIT45.as_ptr() as *const c_char,
        None => {
            c_invalid_arg!("invalid ode_solver_type");
            ptr::null()
        }
    }
}
