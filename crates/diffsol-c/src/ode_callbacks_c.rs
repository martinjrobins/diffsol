use std::ptr;

use crate::c_error;
use crate::ode::OdeWrapper;
use crate::ode_c::parse_ode_new_common_args;
use crate::ode_callbacks::{
    CallbackScalar, DiffsolOdeCallbacksF32, DiffsolOdeCallbacksF64, OdeCallbacksDesc,
};

/// Shared implementation of the `diffsol_ode_new_callbacks_f64` and `diffsol_ode_new_callbacks_f32` constructors.
///
/// # Safety
/// `callbacks` must be null, or it must point to a valid `OdeCallbacksDesc<T>` that satisfies
/// the conditions of [crate::ode_callbacks::OdeCallbacks::new].
unsafe fn new_callbacks<T: CallbackScalar>(
    callbacks: *const OdeCallbacksDesc<T>,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
    nbatch: usize,
) -> *mut OdeWrapper {
    if callbacks.is_null() {
        crate::c_invalid_arg!("callbacks is null");
        return ptr::null_mut();
    }
    // copy the descriptor: the caller's struct need not outlive this call
    let callbacks = unsafe { *callbacks };
    let Some((matrix_type, linear_solver, ode_solver)) =
        parse_ode_new_common_args(matrix_type, linear_solver, ode_solver)
    else {
        // ownership of `user_data` has passed to us, so free it on every failure path
        // SAFETY: nothing else has taken ownership of `user_data` yet
        unsafe { callbacks.free_user_data() };
        return ptr::null_mut();
    };
    if let Err(message) = callbacks.check_nbatch(nbatch) {
        crate::c_invalid_arg!(&message);
        // SAFETY: nothing else has taken ownership of `user_data` yet
        unsafe { callbacks.free_user_data() };
        return ptr::null_mut();
    }
    // from here on `OdeCallbacks` owns `user_data`, and frees it if construction fails
    // SAFETY: the caller guarantees the conditions of `OdeCallbacks::new`
    let ode = unsafe {
        OdeWrapper::new_callbacks(callbacks, matrix_type, linear_solver, ode_solver, nbatch)
    };
    match ode {
        Ok(ode) => Box::into_raw(Box::new(ode)),
        Err(err) => {
            c_error!(&format!("{}", err));
            ptr::null_mut()
        }
    }
}

/// Construct an ODE wrapper from equations that `double` precision C callbacks give.
///
/// The `ode_callbacks` module documentation gives the meaning and the array lengths of each
/// callback. Only `rhs` and `init` are necessary. The other callbacks can be `NULL`. If a solve
/// needs a callback that is `NULL`, the solve fails. The error gives the name of the callback.
///
/// The ODE has `nbatch` lanes: independent copies of the system, each with its own parameters
/// and state, that the solver integrates together. `nbatch` must be at least 1. The library
/// calls each callback one time for each lane. The `root` and `reset` callbacks are not
/// available when `nbatch > 1`.
///
/// The library takes ownership of `callbacks->user_data`. If `callbacks->free_user_data` is not
/// `NULL`, the library calls it one time. It calls it when `diffsol_ode_free` frees the wrapper.
/// If the construction fails, it calls it before this function returns `NULL`.
///
/// # Safety
/// - `callbacks` must be null, or it must point to a valid `DiffsolOdeCallbacksF64`.
/// - Each function pointer must be null or a valid function with the documented signature.
/// - Each function must stay valid until the wrapper is freed.
/// - Each function must read and write only the documented array lengths.
/// - Do not give the same `user_data` with a `free_user_data` to two wrappers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_new_callbacks_f64(
    callbacks: *const DiffsolOdeCallbacksF64,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
    nbatch: usize,
) -> *mut OdeWrapper {
    unsafe { new_callbacks(callbacks, matrix_type, linear_solver, ode_solver, nbatch) }
}

/// Construct an ODE wrapper from equations that `float` precision C callbacks give.
///
/// This function is the same as `diffsol_ode_new_callbacks_f64`, but each scalar is a `float`.
/// The solver also uses single precision.
///
/// # Safety
/// See `diffsol_ode_new_callbacks_f64`, with `DiffsolOdeCallbacksF32` in place of
/// `DiffsolOdeCallbacksF64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_new_callbacks_f32(
    callbacks: *const DiffsolOdeCallbacksF32,
    matrix_type: i32,
    linear_solver: i32,
    ode_solver: i32,
    nbatch: usize,
) -> *mut OdeWrapper {
    unsafe { new_callbacks(callbacks, matrix_type, linear_solver, ode_solver, nbatch) }
}
