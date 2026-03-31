use crate::ode_options::OdeSolverOptions;
use crate::{c_getter_simple, c_invalid_arg, c_setter_simple};

/// Free an ODE options object previously returned by this library.
///
/// # Safety
/// `options` must be either null or a pointer returned by this library that has
/// not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ode_options_free(options: *mut OdeSolverOptions) {
    if options.is_null() {
        c_invalid_arg!("ode options is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(options));
    }
}

c_getter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    max_nonlinear_solver_iterations
);

c_setter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    max_nonlinear_solver_iterations
);

c_getter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    max_error_test_failures
);

c_setter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    max_error_test_failures
);

c_getter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    update_jacobian_after_steps
);

c_setter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    update_jacobian_after_steps
);

c_getter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    update_rhs_jacobian_after_steps
);

c_setter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    usize,
    update_rhs_jacobian_after_steps
);

c_getter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    f64,
    threshold_to_update_jacobian
);

c_setter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    f64,
    threshold_to_update_jacobian
);

c_getter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    f64,
    threshold_to_update_rhs_jacobian
);

c_setter_simple!(
    diffsol_ode_options,
    OdeSolverOptions,
    f64,
    threshold_to_update_rhs_jacobian
);

c_getter_simple!(diffsol_ode_options, OdeSolverOptions, f64, min_timestep);

c_setter_simple!(diffsol_ode_options, OdeSolverOptions, f64, min_timestep);
