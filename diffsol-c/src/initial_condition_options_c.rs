use crate::initial_condition_options::InitialConditionSolverOptions;
use crate::{c_getter_simple, c_invalid_arg, c_setter_simple};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_ic_options_free(options: *mut InitialConditionSolverOptions) {
    if options.is_null() {
        c_invalid_arg!("ic options is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(options));
    }
}

c_getter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    i32,
    use_linesearch
);

c_setter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    i32,
    use_linesearch
);

c_getter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    usize,
    max_linesearch_iterations
);

c_setter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    usize,
    max_linesearch_iterations
);

c_getter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    usize,
    max_newton_iterations
);

c_setter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    usize,
    max_newton_iterations
);

c_getter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    usize,
    max_linear_solver_setups
);

c_setter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    usize,
    max_linear_solver_setups
);

c_getter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    f64,
    step_reduction_factor
);

c_setter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    f64,
    step_reduction_factor
);

c_getter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    f64,
    armijo_constant
);

c_setter_simple!(
    diffsol_ic_options,
    InitialConditionSolverOptions,
    f64,
    armijo_constant
);
