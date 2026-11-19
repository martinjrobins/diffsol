use crate::{problem_implicit, LS};
use diffsol::OdeSolverMethod;

pub fn solver_options() {
    let mut problem = problem_implicit();

    // options for every solver that you create from this problem
    problem.ode_options.max_number_timesteps = 10_000;
    problem.ode_options.max_nonlinear_solver_iterations = 5;
    problem.ic_options.max_newton_iterations = 20;

    let mut solver = problem.bdf::<LS>().unwrap();

    // limits for this solver only, after you create it
    solver.config_mut().maximum_timestep_growth = 5.0;

    let (_ys, _ts, _stop_reason) = solver.solve(10.0).unwrap();

    let stats = solver.get_statistics();
    println!(
        "steps: {}, error test failures: {}, nonlinear iterations: {}, linear solver setups: {}",
        stats.number_of_steps,
        stats.number_of_error_test_failures,
        stats.number_of_nonlinear_solver_iterations,
        stats.number_of_linear_solver_setups,
    );
}
