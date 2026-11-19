use crate::{problem_implicit, LS};
use diffsol::{error::OdeSolverError, DiffsolError, OdeSolverMethod};

pub fn solve_error_handling() {
    let mut problem = problem_implicit();
    problem.ode_options.max_number_timesteps = 10;
    let mut solver = problem.bdf::<LS>().unwrap();

    match solver.solve(10.0) {
        Ok((_ys, ts, _stop_reason)) => println!("solved to t = {}", ts.last().unwrap()),
        Err(DiffsolError::OdeSolverError(OdeSolverError::TooManyTimesteps { time, num_steps })) => {
            println!("stopped after {num_steps} steps at t = {time}");

            // the solver keeps the state of its last step: increase the limit and continue
            solver.config_mut().maximum_number_timesteps = 5000;
            let (_ys, ts, _stop_reason) = solver.solve(10.0).unwrap();
            println!(
                "continued from t = {} to t = {}",
                ts.first().unwrap(),
                ts.last().unwrap()
            );
        }
        Err(e) => panic!("solve failed: {e}"),
    }
}
