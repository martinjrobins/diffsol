use diffsol::{
    MatrixCommon, NalgebraLU, NalgebraMat, OdeBuilder, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverProblem, OdeSolverStopReason, Op,
};
use std::time::{Duration, Instant};

type M = NalgebraMat<f64>;
type LS = NalgebraLU<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

const EVENT_INTERVAL: f64 = 0.1;
const NUM_EVENTS: usize = 101;
const LAMBDA_SLOW: f64 = 1.0;
const LAMBDA_FAST: f64 = 1000.0;
const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;
const TIMING_RUNS: usize = 11;

struct RunResult {
    solver: &'static str,
    elapsed: Duration,
    steps: usize,
    rejected_steps: usize,
    rhs_evals: usize,
    max_event_error: f64,
}

fn problem() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL, ATOL])
        .rhs_implicit(
            |y, _p, _t, dy| {
                dy[0] = -LAMBDA_SLOW * y[0];
                dy[1] = -LAMBDA_FAST * y[1];
            },
            |_y, _p, _t, v, jv| {
                jv[0] = -LAMBDA_SLOW * v[0];
                jv[1] = -LAMBDA_FAST * v[1];
            },
        )
        .init(
            |_p, _t, y| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            2,
        )
        .build()
        .unwrap()
}

fn advance_exact(y: [f64; 2]) -> [f64; 2] {
    [
        (-LAMBDA_SLOW * EVENT_INTERVAL).exp() * y[0] + 1.0,
        (-LAMBDA_FAST * EVENT_INTERVAL).exp() * y[1] + 1.0,
    ]
}

fn max_abs_error(actual: [f64; 2], reference: [f64; 2]) -> f64 {
    (actual[0] - reference[0])
        .abs()
        .max((actual[1] - reference[1]).abs())
}

fn run_bdf_once() -> RunResult {
    let problem = problem();
    let mut solver = problem.bdf::<LS>().unwrap();
    let started = Instant::now();
    let mut reference = [1.0, 1.0];
    let mut max_event_error: f64 = 0.0;

    let state = solver.state_mut();
    state.y[0] += 1.0;
    state.y[1] += 1.0;
    state.dy[0] = -LAMBDA_SLOW * state.y[0];
    state.dy[1] = -LAMBDA_FAST * state.y[1];

    for event_index in 1..NUM_EVENTS {
        solver
            .set_stop_time(EVENT_INTERVAL * event_index as f64)
            .unwrap();
        while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}

        let state = solver.state_mut();
        state.y[0] += 1.0;
        state.y[1] += 1.0;
        state.dy[0] = -LAMBDA_SLOW * state.y[0];
        state.dy[1] = -LAMBDA_FAST * state.y[1];

        reference = advance_exact(reference);
        let state = solver.state();
        max_event_error = max_event_error.max(max_abs_error([state.y[0], state.y[1]], reference));
    }

    let elapsed = started.elapsed();
    let statistics = solver.get_statistics();
    RunResult {
        solver: "BDF",
        elapsed,
        steps: statistics.number_of_steps,
        rejected_steps: statistics.number_of_error_test_failures,
        rhs_evals: solver.problem().eqn.statistics().number_of_calls,
        max_event_error,
    }
}

fn run_esdirk34_once() -> RunResult {
    let problem = problem();
    let mut solver = problem.esdirk34::<LS>().unwrap();
    let started = Instant::now();
    let mut reference = [1.0, 1.0];
    let mut max_event_error: f64 = 0.0;

    let state = solver.state_mut();
    state.y[0] += 1.0;
    state.y[1] += 1.0;
    state.dy[0] = -LAMBDA_SLOW * state.y[0];
    state.dy[1] = -LAMBDA_FAST * state.y[1];

    for event_index in 1..NUM_EVENTS {
        solver
            .set_stop_time(EVENT_INTERVAL * event_index as f64)
            .unwrap();
        while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}

        let state = solver.state_mut();
        state.y[0] += 1.0;
        state.y[1] += 1.0;
        state.dy[0] = -LAMBDA_SLOW * state.y[0];
        state.dy[1] = -LAMBDA_FAST * state.y[1];

        reference = advance_exact(reference);
        let state = solver.state();
        max_event_error = max_event_error.max(max_abs_error([state.y[0], state.y[1]], reference));
    }

    let elapsed = started.elapsed();
    let statistics = solver.get_statistics();
    RunResult {
        solver: "ESDIRK34",
        elapsed,
        steps: statistics.number_of_steps,
        rejected_steps: statistics.number_of_error_test_failures,
        rhs_evals: solver.problem().eqn.statistics().number_of_calls,
        max_event_error,
    }
}

fn median(mut runs: Vec<RunResult>) -> RunResult {
    runs.sort_by_key(|run| run.elapsed);
    runs.swap_remove(runs.len() / 2)
}

fn print_results(results: &[RunResult]) {
    println!(
        "{:<9} {:>15} {:>8} {:>9} {:>8} {:>15}",
        "solver", "median time (ms)", "steps", "rejected", "RHS", "max event error"
    );
    for result in results {
        println!(
            "{:<9} {:>15.3} {:>8} {:>9} {:>8} {:>15.3e}",
            result.solver,
            result.elapsed.as_secs_f64() * 1e3,
            result.steps,
            result.rejected_steps,
            result.rhs_evals,
            result.max_event_error,
        );
    }
}

fn main() {
    run_bdf_once();
    run_esdirk34_once();
    let bdf_runs = (0..TIMING_RUNS).map(|_| run_bdf_once()).collect();
    let esdirk34_runs = (0..TIMING_RUNS).map(|_| run_esdirk34_once()).collect();
    print_results(&[median(bdf_runs), median(esdirk34_runs)]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_solvers_match_the_exact_recurrence() {
        for result in [run_bdf_once(), run_esdirk34_once()] {
            assert!(
                result.max_event_error < 100.0 * RTOL,
                "{} maximum event error was {}",
                result.solver,
                result.max_event_error
            );
        }
    }
}
