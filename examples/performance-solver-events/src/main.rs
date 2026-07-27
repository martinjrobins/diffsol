use diffsol::{
    MatrixCommon, NalgebraLU, NalgebraMat, OdeBuilder, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverProblem, OdeSolverStopReason,
};
use std::{
    fmt::Write as _,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

// ANCHOR: types
type M = NalgebraMat<f64>;
type LS = NalgebraLU<f64>;
// ANCHOR_END: types
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

const EVENT_INTERVAL: f64 = 0.05;
const NUM_EVENTS: usize = 101;
const LAMBDA_SLOW: f64 = 1.0;
const LAMBDA_FAST: f64 = 1000.0;
const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;
const TIMING_RUNS: usize = 11;

struct RunResult {
    solver: &'static str,
    elapsed: Duration,
    steps: usize,
    rejected_steps: usize,
    rhs_evals: usize,
    #[cfg(test)]
    final_state: [f64; 2],
}

fn problem() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    // ANCHOR: model
    OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL, ATOL])
        .rhs_implicit(
            |y, _p, _t, dy| {
                dy[0] = -LAMBDA_SLOW * y[0];
                dy[1] = (LAMBDA_FAST - LAMBDA_SLOW) * y[0] - LAMBDA_FAST * y[1];
            },
            |_y, _p, _t, v, jv| {
                jv[0] = -LAMBDA_SLOW * v[0];
                jv[1] = (LAMBDA_FAST - LAMBDA_SLOW) * v[0] - LAMBDA_FAST * v[1];
            },
        )
        .init(
            |_p, _t, y| {
                y[0] = 1.0;
                y[1] = 1.0;
            },
            2,
        )
        .build()
        .unwrap()
    // ANCHOR_END: model
}

fn run_bdf(
    problem: OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>>,
) -> RunResult {
    // ANCHOR: run_bdf
    let mut solver = problem.bdf::<LS>().unwrap();
    let started = Instant::now();

    for event_index in 1..NUM_EVENTS {
        solver
            .set_stop_time(EVENT_INTERVAL * event_index as f64)
            .unwrap();
        while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}

        let state = solver.state_mut();
        state.y[0] += 1.0;
        state.y[1] += 1.0;
        state.dy[0] = -LAMBDA_SLOW * state.y[0];
        state.dy[1] = (LAMBDA_FAST - LAMBDA_SLOW) * state.y[0] - LAMBDA_FAST * state.y[1];
    }

    let elapsed = started.elapsed();
    let statistics = solver.get_statistics();
    #[cfg(test)]
    let state = solver.state();
    RunResult {
        solver: "BDF",
        elapsed,
        steps: statistics.number_of_steps,
        rejected_steps: statistics.number_of_error_test_failures,
        rhs_evals: solver.problem().eqn.statistics().number_of_calls,
        #[cfg(test)]
        final_state: [state.y[0], state.y[1]],
    }
    // ANCHOR_END: run_bdf
}

fn run_esdirk34(
    problem: OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>>,
) -> RunResult {
    // ANCHOR: run_esdirk34
    let mut solver = problem.esdirk34::<LS>().unwrap();
    let started = Instant::now();

    for event_index in 1..NUM_EVENTS {
        solver
            .set_stop_time(EVENT_INTERVAL * event_index as f64)
            .unwrap();
        while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}

        let state = solver.state_mut();
        state.y[0] += 1.0;
        state.y[1] += 1.0;
        state.dy[0] = -LAMBDA_SLOW * state.y[0];
        state.dy[1] = (LAMBDA_FAST - LAMBDA_SLOW) * state.y[0] - LAMBDA_FAST * state.y[1];
    }

    let elapsed = started.elapsed();
    let statistics = solver.get_statistics();
    #[cfg(test)]
    let state = solver.state();
    RunResult {
        solver: "ESDIRK34",
        elapsed,
        steps: statistics.number_of_steps,
        rejected_steps: statistics.number_of_error_test_failures,
        rhs_evals: solver.problem().eqn.statistics().number_of_calls,
        #[cfg(test)]
        final_state: [state.y[0], state.y[1]],
    }
    // ANCHOR_END: run_esdirk34
}

fn median(mut runs: Vec<RunResult>) -> RunResult {
    runs.sort_by_key(|run| run.elapsed);
    runs.swap_remove(runs.len() / 2)
}

// ANCHOR: comparison
fn main() {
    run_bdf(problem());
    run_esdirk34(problem());
    let bdf_runs = (0..TIMING_RUNS).map(|_| run_bdf(problem())).collect();
    let esdirk34_runs = (0..TIMING_RUNS).map(|_| run_esdirk34(problem())).collect();
    let results = [median(bdf_runs), median(esdirk34_runs)];

    println!(
        "{:<9} {:>15} {:>8} {:>9} {:>8}",
        "solver", "median time (ms)", "steps", "rejected", "RHS"
    );
    for result in &results {
        println!(
            "{:<9} {:>15.3} {:>8} {:>9} {:>8}",
            result.solver,
            result.elapsed.as_secs_f64() * 1e3,
            result.steps,
            result.rejected_steps,
            result.rhs_evals,
        );
    }

    let mut html = String::from(
        "<table class=\"performance-results\">\n\
         <thead>\n\
         <tr><th>Solver</th><th>Median time (ms)</th><th>Steps</th><th>Rejects</th><th>RHS evals</th></tr>\n\
         </thead>\n\
         <tbody>\n",
    );
    for result in &results {
        writeln!(
            html,
            "<tr><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            result.solver,
            result.elapsed.as_secs_f64() * 1e3,
            result.steps,
            result.rejected_steps,
            result.rhs_evals,
        )
        .unwrap();
    }
    html.push_str("</tbody>\n</table>\n");
    let results_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/performance/events_and_multistep_solvers_results.html");
    fs::write(results_path, html).expect("failed to write the results HTML file");
}
// ANCHOR_END: comparison

#[cfg(test)]
mod tests {
    use super::*;

    fn advance_exact(y: [f64; 2]) -> [f64; 2] {
        [
            (-LAMBDA_SLOW * EVENT_INTERVAL).exp() * y[0] + 1.0,
            (-LAMBDA_SLOW * EVENT_INTERVAL).exp() * y[1] + 1.0,
        ]
    }

    fn max_abs_error(actual: [f64; 2], reference: [f64; 2]) -> f64 {
        (actual[0] - reference[0])
            .abs()
            .max((actual[1] - reference[1]).abs())
    }

    fn exact_final_state() -> [f64; 2] {
        let mut state = [1.0, 1.0];
        for _ in 1..NUM_EVENTS {
            state = advance_exact(state);
        }
        state
    }

    #[test]
    fn both_solvers_match_the_exact_recurrence() {
        let reference = exact_final_state();
        for result in [run_bdf(problem()), run_esdirk34(problem())] {
            assert!(
                max_abs_error(result.final_state, reference) < 1000.0 * RTOL,
                "{} final state {:?} differed from reference {:?}",
                result.solver,
                result.final_state,
                reference,
            );
        }
    }
}
