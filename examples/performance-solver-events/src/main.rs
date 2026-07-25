use diffsol::{
    MatrixCommon, NalgebraLU, NalgebraMat, OdeBuilder, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverProblem, OdeSolverStopReason,
};
use std::{
    cell::Cell,
    rc::Rc,
    time::{Duration, Instant},
};

type M = NalgebraMat<f64>;
type LS = NalgebraLU<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

const EVENT_INTERVAL: f64 = 0.05;
const NUM_SEGMENTS: usize = 100;
const ALPHA_1: f64 = std::f64::consts::PI * std::f64::consts::PI;
const ALPHA_2: f64 = 1600.0 * ALPHA_1;
const RTOL: f64 = 1e-4;
const ATOL: f64 = 1e-4;

struct RunResult {
    solver: &'static str,
    elapsed: Duration,
    steps: usize,
    rejected_steps: usize,
    rhs_evals: usize,
    max_event_error: f64,
    final_state: [f64; 2],
}

fn problem(
    forcing: Rc<Cell<f64>>,
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    let mut problem = OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL, ATOL])
        .rhs_implicit(
            move |y, _p, _t, dy| {
                let q = forcing.get();
                dy[0] = -ALPHA_1 * y[0] + q;
                dy[1] = -ALPHA_2 * y[1] + 0.1 * q;
            },
            |_y, _p, _t, v, jv| {
                jv[0] = -ALPHA_1 * v[0];
                jv[1] = -ALPHA_2 * v[1];
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
        .unwrap();
    // The repeated BDF restarts can each trigger a recoverable Newton failure.
    problem.ode_options.max_nonlinear_solver_failures = NUM_SEGMENTS * 2;
    problem
}

fn forcing(segment: usize) -> f64 {
    if segment.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

fn advance_exact(y: [f64; 2], q: f64) -> [f64; 2] {
    fn advance_component(y: f64, alpha: f64, coefficient: f64, q: f64) -> f64 {
        let decay = (-alpha * EVENT_INTERVAL).exp();
        decay * y + coefficient * q / alpha * (1.0 - decay)
    }

    [
        advance_component(y[0], ALPHA_1, 1.0, q),
        advance_component(y[1], ALPHA_2, 0.1, q),
    ]
}

fn max_abs_error(actual: [f64; 2], reference: [f64; 2]) -> f64 {
    (actual[0] - reference[0])
        .abs()
        .max((actual[1] - reference[1]).abs())
}

fn run<'a, Eqn, Solver, Statistics>(
    solver_name: &'static str,
    mut solver: Solver,
    active_forcing: Rc<Cell<f64>>,
    statistics: Statistics,
) -> RunResult
where
    Eqn: OdeEquationsImplicit<M = M, V = V, T = T, C = C> + 'a,
    Solver: OdeSolverMethod<'a, Eqn>,
    Statistics: FnOnce(&Solver) -> (usize, usize),
{
    let started = Instant::now();
    let mut reference = [0.0, 0.0];
    let mut max_event_error: f64 = 0.0;

    for segment in 0..NUM_SEGMENTS {
        active_forcing.set(forcing(segment));
        let event_time = EVENT_INTERVAL * (segment + 1) as f64;
        solver.set_stop_time(event_time).unwrap();
        loop {
            match solver.step().unwrap() {
                OdeSolverStopReason::InternalTimestep => {}
                OdeSolverStopReason::TstopReached => break,
                OdeSolverStopReason::RootFound(_, _) => unreachable!("the model has no roots"),
            }
        }

        reference = advance_exact(reference, active_forcing.get());
        let state = solver.state();
        max_event_error = max_event_error.max(max_abs_error([state.y[0], state.y[1]], reference));
        drop(state);

        if segment + 1 < NUM_SEGMENTS {
            let q_next = forcing(segment + 1);
            active_forcing.set(q_next);
            // The state is continuous, but the new forcing makes dy discontinuous.
            // Mutating the state restarts BDF's history at first order.
            let state = solver.state_mut();
            state.dy[0] = -ALPHA_1 * state.y[0] + q_next;
            state.dy[1] = -ALPHA_2 * state.y[1] + 0.1 * q_next;
        }
    }

    let elapsed = started.elapsed();
    let state = solver.state();
    let final_state = [state.y[0], state.y[1]];
    let (steps, rejected_steps) = statistics(&solver);
    let equation_stats = solver.problem().eqn.statistics();

    RunResult {
        solver: solver_name,
        elapsed,
        steps,
        rejected_steps,
        rhs_evals: equation_stats.number_of_calls,
        max_event_error,
        final_state,
    }
}

fn print_results(results: &[RunResult]) {
    println!(
        "{:<9} {:>10} {:>8} {:>9} {:>8} {:>15} {:>24}",
        "solver", "time (ms)", "steps", "rejected", "RHS", "max event error", "final state"
    );
    for result in results {
        println!(
            "{:<9} {:>10.3} {:>8} {:>9} {:>8} {:>15.3e} [{:>9.5}, {:>9.5}]",
            result.solver,
            result.elapsed.as_secs_f64() * 1e3,
            result.steps,
            result.rejected_steps,
            result.rhs_evals,
            result.max_event_error,
            result.final_state[0],
            result.final_state[1],
        );
    }
}

fn main() {
    let bdf_forcing = Rc::new(Cell::new(forcing(0)));
    let tr_bdf2_forcing = Rc::new(Cell::new(forcing(0)));
    let esdirk34_forcing = Rc::new(Cell::new(forcing(0)));
    let results = [
        run(
            "BDF",
            problem(bdf_forcing.clone()).bdf::<LS>().unwrap(),
            bdf_forcing,
            |solver| {
                let stats = solver.get_statistics();
                (stats.number_of_steps, stats.number_of_error_test_failures)
            },
        ),
        run(
            "TR-BDF2",
            problem(tr_bdf2_forcing.clone()).tr_bdf2::<LS>().unwrap(),
            tr_bdf2_forcing,
            |solver| {
                let stats = solver.get_statistics();
                (stats.number_of_steps, stats.number_of_error_test_failures)
            },
        ),
        run(
            "ESDIRK34",
            problem(esdirk34_forcing.clone()).esdirk34::<LS>().unwrap(),
            esdirk34_forcing,
            |solver| {
                let stats = solver.get_statistics();
                (stats.number_of_steps, stats.number_of_error_test_failures)
            },
        ),
    ];
    print_results(&results);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_solvers_match_the_exact_recurrence() {
        let bdf_forcing = Rc::new(Cell::new(forcing(0)));
        let tr_bdf2_forcing = Rc::new(Cell::new(forcing(0)));
        let esdirk34_forcing = Rc::new(Cell::new(forcing(0)));
        let results = [
            run(
                "BDF",
                problem(bdf_forcing.clone()).bdf::<LS>().unwrap(),
                bdf_forcing,
                |solver| {
                    let stats = solver.get_statistics();
                    (stats.number_of_steps, stats.number_of_error_test_failures)
                },
            ),
            run(
                "TR-BDF2",
                problem(tr_bdf2_forcing.clone()).tr_bdf2::<LS>().unwrap(),
                tr_bdf2_forcing,
                |solver| {
                    let stats = solver.get_statistics();
                    (stats.number_of_steps, stats.number_of_error_test_failures)
                },
            ),
            run(
                "ESDIRK34",
                problem(esdirk34_forcing.clone()).esdirk34::<LS>().unwrap(),
                esdirk34_forcing,
                |solver| {
                    let stats = solver.get_statistics();
                    (stats.number_of_steps, stats.number_of_error_test_failures)
                },
            ),
        ];

        for result in results {
            assert!(
                result.max_event_error < 1e-7,
                "{} maximum event error was {}",
                result.solver,
                result.max_event_error
            );
        }
    }
}
