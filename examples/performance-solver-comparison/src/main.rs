use diffsol::{
    DenseMatrix, MatrixCommon, NalgebraLU, NalgebraMat, OdeBuilder, OdeEquations,
    OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem,
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

const RTOL: f64 = 1e-6;
const ATOL: f64 = 1e-8;

struct RunResult {
    problem: &'static str,
    solver: &'static str,
    elapsed: Duration,
    value: f64,
    steps: usize,
    error_test_failures: usize,
    rhs_evals: usize,
}

fn logistic_explicit() -> OdeSolverProblem<impl OdeEquations<M = M, V = V, T = T, C = C>> {
    // ANCHOR: logistic_explicit
    OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL])
        .rhs(|y, _p, _t, dy| dy[0] = 2.0 * y[0] * (1.0 - y[0] / 10.0))
        .init(|_p, _t, y| y[0] = 0.5, 1)
        .build()
        .unwrap()
    // ANCHOR_END: logistic_explicit
}

fn logistic_implicit() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    // ANCHOR: logistic_implicit
    OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL])
        .rhs_implicit(
            |y, _p, _t, dy| dy[0] = 2.0 * y[0] * (1.0 - y[0] / 10.0),
            |y, _p, _t, v, jv| jv[0] = (2.0 - 0.4 * y[0]) * v[0],
        )
        .init(|_p, _t, y| y[0] = 0.5, 1)
        .build()
        .unwrap()
    // ANCHOR_END: logistic_implicit
}

fn relaxation_explicit() -> OdeSolverProblem<impl OdeEquations<M = M, V = V, T = T, C = C>> {
    // ANCHOR: relaxation_explicit
    OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL])
        .rhs(|y, _p, t, dy| dy[0] = -1000.0 * (y[0] - t.cos()) - t.sin())
        .init(|_p, _t, y| y[0] = 0.0, 1)
        .build()
        .unwrap()
    // ANCHOR_END: relaxation_explicit
}

fn relaxation_implicit() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>>
{
    // ANCHOR: relaxation_implicit
    OdeBuilder::<M>::new()
        .rtol(RTOL)
        .atol([ATOL])
        .rhs_implicit(
            |y, _p, t, dy| dy[0] = -1000.0 * (y[0] - t.cos()) - t.sin(),
            |_y, _p, _t, v, jv| jv[0] = -1000.0 * v[0],
        )
        .init(|_p, _t, y| y[0] = 0.0, 1)
        .build()
        .unwrap()
    // ANCHOR_END: relaxation_implicit
}

#[cfg(test)]
fn logistic_exact(t: f64) -> f64 {
    10.0 / (1.0 + 19.0 * (-2.0 * t).exp())
}

#[cfg(test)]
fn relaxation_exact(t: f64) -> f64 {
    t.cos() - (-1000.0 * t).exp()
}

fn run_tsit45(
    problem_name: &'static str,
    problem: OdeSolverProblem<impl OdeEquations<M = M, V = V, T = T, C = C>>,
    final_time: f64,
) -> RunResult {
    // ANCHOR: solve_tsit45
    let mut solver = problem.tsit45().unwrap();
    let started = Instant::now();
    let (solution, _) = solver.solve_dense(&[final_time]).unwrap();
    let elapsed = started.elapsed();
    let value = solution.column(0)[0];
    let solver_stats = solver.get_statistics();
    let equation_stats = solver.problem().eqn.statistics();

    RunResult {
        problem: problem_name,
        solver: "Tsit45",
        elapsed,
        value,
        steps: solver_stats.number_of_steps,
        error_test_failures: solver_stats.number_of_error_test_failures,
        rhs_evals: equation_stats.number_of_calls,
    }
    // ANCHOR_END: solve_tsit45
}

fn run_bdf(
    problem_name: &'static str,
    problem: OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>>,
    final_time: f64,
) -> RunResult {
    // ANCHOR: solve_bdf
    let mut solver = problem.bdf::<LS>().unwrap();
    let started = Instant::now();
    let (solution, _) = solver.solve_dense(&[final_time]).unwrap();
    let elapsed = started.elapsed();
    let value = solution.column(0)[0];
    let solver_stats = solver.get_statistics();
    let equation_stats = solver.problem().eqn.statistics();

    RunResult {
        problem: problem_name,
        solver: "BDF",
        elapsed,
        value,
        steps: solver_stats.number_of_steps,
        error_test_failures: solver_stats.number_of_error_test_failures,
        rhs_evals: equation_stats.number_of_calls,
    }
    // ANCHOR_END: solve_bdf
}

fn print_results(results: &[RunResult]) {
    println!(
        "{:<18} {:<7} {:>10} {:>14} {:>8} {:>9} {:>8}",
        "problem", "solver", "time (ms)", "final value", "steps", "rejected", "RHS",
    );
    for result in results {
        println!(
            "{:<18} {:<7} {:>10.3} {:>14.8} {:>8} {:>9} {:>8}",
            result.problem,
            result.solver,
            result.elapsed.as_secs_f64() * 1e3,
            result.value,
            result.steps,
            result.error_test_failures,
            result.rhs_evals,
        );
    }
}

fn results_html_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/performance/stiff_vs_non_stiff_results.html")
}

fn write_results_html(results: &[RunResult]) -> std::io::Result<()> {
    let mut html = String::from(
        "<table class=\"performance-results\">\n\
         <thead>\n\
         <tr><th>Problem</th><th>Solver</th><th>Time (ms)</th><th>Final</th><th>Steps</th><th>Rejects</th><th>RHS evals</th></tr>\n\
         </thead>\n\
         <tbody>\n",
    );
    for result in results {
        writeln!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.8}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            result.problem,
            result.solver,
            result.elapsed.as_secs_f64() * 1e3,
            result.value,
            result.steps,
            result.error_test_failures,
            result.rhs_evals,
        )
        .unwrap();
    }
    html.push_str("</tbody>\n</table>\n");
    fs::write(results_html_path(), html)
}

// ANCHOR: comparison
fn main() {
    let results = [
        run_tsit45("logistic", logistic_explicit(), 5.0),
        run_bdf("logistic", logistic_implicit(), 5.0),
        run_tsit45("rapid relaxation", relaxation_explicit(), 10.0),
        run_bdf("rapid relaxation", relaxation_implicit(), 10.0),
    ];
    print_results(&results);
    write_results_html(&results).expect("failed to write the results HTML file");
}
// ANCHOR_END: comparison

#[cfg(test)]
mod tests {
    use super::*;

    fn final_value(solution: <V as diffsol::DefaultDenseMatrix>::M) -> f64 {
        solution.column(0)[0]
    }

    #[test]
    fn all_solvers_match_the_exact_solutions() {
        let problem = logistic_explicit();
        let mut solver = problem.tsit45().unwrap();
        let (solution, _) = solver.solve_dense(&[5.0]).unwrap();
        assert!((final_value(solution) - logistic_exact(5.0)).abs() < 1e-5);

        let problem = logistic_implicit();
        let mut solver = problem.bdf::<LS>().unwrap();
        let (solution, _) = solver.solve_dense(&[5.0]).unwrap();
        assert!((final_value(solution) - logistic_exact(5.0)).abs() < 1e-5);

        let problem = relaxation_explicit();
        let mut solver = problem.tsit45().unwrap();
        let (solution, _) = solver.solve_dense(&[10.0]).unwrap();
        assert!((final_value(solution) - relaxation_exact(10.0)).abs() < 1e-5);

        let problem = relaxation_implicit();
        let mut solver = problem.bdf::<LS>().unwrap();
        let (solution, _) = solver.solve_dense(&[10.0]).unwrap();
        assert!((final_value(solution) - relaxation_exact(10.0)).abs() < 1e-5);
    }
}
