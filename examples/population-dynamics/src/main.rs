use diffsol::{
    CraneliftJitModule, MatrixCommon, NalgebraVec, OdeBuilder, OdeEquations, OdeEquationsImplicit,
    OdeSolverMethod, OdeSolverProblem, Vector,
};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
// ANCHOR: types
type M = diffsol::NalgebraMat<f64>;
type LS = diffsol::NalgebraLU<f64>;
type CG = CraneliftJitModule;
// ANCHOR_END: types
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

fn main() {
    solve(problem_diffsl(1.0));
    solve_explicit(problem_rust_closures_explicit(1.0));
    phase_plane(problem_rust_closures(1.0));
}

fn problem_diffsl(
    y0: T,
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    // ANCHOR: problem_diffsl
    OdeBuilder::<M>::new()
        .p([y0])
        .build_from_diffsl::<CG>(
            "
        in { y0 = 1.0 }
        a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
        u_i {
            y1 = y0,
            y2 = y0,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }
    ",
        )
        .unwrap()
    // ANCHOR_END: problem_diffsl
}

fn problem_rust_closures(
    y0: T,
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    // ANCHOR: problem_rust_closures
    OdeBuilder::<M>::new()
        .p([y0])
        .rhs_implicit(
            |y, _p, _t, out| {
                out[0] = 2.0 / 3.0 * y[0] - 4.0 / 3.0 * y[0] * y[1];
                out[1] = y[0] * y[1] - y[1];
            },
            |y, _p, _t, v, out| {
                out[0] = 2.0 / 3.0 * v[0] - 4.0 / 3.0 * (v[0] * y[1] + y[0] * v[1]);
                out[1] = v[0] * y[1] + y[0] * v[1] - v[1];
            },
        )
        .init(
            |p, _t, y| {
                y[0] = p[0];
                y[1] = p[0];
            },
            2,
        )
        .build()
        .unwrap()
    // ANCHOR_END: problem_rust_closures
}

fn problem_rust_closures_explicit(
    y0: T,
) -> OdeSolverProblem<impl OdeEquations<M = M, V = V, T = T, C = C>> {
    // ANCHOR: problem_rust_closures_explicit
    OdeBuilder::<M>::new()
        .p([y0])
        .rhs(|y, _p, _t, out| {
            out[0] = 2.0 / 3.0 * y[0] - 4.0 / 3.0 * y[0] * y[1];
            out[1] = y[0] * y[1] - y[1];
        })
        .init(
            |p, _t, y| {
                y[0] = p[0];
                y[1] = p[0];
            },
            2,
        )
        .build()
        .unwrap()
    // ANCHOR_END: problem_rust_closures_explicit
}

fn solve<Eqn>(problem: OdeSolverProblem<Eqn>)
where
    Eqn: OdeEquationsImplicit<M = M, V = V, T = T, C = C>,
{
    // ANCHOR: solve
    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, ts, _stop_reason) = solver.solve(40.0).unwrap();

    let prey: Vec<_> = ys.inner()[0].row(0).into_iter().copied().collect();
    let predator: Vec<_> = ys.inner()[0].row(1).into_iter().copied().collect();
    let time: Vec<_> = ts.into_iter().collect();

    let prey = Scatter::new(time.clone(), prey)
        .mode(Mode::Lines)
        .name("Prey");
    let predator = Scatter::new(time, predator)
        .mode(Mode::Lines)
        .name("Predator");

    let mut plot = Plot::new();
    plot.add_trace(prey);
    plot.add_trace(predator);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new().title("population"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("prey-predator"));
    fs::write("book/src/primer/images/prey-predator.html", plot_html)
        .expect("Unable to write file");
    // ANCHOR_END: solve
}

fn solve_explicit<Eqn>(problem: OdeSolverProblem<Eqn>)
where
    Eqn: OdeEquations<M = M, V = V, T = T, C = C>,
{
    // ANCHOR: solve_explicit
    let mut solver = problem.tsit45().unwrap();
    let (ys, ts, _stop_reason) = solver.solve(40.0).unwrap();

    let prey: Vec<_> = ys.inner()[0].row(0).into_iter().copied().collect();
    let predator: Vec<_> = ys.inner()[0].row(1).into_iter().copied().collect();
    let time: Vec<_> = ts.into_iter().collect();

    let prey = Scatter::new(time.clone(), prey)
        .mode(Mode::Lines)
        .name("Prey");
    let predator = Scatter::new(time, predator)
        .mode(Mode::Lines)
        .name("Predator");

    let mut plot = Plot::new();
    plot.add_trace(prey);
    plot.add_trace(predator);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new().title("population"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("prey-predator"));
    fs::write("book/src/primer/images/prey-predator.html", plot_html)
        .expect("Unable to write file");
    // ANCHOR_END: solve_explicit
}

fn phase_plane<Eqn>(mut problem: OdeSolverProblem<Eqn>)
where
    Eqn: OdeEquationsImplicit<M = M, V = V, T = T, C = C>,
{
    // ANCHOR: phase_plane
    let mut plot = Plot::new();
    for y0 in (1..6).map(f64::from) {
        let p = NalgebraVec::from_element(1, y0, *problem.context());
        problem.eqn_mut().set_params(&p);

        let mut solver = problem.bdf::<LS>().unwrap();
        let (ys, _ts, _stop_reason) = solver.solve(40.0).unwrap();

        let prey: Vec<_> = ys.inner()[0].row(0).into_iter().copied().collect();
        let predator: Vec<_> = ys.inner()[0].row(1).into_iter().copied().collect();

        let phase = Scatter::new(prey, predator)
            .mode(Mode::Lines)
            .name(format!("y0 = {y0}"));
        plot.add_trace(phase);
    }

    let layout = Layout::new()
        .x_axis(Axis::new().title("x"))
        .y_axis(Axis::new().title("y"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("prey-predator2"));
    fs::write("book/src/primer/images/prey-predator2.html", plot_html)
        .expect("Unable to write file");
    // ANCHOR_END: phase_plane
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{DenseMatrix, VectorView};

    #[test]
    fn all_problem_definitions_agree_within_default_tolerances() {
        let times: Vec<_> = (0..=400).map(|i| i as f64 * 0.1).collect();

        let mut diffsl_problem = problem_diffsl(1.0);
        let rtol = diffsl_problem.rtol;
        let atol = diffsl_problem.atol.clone();
        // Tighter integration keeps accumulated errors below the default comparison tolerance.
        diffsl_problem.rtol = 1e-8;
        diffsl_problem.atol.fill(1e-8);
        let mut diffsl_solver = diffsl_problem.bdf::<LS>().unwrap();
        let (diffsl_solution, _) = diffsl_solver.solve_dense(&times).unwrap();

        let mut closures_problem = problem_rust_closures(1.0);
        closures_problem.rtol = 1e-8;
        closures_problem.atol.fill(1e-8);
        let mut closures_solver = closures_problem.bdf::<LS>().unwrap();
        let (closures_solution, _) = closures_solver.solve_dense(&times).unwrap();

        let mut explicit_problem = problem_rust_closures_explicit(1.0);
        explicit_problem.rtol = 1e-8;
        explicit_problem.atol.fill(1e-8);
        let mut explicit_solver = explicit_problem.tsit45().unwrap();
        let (explicit_solution, _) = explicit_solver.solve_dense(&times).unwrap();

        for col in 0..times.len() {
            let diffsl = diffsl_solution.column(col).into_owned();
            let closures = closures_solution.column(col).into_owned();
            let explicit = explicit_solution.column(col).into_owned();
            diffsl.assert_eq_norm(&closures, &atol, rtol, 2.0);
            diffsl.assert_eq_norm(&explicit, &atol, rtol, 2.0);
            closures.assert_eq_norm(&explicit, &atol, rtol, 2.0);
        }
    }
}
