use diffsol::{
    CraneliftJitModule, MatrixCommon, OdeBuilder, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverProblem,
};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = diffsol::NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    solve(problem_diffsl());
    let _ = problem_rust_closures();
}

fn problem_diffsl() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    // ANCHOR: problem_diffsl
    OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
        k { 1.0 } m { 1.0 } c { 0.1 }
        u_i {
            x = 1,
            v = 0,
        }
        F_i {
            v,
            -k/m * x - c/m * v,
        }
    ",
        )
        .unwrap()
    // ANCHOR_END: problem_diffsl
}

fn problem_rust_closures() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>>
{
    // ANCHOR: problem_rust_closures
    OdeBuilder::<M>::new()
        .rhs_implicit(
            |y, _p, _t, out| {
                out[0] = y[1];
                out[1] = -y[0] - 0.1 * y[1];
            },
            |_y, _p, _t, v, out| {
                out[0] = v[1];
                out[1] = -v[0] - 0.1 * v[1];
            },
        )
        .init(
            |_p, _t, y| {
                y[0] = 1.0;
                y[1] = 0.0;
            },
            2,
        )
        .build()
        .unwrap()
    // ANCHOR_END: problem_rust_closures
}

fn solve<Eqn>(problem: OdeSolverProblem<Eqn>)
where
    Eqn: OdeEquationsImplicit<M = M, V = V, T = T, C = C>,
{
    // ANCHOR: solve
    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, ts, _stop_reason) = solver.solve(40.0).unwrap();

    let x: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
    let time: Vec<_> = ts.into_iter().collect();

    let x_line = Scatter::new(time.clone(), x).mode(Mode::Lines);

    let mut plot = Plot::new();
    plot.add_trace(x_line);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new().title("x"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("sping-mass-system"));
    fs::write("book/src/primer/images/spring-mass-system.html", plot_html)
        .expect("Unable to write file");
    // ANCHOR_END: solve
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{DenseMatrix, Vector, VectorView};

    #[test]
    fn diffsl_and_rust_closures_agree() {
        let times: Vec<_> = (0..=400).map(|i| i as f64 * 0.1).collect();

        let diffsl_problem = problem_diffsl();
        let rtol = diffsl_problem.rtol;
        let atol = diffsl_problem.atol.clone();
        let mut diffsl_solver = diffsl_problem.bdf::<LS>().unwrap();
        let (diffsl_solution, _) = diffsl_solver.solve_dense(&times).unwrap();

        let closures_problem = problem_rust_closures();
        let mut closures_solver = closures_problem.bdf::<LS>().unwrap();
        let (closures_solution, _) = closures_solver.solve_dense(&times).unwrap();

        for col in 0..times.len() {
            let diffsl = diffsl_solution.column(col).into_owned();
            let closures = closures_solution.column(col).into_owned();
            diffsl.assert_eq_norm(&closures, &atol, rtol, 2.0);
        }
    }
}
