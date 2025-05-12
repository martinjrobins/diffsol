use diffsol::{CraneliftJitModule, MatrixCommon, OdeBuilder, OdeSolverMethod};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let problem = OdeBuilder::<M>::new()
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
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, ts) = solver.solve(40.0).unwrap();

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
    fs::write("../src/primer/images/spring-mass-system.html", plot_html)
        .expect("Unable to write file");
}
