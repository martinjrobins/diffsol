use diffsol::{CraneliftJitModule, MatrixCommon, OdeBuilder, OdeSolverMethod};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let model = "
        restitution { 0.8 } xeps { 1e-12 }
        g { 9.81 } h { 10.0 }
        u_i {
            position = h,
            velocity = 0,
        }
        F_i {
            velocity,
            -g,
        }
        stop_i {
            position,
        }
        reset_i {
            xeps,
            -restitution * velocity,
        }
    ";
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(&model)
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();

    let final_time = 10.0;
    let (ys, t, _stop_reason) = solver.solve(final_time).unwrap();
    let x: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
    let v: Vec<_> = ys.inner().row(1).into_iter().copied().collect();

    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(t.clone(), x).mode(Mode::Lines).name("x"));
    plot.add_trace(Scatter::new(t, v).mode(Mode::Lines).name("v"));
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("t"))
            .y_axis(Axis::new()),
    );

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/primer/images/bouncing-ball-declarative.html");
    let plot_html = plot.to_inline_html(Some("bouncing-ball-declarative"));
    fs::write(output_path, plot_html).expect("Unable to write file");
}
