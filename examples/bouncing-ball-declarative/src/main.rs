use diffsol::{CraneliftJitModule, OdeBuilder, OdeSolverMethod, OdeSolverStopReason};
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

    let mut x = Vec::new();
    let mut v = Vec::new();
    let mut t = Vec::new();
    let final_time = 10.0;

    x.push(solver.state().y[0]);
    v.push(solver.state().y[1]);
    t.push(solver.state().t);

    solver.set_stop_time(final_time).unwrap();
    loop {
        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => {}
            Ok(OdeSolverStopReason::RootFound(t_root, _)) => {
                solver.state_mut_back(t_root).unwrap();
                solver.apply_reset().expect("reset must apply cleanly");
                if solver.state().t < final_time {
                    solver.set_stop_time(final_time).unwrap();
                } else {
                    break;
                }
            }
            Ok(OdeSolverStopReason::TstopReached) => break,
            Err(err) => panic!("unexpected solver error: {err}"),
        }
        x.push(solver.state().y[0]);
        v.push(solver.state().y[1]);
        t.push(solver.state().t);
    }

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
