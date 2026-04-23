use diffsol::{
    CraneliftJitModule, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverState,
    OdeSolverStopReason,
};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let dose = 1000.0;
    let model = format!(
        r#"
        Vc {{ 1000.0 }} Vp1 {{ 1000.0 }} CL {{ 100.0 }} Qp1 {{ 50.0 }} dose {{ {dose} }}
        u_i {{
            qc = dose,
            qp1 = 0,
        }}
        F_i {{
            -qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }}
        stop_i {{
            t - 6.0,
            t - 12.0,
            t - 18.0,
        }}
        reset_i {{
            qc + dose,
            qp1,
        }}
    "#
    );
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(&model)
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();

    let mut q_c = Vec::new();
    let mut q_p1 = Vec::new();
    let mut time = Vec::new();
    let final_time = 24.0;

    q_c.push(solver.state().y[0]);
    q_p1.push(solver.state().y[1]);
    time.push(solver.state().t);

    solver.set_stop_time(final_time).unwrap();
    loop {
        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => {}
            Ok(OdeSolverStopReason::RootFound(t_root, _)) => {
                solver.state_mut_back(t_root).unwrap();
                let mut state = solver.state_clone();
                {
                    let problem = solver.problem();
                    let reset = problem.eqn.reset().expect("reset must be configured");
                    state
                        .state_mut_op(&problem.eqn, &reset)
                        .expect("reset must apply cleanly");
                }
                solver.set_state(state);
                if solver.state().t < final_time {
                    solver.set_stop_time(final_time).unwrap();
                } else {
                    break;
                }
            }
            Ok(OdeSolverStopReason::TstopReached) => break,
            Err(err) => panic!("unexpected solver error: {err}"),
        }
        q_c.push(solver.state().y[0]);
        q_p1.push(solver.state().y[1]);
        time.push(solver.state().t);
    }

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(time.clone(), q_c)
            .mode(Mode::Lines)
            .name("q_c"),
    );
    plot.add_trace(Scatter::new(time, q_p1).mode(Mode::Lines).name("q_p1"));
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("t [h]"))
            .y_axis(Axis::new().title("amount [ng]")),
    );

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/primer/images/drug-delivery-declarative.html");
    let plot_html = plot.to_inline_html(Some("drug-delivery-declarative"));
    fs::write(output_path, plot_html).expect("Unable to write file");
}
