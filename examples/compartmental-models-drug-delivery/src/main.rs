use diffsol::{CraneliftJitModule, OdeBuilder, OdeSolverMethod, OdeSolverStopReason};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
        Vc { 1000.0 } Vp1 { 1000.0 } CL { 100.0 } Qp1 { 50.0 }
        u_i {
            qc = 0,
            qp1 = 0,
        }
        F_i {
            - qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }
    ",
        )
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let doses = vec![(0.0, 1000.0), (6.0, 1000.0), (12.0, 1000.0), (18.0, 1000.0)];

    let mut q_c = Vec::new();
    let mut q_p1 = Vec::new();
    let mut time = Vec::new();

    // apply the first dose and save the initial state
    solver.state_mut().y[0] = doses[0].1;
    q_c.push(solver.state().y[0]);
    q_p1.push(solver.state().y[1]);
    time.push(0.0);

    // solve and apply the remaining doses
    for (t, dose) in doses.into_iter().skip(1) {
        solver.set_stop_time(t).unwrap();
        loop {
            let ret = solver.step();
            q_c.push(solver.state().y[0]);
            q_p1.push(solver.state().y[1]);
            time.push(solver.state().t);
            match ret {
                Ok(OdeSolverStopReason::InternalTimestep) => continue,
                Ok(OdeSolverStopReason::TstopReached) => break,
                _ => panic!("unexpected solver error"),
            }
        }
        solver.state_mut().y[0] += dose;
    }
    let mut plot = Plot::new();
    let q_c = Scatter::new(time.clone(), q_c)
        .mode(Mode::Lines)
        .name("q_c");
    let q_p1 = Scatter::new(time, q_p1).mode(Mode::Lines).name("q_p1");
    plot.add_trace(q_c);
    plot.add_trace(q_p1);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t [h]"))
        .y_axis(Axis::new().title("amount [ng]"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("drug-delivery"));
    fs::write("../src/primer/images/drug-delivery.html", plot_html).expect("Unable to write file");
}
