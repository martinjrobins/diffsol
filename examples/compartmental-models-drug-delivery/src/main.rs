use diffsol::{CraneliftJitModule, OdeBuilder, OdeSolverMethod, OdeSolverStopReason};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};
type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
        Vc { 1000.0 } Vp1 { 1000.0 } CL { 100.0 } Qp1 { 50.0 }
        u_i {
            centralamount = 0,
            peripheralamount = 0,
        }
        F_i {
            - centralamount / Vc * CL - Qp1 * (centralamount / Vc - peripheralamount / Vp1),
            Qp1 * (centralamount / Vc - peripheralamount / Vp1),
        }
    ",
        )
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let doses = vec![(0.0, 1000.0), (6.0, 1000.0), (12.0, 1000.0), (18.0, 1000.0)];

    let mut central_amount = Vec::new();
    let mut peripheral_amount = Vec::new();
    let mut time = Vec::new();

    // apply the first dose and save the initial state
    solver.state_mut().y[0] = doses[0].1;
    central_amount.push(solver.state().y[0]);
    peripheral_amount.push(solver.state().y[1]);
    time.push(0.0);

    // solve to each remaining dose time and apply the dose,
    // then solve to the final time (a zero dose at t = 24 h)
    let final_time = 24.0;
    let segments = doses
        .into_iter()
        .skip(1)
        .chain(std::iter::once((final_time, 0.0)));
    for (t, dose) in segments {
        solver.set_stop_time(t).unwrap();
        loop {
            let ret = solver.step();
            central_amount.push(solver.state().y[0]);
            peripheral_amount.push(solver.state().y[1]);
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
    let central_amount = Scatter::new(time.clone(), central_amount)
        .mode(Mode::Lines)
        .name("central_amount");
    let peripheral_amount = Scatter::new(time, peripheral_amount)
        .mode(Mode::Lines)
        .name("peripheral_amount");
    plot.add_trace(central_amount);
    plot.add_trace(peripheral_amount);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t [h]"))
        .y_axis(Axis::new().title("amount [ng]"));
    plot.set_layout(layout);
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/primer/images/drug-delivery.html");
    let plot_html = plot.to_inline_html(Some("drug-delivery"));
    fs::write(output_path, plot_html).expect("Unable to write file");
}
