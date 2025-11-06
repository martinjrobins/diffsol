use diffsol::{
    CraneliftJitModule, FaerSparseLU, FaerSparseMat, FaerVec, NonLinearOp, OdeBuilder,
    OdeEquations, OdeSolverMethod, OdeSolverStopReason, Vector,
};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = FaerSparseMat<f64>;
type V = FaerVec<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftJitModule;

fn main() {
    let file = std::fs::read_to_string("../src/primer/src/spm.ds").unwrap();

    let mut problem = OdeBuilder::<M>::new()
        .p([1.0])
        .build_from_diffsl::<CG>(&file)
        .unwrap();
    let currents = vec![0.6, 0.8, 1.0, 1.2, 1.4];
    let final_time = 3600.0;
    let delta_t = 3.0;

    let mut plot = Plot::new();
    for current in currents {
        problem
            .eqn
            .set_params(&V::from_vec(vec![current], *problem.context()));

        let mut solver = problem.bdf::<LS>().unwrap();
        let mut v = Vec::new();
        let mut t = Vec::new();

        // save the initial output
        let mut out = problem
            .eqn
            .out()
            .unwrap()
            .call(solver.state().y, solver.state().t);
        v.push(out[0]);
        t.push(0.0);

        // solve until the final time is reached
        // or we reach the stop condition
        solver.set_stop_time(final_time).unwrap();
        let mut next_output_time = delta_t;
        let mut finished = false;
        while !finished {
            let curr_t = match solver.step() {
                Ok(OdeSolverStopReason::InternalTimestep) => solver.state().t,
                Ok(OdeSolverStopReason::RootFound(t)) => {
                    finished = true;
                    t
                }
                Ok(OdeSolverStopReason::TstopReached) => {
                    finished = true;
                    final_time
                }
                Err(_) => panic!("unexpected solver error"),
            };
            while curr_t > next_output_time {
                let y = solver.interpolate(next_output_time).unwrap();
                problem
                    .eqn
                    .out()
                    .unwrap()
                    .call_inplace(&y, next_output_time, &mut out);
                v.push(out[0]);
                t.push(next_output_time);
                next_output_time += delta_t;
            }
        }

        let voltage = Scatter::new(t, v)
            .mode(Mode::Lines)
            .name(format!("current = {current} A"));
        plot.add_trace(voltage);
    }

    let layout = Layout::new()
        .x_axis(Axis::new().title("t [sec]"))
        .y_axis(Axis::new().title("voltage [V]"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("battery-simulation"));
    fs::write("../src/primer/images/battery-simulation.html", plot_html)
        .expect("Unable to write file");
}
