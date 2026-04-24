use diffsol::{CraneliftJitModule, OdeBuilder, OdeSolverMethod, OdeSolverStopReason, Vector};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let e = 0.8;
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
        g { 9.81 } h { 10.0 }
        u_i {
            position = h,
            velocity = 0,
        }
        F_i {
            velocity,
            -g,
        }
        stop {
            position,
        }
    ",
        )
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();

    let mut position = Vec::new();
    let mut velocity = Vec::new();
    let mut t = Vec::new();
    let final_time = 10.0;

    // save the initial state
    position.push(solver.state().y[0]);
    velocity.push(solver.state().y[1]);
    t.push(0.0);

    // solve until the final time is reached
    solver.set_stop_time(final_time).unwrap();
    loop {
        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => (),
            Ok(OdeSolverStopReason::RootFound(t, _idx)) => {
                // get the state when the event occurred
                let mut y = solver.interpolate(t).unwrap();

                // update the velocity of the ball
                y[1] *= -e;

                // make sure the ball is above the ground
                y[0] = y[0].max(f64::EPSILON);

                // set the state to the updated state
                solver.state_mut().y.copy_from(&y);
                solver.state_mut().dy[0] = y[1];
                *solver.state_mut().t = t;
            }
            Ok(OdeSolverStopReason::TstopReached) => break,
            Err(_) => panic!("unexpected solver error"),
        }
        position.push(solver.state().y[0]);
        velocity.push(solver.state().y[1]);
        t.push(solver.state().t);
    }
    let mut plot = Plot::new();
    let position = Scatter::new(t.clone(), position)
        .mode(Mode::Lines)
        .name("position");
    let velocity = Scatter::new(t, velocity).mode(Mode::Lines).name("velocity");
    plot.add_trace(position);
    plot.add_trace(velocity);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new());
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("bouncing-ball"));
    fs::write("../src/primer/images/bouncing-ball.html", plot_html).expect("Unable to write file");
}
