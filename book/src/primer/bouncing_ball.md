# Example: Bouncing Ball

Modelling a bouncing ball is a simple and intuitive example of a system with discrete events. The ball is dropped from a height \\(h\\) and bounces off the ground with a coefficient of restitution \\(e\\). When the ball hits the ground, its velocity is reversed and scaled by the coefficient of restitution, and the ball rises and then continues to fall until it hits the ground again. This process repeats until halted.

The second order ODE that describes the motion of the ball is given by:

\\[
\frac{d^2x}{dt^2} = -g
\\]

where \\(x\\) is the position of the ball, \\(t\\) is time, and \\(g\\) is the acceleration due to gravity. We can rewrite this as a system of two first order ODEs by introducing a new variable for the velocity of the ball:

\\[
\begin{align*}
\frac{dx}{dt} &= v \\\\
\frac{dv}{dt} &= -g
\end{align*}
\\]

where \\(v = \frac{dx}{dt}\\) is the velocity of the ball. This is a system of two first order ODEs, which can be written in vector form as:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

where

\\[
\mathbf{y} = \begin{bmatrix} x \\\\ v \end{bmatrix}
\\]

and

\\[
\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} v \\\\ -g \end{bmatrix}
\\]

The initial conditions for the ball, including the height from which it is dropped and its initial velocity, are given by:

\\[
\mathbf{y}(0) = \begin{bmatrix} h \\\\ 0 \end{bmatrix}
\\]

When the ball hits the ground, we need to update the velocity of the ball according to the coefficient of restitution, which is the ratio of the velocity after the bounce to the velocity before the bounce. The velocity after the bounce \\(v'\\) is given by:

\\[
v' = -e v
\\]

where \\(e\\) is the coefficient of restitution. However, to implement this in our ODE solver, we need to detect when the ball hits the ground. We can do this by using DiffSol's event handling feature, which allows us to specify a function that is equal to zero when the event occurs, i.e. when the ball hits the ground. This function \\(g(\mathbf{y}, t)\\) is called an event or root function, and for our bouncing ball problem, it is given by:

\\[
g(\mathbf{y}, t) = x
\\]

where \\(x\\) is the position of the ball. When the ball hits the ground, the event function will be zero and DiffSol will stop the integration, and we can update the velocity of the ball accordingly.

In code, the bouncing ball problem can be solved using DiffSol as follows:

```rust
# fn main() {
# use std::fs;
use diffsol::{
    DiffSl, CraneliftModule, OdeBuilder, OdeSolverMethod, OdeSolverStopReason,
};
use plotly::{
    Plot, Scatter, common::Mode, layout::Layout, layout::Axis
};
type M = nalgebra::DMatrix<f64>;
type CG = CraneliftModule;
type LS = diffsol::NalgebraLU<f64>;
        
let eqn = DiffSl::<M, CG>::compile("
    g { 9.81 } h { 10.0 }
    u_i {
        x = h,
        v = 0,
    }
    F_i {
        v,
        -g,
    }
    stop {
        x,
    }
").unwrap();

let e = 0.8;
let problem = OdeBuilder::<M>::new().build_from_eqn(eqn).unwrap();
let mut solver = problem.bdf::<LS>().unwrap();

let mut x = Vec::new();
let mut v = Vec::new();
let mut t = Vec::new();
let final_time = 10.0;

// save the initial state
x.push(solver.state().y[0]);
v.push(solver.state().y[1]);
t.push(0.0);

// solve until the final time is reached
solver.set_stop_time(final_time).unwrap();
loop {
    match solver.step() {
        Ok(OdeSolverStopReason::InternalTimestep) => (),
        Ok(OdeSolverStopReason::RootFound(t)) => {
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
        },
        Ok(OdeSolverStopReason::TstopReached) => break,
        Err(_) => panic!("unexpected solver error"),
    }
    x.push(solver.state().y[0]);
    v.push(solver.state().y[1]);
    t.push(solver.state().t);
}
let mut plot = Plot::new();
let x = Scatter::new(t.clone(), x).mode(Mode::Lines).name("x");
let v = Scatter::new(t, v).mode(Mode::Lines).name("v");
plot.add_trace(x);
plot.add_trace(v);

let layout = Layout::new()
    .x_axis(Axis::new().title("t"))
    .y_axis(Axis::new());
plot.set_layout(layout);
let plot_html = plot.to_inline_html(Some("bouncing-ball"));
# fs::write("../src/primer/images/bouncing-ball.html", plot_html).expect("Unable to write file");
# }
```
{{#include images/bouncing-ball.html}}