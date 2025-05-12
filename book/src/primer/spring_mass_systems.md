# Example: Spring-mass systems

We will model a [damped spring-mass system](https://en.wikipedia.org/wiki/Mass-spring-damper_model) using a second order ODE. The system consists of a mass \\(m\\) attached to a spring with spring constant \\(k\\), and a damping force proportional to the velocity of the mass with damping coefficient \\(c\\). 

<img src="images/Mass_spring_damper.svg" width="350">

The equation of motion for the mass can be written as:

\\[
m \frac{d^2x}{dt^2} = -k x - c \frac{dx}{dt}
\\]

where \\(x\\) is the position of the mass, \\(t\\) is time, and the negative sign on the right hand side indicates that the spring force and damping force act in the opposite direction to the displacement of the mass. 

We can convert this to a system of two first order ODEs by introducing a new variable for the velocity of the mass:

\\[
\begin{align*}
\frac{dx}{dt} &= v \\\\
\frac{dv}{dt} &= -\frac{k}{m} x - \frac{c}{m} v
\end{align*}
\\]

where \\(v = \frac{dx}{dt}\\) is the velocity of the mass.

We can solve this system of ODEs using Diffsol with the following code:

```rust
# fn main() {
# use std::fs;
use diffsol::{
    DiffSl, CraneliftModule, OdeBuilder, OdeSolverMethod
};
use plotly::{
    Plot, Scatter, common::Mode, layout::Layout, layout::Axis
};
type M = nalgebra::DMatrix<f64>;
type CG = CraneliftModule;
type LS = diffsol::NalgebraLU<f64>;
        
let eqn = DiffSl::<M, CG>::compile("
    k { 1.0 } m { 1.0 } c { 0.1 }
    u_i {
        x = 1,
        v = 0,
    }
    F_i {
        v,
        -k/m * x - c/m * v,
    }
").unwrap();
let problem = OdeBuilder::<M>::new()
    .build_from_eqn(eqn).unwrap();
let mut solver = problem.bdf::<LS>().unwrap();
let (ys, ts) = solver.solve(40.0).unwrap();

let x: Vec<_> = ys.row(0).into_iter().copied().collect();
let time: Vec<_> = ts.into_iter().collect();

let x_line = Scatter::new(time.clone(), x).mode(Mode::Lines);

let mut plot = Plot::new();
plot.add_trace(x_line);

let layout = Layout::new()
    .x_axis(Axis::new().title("t"))
    .y_axis(Axis::new().title("x"));
plot.set_layout(layout);
let plot_html = plot.to_inline_html(Some("sping-mass-system"));
# fs::write("../src/primer/images/spring-mass-system.html", plot_html).expect("Unable to write file");
# }
```
{{#include images/spring-mass-system.html}}
