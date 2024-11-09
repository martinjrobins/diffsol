# Population Dynamics - Predator-Prey Model

In this example, we will model the population dynamics of a predator-prey system using a set of first order ODEs. The [Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) are a classic example of a predator-prey model, and describe the interactions between two species in an ecosystem. The first species is a prey species, which we will call \\(x\\), and the second species is a predator species, which we will call \\(y\\).

The rate of change of the prey population is governed by two terms: growth and predation. The growth term represents the natural increase in the prey population in the absence of predators, and is proportional to the current population of prey. The predation term represents the rate at which the predators consume the prey, and is proportional to the product of the prey and predator populations. The rate of change of the prey population can be written as:

\\[
\frac{dx}{dt} = a x - b x y 
\\]

where \\(a\\) is the growth rate of the prey population, and \\(b\\) is the predation rate.

The rate of change of the predator population is also governed by two terms: death and growth. The death term represents the natural decrease in the predator population in the absence of prey, and is proportional to the current population of predators. The growth term represents the rate at which the predators reproduce, and is proportional to the product of the prey and predator populations, since the predators need to consume the prey to reproduce. The rate of change of the predator population can be written as:

\\[
\frac{dy}{dt} = c x y - d y
\\]

where \\(c\\) is the reproduction rate of the predators, and \\(d\\) is the death rate.

The Lotka-Volterra equations are a simple model of predator-prey dynamics, and make several assumptions that may not hold in real ecosystems. For example, the model assumes that the prey population grows exponentially in the absence of predators, that the predator population decreases linearly in the absence of prey, and that the spatial distribution of the species has no effect. Despite these simplifications, the Lotka-Volterra equations capture some of the essential features of predator-prey interactions, such as oscillations in the populations and the dependence of each species on the other. When modelling with ODEs, it is important to consider the simplest model that captures the behaviour of interest, and to be aware of the assumptions that underlie the model.

Putting the two equations together, we have a system of two first order ODEs:

\\[
\frac{x}{dt} = a x - b x y \\\\
\frac{y}{dt} = c x y - d y
\\]

which can be written in vector form as:

\\[
\begin{bmatrix}
\frac{dx}{dt} \\\\
\frac{dy}{dt}
\end{bmatrix} = \begin{bmatrix}
a x - b x y \\\\
c x y - d y
\end{bmatrix}
\\]

or in the general form of a first order ODE system:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

where 

\\[\mathbf{y} = \begin{bmatrix} x \\\\ y \end{bmatrix} \\] 

and 

\\[\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} a x - b x y \\\\ c x y - d y \end{bmatrix}\\]

We also have initial conditions for the populations at time \\(t = 0\\). We can set both populations to 1 at the start like so:

\\[
\mathbf{y}(0) = \begin{bmatrix} 1 \\\\ 1 \end{bmatrix}
\\]

Let's solve this system of ODEs using the DiffSol crate. We will use the [DiffSL](https://martinjrobins.github.io/diffsl/) domain-specific language to specify the problem. We could have also used [Rust closures](specify/ode_equations.md), but this allows us to illustrate the modelling process with a minimum of Rust syntax.

```rust
# fn main() {
# use std::fs;
use diffsol::{DiffSl, CraneliftModule, OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod};
use plotly::{Plot, Scatter, common::Mode, layout::Layout, layout::Axis};
type M = nalgebra::DMatrix<f64>;
type CG = CraneliftModule; // use LlvmModule for LLVM, this has a faster runtime but requires LLVM to be installed
        
let eqn = DiffSl::<M, CG>::compile("
    a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
    u_i {
        y1 = 1,
        y2 = 1,
    }
    F_i {
        a * y1 - b * y1 * y2,
        c * y1 * y2 - d * y2,
    }
").unwrap();
let problem = OdeBuilder::new()
    .build_from_eqn(eqn).unwrap();
let mut solver = Bdf::default();
let state = OdeSolverState::new(&problem, &solver).unwrap();
let (ys, ts) = solver.solve(&problem, state, 40.0).unwrap();

let prey: Vec<_> = ys.row(0).into_iter().copied().collect();
let predator: Vec<_> = ys.row(1).into_iter().copied().collect();
let time: Vec<_> = ts.into_iter().collect();

let prey = Scatter::new(time.clone(), prey).mode(Mode::Lines).name("Prey");
let predator = Scatter::new(time, predator).mode(Mode::Lines).name("Predator");

let mut plot = Plot::new();
plot.add_trace(prey);
plot.add_trace(predator);

let layout = Layout::new()
    .x_axis(Axis::new().title("t"))
    .y_axis(Axis::new().title("population"));
plot.set_layout(layout);

let plot_html = plot.to_inline_html(Some("prey-predator"));
# fs::write("../src/primer/images/prey-predator.html", plot_html).expect("Unable to write file");
# }
```
{{#include images/prey-predator.html}}

A phase plane plot of the predator-prey system is a useful visualisation of the dynamics of the system. This plot shows the prey population on the x-axis and the predator population on the y-axis. Trajectories in the phase plane represent the evolution of the populations over time. Lets refrase the equations to introduce a new parameter \\(y_0\\) which is the initial predator and prey population. We can then plot the phase plane for different values of \\(y_0\\) to see how the system behaves for different initial conditions.

Our initial conditions are now:

\\[
\mathbf{y}(0) = \begin{bmatrix} y_0 \\\\ y_0 \end{bmatrix}
\\]

so we can solve this system for different values of \\(y_0\\) and plot the phase plane for each case. We will use similar code as above, but we will introduce our new parameter and loop over different values of \\(y_0\\)

```rust
# fn main() {
# use std::fs;
use diffsol::{DiffSl, CraneliftModule, OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod};
use plotly::{Plot, Scatter, common::Mode, layout::Layout, layout::Axis};
type M = nalgebra::DMatrix<f64>;
type CG = CraneliftModule; // use LlvmModule for LLVM, this has a faster runtime but requires LLVM to be installed
        
let eqn = DiffSl::<M, CG>::compile("
    in = [ y0 ]
    y0 { 1.0 }
    a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
    u_i {
        y1 = y0,
        y2 = y0,
    }
    F_i {
        a * y1 - b * y1 * y2,
        c * y1 * y2 - d * y2,
    }
").unwrap();

let mut plot = Plot::new();

for y0 in 1..6.map(f64::from) {
    let problem = OdeBuilder::new()
        .p([y0])
        .build_from_eqn(eqn.clone()).unwrap();
    let mut solver = Bdf::default();
    let state = OdeSolverState::new(&problem, &solver).unwrap();
    let (ys, ts) = solver.solve(&problem, state, 40.0).unwrap();

    let prey: Vec<_> = ys.row(0).into_iter().copied().collect();
    let predator: Vec<_> = ys.row(1).into_iter().copied().collect();
    let time: Vec<_> = ts.into_iter().collect();

    let phase = Scatter::new(prey, predator).mode(Mode::Lines).name(format!("y0 = {}", y0));
    plot.add_trace(phase);
}

let layout = Layout::new()
    .x_axis(Axis::new().title("x"))
    .y_axis(Axis::new().title("y"));
plot.set_layout(layout);

let plot_html = plot.to_inline_html(Some("prey-predator2"));
# fs::write("../src/primer/images/prey-predator2.html", plot_html).expect("Unable to write file");
# }
```
{{#include images/prey-predator.html}}


