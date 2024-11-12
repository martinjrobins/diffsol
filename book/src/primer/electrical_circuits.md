# Example: Electrical Circuits

Lets consider the following low-pass LRC filter circuit:

```plaintext
    +---L---+---C---+
    |       |       |
V_s =       R       |
    |       |       |
    +-------+-------+
```

The circuit consists of a resistor \\(R\\), an inductor \\(L\\), and a capacitor \\(C\\) connected in series to a voltage source \\(V_s\\). The voltage across the resistor \\(V\\) is given by Ohm's law:

\\[
V = R i_R
\\]

where \\(i_R\\) is the current flowing through the resistor. The voltage across the inductor is given by:

\\[
\frac{di_L}{dt} = \frac{V_s - V}{L}
\\]

where \\(di_L/dt\\) is the rate of change of current with respect to time. The voltage across the capacitor is the same as the voltage across the resistor and the equation for an ideal capacitor is:

\\[
\frac{dV}{dt} = \frac{i_C}{C}
\\]

where \\(i_C\\) is the current flowing through the capacitor. The sum of the currents flowing into and out of the top-center node of the circuit must be zero according to Kirchhoff's current law:


\\[
i_L = i_R + i_C
\\]

Thus we have a system of two differential equations and two algebraic equation that describe the evolution of the currents through the resistor, inductor, and capacitor, and the voltage across the resistor. We can write these equations in the general form:

\\[
M \frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

where

\\[
\mathbf{y} = \begin{bmatrix} i_R \\\\ i_L \\\\ i_C \\\\ V \end{bmatrix}
\\]

and

\\[
\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} V - R i_R \\\\ \frac{V_s - V}{L} \\\\ i_L - i_R - i_C \\\\ \frac{i_C}{C} \end{bmatrix}
\\]

The mass matrix \\(M\\) has one on the diagonal for the differential equations and zero for the algebraic equation. 

\\[
M = \begin{bmatrix} 0 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \end{bmatrix}
\\]

Instead of providing the mass matrix explicitly, the DiffSL language specifies the multiplication of the mass matrix with the derivative term, \\(M \frac{d\mathbf{y}}{dt}\\), which is given by:

\\[
M \frac{d\mathbf{y}}{dt} = \begin{bmatrix} 0 \\\\ \frac{di_L}{dt} \\\\ 0 \\\\ \frac{dV}{dt} \end{bmatrix}
\\]

The initial conditions for the system are:

\\[
\mathbf{y}(0) = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix}
\\]

The voltage source \\(V_s\\) acts as a forcing function for the system, and we can specify this as sinusoidal function of time.

\\[
    V_s(t) = V_0 \sin(omega t)
\\]

where \\(omega\\) is the angular frequency of the source. Since this is a low-pass filter, we will choose a high frequency for the source, say \\(omega = 200\\), to demonstrate the filtering effect of the circuit.

We can solve this system of equations using DiffSol and plot the current and voltage across the resistor as a function of time. 

```rust
# fn main() {
# use std::fs;
use diffsol::{
    DiffSl, CraneliftModule, OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod
};
use plotly::{
    Plot, Scatter, common::Mode, layout::Layout, layout::Axis
};
type M = nalgebra::DMatrix<f64>;
type CG = CraneliftModule;
        
let eqn = DiffSl::<M, CG>::compile("
    R { 100.0 } L { 1.0 } C { 0.001 } V0 { 10 } omega { 100.0 }
    Vs { V0 * sin(omega * t) }
    u_i {
        iR = 0,
        iL = 0,
        iC = 0,
        V = 0,
    }
    dudt_i {
        diRdt = 0,
        diLdt = 0,
        diCdt = 0,
        dVdt = 0,
    }
    M_i {
        0,
        diLdt,
        0,
        dVdt,
    }
    F_i {
        V - R * iR,
        (Vs - V) / L,
        iL - iR - iC,
        iC / C,
    }
    out_i {
        iR,
    }
").unwrap();
let problem = OdeBuilder::new()
    .build_from_eqn(eqn).unwrap();
let mut solver = Bdf::default();
let state = OdeSolverState::new(&problem, &solver).unwrap();
let (ys, ts) = solver.solve(&problem, state, 1.0).unwrap();

let ir: Vec<_> = ys.row(0).into_iter().copied().collect();
let t: Vec<_> = ts.into_iter().collect();

let ir = Scatter::new(t.clone(), ir).mode(Mode::Lines);

let mut plot = Plot::new();
plot.add_trace(ir);

let layout = Layout::new()
    .x_axis(Axis::new().title("t"))
    .y_axis(Axis::new().title("current"));
plot.set_layout(layout);
let plot_html = plot.to_inline_html(Some("electrical-circuit"));
# fs::write("../src/primer/images/electrical-circuit.html", plot_html).expect("Unable to write file");
# }
```
{{#include images/electrical-circuit.html}}
