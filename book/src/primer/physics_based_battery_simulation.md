# Physics-based Battery Simulation

Traditional battery models are based on equivalent circuit models, similar to the circuit modelled in section [Electrical Circuits](./electrical_circuits.md). 
These models are simple and computationally efficient, but they lack the ability to capture all of the complex electrochemical processes that occur in a battery.
Physics-based models, on the other hand, are based on the electrochemical processes that occur in the battery, and can provide a more detailed description of the battery's behaviour.
They are parameterized by physical properties of the battery, such as the diffusion coefficients of lithium ions in the electrodes, the reaction rate constants, and the surface area of the electrodes,
and can be used to predict the battery's performance under different operating conditions, once these parameters are known.

The Single Particle Model (SPM) is a physics-based model of a lithium-ion battery. It describes the diffusion of lithium ions in the positive and negative electrodes of the battery over a 1D radial domain, assuming that the properties of the electrodes are uniform across the thickness of the electrode. Here we will describe the equations that govern the SPM, and show how to solve them at different current rates to calculate the terminal voltage of the battery.

## The Single Particle Model state equations

The SPM model only needs to solve for the concentration of lithium ions in the positive and negative electrodes, $c_n$ and $c_p$. The diffusion of lithium ions in each electrode particle $c_i$ is given by:

$$
\frac{\partial c_i}{\partial t} = \nabla \cdot (D_i \nabla c_i)
$$

subject to the following boundary and initial conditions:

$$
\left.\frac{\partial c_i}{\partial r}\right\vert_{r=0} = 0, \quad \left.\frac{\partial c}{\partial r}\right\vert_{r=R_i} = -j_i, \quad \left.c\right\vert_{t=0} = c^0_i
$$

where $c_i$ is the concentration of lithium ions in the positive ($i=n$) or negative ($i=p$) electrode, $D_i$ is the diffusion coefficient, $j_i$ is the interfacial current density, and $c^0_i$ is the concentration at the particle surface.

The  fluxes of lithium ions in the positive and negative electrodes $j_i$ are dependent on the applied current $I$:

$$
\begin{align*}
j_n &= \frac{I}{a_n \delta_n F \mathcal{A}}, \qquad
j_p &= \frac{-I}{a_p \delta_p F \mathcal{A}},
\end{align*}
$$

where $a_i = 3 \epsilon_i / R_i$ is the specific surface area of the electrode, $\epsilon_i$ is the volume fraction of active material, $\delta_i$ is the thickness of the electrode, $F$ is the Faraday constant, and $\mathcal{A}$ is the electrode surface area.

## Output variables for the Single Particle Model

Now that we have defined the equations to solve, we turn to the output variables that we need to calculate from the state variables $c_n$ and $c_p$. The terminal voltage of the battery is given by:

$$
V = U_p(x_p^s) - U_n(x_n^s) + \eta_p - \eta_n
$$

where $U_i$ is the open circuit potential (OCP) of the electrode, $x_i^s = c_i(r=R_i) / c_i^{max}$ is the surface stoichiometry, and $\eta_i$ is the overpotential.

Assuming Butler-Volmer kinetics and $\alpha_i = 0.5$, the overpotential is given by:

$$
\eta_i = \frac{2RT}{F} \sinh^{-1} \left( \frac{j_i F}{2i_{0,i}} \right)
$$

where the exchange current density $i_{0,i}$ is given by:

$$
i_{0,i} = k_i F \sqrt{c_e} \sqrt{c_i(r=R_i)} \sqrt{c_i^{max} - c_i(r=R_i)}
$$

where $c_e$ is the concentration of lithium ions in the electrolyte, and $k_i$ is the reaction rate constant.

## Stopping conditions


## Solving the Single Particle Model using DiffSol

Rather than write down the equations ourselves, we will rely on the [PyBaMM library](https://pybamm.org/) to generate the equations for us. PyBaMM is a Python library that can generate a wide variety of physics-based battery models, using different parameterisations, physics and operating conditions. Combined with [a tool](https://github.com/martinjrobins/pybamm2diffsl) that takes a PyBaMM model and writes it out in the DiffSL language, we can generate [a DiffSL file](src/spm.ds) that can be used to solve the equations of the SPM model described above. We can then use the DiffSol crate to solve the model and calculate the terminal voltage of the battery over a range of current rates.

The code below reads in the DiffSL file, compiles it, and then solves the equation for different current rates. We wish to stop the simulation when either the final time is reached, or when one of the stopping conditions is met. We will output the terminal voltage of the battery at regular intervals during the simulation, because the terminal voltage can change more rapidly than the state variables $c_n$ and $c_p$, particularly during the "knee" of the discharge curve.

The discretised equations result in sparse matrices, so we use the sparse matrix and linear solver modules provided by the [faer](https://github.com/sarah-quinones/faer-rs) crate to solve the equations efficiently.


```rust
use diffsol::{
    DiffSl, OdeBuilder, CraneliftModule, SparseColMat, FaerSparseLU, 
    OdeSolverMethod, OdeSolverStopReason, OdeEquations, NonLinearOp,
};
use plotly::{
    Plot, Scatter, common::Mode, layout::Layout, layout::Axis
};
# use std::fs;
# fn main() {
type M = SparseColMat<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftModule;

let file = std::fs::read_to_string("../src/primer/src/spm.ds").unwrap();

let eqn = DiffSl::<M, CG>::compile(&file).unwrap();
let mut problem = OdeBuilder::<M>::new()
    .p([1.0])
    .build_from_eqn(eqn)
    .unwrap();
let currents = vec![0.6, 0.8, 1.0, 1.2, 1.4];
let final_time = 3600.0;
let delta_t = 3.0;
    
let mut plot = Plot::new();
for current in currents {
    problem.eqn.set_params(&faer::Col::from_fn(1, |_| current));

    let mut solver = problem.bdf::<LS>().unwrap();
    let mut v = Vec::new();
    let mut t = Vec::new();

    // save the initial output
    let mut out = problem.eqn.out().unwrap().call(solver.state().y, solver.state().t);
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
            },
            Ok(OdeSolverStopReason::TstopReached) => {
                finished = true;
                final_time
            },
            Err(_) => panic!("unexpected solver error"),
        };
        while curr_t > next_output_time {
            let y = solver.interpolate(next_output_time).unwrap();
            problem.eqn.out().unwrap().call_inplace(&y, next_output_time, &mut out);
            v.push(out[0]);
            t.push(next_output_time);
            next_output_time += delta_t;
        }
    }

    let voltage = Scatter::new(t, v)
        .mode(Mode::Lines)
        .name(format!("current = {} A", current));
    plot.add_trace(voltage);
}

let layout = Layout::new()
    .x_axis(Axis::new().title("t [sec]"))
    .y_axis(Axis::new().title("voltage [V]"));
plot.set_layout(layout);
let plot_html = plot.to_inline_html(Some("battery-simulation"));
# fs::write("../src/primer/images/battery-simulation.html", plot_html).expect("Unable to write file");
# }
```

{{#include images/battery-simulation.html}}

