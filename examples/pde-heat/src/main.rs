use diffsol::{
    CraneliftJitModule, FaerSparseLU, FaerSparseMat, MatrixCommon, OdeBuilder, OdeSolverMethod,
};
use plotly::{
    layout::{Axis, Layout},
    Plot, Surface,
};
use std::fs;

type M = FaerSparseMat<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftJitModule;

fn main() {
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
    D { 0.1 }
    h { 1.0 / 21.0}
    g { 0.0 }
    m { 1.0 }
    A_ij {
        (0..20, 1..21): 1.0,
        (0..21, 0..21): -2.0,
        (1..21, 0..20): 1.0,
    }
    b_i { 
        (0): g,
        (1:20): 0.0,
        (20): g,
    }
    u_i {
        (0:5): g,
        (5:15): g + m,
        (15:21): g,
    }
    heat_i { A_ij * u_j }
    F_i {
        D * (heat_i + b_i) / (h * h)
    }",
        )
        .unwrap();
    let times = (0..100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>();
    let mut solver = problem.bdf::<LS>().unwrap();
    let sol = solver.solve_dense(&times).unwrap();

    let x = (1..=20).map(|i| i as f64 / 21.0).collect::<Vec<f64>>();
    let y = times;
    let z = sol
        .inner()
        .col_iter()
        .map(|v| v.iter().copied().collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();
    let trace = Surface::new(z).x(x).y(y);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    let layout = Layout::new()
        .x_axis(Axis::new().title("x"))
        .y_axis(Axis::new().title("t"))
        .z_axis(Axis::new().title("u"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("heat-equation"));
    fs::write("book/src/primer/images/heat-equation.html", plot_html)
        .expect("Unable to write file");
}
