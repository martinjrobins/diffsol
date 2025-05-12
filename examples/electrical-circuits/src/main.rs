use diffsol::{CraneliftJitModule, MatrixCommon, OdeBuilder, OdeSolverMethod};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
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
    ",
        )
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, ts) = solver.solve(1.0).unwrap();

    let ir: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
    let t: Vec<_> = ts.into_iter().collect();

    let ir = Scatter::new(t.clone(), ir).mode(Mode::Lines);

    let mut plot = Plot::new();
    plot.add_trace(ir);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new().title("current"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("electrical-circuit"));
    fs::write("../src/primer/images/electrical-circuit.html", plot_html)
        .expect("Unable to write file");
}
