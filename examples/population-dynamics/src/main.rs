use diffsol::{
    CraneliftJitModule, MatrixCommon, NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod,
    Vector,
};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::fs;
type M = diffsol::NalgebraMat<f64>;
type LS = diffsol::NalgebraLU<f64>;
type CG = CraneliftJitModule;

fn main() {
    solve();
    phase_plane();
}

fn solve() {
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(
            "
        a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
        u_i {
            y1 = 1,
            y2 = 1,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }
    ",
        )
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, ts) = solver.solve(40.0).unwrap();

    let prey: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
    let predator: Vec<_> = ys.inner().row(1).into_iter().copied().collect();
    let time: Vec<_> = ts.into_iter().collect();

    let prey = Scatter::new(time.clone(), prey)
        .mode(Mode::Lines)
        .name("Prey");
    let predator = Scatter::new(time, predator)
        .mode(Mode::Lines)
        .name("Predator");

    let mut plot = Plot::new();
    plot.add_trace(prey);
    plot.add_trace(predator);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new().title("population"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("prey-predator"));
    fs::write("book/src/primer/images/prey-predator.html", plot_html)
        .expect("Unable to write file");
}

fn phase_plane() {
    let mut problem = OdeBuilder::<M>::new()
        .p([1.0])
        .build_from_diffsl::<CG>(
            "
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
    ",
        )
        .unwrap();

    let mut plot = Plot::new();
    for y0 in (1..6).map(f64::from) {
        let p = NalgebraVec::from_element(1, y0, *problem.context());
        problem.eqn_mut().set_params(&p);

        let mut solver = problem.bdf::<LS>().unwrap();
        let (ys, _ts) = solver.solve(40.0).unwrap();

        let prey: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
        let predator: Vec<_> = ys.inner().row(1).into_iter().copied().collect();

        let phase = Scatter::new(prey, predator)
            .mode(Mode::Lines)
            .name(format!("y0 = {y0}"));
        plot.add_trace(phase);
    }

    let layout = Layout::new()
        .x_axis(Axis::new().title("x"))
        .y_axis(Axis::new().title("y"));
    plot.set_layout(layout);
    let plot_html = plot.to_inline_html(Some("prey-predator2"));
    fs::write("book/src/primer/images/prey-predator2.html", plot_html)
        .expect("Unable to write file");
}
