use diffsol::{CraneliftJitModule, MatrixCommon, OdeBuilder, OdeSolverMethod};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let dose = 1000.0;
    let model = format!(
        r#"
        Vc {{ 1000.0 }} Vp1 {{ 1000.0 }} CL {{ 100.0 }} Qp1 {{ 50.0 }} dose {{ {dose} }}
        u_i {{
            qc = dose,
            qp1 = 0,
        }}
        F_i {{
            -qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }}
        stop_i {{
            t - 6.0,
            t - 12.0,
            t - 18.0,
        }}
        reset_i {{
            qc + dose,
            qp1,
        }}
    "#
    );
    let problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(&model)
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();

    let final_time = 24.0;
    let (ys, time, _stop_reason) = solver.solve(final_time).unwrap();
    let q_c: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
    let q_p1: Vec<_> = ys.inner().row(1).into_iter().copied().collect();

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(time.clone(), q_c)
            .mode(Mode::Lines)
            .name("q_c"),
    );
    plot.add_trace(Scatter::new(time, q_p1).mode(Mode::Lines).name("q_p1"));
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("t [h]"))
            .y_axis(Axis::new().title("amount [ng]")),
    );

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/primer/images/drug-delivery-declarative.html");
    let plot_html = plot.to_inline_html(Some("drug-delivery-declarative"));
    fs::write(output_path, plot_html).expect("Unable to write file");
}
