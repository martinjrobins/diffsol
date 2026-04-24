use diffsol::{
    LlvmModule, MatrixCommon, OdeBuilder, OdeEquations, Op, SensitivitiesOdeSolverMethod, Vector,
};
use plotly::{common::AxisSide, common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type V = diffsol::NalgebraVec<f64>;
type CG = LlvmModule;
type LS = diffsol::NalgebraLU<f64>;

fn trapezoid_integral(ts: &[f64], values: &[f64]) -> f64 {
    ts.windows(2)
        .zip(values.windows(2))
        .map(|(t, y)| 0.5 * (t[1] - t[0]) * (y[0] + y[1]))
        .sum()
}

pub fn main() {
    let model = r#"
        in_i { dose = 1000.0 }
        Vc { 1000.0 } Vp1 { 1000.0 } CL { 100.0 } Qp1 { 50.0 }
        u_i {
            qc = dose,
            qp1 = 0,
        }
        F_i {
            -qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }
        stop_i {
            t - 6.0,
            t - 12.0,
            t - 18.0,
        }
        reset_i {
            qc + dose,
            qp1,
        }
        out_i {
            pow(qc / Vc, 2),
        }
    "#;

    let t_eval = (0..=240)
        .map(|i| f64::from(i) * 24.0 / 240.0)
        .collect::<Vec<_>>();
    let dose_levels = (0..=12)
        .map(|i| 250.0 + f64::from(i) * 125.0)
        .collect::<Vec<_>>();

    let mut problem = OdeBuilder::<M>::new()
        .p([dose_levels[0]])
        .sens_atol([1e-8])
        .sens_rtol(1e-8)
        .build_from_diffsl::<CG>(model)
        .unwrap();

    let mut aucs = Vec::with_capacity(dose_levels.len());
    let mut auc_grads = Vec::with_capacity(dose_levels.len());

    for &dose in &dose_levels {
        let ctx = *problem.eqn().context();
        problem.eqn_mut().set_params(&V::from_vec(vec![dose], ctx));

        let mut solver = problem.bdf_sens::<LS>().unwrap();
        let (concentration_squared, concentration_squared_sens, _stop_reason) =
            solver.solve_dense_sensitivities(&t_eval).unwrap();

        let c2 = concentration_squared
            .inner()
            .row(0)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let dc2_ddose = concentration_squared_sens[0]
            .inner()
            .row(0)
            .iter()
            .copied()
            .collect::<Vec<_>>();

        aucs.push(trapezoid_integral(&t_eval, &c2));
        auc_grads.push(trapezoid_integral(&t_eval, &dc2_ddose));
    }

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(dose_levels.clone(), aucs)
            .mode(Mode::LinesMarkers)
            .name("AUC2"),
    );
    plot.add_trace(
        Scatter::new(dose_levels, auc_grads)
            .mode(Mode::LinesMarkers)
            .name("dAUC2/ddose")
            .y_axis("y2"),
    );
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("dose [ng]"))
            .y_axis(Axis::new().title("AUC2 [ng^2 h / mL^2]"))
            .y_axis2(
                Axis::new()
                    .title("dAUC2/ddose [ng h / mL^2]")
                    .overlaying("y")
                    .side(AxisSide::Right)
                    .position(1.0),
            ),
    );

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/primer/images/drug-delivery-dose-sensitivity.html");
    let plot_html = plot.to_inline_html(Some("drug-delivery-dose-sensitivity"));
    fs::write(output_path, plot_html).expect("Unable to write file");
}
