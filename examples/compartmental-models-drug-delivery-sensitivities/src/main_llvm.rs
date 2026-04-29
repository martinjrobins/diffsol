use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, LlvmModule, MatrixCommon, OdeBuilder, OdeEquations,
    OdeSolverMethod, OdeSolverState, Op, Vector,
};
use plotly::{common::AxisSide, common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type V = diffsol::NalgebraVec<f64>;
type CG = LlvmModule;
type LS = diffsol::NalgebraLU<f64>;

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

    let final_time = 24.0;
    let dose_levels = (0..=12)
        .map(|i| 250.0 + f64::from(i) * 125.0)
        .collect::<Vec<_>>();

    let mut problem = OdeBuilder::<M>::new()
        .p([dose_levels[0]])
        .sens_atol([1e-8])
        .sens_rtol(1e-8)
        .out_atol([1e-8])
        .out_rtol(1e-8)
        .integrate_out(true)
        .build_from_diffsl::<CG>(model)
        .unwrap();

    let mut aucs = Vec::with_capacity(dose_levels.len());
    let mut auc_grads = Vec::with_capacity(dose_levels.len());

    for &dose in &dose_levels {
        let ctx = *problem.eqn().context();
        problem.eqn_mut().set_params(&V::from_vec(vec![dose], ctx));

        let mut solver = problem.bdf::<LS>().unwrap();
        let (checkpoints, integrated_output, _output_times, _stop_reason) =
            solver.solve_with_checkpointing(final_time, None).unwrap();
        let auc = integrated_output.get_index(0, integrated_output.ncols() - 1);

        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpoints, Some(solver), None)
            .unwrap();
        let (adjoint_state, _) = adjoint_solver
            .solve_adjoint_backwards_pass(&[], &[])
            .unwrap();

        aucs.push(auc);
        auc_grads.push(adjoint_state.as_ref().sg[0].get_index(0));
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
