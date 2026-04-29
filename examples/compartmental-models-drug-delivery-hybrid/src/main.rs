use diffsol::{
    CraneliftJitModule, MatrixCommon, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverState,
    OdeSolverStopReason, Solution,
};
use plotly::{common::Mode, layout::Axis, layout::Layout, Plot, Scatter};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let model = "
        Vc { 1000.0 } Vp1 { 1000.0 } CL { 100.0 } Qp1 { 50.0 }
        doses_i { 1000.0, 500.0, 2000.0, 1500.0 }
        u_i {
            qc = doses_i[0],
            qp1 = 0,
        }
        F_i {
            -qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }
        stop_i {
            1.0
            t - 6.0,
            t - 12.0,
            t - 18.0,
        }
        reset_i {
            qc + doses_i[N],
            qp1,
        }
    ";
    let mut problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(model)
        .unwrap();

    let final_time = 24.0;
    let mut soln = Solution::new(final_time);
    let mut state = problem.bdf_state::<LS>().unwrap();

    while !soln.is_complete() {
        state = problem
            .bdf_solver::<LS>(state)
            .unwrap()
            .solve_soln(&mut soln)
            .unwrap()
            .into_state();
        if let Some(OdeSolverStopReason::RootFound(_t_root, root_idx)) = soln.stop_reason {
            problem.eqn_mut().set_model_index(root_idx);
            state.as_mut().apply_reset(problem.eqn()).unwrap();
        }
    }

    let time = soln.ts.clone();
    let q_c: Vec<_> = soln.ys.inner().row(0).into_iter().copied().collect();
    let q_p1: Vec<_> = soln.ys.inner().row(1).into_iter().copied().collect();

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
