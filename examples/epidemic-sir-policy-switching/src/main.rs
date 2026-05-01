use diffsol::{
    CraneliftJitModule, MatrixCommon, OdeBuilder, OdeEquations, OdeSolverMethod,
    OdeSolverStopReason, Solution,
};
use plotly::{
    common::{AxisSide, Line, LineShape, Mode},
    layout::{Axis, Layout},
    Plot, Scatter,
};
use std::{fs, path::PathBuf};

type M = diffsol::NalgebraMat<f64>;
type CG = CraneliftJitModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let model = "
        population { 1000.0 }
        gamma { 0.1 }
        beta_i {
            0.3,
            0.08,
        }
        threshold_i {
            20.0,
            100.0,
        }
        u_i {
            S = 999.0,
            I = 1.0,
            R = 0.0,
        }
        F_i {
            -beta_i[N] * S * I / population,
            beta_i[N] * S * I / population - gamma * I,
            gamma * I,
        }
        stop_i {
            I - threshold_i,
        }
    ";
    let mut problem = OdeBuilder::<M>::new()
        .build_from_diffsl::<CG>(model)
        .unwrap();

    let final_time = 300.0;
    let mut soln = Solution::new(final_time);
    let mut state = problem.bdf_state::<LS>().unwrap();
    let mut switch_times = vec![(0.0, 0)];

    while !soln.is_complete() {
        state = problem
            .bdf_solver::<LS>(state)
            .unwrap()
            .solve_soln(&mut soln)
            .unwrap()
            .into_state();
        if let Some(OdeSolverStopReason::RootFound(t_root, root_idx)) = soln.stop_reason {
            switch_times.push((t_root, root_idx));
            problem.eqn_mut().set_model_index(root_idx);
        }
    }

    let time = soln.ts.clone();
    let susceptible: Vec<_> = soln.ys.inner().row(0).into_iter().copied().collect();
    let infected: Vec<_> = soln.ys.inner().row(1).into_iter().copied().collect();
    let recovered: Vec<_> = soln.ys.inner().row(2).into_iter().copied().collect();
    let lockdown: Vec<_> = time
        .iter()
        .map(|t| {
            switch_times
                .iter()
                .rev()
                .find(|(switch_time, _)| switch_time <= t)
                .map(|(_, index)| *index as f64)
                .unwrap_or(0.0)
        })
        .collect();

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(time.clone(), susceptible)
            .mode(Mode::Lines)
            .name("S"),
    );
    plot.add_trace(
        Scatter::new(time.clone(), infected)
            .mode(Mode::Lines)
            .name("I"),
    );
    plot.add_trace(
        Scatter::new(time.clone(), recovered)
            .mode(Mode::Lines)
            .name("R"),
    );
    plot.add_trace(
        Scatter::new(time, lockdown)
            .mode(Mode::Lines)
            .name("lockdown")
            .line(Line::new().shape(LineShape::Hv))
            .y_axis("y2"),
    );
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("t [days]"))
            .y_axis(Axis::new().title("people"))
            .y_axis2(
                Axis::new()
                    .title("lockdown")
                    .range(vec![-0.05, 1.05])
                    .overlaying("y")
                    .side(AxisSide::Right)
                    .position(1.0),
            ),
    );

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/primer/images/sir-policy-switching.html");
    let plot_html = plot.to_inline_html(Some("sir-policy-switching"));
    fs::write(output_path, plot_html).expect("Unable to write file");
}
