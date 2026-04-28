#![cfg(feature = "diffsl-llvm")]

mod common;

use common::{
    assert_close, logistic_integral, logistic_state_dr, matrix_host, vector_host, ASSERT_TOL,
    LOGISTIC_X0,
};
use diffsol_c::host_array::FromHostArray;
use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};

fn make_no_reset_stop_ode(ode_solver: OdeSolverType) -> OdeWrapper {
    let logistic_stop_model = r#"
        in_i { r = 1, k = 1, y0 = 0.1 }
        u_i { y = y0 }
        dudt_i { dydt = 0 }
        F_i { r * y * (1.0 - y / k) }
        stop_i { y - 0.6 }
        out_i { r * y }
    "#;
    let ode = OdeWrapper::new_jit(
        logistic_stop_model,
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        ode_solver,
    )
    .unwrap();
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();
    ode
}

fn logistic_stop_time(x0: f64, r: f64, k: f64, y_stop: f64) -> f64 {
    let initial_odds = x0 / (k - x0);
    let stop_odds = y_stop / (k - y_stop);
    (stop_odds / initial_odds).ln() / r
}

#[test]
fn bdf_no_reset_stop_terminates_forward_solves_at_first_root() {
    let params = [2.0, 1.0, LOGISTIC_X0];
    let final_time = 2.0;
    let root_time = logistic_stop_time(LOGISTIC_X0, params[0], params[1], 0.6);
    let t_eval = [0.5, 1.0, 1.5, final_time];

    let ode = make_no_reset_stop_ode(OdeSolverType::Bdf);
    let solution = ode.solve(vector_host(&params), final_time).unwrap();
    let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();
    let ys = solution.get_ys().unwrap();
    let ys = ys.as_array::<f64>().unwrap();
    assert_close(*ts.last().unwrap(), root_time, 5e-4, "solve stop time");
    assert_close(
        ys[(0, ys.ncols() - 1)],
        params[0] * 0.6,
        5e-4,
        "solve stop value",
    );
    assert!(*ts.last().unwrap() < final_time);

    let dense_solution = ode
        .solve_dense(vector_host(&params), vector_host(&t_eval))
        .unwrap();
    let dense_ts = Vec::<f64>::from_host_array(dense_solution.get_ts().unwrap()).unwrap();
    let dense_ys = dense_solution.get_ys().unwrap();
    let dense_ys = dense_ys.as_array::<f64>().unwrap();
    assert_eq!(dense_ts.len(), 3);
    assert_close(dense_ts[0], 0.5, ASSERT_TOL, "dense first time");
    assert_close(dense_ts[1], 1.0, ASSERT_TOL, "dense second time");
    assert_close(dense_ts[2], root_time, 5e-4, "dense stop time");
    assert_eq!(dense_ys.ncols(), dense_ts.len());
    assert_close(
        dense_ys[(0, dense_ys.ncols() - 1)],
        params[0] * 0.6,
        5e-4,
        "dense stop value",
    );

    let sens_solution = ode
        .solve_fwd_sens(vector_host(&params), vector_host(&t_eval))
        .unwrap();
    let sens_ts = Vec::<f64>::from_host_array(sens_solution.get_ts().unwrap()).unwrap();
    let sens_ys = sens_solution.get_ys().unwrap();
    let sens_ys = sens_ys.as_array::<f64>().unwrap();
    let sens = sens_solution.get_sens().unwrap();
    assert_eq!(sens_ts.len(), dense_ts.len());
    for (i, (&actual, &expected)) in sens_ts.iter().zip(dense_ts.iter()).enumerate() {
        assert_close(actual, expected, 5e-4, &format!("sensitivity time {i}"));
    }
    assert_eq!(sens_ys.ncols(), sens_ts.len());
    assert_eq!(sens.len(), params.len());
    let sens_r = sens[0].as_array::<f64>().unwrap();
    assert_close(
        sens_r[(0, sens_r.ncols() - 1)],
        0.6 + params[0] * logistic_state_dr(LOGISTIC_X0, params[0], root_time),
        5e-4,
        "sensitivity stop value",
    );
    for sens_component in sens {
        let sens_component = sens_component.as_array::<f64>().unwrap();
        assert_eq!(sens_component.ncols(), sens_ts.len());
    }

    let (integral, gradient) = ode
        .solve_continuous_adjoint(vector_host(&params), final_time)
        .unwrap();
    let integral = Vec::<f64>::from_host_array(integral).unwrap();
    let gradient = gradient.as_array::<f64>().unwrap();
    assert_close(
        integral[0],
        params[0] * logistic_integral(LOGISTIC_X0, params[0], root_time),
        5e-4,
        "continuous adjoint stop integral",
    );
    assert_eq!(gradient.shape(), &[params.len(), 1]);

    let (adjoint_solution, checkpoint) = ode
        .solve_adjoint_fwd(vector_host(&params), vector_host(&t_eval))
        .unwrap();
    let adjoint_ts = Vec::<f64>::from_host_array(adjoint_solution.get_ts().unwrap()).unwrap();
    let adjoint_ys = adjoint_solution.get_ys().unwrap();
    let adjoint_ys = adjoint_ys.as_array::<f64>().unwrap();
    assert_eq!(adjoint_ts.len(), dense_ts.len());
    for (i, (&actual, &expected)) in adjoint_ts.iter().zip(dense_ts.iter()).enumerate() {
        assert_close(actual, expected, 5e-4, &format!("adjoint time {i}"));
    }
    assert_eq!(adjoint_ys.ncols(), adjoint_ts.len());

    let gradient = ode
        .solve_adjoint_bkwd(
            &adjoint_solution,
            &checkpoint,
            matrix_host(1, adjoint_ts.len(), &vec![0.0; adjoint_ts.len()]),
        )
        .unwrap();
    assert_eq!(
        gradient.as_array::<f64>().unwrap().shape(),
        &[params.len(), 1]
    );
}
