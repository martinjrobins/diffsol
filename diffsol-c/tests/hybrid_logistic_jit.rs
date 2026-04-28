#![cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]

mod common;

use common::{
    all_ode_solvers, assert_close, available_jit_backends, hybrid_logistic_period,
    hybrid_logistic_state, hybrid_logistic_state_dr, vector_host, ASSERT_TOL,
};
use diffsol_c::host_array::FromHostArray;
use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};

fn hybrid_logistic_diffsl_code() -> &'static str {
    r#"
        in_i { r = 1 }
        u_i { y = 0.1 }
        dudt_i { dydt = 0 }
        F_i { (r * y) * (1 - y) }
        stop_i { y - 0.9 }
        reset_i { 0.1 }
        out_i { y }
    "#
}

fn make_hybrid_ode(
    jit_backend: JitBackendType,
    matrix_type: MatrixType,
    ode_solver: OdeSolverType,
) -> OdeWrapper {
    OdeWrapper::new_jit(
        hybrid_logistic_diffsl_code(),
        jit_backend,
        ScalarType::F64,
        matrix_type,
        LinearSolverType::Default,
        ode_solver,
    )
    .unwrap()
}

fn hybrid_t_eval() -> [f64; 7] {
    [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.8]
}

fn assert_hybrid_solution_matches_piecewise_logistic_diffsl_model(
    jit_backend: JitBackendType,
    ode_solver: OdeSolverType,
) {
    let r = 2.0;
    let final_time = 5.0;
    let tau = hybrid_logistic_period(r);
    let ode = make_hybrid_ode(jit_backend, MatrixType::NalgebraDense, ode_solver);
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();
    assert_eq!(ode.get_nstates().unwrap(), 1);
    assert_eq!(ode.get_nparams().unwrap(), 1);
    assert_eq!(ode.get_nout().unwrap(), 1);
    assert!(ode.has_stop().unwrap());

    let solution = ode.solve(vector_host(&[r]), final_time).unwrap();
    let ys = solution.get_ys().unwrap();
    let ys = ys.as_array::<f64>().unwrap();
    let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

    assert_eq!(ys.nrows(), 1);
    assert_eq!(ys.ncols(), ts.len());
    assert!(!ts.is_empty(), "expected hybrid solve to produce output");
    assert_close(
        *ts.last().unwrap(),
        final_time,
        ASSERT_TOL,
        "jit hybrid final time",
    );
    assert_close(
        ys[(0, ys.ncols() - 1)],
        hybrid_logistic_state(r, final_time),
        5e-4,
        "jit hybrid final value",
    );
    assert!(ts.iter().any(|&t| (t - tau).abs() < 1e-3));
    assert!(ts.iter().any(|&t| (t - 2.0 * tau).abs() < 1e-3));
    for (col, &t) in ts.iter().enumerate() {
        if ((t / tau).round() * tau - t).abs() < 1e-3 {
            continue;
        }
        assert_close(
            ys[(0, col)],
            hybrid_logistic_state(r, t),
            5e-4,
            &format!("jit hybrid value[{col}]"),
        );
    }
}

fn assert_hybrid_dense_solution_matches_piecewise_logistic_diffsl_model(
    jit_backend: JitBackendType,
    ode_solver: OdeSolverType,
) {
    let r = 2.0;
    let t_eval = hybrid_t_eval();
    let ode = make_hybrid_ode(jit_backend, MatrixType::NalgebraDense, ode_solver);
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let solution = ode
        .solve_dense(vector_host(&[r]), vector_host(&t_eval))
        .unwrap();
    let ys = solution.get_ys().unwrap();
    let ys = ys.as_array::<f64>().unwrap();
    let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

    assert_eq!(ys.nrows(), 1);
    assert_eq!(ys.ncols(), t_eval.len());
    assert_eq!(ts, t_eval);
    for (col, &t) in t_eval.iter().enumerate() {
        assert_close(
            ys[(0, col)],
            hybrid_logistic_state(r, t),
            5e-4,
            &format!("jit hybrid dense value[{col}]"),
        );
    }
}

#[cfg(feature = "diffsl-llvm")]
fn assert_hybrid_forward_sensitivities_match_piecewise_logistic_diffsl_model(
    ode_solver: OdeSolverType,
) {
    let r = 2.0;
    let t_eval = hybrid_t_eval();
    let ode = make_hybrid_ode(JitBackendType::Llvm, MatrixType::NalgebraDense, ode_solver);
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let solution = ode
        .solve_fwd_sens(vector_host(&[r]), vector_host(&t_eval))
        .unwrap();
    let ys = solution.get_ys().unwrap();
    let ys = ys.as_array::<f64>().unwrap();
    let sens = solution.get_sens().unwrap();

    assert_eq!(ys.nrows(), 1);
    assert_eq!(ys.ncols(), t_eval.len());
    assert_eq!(sens.len(), 1);
    let sens_values = sens[0].as_array::<f64>().unwrap();
    assert_eq!(sens_values.nrows(), 1);
    assert_eq!(sens_values.ncols(), t_eval.len());
    for (col, &t) in t_eval.iter().enumerate() {
        assert_close(
            ys[(0, col)],
            hybrid_logistic_state(r, t),
            5e-4,
            &format!("jit hybrid sens value[{col}]"),
        );
        assert_close(
            sens_values[(0, col)],
            hybrid_logistic_state_dr(r, t),
            5e-4,
            &format!("jit hybrid sensitivity[{col}]"),
        );
    }
}

#[test]
fn hybrid_solution_matches_piecewise_logistic_diffsl_model() {
    for jit_backend in available_jit_backends() {
        for ode_solver in all_ode_solvers() {
            assert_hybrid_solution_matches_piecewise_logistic_diffsl_model(jit_backend, ode_solver);
        }
    }
}

#[test]
fn hybrid_dense_solution_matches_piecewise_logistic_diffsl_model() {
    for jit_backend in available_jit_backends() {
        for ode_solver in all_ode_solvers() {
            assert_hybrid_dense_solution_matches_piecewise_logistic_diffsl_model(
                jit_backend,
                ode_solver,
            );
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn hybrid_forward_sensitivities_match_piecewise_logistic_diffsl_model() {
    for ode_solver in all_ode_solvers() {
        assert_hybrid_forward_sensitivities_match_piecewise_logistic_diffsl_model(ode_solver);
    }
}
