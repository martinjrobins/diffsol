#![cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]

mod common;

#[cfg(feature = "diffsl-llvm")]
use common::matrix_host;
use common::{
    assert_close, available_jit_backends, logistic_integral, logistic_integral_dr, logistic_state,
    logistic_state_dr, logistic_time_reset_diffsl_code, vector_host, ASSERT_TOL,
};
use diffsol_c::host_array::FromHostArray;
use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};

const RESET_TIME: f64 = 0.5;
const RESET_Y: f64 = 0.2;
const INITIAL_Y: f64 = 0.1;

fn matrix_solver_cases() -> [(MatrixType, OdeSolverType, LinearSolverType); 8] {
    [
        (
            MatrixType::FaerDense,
            OdeSolverType::Bdf,
            LinearSolverType::Default,
        ),
        (
            MatrixType::FaerDense,
            OdeSolverType::Bdf,
            LinearSolverType::Lu,
        ),
        (
            MatrixType::FaerDense,
            OdeSolverType::Esdirk34,
            LinearSolverType::Default,
        ),
        (
            MatrixType::FaerDense,
            OdeSolverType::Esdirk34,
            LinearSolverType::Lu,
        ),
        (
            MatrixType::FaerSparse,
            OdeSolverType::TrBdf2,
            LinearSolverType::Default,
        ),
        (
            MatrixType::FaerSparse,
            OdeSolverType::TrBdf2,
            LinearSolverType::Lu,
        ),
        (
            MatrixType::NalgebraDense,
            OdeSolverType::Tsit45,
            LinearSolverType::Default,
        ),
        (
            MatrixType::NalgebraDense,
            OdeSolverType::Tsit45,
            LinearSolverType::Lu,
        ),
    ]
}

fn make_time_reset_ode(
    jit_backend: JitBackendType,
    scalar_type: ScalarType,
    matrix_type: MatrixType,
    ode_solver: OdeSolverType,
    linear_solver: LinearSolverType,
) -> OdeWrapper {
    let ode = OdeWrapper::new_jit(
        logistic_time_reset_diffsl_code(),
        jit_backend,
        scalar_type,
        matrix_type,
        linear_solver,
        ode_solver,
    )
    .unwrap();
    let tol = match scalar_type {
        ScalarType::F64 => 1e-8,
        ScalarType::F32 => 1e-6,
    };
    ode.set_rtol(tol).unwrap();
    ode.set_atol(tol).unwrap();
    ode
}

fn time_reset_state(r: f64, t: f64) -> f64 {
    if t <= RESET_TIME {
        logistic_state(INITIAL_Y, r, t)
    } else {
        logistic_state(RESET_Y, r, t - RESET_TIME)
    }
}

fn time_reset_state_dr(r: f64, t: f64) -> f64 {
    if t <= RESET_TIME {
        logistic_state_dr(INITIAL_Y, r, t)
    } else {
        logistic_state_dr(RESET_Y, r, t - RESET_TIME)
    }
}

fn time_reset_integral(r: f64, final_time: f64) -> f64 {
    if final_time <= RESET_TIME {
        logistic_integral(INITIAL_Y, r, final_time)
    } else {
        logistic_integral(INITIAL_Y, r, RESET_TIME)
            + logistic_integral(RESET_Y, r, final_time - RESET_TIME)
    }
}

fn time_reset_integral_dr(r: f64, final_time: f64) -> f64 {
    if final_time <= RESET_TIME {
        logistic_integral_dr(INITIAL_Y, r, final_time)
    } else {
        logistic_integral_dr(INITIAL_Y, r, RESET_TIME)
            + logistic_integral_dr(RESET_Y, r, final_time - RESET_TIME)
    }
}

fn assert_time_reset_dense_solution(ode: &OdeWrapper, r: f64, t_eval: &[f64]) {
    let solution = ode
        .solve_dense(vector_host(&[r]), vector_host(t_eval))
        .unwrap();
    let ys = Vec::<Vec<f64>>::from_host_array(solution.get_ys().unwrap()).unwrap();
    let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

    assert_eq!(ys.len(), 1);
    assert_eq!(ys[0].len(), t_eval.len());
    assert_eq!(ts, t_eval);
    for (col, &t) in t_eval.iter().enumerate() {
        assert_close(
            ys[0][col],
            time_reset_state(r, t),
            5e-4,
            &format!("time-reset dense value[{col}]"),
        );
    }
}

#[test]
fn logistic_time_reset_solve_matches_piecewise_solution() {
    let r = 2.0;
    let final_time = 1.0;
    for jit_backend in available_jit_backends() {
        for scalar_type in [ScalarType::F64, ScalarType::F32] {
            for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
                let ode = make_time_reset_ode(
                    jit_backend,
                    scalar_type,
                    matrix_type,
                    ode_solver,
                    linear_solver,
                );
                assert_eq!(ode.get_nstates().unwrap(), 1);
                assert_eq!(ode.get_nparams().unwrap(), 1);
                assert_eq!(ode.get_nout().unwrap(), 1);
                assert!(ode.has_stop().unwrap());

                let solution = ode.solve(vector_host(&[r]), final_time).unwrap();
                let ys = Vec::<Vec<f64>>::from_host_array(solution.get_ys().unwrap()).unwrap();
                let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

                assert_eq!(ys.len(), 1);
                assert_eq!(ys[0].len(), ts.len());
                assert!(!ts.is_empty(), "expected solve() to record output");
                assert_close(
                    *ts.last().unwrap(),
                    final_time,
                    ASSERT_TOL,
                    "time-reset solve final time",
                );
                assert!(ts.iter().any(|&t| (t - RESET_TIME).abs() < 1e-3));
                assert!(ts.iter().any(|&t| t > RESET_TIME + 1e-3));

                for (col, &t) in ts.iter().enumerate() {
                    if (t - RESET_TIME).abs() < 1e-3 {
                        continue;
                    }
                    assert_close(
                        ys[0][col],
                        time_reset_state(r, t),
                        5e-4,
                        &format!("time-reset solve value[{col}]"),
                    );
                }
            }
        }
    }
}

#[test]
fn logistic_time_reset_solve_dense_matches_piecewise_solution() {
    let r = 2.0;
    let t_eval = [0.25, 0.75, 1.0];
    for jit_backend in available_jit_backends() {
        for scalar_type in [ScalarType::F64, ScalarType::F32] {
            for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
                let ode = make_time_reset_ode(
                    jit_backend,
                    scalar_type,
                    matrix_type,
                    ode_solver,
                    linear_solver,
                );
                assert_time_reset_dense_solution(&ode, r, &t_eval);
            }
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn logistic_time_reset_forward_sensitivities_match_piecewise_solution() {
    let r = 2.0;
    let t_eval = [0.25, 0.75, 1.0];
    for jit_backend in available_jit_backends() {
        if jit_backend != JitBackendType::Llvm {
            continue;
        }
        for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
            let ode = make_time_reset_ode(
                jit_backend,
                ScalarType::F64,
                matrix_type,
                ode_solver,
                linear_solver,
            );

            let solution = ode
                .solve_fwd_sens(vector_host(&[r]), vector_host(&t_eval))
                .unwrap();
            let ys = solution.get_ys().unwrap();
            let ys = ys.as_array::<f64>().unwrap();
            let sens = solution.get_sens().unwrap();
            assert_eq!(sens.len(), 1);
            let sens_r = sens[0].as_array::<f64>().unwrap();

            assert_eq!(ys.nrows(), 1);
            assert_eq!(ys.ncols(), t_eval.len());
            assert_eq!(sens_r.nrows(), 1);
            assert_eq!(sens_r.ncols(), t_eval.len());
            for (col, &t) in t_eval.iter().enumerate() {
                assert_close(
                    ys[(0, col)],
                    time_reset_state(r, t),
                    5e-4,
                    &format!("time-reset fwd sens value[{col}]"),
                );
                assert_close(
                    sens_r[(0, col)],
                    time_reset_state_dr(r, t),
                    5e-4,
                    &format!("time-reset fwd sens dr[{col}]"),
                );
            }
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn logistic_time_reset_continuous_adjoint_matches_piecewise_integral() {
    let r = 2.0;
    let final_time = 1.0;
    for jit_backend in available_jit_backends() {
        if jit_backend != JitBackendType::Llvm {
            continue;
        }
        for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
            let ode = make_time_reset_ode(
                jit_backend,
                ScalarType::F64,
                matrix_type,
                ode_solver,
                linear_solver,
            );

            let (integral, gradient) = ode
                .solve_continuous_adjoint(vector_host(&[r]), final_time)
                .unwrap();
            let integral = Vec::<f64>::from_host_array(integral).unwrap();
            let gradient = gradient.as_array::<f64>().unwrap();

            assert_eq!(integral.len(), 1);
            assert_close(
                integral[0],
                time_reset_integral(r, final_time),
                1e-4,
                "time-reset continuous adjoint integral",
            );
            assert_eq!(gradient.nrows(), 1);
            assert_eq!(gradient.ncols(), 1);
            assert_close(
                gradient[(0, 0)],
                time_reset_integral_dr(r, final_time),
                5e-4,
                "time-reset continuous adjoint gradient",
            );
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn logistic_time_reset_split_adjoint_matches_analytical_gradient() {
    let fit_r = 2.0;
    let data_r = 1.5;
    let t_eval = [0.25, 0.75, 1.0];
    let data_values: Vec<f64> = t_eval
        .iter()
        .map(|&t| time_reset_state(data_r, t))
        .collect();
    for jit_backend in available_jit_backends() {
        if jit_backend != JitBackendType::Llvm {
            continue;
        }
        for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
            let ode = make_time_reset_ode(
                jit_backend,
                ScalarType::F64,
                matrix_type,
                ode_solver,
                linear_solver,
            );

            let (solution, checkpoint) = ode
                .solve_adjoint_fwd(vector_host(&[fit_r]), vector_host(&t_eval))
                .unwrap();
            let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();
            assert_eq!(ts, t_eval);
            let ys = solution.get_ys().unwrap();
            let ys = ys.as_array::<f64>().unwrap();

            let dgdu_values: Vec<f64> = (0..t_eval.len())
                .map(|col| 2.0 * (ys[(0, col)] - data_values[col]))
                .collect();
            let split_gradient = ode
                .solve_adjoint_bkwd(
                    &solution,
                    &checkpoint,
                    matrix_host(1, t_eval.len(), &dgdu_values),
                )
                .unwrap();
            let split_gradient = split_gradient.as_array::<f64>().unwrap();

            let expected_gradient: f64 = t_eval
                .iter()
                .zip(data_values.iter())
                .map(|(&t, &data_y)| {
                    let fit_y = time_reset_state(fit_r, t);
                    2.0 * (fit_y - data_y) * time_reset_state_dr(fit_r, t)
                })
                .sum();

            assert_eq!(split_gradient.nrows(), 1);
            assert_eq!(split_gradient.ncols(), 1);
            assert_close(
                split_gradient[(0, 0)],
                expected_gradient,
                5e-4,
                "time-reset split adjoint gradient",
            );
        }
    }
}
