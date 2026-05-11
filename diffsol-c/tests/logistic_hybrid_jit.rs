#![cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]

mod common;

#[cfg(feature = "diffsl-llvm")]
use common::matrix_host;
use common::{
    assert_close, available_jit_backends, logistic_integral, logistic_integral_dr, logistic_state,
    logistic_state_dr, vector_host, ASSERT_TOL, LOGISTIC_X0,
};
use diffsol_c::host_array::FromHostArray;
use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};

const A_VALUES: [f64; 4] = [0.8, 1.0, 1.2, 1.4];
const B_VALUES: [f64; 4] = [0.7, 0.9, 1.2, 1.5];
const RESET_SCALE: f64 = 0.2;
const STOP_TIMES: [f64; 3] = [0.5, 1.0, 1.5];

fn logistic_hybrid_diffsl_code() -> &'static str {
    r#"
        in_i { r = 1 }
        a_i {
            0.8,
            1.0,
            1.2,
            1.4,
        }
        b_i {
            0.7,
            0.9,
            1.2,
            1.5,
        }
        u_i { y = 0.1 }
        dudt_i { dydt = 0 }
        F_i { a_i[N] * r * y * (1.0 - y) }
        stop_i {
            1.0,
            t - 0.5,
            t - 1.0,
            t - 1.5,
        }
        reset_i { b_i[N] * 0.2 }
        out_i { y }
    "#
}

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

fn make_hybrid_ode(
    jit_backend: JitBackendType,
    scalar_type: ScalarType,
    matrix_type: MatrixType,
    ode_solver: OdeSolverType,
    linear_solver: LinearSolverType,
) -> OdeWrapper {
    let ode = OdeWrapper::new_jit(
        logistic_hybrid_diffsl_code(),
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

fn hybrid_segment(t: f64) -> (usize, f64, f64) {
    if t <= STOP_TIMES[0] {
        (0, LOGISTIC_X0, t)
    } else if t <= STOP_TIMES[1] {
        (1, B_VALUES[1] * RESET_SCALE, t - STOP_TIMES[0])
    } else if t <= STOP_TIMES[2] {
        (2, B_VALUES[2] * RESET_SCALE, t - STOP_TIMES[1])
    } else {
        (3, B_VALUES[3] * RESET_SCALE, t - STOP_TIMES[2])
    }
}

fn hybrid_state(r: f64, t: f64) -> f64 {
    let (segment, y0, local_t) = hybrid_segment(t);
    logistic_state(y0, A_VALUES[segment] * r, local_t)
}

#[cfg(feature = "diffsl-llvm")]
fn hybrid_state_dr(r: f64, t: f64) -> f64 {
    let (segment, y0, local_t) = hybrid_segment(t);
    A_VALUES[segment] * logistic_state_dr(y0, A_VALUES[segment] * r, local_t)
}

#[cfg(feature = "diffsl-llvm")]
fn hybrid_integral(r: f64, final_time: f64) -> f64 {
    let mut total = 0.0;
    let mut start = 0.0;
    let mut y0 = LOGISTIC_X0;
    for segment in 0..A_VALUES.len() {
        let end = if segment < STOP_TIMES.len() {
            STOP_TIMES[segment]
        } else {
            final_time
        };
        if final_time <= start {
            break;
        }
        let duration = final_time.min(end) - start;
        if duration > 0.0 {
            total += logistic_integral(y0, A_VALUES[segment] * r, duration);
        }
        if final_time <= end {
            break;
        }
        start = end;
        y0 = B_VALUES[segment + 1] * RESET_SCALE;
    }
    total
}

#[cfg(feature = "diffsl-llvm")]
fn hybrid_integral_dr(r: f64, final_time: f64) -> f64 {
    let mut total = 0.0;
    let mut start = 0.0;
    let mut y0 = LOGISTIC_X0;
    for segment in 0..A_VALUES.len() {
        let end = if segment < STOP_TIMES.len() {
            STOP_TIMES[segment]
        } else {
            final_time
        };
        if final_time <= start {
            break;
        }
        let duration = final_time.min(end) - start;
        if duration > 0.0 {
            total += A_VALUES[segment] * logistic_integral_dr(y0, A_VALUES[segment] * r, duration);
        }
        if final_time <= end {
            break;
        }
        start = end;
        y0 = B_VALUES[segment + 1] * RESET_SCALE;
    }
    total
}

fn assert_hybrid_dense_solution(ode: &OdeWrapper, r: f64, t_eval: &[f64]) {
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
            hybrid_state(r, t),
            5e-4,
            &format!("hybrid dense value[{col}]"),
        );
    }
}

#[test]
fn logistic_hybrid_solve_matches_piecewise_solution() {
    let r = 2.0;
    let final_time = 2.0;
    for jit_backend in available_jit_backends() {
        for scalar_type in [ScalarType::F64, ScalarType::F32] {
            for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
                let ode = make_hybrid_ode(
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
                    "hybrid solve final time",
                );
                for &stop_time in &STOP_TIMES {
                    assert!(
                        ts.iter().any(|&t| (t - stop_time).abs() < 1e-3),
                        "expected solve output at stop time {stop_time}"
                    );
                }

                for (col, &t) in ts.iter().enumerate() {
                    if STOP_TIMES
                        .iter()
                        .any(|&stop_time| (t - stop_time).abs() < 1e-3)
                    {
                        continue;
                    }
                    assert_close(
                        ys[0][col],
                        hybrid_state(r, t),
                        5e-4,
                        &format!("hybrid solve value[{col}]"),
                    );
                }
            }
        }
    }
}

#[test]
fn logistic_hybrid_solve_dense_matches_piecewise_solution() {
    let r = 2.0;
    let t_eval = [0.25, 0.75, 1.25, 1.75, 2.0];
    for jit_backend in available_jit_backends() {
        for scalar_type in [ScalarType::F64, ScalarType::F32] {
            for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
                let ode = make_hybrid_ode(
                    jit_backend,
                    scalar_type,
                    matrix_type,
                    ode_solver,
                    linear_solver,
                );
                assert_hybrid_dense_solution(&ode, r, &t_eval);
            }
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn logistic_hybrid_forward_sensitivities_match_piecewise_solution() {
    let r = 2.0;
    let t_eval = [0.25, 0.75, 1.25, 1.75, 2.0];
    for jit_backend in available_jit_backends() {
        if jit_backend != JitBackendType::Llvm {
            continue;
        }
        for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
            let ode = make_hybrid_ode(
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
                    hybrid_state(r, t),
                    5e-4,
                    &format!("hybrid fwd sens value[{col}]"),
                );
                assert_close(
                    sens_r[(0, col)],
                    hybrid_state_dr(r, t),
                    5e-4,
                    &format!("hybrid fwd sens dr[{col}]"),
                );
            }
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn logistic_hybrid_continuous_adjoint_matches_piecewise_integral() {
    let r = 2.0;
    let final_time = 2.0;
    for jit_backend in available_jit_backends() {
        if jit_backend != JitBackendType::Llvm {
            continue;
        }
        for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
            let ode = make_hybrid_ode(
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
                hybrid_integral(r, final_time),
                5e-4,
                "hybrid continuous adjoint integral",
            );
            assert_eq!(gradient.nrows(), 1);
            assert_eq!(gradient.ncols(), 1);
            assert_close(
                gradient[(0, 0)],
                hybrid_integral_dr(r, final_time),
                5e-4,
                "hybrid continuous adjoint gradient",
            );
        }
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn logistic_hybrid_split_adjoint_matches_analytical_gradient() {
    let fit_r = 2.0;
    let data_r = 1.5;
    let t_eval = [0.25, 0.75, 1.25, 1.75, 2.0];
    let data_values: Vec<f64> = t_eval.iter().map(|&t| hybrid_state(data_r, t)).collect();
    for jit_backend in available_jit_backends() {
        if jit_backend != JitBackendType::Llvm {
            continue;
        }
        for (matrix_type, ode_solver, linear_solver) in matrix_solver_cases() {
            let ode = make_hybrid_ode(
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
                    let fit_y = hybrid_state(fit_r, t);
                    2.0 * (fit_y - data_y) * hybrid_state_dr(fit_r, t)
                })
                .sum();

            assert_eq!(split_gradient.nrows(), 1);
            assert_eq!(split_gradient.ncols(), 1);
            assert_close(
                split_gradient[(0, 0)],
                expected_gradient,
                5e-4,
                "hybrid split adjoint gradient",
            );
        }
    }
}
