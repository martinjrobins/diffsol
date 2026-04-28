#![cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]

mod common;

#[cfg(feature = "diffsl-llvm")]
use common::matrix_host;
use common::{
    assert_close, assert_solution_tail, available_jit_backends, logistic_diffsl_code,
    logistic_integral, logistic_state, logistic_state_dr, vector_host, ASSERT_TOL, LOGISTIC_X0,
};
use diffsol_c::host_array::FromHostArray;
use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};

fn make_ode(
    jit_backend: JitBackendType,
    scalar_type: ScalarType,
    matrix_type: MatrixType,
    ode_solver: OdeSolverType,
) -> OdeWrapper {
    OdeWrapper::new_jit(
        logistic_diffsl_code(),
        jit_backend,
        scalar_type,
        matrix_type,
        LinearSolverType::Default,
        ode_solver,
    )
    .unwrap()
}

fn serialized_linear_solver(matrix_type: MatrixType) -> LinearSolverType {
    match matrix_type {
        MatrixType::NalgebraDense | MatrixType::FaerDense => LinearSolverType::Lu,
        MatrixType::FaerSparse => LinearSolverType::Default,
    }
}

fn configure_serialized_ode(ode: &OdeWrapper, matrix_type: MatrixType) {
    ode.set_linear_solver(serialized_linear_solver(matrix_type))
        .unwrap();
    ode.set_ode_solver(OdeSolverType::TrBdf2).unwrap();
    ode.set_rtol(1e-7).unwrap();
    ode.set_atol(1e-9).unwrap();
    ode.set_t0(0.125).unwrap();
    ode.set_h0(0.25).unwrap();
    ode.set_integrate_out(true).unwrap();
    ode.set_sens_rtol(Some(1e-5)).unwrap();
    ode.set_sens_atol(Some(1e-6)).unwrap();
    ode.set_out_rtol(None).unwrap();
    ode.set_out_atol(None).unwrap();
    ode.set_param_rtol(Some(2e-5)).unwrap();
    ode.set_param_atol(Some(2e-6)).unwrap();

    let ic_options = ode.get_ic_options();
    ic_options.set_use_linesearch(true).unwrap();
    ic_options.set_max_linesearch_iterations(13).unwrap();
    ic_options.set_max_newton_iterations(17).unwrap();
    ic_options.set_max_linear_solver_setups(19).unwrap();
    ic_options.set_step_reduction_factor(0.5).unwrap();
    ic_options.set_armijo_constant(1e-4).unwrap();

    let options = ode.get_options();
    options.set_max_nonlinear_solver_iterations(23).unwrap();
    options.set_max_error_test_failures(29).unwrap();
    options.set_update_jacobian_after_steps(31).unwrap();
    options.set_update_rhs_jacobian_after_steps(37).unwrap();
    options.set_threshold_to_update_jacobian(1e-3).unwrap();
    options.set_threshold_to_update_rhs_jacobian(2e-3).unwrap();
    options.set_min_timestep(1e-4).unwrap();
}

fn scalar_value(value: f64, scalar_type: ScalarType) -> f64 {
    match scalar_type {
        ScalarType::F32 => (value as f32) as f64,
        ScalarType::F64 => value,
    }
}

fn assert_serialization_roundtrip(
    jit_backend: JitBackendType,
    scalar_type: ScalarType,
    matrix_type: MatrixType,
) {
    let ode = make_ode(jit_backend, scalar_type, matrix_type, OdeSolverType::Bdf);
    configure_serialized_ode(&ode, matrix_type);

    #[cfg(feature = "diffsl-cranelift")]
    if jit_backend == JitBackendType::Cranelift {
        let err = serde_json::to_string(&ode).unwrap_err().to_string();
        assert!(err.contains("not supported for Cranelift"));
        return;
    }

    let y0_before = Vec::<f64>::from_host_array(ode.y0(vector_host(&[2.0])).unwrap()).unwrap();
    let rhs_before = Vec::<f64>::from_host_array(
        ode.rhs(vector_host(&[2.0]), 0.0, vector_host(&[0.25]))
            .unwrap(),
    )
    .unwrap();

    let encoded = serde_json::to_string(&ode).unwrap();
    let decoded: OdeWrapper = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.get_jit_backend().unwrap(), Some(jit_backend));
    assert_eq!(decoded.get_code().unwrap(), logistic_diffsl_code());
    assert_eq!(decoded.get_scalar_type().unwrap(), scalar_type);
    assert_eq!(decoded.get_matrix_type().unwrap(), matrix_type);
    assert_eq!(
        decoded.get_linear_solver().unwrap(),
        serialized_linear_solver(matrix_type)
    );
    assert_eq!(decoded.get_ode_solver().unwrap(), OdeSolverType::TrBdf2);
    assert_close(
        decoded.get_rtol().unwrap(),
        scalar_value(1e-7, scalar_type),
        1e-12,
        "serialized rtol",
    );
    assert_close(
        decoded.get_atol().unwrap(),
        scalar_value(1e-9, scalar_type),
        1e-12,
        "serialized atol",
    );
    assert_close(
        decoded.get_t0().unwrap(),
        scalar_value(0.125, scalar_type),
        1e-12,
        "serialized t0",
    );
    assert_close(
        decoded.get_h0().unwrap(),
        scalar_value(0.25, scalar_type),
        1e-12,
        "serialized h0",
    );
    assert!(decoded.get_integrate_out().unwrap());
    assert_close(
        decoded.get_sens_rtol().unwrap().unwrap(),
        scalar_value(1e-5, scalar_type),
        1e-12,
        "serialized sens_rtol",
    );
    assert_close(
        decoded.get_sens_atol().unwrap().unwrap(),
        scalar_value(1e-6, scalar_type),
        1e-12,
        "serialized sens_atol",
    );
    assert_eq!(decoded.get_out_rtol().unwrap(), None);
    assert_eq!(decoded.get_out_atol().unwrap(), None);
    assert_close(
        decoded.get_param_rtol().unwrap().unwrap(),
        scalar_value(2e-5, scalar_type),
        1e-12,
        "serialized param_rtol",
    );
    assert_close(
        decoded.get_param_atol().unwrap().unwrap(),
        scalar_value(2e-6, scalar_type),
        1e-12,
        "serialized param_atol",
    );

    let ic_options = decoded.get_ic_options();
    assert!(ic_options.get_use_linesearch().unwrap());
    assert_eq!(ic_options.get_max_linesearch_iterations().unwrap(), 13);
    assert_eq!(ic_options.get_max_newton_iterations().unwrap(), 17);
    assert_eq!(ic_options.get_max_linear_solver_setups().unwrap(), 19);
    assert_close(
        ic_options.get_step_reduction_factor().unwrap(),
        scalar_value(0.5, scalar_type),
        1e-12,
        "serialized step_reduction_factor",
    );
    assert_close(
        ic_options.get_armijo_constant().unwrap(),
        scalar_value(1e-4, scalar_type),
        1e-12,
        "serialized armijo_constant",
    );

    let options = decoded.get_options();
    assert_eq!(options.get_max_nonlinear_solver_iterations().unwrap(), 23);
    assert_eq!(options.get_max_error_test_failures().unwrap(), 29);
    assert_eq!(options.get_update_jacobian_after_steps().unwrap(), 31);
    assert_eq!(options.get_update_rhs_jacobian_after_steps().unwrap(), 37);
    assert_close(
        options.get_threshold_to_update_jacobian().unwrap(),
        scalar_value(1e-3, scalar_type),
        1e-12,
        "serialized threshold_to_update_jacobian",
    );
    assert_close(
        options.get_threshold_to_update_rhs_jacobian().unwrap(),
        scalar_value(2e-3, scalar_type),
        1e-12,
        "serialized threshold_to_update_rhs_jacobian",
    );
    assert_close(
        options.get_min_timestep().unwrap(),
        scalar_value(1e-4, scalar_type),
        1e-12,
        "serialized min_timestep",
    );

    let y0_after = Vec::<f64>::from_host_array(decoded.y0(vector_host(&[2.0])).unwrap()).unwrap();
    let rhs_after = Vec::<f64>::from_host_array(
        decoded
            .rhs(vector_host(&[2.0]), 0.0, vector_host(&[0.25]))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(y0_after, y0_before);
    assert_close(
        rhs_after[0],
        rhs_before[0],
        ASSERT_TOL,
        "serialized rhs matches",
    );

    decoded
        .set_linear_solver(serialized_linear_solver(matrix_type))
        .unwrap();
    decoded.set_t0(0.0).unwrap();
    decoded.set_integrate_out(false).unwrap();
    let t_eval = [0.25, 0.5, 1.0];
    let solution = decoded
        .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
        .unwrap();
    assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
}

fn assert_runtime_dispatch(jit_backend: JitBackendType, matrix_type: MatrixType) {
    let ode = make_ode(
        jit_backend,
        ScalarType::F64,
        matrix_type,
        OdeSolverType::Bdf,
    );
    assert_eq!(ode.get_matrix_type().unwrap(), matrix_type);
    assert_eq!(ode.get_code().unwrap(), logistic_diffsl_code());
    assert_eq!(ode.get_nstates().unwrap(), 1);
    assert_eq!(ode.get_nparams().unwrap(), 1);
    assert_eq!(ode.get_nout().unwrap(), 1);
    assert!(!ode.has_stop().unwrap());

    let y0 = ode.y0(vector_host(&[2.0])).unwrap();
    assert_eq!(Vec::<f64>::from_host_array(y0).unwrap(), vec![LOGISTIC_X0]);

    let rhs = ode
        .rhs(vector_host(&[2.0]), 0.0, vector_host(&[0.25]))
        .unwrap();
    assert_close(
        Vec::<f64>::from_host_array(rhs).unwrap()[0],
        0.375,
        ASSERT_TOL,
        "jit rhs(0.25)",
    );

    let rhs_jac_mul = ode
        .rhs_jac_mul(
            vector_host(&[2.0]),
            0.0,
            vector_host(&[0.25]),
            vector_host(&[3.0]),
        )
        .unwrap();
    assert_close(
        Vec::<f64>::from_host_array(rhs_jac_mul).unwrap()[0],
        3.0,
        ASSERT_TOL,
        "jit rhs_jac_mul(0.25, 3.0)",
    );
}

fn assert_solver_dense_solution(
    jit_backend: JitBackendType,
    scalar_type: ScalarType,
    matrix_type: MatrixType,
    ode_solver: OdeSolverType,
) {
    let ode = make_ode(jit_backend, scalar_type, matrix_type, ode_solver);
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let t_eval = [0.25, 0.5, 1.0];
    let solution = ode
        .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
        .unwrap();

    assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
}

#[test]
fn runtime_dispatch_matches_requested_matrix_type_from_diffsl() {
    for jit_backend in available_jit_backends() {
        for matrix_type in [
            MatrixType::NalgebraDense,
            MatrixType::FaerDense,
            MatrixType::FaerSparse,
        ] {
            assert_runtime_dispatch(jit_backend, matrix_type);
        }
    }
}

#[test]
fn dense_solution_matches_logistic_solution_from_diffsl() {
    for jit_backend in available_jit_backends() {
        for scalar_type in [ScalarType::F64, ScalarType::F32] {
            for (matrix_type, solver) in [
                (MatrixType::FaerDense, OdeSolverType::Esdirk34),
                (MatrixType::FaerSparse, OdeSolverType::TrBdf2),
                (MatrixType::NalgebraDense, OdeSolverType::Tsit45),
            ] {
                assert_solver_dense_solution(jit_backend, scalar_type, matrix_type, solver);
            }
        }
    }
}

#[test]
fn bdf_dense_solution_matches_logistic_diffsl_model() {
    for jit_backend in available_jit_backends() {
        let ode = make_ode(
            jit_backend,
            ScalarType::F64,
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    }
}

#[test]
fn bdf_solution_matches_logistic_diffsl_model() {
    for jit_backend in available_jit_backends() {
        let x0 = LOGISTIC_X0;
        let r = 2.0;
        let ode = make_ode(
            jit_backend,
            ScalarType::F64,
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let final_time = 1.0;
        let solution = ode.solve(vector_host(&[r]), final_time).unwrap();

        let ys = solution.get_ys().unwrap();
        let ys = ys.as_array::<f64>().unwrap();
        let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

        assert_eq!(ys.nrows(), 1);
        assert_eq!(ys.ncols(), ts.len());
        assert!(
            !ts.is_empty(),
            "expected solve() to record at least one time point"
        );
        assert_close(
            *ts.last().unwrap(),
            final_time,
            ASSERT_TOL,
            "solve final time",
        );
        for (i, &t) in ts.iter().enumerate() {
            assert_close(
                ys[(0, i)],
                logistic_state(x0, r, t),
                5e-4,
                &format!("solve value[{i}]"),
            );
        }
    }
}

#[cfg_attr(
    all(target_os = "macos", target_arch = "x86_64"),
    ignore = "from_external_object is unsupported on Intel macOS due to unsupported relocations"
)]
#[test]
fn serialization_roundtrip_restores_full_solver_state() {
    for jit_backend in available_jit_backends() {
        for scalar_type in [ScalarType::F64, ScalarType::F32] {
            for matrix_type in [MatrixType::NalgebraDense, MatrixType::FaerSparse] {
                assert_serialization_roundtrip(jit_backend, scalar_type, matrix_type);
            }
        }
    }
}

#[cfg(all(feature = "diffsl-llvm", not(feature = "diffsl-cranelift")))]
#[test]
fn deserialization_rejects_unavailable_jit_backend() {
    let ode = make_ode(
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Bdf,
    );
    let mut value = serde_json::to_value(&ode).unwrap();
    value["jit_backend"] = serde_json::Value::String("cranelift".to_string());
    let err = serde_json::from_value::<OdeWrapper>(value)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("unknown variant"));
}

#[cfg(all(feature = "diffsl-cranelift", not(feature = "diffsl-llvm")))]
#[test]
fn deserialization_rejects_unavailable_jit_backend() {
    let ode = make_ode(
        JitBackendType::Cranelift,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Bdf,
    );
    let mut value = serde_json::to_value(&ode).unwrap();
    value["jit_backend"] = serde_json::Value::String("llvm".to_string());
    let err = serde_json::from_value::<OdeWrapper>(value)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("unknown variant"));
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn bdf_forward_sensitivities_match_logistic_derivative_from_diffsl() {
    let ode = make_ode(
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Bdf,
    );
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let t_eval = [0.25, 0.5, 1.0];
    let solution = ode
        .solve_fwd_sens(vector_host(&[2.0]), vector_host(&t_eval))
        .unwrap();

    assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    let sens = solution.get_sens().unwrap();
    assert_eq!(sens.len(), 1);
    let sens_values = sens[0].as_array::<f64>().unwrap();
    assert_eq!(sens_values.nrows(), 1);
    assert_eq!(sens_values.ncols(), t_eval.len());
    for (i, &t) in t_eval.iter().enumerate() {
        assert_close(
            sens_values[(0, i)],
            logistic_state_dr(LOGISTIC_X0, 2.0, t),
            ASSERT_TOL,
            &format!("jit sensitivity[{i}]"),
        );
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn continuous_adjoint_returns_integral_and_parameter_gradient() {
    let ode = make_ode(
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Bdf,
    );
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let r = 2.0;
    let final_time = 1.0;
    let (integral, gradient) = ode
        .solve_continuous_adjoint(vector_host(&[r]), final_time)
        .unwrap();
    let integral = Vec::<f64>::from_host_array(integral).unwrap();
    let gradient = gradient.as_array::<f64>().unwrap();

    let step = 1e-6;
    let expected_gradient = (logistic_integral(LOGISTIC_X0, r + step, final_time)
        - logistic_integral(LOGISTIC_X0, r - step, final_time))
        / (2.0 * step);

    assert_eq!(integral.len(), 1);
    assert_close(
        integral[0],
        logistic_integral(LOGISTIC_X0, r, final_time),
        5e-5,
        "continuous adjoint integral",
    );
    assert_eq!(gradient.nrows(), 1);
    assert_eq!(gradient.ncols(), 1);
    assert_close(
        gradient[(0, 0)],
        expected_gradient,
        5e-4,
        "continuous adjoint gradient",
    );
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn split_adjoint_matches_finite_difference_gradient() {
    let ode = make_ode(
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Bdf,
    );
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let t_eval = [0.0, 0.25, 0.5, 1.0];
    let fit_r = 2.0;
    let data_r = 1.5;
    let data_values: Vec<f64> = t_eval
        .iter()
        .map(|&t| logistic_state(LOGISTIC_X0, data_r, t))
        .collect();
    let (solution, checkpoint) = ode
        .solve_adjoint_fwd(vector_host(&[fit_r]), vector_host(&t_eval))
        .unwrap();
    assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, fit_r, 5e-4);

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

    let objective = |r: f64| -> f64 {
        let solution = ode
            .solve_dense(vector_host(&[r]), vector_host(&t_eval))
            .unwrap();
        let ys = solution.get_ys().unwrap();
        let ys = ys.as_array::<f64>().unwrap();
        (0..t_eval.len())
            .map(|col| {
                let residual = ys[(0, col)] - data_values[col];
                residual * residual
            })
            .sum()
    };
    let step = 1e-6;
    let finite_difference_gradient =
        (objective(fit_r + step) - objective(fit_r - step)) / (2.0 * step);

    assert_eq!(split_gradient.nrows(), 1);
    assert_eq!(split_gradient.ncols(), 1);
    assert_close(
        split_gradient[(0, 0)],
        finite_difference_gradient,
        5e-5,
        "split adjoint gradient",
    );
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn split_adjoint_rejects_invalid_dgdu_shapes() {
    let ode = make_ode(
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Bdf,
    );
    let t_eval = [0.0, 0.25, 0.5, 1.0];
    let (solution, checkpoint) = ode
        .solve_adjoint_fwd(vector_host(&[2.0]), vector_host(&t_eval))
        .unwrap();

    let wrong_rows = matrix_host(2, t_eval.len(), &[0.0; 8]);
    let err = match ode.solve_adjoint_bkwd(&solution, &checkpoint, wrong_rows) {
        Ok(_) => panic!("expected invalid dgdu row count to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("Expected dgdu_eval to have 1 rows"));

    let wrong_cols = matrix_host(1, t_eval.len() - 1, &[0.0; 3]);
    let err = match ode.solve_adjoint_bkwd(&solution, &checkpoint, wrong_cols) {
        Ok(_) => panic!("expected invalid dgdu column count to fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("Expected dgdu_eval to have 4 columns"));
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn tsit45_split_adjoint_backward_returns_gradient() {
    let ode = make_ode(
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        OdeSolverType::Tsit45,
    );
    let t_eval = [0.0, 0.25, 0.5, 1.0];
    let (solution, checkpoint) = ode
        .solve_adjoint_fwd(vector_host(&[2.0]), vector_host(&t_eval))
        .unwrap();
    let gradient = ode
        .solve_adjoint_bkwd(
            &solution,
            &checkpoint,
            matrix_host(1, t_eval.len(), &[0.0; 4]),
        )
        .unwrap();
    let gradient = gradient.as_array::<f64>().unwrap();
    assert_eq!(gradient.shape(), &[1, 1]);
    assert!(gradient[(0, 0)].is_finite());
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn bdf_split_adjoint_matches_finite_difference_gradient_for_logistic_model() {
    let logistic_model = r#"
        in_i { r = 1, k = 1, y0 = 0.1 }
        u { y0 }
        F { r * u * (1.0 - u / k) }
    "#;
    let ode = OdeWrapper::new_jit(
        logistic_model,
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let t_eval = [0.0, 0.1, 0.3, 0.7, 1.0];
    let data_params = [1.2, 0.9, 0.2];
    let fit_params = [0.8, 1.3, 0.12];
    let fd_step = 1e-6;

    let data_solution = ode
        .solve_dense(vector_host(&data_params), vector_host(&t_eval))
        .unwrap();
    let data_ys = data_solution.get_ys().unwrap();
    let data_ys = data_ys.as_array::<f64>().unwrap();
    let data_values: Vec<f64> = (0..t_eval.len()).map(|col| data_ys[(0, col)]).collect();

    let objective_from_dense = |params: [f64; 3]| -> f64 {
        let solution = ode
            .solve_dense(vector_host(&params), vector_host(&t_eval))
            .unwrap();
        let ys = solution.get_ys().unwrap();
        let ys = ys.as_array::<f64>().unwrap();
        (0..t_eval.len())
            .map(|col| {
                let residual = ys[(0, col)] - data_values[col];
                residual * residual
            })
            .sum()
    };

    let objective_fd = objective_from_dense(fit_params);
    let mut finite_difference_gradient = [0.0; 3];
    for i in 0..fit_params.len() {
        let mut plus = fit_params;
        let mut minus = fit_params;
        let step = fd_step * fit_params[i].abs().max(1.0);
        plus[i] += step;
        minus[i] -= step;
        finite_difference_gradient[i] =
            (objective_from_dense(plus) - objective_from_dense(minus)) / (2.0 * step);
    }

    let ode_adj = OdeWrapper::new_jit(
        logistic_model,
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    ode_adj.set_rtol(1e-8).unwrap();
    ode_adj.set_atol(1e-8).unwrap();

    let (solution, checkpoint) = ode_adj
        .solve_adjoint_fwd(vector_host(&fit_params), vector_host(&t_eval))
        .unwrap();
    let ys = solution.get_ys().unwrap();
    let ys = ys.as_array::<f64>().unwrap();
    let dgdu_values: Vec<f64> = (0..t_eval.len())
        .map(|col| 2.0 * (ys[(0, col)] - data_values[col]))
        .collect();
    let adjoint_gradient = ode_adj
        .solve_adjoint_bkwd(
            &solution,
            &checkpoint,
            matrix_host(1, t_eval.len(), &dgdu_values),
        )
        .unwrap();
    let adjoint_gradient = adjoint_gradient.as_array::<f64>().unwrap();

    assert!(objective_fd.is_finite());
    assert_eq!(adjoint_gradient.nrows(), 3);
    assert_eq!(adjoint_gradient.ncols(), 1);
    for i in 0..adjoint_gradient.nrows() {
        assert_close(
            adjoint_gradient[(i, 0)],
            finite_difference_gradient[i],
            5e-4,
            &format!("split adjoint gradient component {i}"),
        );
    }
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn bdf_continuous_adjoint_matches_finite_difference_gradient_for_logistic_integral() {
    let logistic_model = r#"
        in_i { r = 1, k = 1, y0 = 0.1 }
        u_i { y = y0 }
        dudt_i { dydt = 0 }
        F_i { r * y * (1.0 - y / k) }
        out_i { y }
    "#;
    let ode = OdeWrapper::new_jit(
        logistic_model,
        JitBackendType::Llvm,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();

    let fit_params = [0.8, 1.3, 0.12];
    let final_time = 1.0;
    let fd_step = 1e-6;
    let objective_from_exact_integral = |params: [f64; 3]| -> f64 {
        let [r, k, y0] = params;
        let a = (k - y0) / y0;
        k * (final_time + ((1.0 + a * (-r * final_time).exp()).ln() - (1.0 + a).ln()) / r)
    };

    let objective_fd = objective_from_exact_integral(fit_params);
    let mut finite_difference_gradient = [0.0; 3];
    for i in 0..fit_params.len() {
        let mut plus = fit_params;
        let mut minus = fit_params;
        let step = fd_step * fit_params[i].abs().max(1.0);
        plus[i] += step;
        minus[i] -= step;
        finite_difference_gradient[i] = (objective_from_exact_integral(plus)
            - objective_from_exact_integral(minus))
            / (2.0 * step);
    }

    let (integral, adjoint_gradient) = ode
        .solve_continuous_adjoint(vector_host(&fit_params), final_time)
        .unwrap();
    let integral = Vec::<f64>::from_host_array(integral).unwrap();
    let adjoint_gradient = adjoint_gradient.as_array::<f64>().unwrap();

    assert!(objective_fd.is_finite());
    assert_eq!(integral.len(), 1);
    assert_close(
        integral[0],
        objective_fd,
        ASSERT_TOL,
        "continuous adjoint logistic integral",
    );
    assert_eq!(adjoint_gradient.nrows(), 3);
    assert_eq!(adjoint_gradient.ncols(), 1);
    for i in 0..adjoint_gradient.nrows() {
        assert_close(
            adjoint_gradient[(i, 0)],
            finite_difference_gradient[i],
            5e-4,
            &format!("continuous adjoint gradient component {i}"),
        );
    }
}
