use std::sync::{Arc, Mutex};

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use crate::jit::JitBackendType;
use crate::{
    error::DiffsolJsError, host_array::HostArray,
    initial_condition_options::InitialConditionSolverOptions, linear_solver_type::LinearSolverType,
    matrix_type::MatrixType, ode_options::OdeSolverOptions, ode_solver_type::OdeSolverType,
    scalar_type::ScalarType, solution::Solution, solution_wrapper::SolutionWrapper, solve::Solve,
};

pub struct Ode {
    pub(crate) solve: Box<dyn Solve>,
    code: String,
    linear_solver: LinearSolverType,
    ode_solver: OdeSolverType,
}

pub struct OdeWrapper(Arc<Mutex<Ode>>);
type SolveCallResult = Result<Box<dyn Solution>, (DiffsolJsError, Option<Box<dyn Solution>>)>;

impl OdeWrapper {
    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolJsError> {
        self.0.lock().map_err(|_| {
            DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on ODE solver".to_string(),
            ))
        })
    }
}

impl OdeWrapper {
    fn build(
        code: String,
        solve: Box<dyn Solve>,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolJsError> {
        solve.check(linear_solver)?;
        Ok(OdeWrapper(Arc::new(Mutex::new(Ode {
            code,
            solve,
            linear_solver,
            ode_solver,
        }))))
    }

    /// Construct an ODE solver backed by externally-provided DiffSL symbols.
    #[cfg(feature = "external")]
    pub(crate) fn new_external(
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolJsError> {
        let solve = crate::solve::solve_factory_external(
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            matrix_type,
            scalar_type,
        )?;
        Self::build(String::new(), solve, linear_solver, ode_solver)
    }

    /// Construct an ODE solver by JIT-compiling DiffSL code immediately.
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    pub(crate) fn new_jit(
        code: &str,
        jit_backend: JitBackendType,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolJsError> {
        let solve = crate::solve::solve_factory_jit(code, jit_backend, matrix_type, scalar_type)?;
        Self::build(code.to_owned(), solve, linear_solver, ode_solver)
    }

    fn run_solve_call<F>(
        py_solve: &mut dyn Solve,
        solution: Option<SolutionWrapper>,
        solve_call: F,
    ) -> Result<SolutionWrapper, DiffsolJsError>
    where
        F: FnOnce(&mut dyn Solve, Option<Box<dyn Solution>>) -> SolveCallResult,
    {
        if let Some(solution) = solution {
            let old_solution = solution.take_solution()?;
            match solve_call(py_solve, Some(old_solution)) {
                Ok(new_solution) => Ok(SolutionWrapper::new(new_solution)),
                Err((err, maybe_old_solution)) => {
                    if let Some(old_solution) = maybe_old_solution {
                        solution.replace_solution(old_solution)?;
                    }
                    Err(err)
                }
            }
        } else {
            let new_solution = solve_call(py_solve, None).map_err(|(err, _)| err)?;
            Ok(SolutionWrapper::new(new_solution))
        }
    }

    /// Matrix type used in the ODE solver. This is fixed after construction.
    pub(crate) fn get_matrix_type(&self) -> Result<MatrixType, DiffsolJsError> {
        Ok(self.guard()?.solve.matrix_type())
    }

    /// Ode solver method, default Bdf (backward differentiation formula).
    pub(crate) fn get_ode_solver(&self) -> Result<OdeSolverType, DiffsolJsError> {
        Ok(self.guard()?.ode_solver)
    }

    pub(crate) fn set_ode_solver(&self, value: OdeSolverType) -> Result<(), DiffsolJsError> {
        self.guard()?.ode_solver = value;
        Ok(())
    }

    /// Linear solver type used in the ODE solver. Set to default to use the
    /// solver's default choice, which is typically an LU solver.
    pub(crate) fn get_linear_solver(&self) -> Result<LinearSolverType, DiffsolJsError> {
        Ok(self.guard()?.linear_solver)
    }

    pub(crate) fn set_linear_solver(&self, value: LinearSolverType) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.check(value)?;
        self.guard()?.linear_solver = value;
        Ok(())
    }

    /// Relative tolerance for the solver, default 1e-6. Governs the error relative to the solution size.
    pub(crate) fn get_rtol(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.rtol())
    }

    pub(crate) fn set_rtol(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_rtol(value);
        Ok(())
    }

    /// Absolute tolerance for the solver, default 1e-6. Governs the error as the solution goes to zero.
    pub(crate) fn get_atol(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.atol())
    }

    pub(crate) fn set_atol(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_atol(value);
        Ok(())
    }

    pub fn get_code(&self) -> Result<String, DiffsolJsError> {
        Ok(self.guard()?.code.clone())
    }

    pub(crate) fn get_ic_options(&self) -> InitialConditionSolverOptions {
        InitialConditionSolverOptions::new(self.0.clone())
    }

    pub(crate) fn get_options(&self) -> OdeSolverOptions {
        OdeSolverOptions::new(self.0.clone())
    }

    /// Get the initial condition vector y0 as a 1D numpy array.
    pub(crate) fn y0(&mut self, params: HostArray) -> Result<HostArray, DiffsolJsError> {
        let mut self_guard = self.guard()?;
        self_guard.solve.y0(params.as_slice()?)
    }

    /// evaluate the right-hand side function at time `t` and state `y`.
    pub(crate) fn rhs(
        &mut self,
        params: HostArray,
        t: f64,
        y: HostArray,
    ) -> Result<HostArray, DiffsolJsError> {
        let mut self_guard = self.guard()?;
        self_guard.solve.rhs(params.as_slice()?, t, y.as_slice()?)
    }

    /// evaluate the right-hand side Jacobian-vector product `Jv`` at time `t` and state `y`.
    pub(crate) fn rhs_jac_mul(
        &mut self,
        params: HostArray,
        t: f64,
        y: HostArray,
        v: HostArray,
    ) -> Result<HostArray, DiffsolJsError> {
        let mut self_guard = self.guard()?;
        self_guard
            .solve
            .rhs_jac_mul(params.as_slice()?, t, y.as_slice()?, v.as_slice()?)
    }

    /// Using the provided state, solve the problem up to time `final_time`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    /// If specified, the config can be used to override the solver method
    /// (Bdf by default) and SolverType (Lu by default) along with other solver
    /// params like `rtol`.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param final_time: end time of solver
    /// :type final_time: float
    /// :return: `(ys, ts)` tuple where `ys` is a 2D array of values at times
    ///     `ts` chosen by the solver
    /// :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    ///
    /// Example:
    ///     >>> print(ode.solve(np.array([]), 0.5))
    #[allow(clippy::type_complexity)]
    pub(crate) fn solve(
        &mut self,
        params: HostArray,
        final_time: f64,
        solution: Option<SolutionWrapper>,
    ) -> Result<SolutionWrapper, DiffsolJsError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        Self::run_solve_call(&mut *self_guard.solve, solution, |py_solve, py_solution| {
            py_solve.solve(method, linear_solver, params, final_time, py_solution)
        })
    }

    /// Using the provided state, solve the problem up to time
    /// `t_eval[t_eval.len()-1]`. Returns 2D array of solution values at
    /// timepoints given by `t_eval`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    /// The config may be optionally specified to override solver settings.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type params: numpy.ndarray
    /// :return: 2D array of values at times `t_eval`
    /// :rtype: numpy.ndarray
    pub(crate) fn solve_dense(
        &mut self,
        params: HostArray,
        t_eval: HostArray,
        solution: Option<SolutionWrapper>,
    ) -> Result<SolutionWrapper, DiffsolJsError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        Self::run_solve_call(&mut *self_guard.solve, solution, |py_solve, py_solution| {
            py_solve.solve_dense(method, linear_solver, params, t_eval, py_solution)
        })
    }

    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`.
    /// Returns 2D array of solution values at timepoints given by `t_eval`.
    /// Also returns a list of 2D arrays of sensitivities at the same timepoints
    /// as the solution.
    /// The number of params must match the expected params in the diffsl code.
    /// The config may be optionally specified to override solver settings.
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type params: numpy.ndarray
    /// :return: 2D array of values at times `t_eval` and a list of 2D arrays of sensitivities at the same timepoints
    /// :rtype: (numpy.ndarray, List[numpy.ndarray])
    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_fwd_sens(
        &mut self,
        params: HostArray,
        t_eval: HostArray,
        solution: Option<SolutionWrapper>,
    ) -> Result<SolutionWrapper, DiffsolJsError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        Self::run_solve_call(&mut *self_guard.solve, solution, |py_solve, py_solution| {
            py_solve.solve_fwd_sens(method, linear_solver, params, t_eval, py_solution)
        })
    }

    /// Using the provided state, solve the adjoint problem for the sum of squares
    /// objective given data at timepoints `t_eval`.
    /// Returns the objective value and a list of 1D arrays of adjoint sensitivities
    /// for each parameter.
    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_sum_squares_adj(
        &mut self,
        params: HostArray,
        data: HostArray,
        t_eval: HostArray,
    ) -> Result<(f64, HostArray), DiffsolJsError> {
        let mut self_guard = self.guard()?;
        let linear_solver = self_guard.linear_solver;
        let ode_solver = self_guard.ode_solver;

        self_guard.solve.solve_sum_squares_adj(
            ode_solver,
            linear_solver,
            ode_solver,
            linear_solver,
            params.as_slice()?,
            data,
            t_eval.as_slice()?,
        )
    }
}

#[cfg(all(test, feature = "diffsl-external-f64"))]
mod tests {
    use crate::host_array::FromHostArray;
    use crate::linear_solver_type::LinearSolverType;
    use crate::scalar_type::ScalarType;
    use crate::test_support::{
        ASSERT_TOL, LOGISTIC_X0, assert_close, assert_current_state, assert_solution_tail,
        logistic_integral, logistic_state, logistic_state_dr, mass_state_deps, rhs_input_deps,
        rhs_state_deps, vector_host,
    };

    use super::*;

    fn make_ode(matrix_type: MatrixType, ode_solver: OdeSolverType) -> OdeWrapper {
        OdeWrapper::new_external(
            rhs_state_deps(),
            rhs_input_deps(),
            mass_state_deps(),
            ScalarType::F64,
            matrix_type,
            LinearSolverType::Default,
            ode_solver,
        )
        .unwrap()
    }

    fn make_seed_solution(ode: &mut OdeWrapper, x0: f64) -> SolutionWrapper {
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&[1e-9]), None)
            .unwrap();
        solution.set_current_state(&[x0]).unwrap();
        assert_current_state(&solution, &[x0], ASSERT_TOL);
        solution
    }

    fn assert_runtime_dispatch(matrix_type: MatrixType) {
        let mut ode = make_ode(matrix_type, OdeSolverType::Bdf);
        assert_eq!(ode.get_matrix_type().unwrap(), matrix_type);

        let y0 = ode.y0(vector_host(&[2.0])).unwrap();
        assert_eq!(Vec::<f64>::from_host_array(y0).unwrap(), vec![LOGISTIC_X0]);

        let rhs = ode
            .rhs(vector_host(&[2.0]), 0.0, vector_host(&[0.25]))
            .unwrap();
        assert_close(
            Vec::<f64>::from_host_array(rhs).unwrap()[0],
            0.375,
            ASSERT_TOL,
            "rhs(0.25)",
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
            "rhs_jac_mul(0.25, 3.0)",
        );
    }

    fn assert_solver_continuation(matrix_type: MatrixType, ode_solver: OdeSolverType) {
        let mut ode = make_ode(matrix_type, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let seed = make_seed_solution(&mut ode, 0.1);
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval), Some(seed))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, 0.1, 2.0, 5e-4);
        assert_current_state(
            &solution,
            &[logistic_state(0.1, 2.0, *t_eval.last().unwrap())],
            5e-4,
        );
    }

    #[test]
    fn runtime_dispatch_matches_requested_matrix_type() {
        for matrix_type in [
            MatrixType::NalgebraDense,
            MatrixType::FaerDense,
            MatrixType::FaerSparse,
        ] {
            assert_runtime_dispatch(matrix_type);
        }
    }

    #[test]
    fn bdf_dense_continuation_matches_logistic_solution() {
        let mut ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval), None)
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
        assert_current_state(
            &solution,
            &[logistic_state(LOGISTIC_X0, 2.0, *t_eval.last().unwrap())],
            5e-4,
        );
    }

    #[test]
    fn esdirk34_dense_continuation_matches_logistic_solution() {
        assert_solver_continuation(MatrixType::FaerDense, OdeSolverType::Esdirk34);
    }

    #[test]
    fn tr_bdf2_sparse_continuation_matches_logistic_solution() {
        assert_solver_continuation(MatrixType::FaerSparse, OdeSolverType::TrBdf2);
    }

    #[test]
    fn tsit45_dense_continuation_matches_logistic_solution() {
        assert_solver_continuation(MatrixType::NalgebraDense, OdeSolverType::Tsit45);
    }

    #[test]
    fn bdf_forward_sensitivities_match_logistic_derivative() {
        let mut ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_fwd_sens(vector_host(&[2.0]), vector_host(&t_eval), None)
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
                &format!("sensitivity[{i}]"),
            );
        }
    }

    #[test]
    fn bdf_sum_squares_adjoint_matches_external_logistic_model() {
        let mut ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let data_values: Vec<f64> = t_eval
            .iter()
            .map(|&t| logistic_integral(LOGISTIC_X0, 2.0, t))
            .collect();
        let data = crate::test_support::matrix_host(1, t_eval.len(), &data_values);
        let (value, sens) = ode
            .solve_sum_squares_adj(vector_host(&[2.0]), data, vector_host(&t_eval))
            .unwrap();
        let grad = Vec::<f64>::from_host_array(sens).unwrap();

        assert_close(value, 0.0, ASSERT_TOL, "sum_squares objective");
        assert_eq!(grad.len(), 1);
        assert_close(grad[0], 0.0, ASSERT_TOL, "sum_squares gradient");
    }
}

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod jit_tests {
    use crate::host_array::FromHostArray;
    use crate::jit::JitBackendType;
    use crate::linear_solver_type::LinearSolverType;
    use crate::scalar_type::ScalarType;
    use crate::test_support::{
        ASSERT_TOL, LOGISTIC_X0, assert_close, assert_current_state, assert_solution_tail,
        available_jit_backends, logistic_diffsl_code, logistic_state, vector_host,
    };
    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::{logistic_integral, logistic_state_dr};

    use super::*;

    fn make_ode(
        jit_backend: JitBackendType,
        matrix_type: MatrixType,
        ode_solver: OdeSolverType,
    ) -> OdeWrapper {
        OdeWrapper::new_jit(
            logistic_diffsl_code(),
            jit_backend,
            ScalarType::F64,
            matrix_type,
            LinearSolverType::Default,
            ode_solver,
        )
        .unwrap()
    }

    fn make_seed_solution(ode: &mut OdeWrapper, x0: f64) -> SolutionWrapper {
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&[1e-9]), None)
            .unwrap();
        solution.set_current_state(&[x0]).unwrap();
        assert_current_state(&solution, &[x0], ASSERT_TOL);
        solution
    }

    fn assert_runtime_dispatch(jit_backend: JitBackendType, matrix_type: MatrixType) {
        let mut ode = make_ode(jit_backend, matrix_type, OdeSolverType::Bdf);
        assert_eq!(ode.get_matrix_type().unwrap(), matrix_type);
        assert_eq!(ode.get_code().unwrap(), logistic_diffsl_code());

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

    fn assert_solver_continuation(
        jit_backend: JitBackendType,
        matrix_type: MatrixType,
        ode_solver: OdeSolverType,
    ) {
        let mut ode = make_ode(jit_backend, matrix_type, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let seed = make_seed_solution(&mut ode, LOGISTIC_X0);
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval), Some(seed))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
        assert_current_state(
            &solution,
            &[logistic_state(LOGISTIC_X0, 2.0, *t_eval.last().unwrap())],
            5e-4,
        );
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
    fn solver_continuation_matches_logistic_solution_from_diffsl() {
        for jit_backend in available_jit_backends() {
            for (matrix_type, solver) in [
                (MatrixType::FaerDense, OdeSolverType::Esdirk34),
                (MatrixType::FaerSparse, OdeSolverType::TrBdf2),
                (MatrixType::NalgebraDense, OdeSolverType::Tsit45),
            ] {
                assert_solver_continuation(jit_backend, matrix_type, solver);
            }
        }
    }

    #[test]
    fn bdf_dense_solution_matches_logistic_diffsl_model() {
        for jit_backend in available_jit_backends() {
            let mut ode = make_ode(jit_backend, MatrixType::NalgebraDense, OdeSolverType::Bdf);
            ode.set_rtol(1e-8).unwrap();
            ode.set_atol(1e-8).unwrap();

            let t_eval = [0.25, 0.5, 1.0];
            let solution = ode
                .solve_dense(vector_host(&[2.0]), vector_host(&t_eval), None)
                .unwrap();

            assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
            assert_current_state(
                &solution,
                &[logistic_state(LOGISTIC_X0, 2.0, *t_eval.last().unwrap())],
                5e-4,
            );
        }
    }

    #[test]
    fn bdf_solution_matches_logistic_diffsl_model() {
        for jit_backend in available_jit_backends() {
            let x0 = LOGISTIC_X0;
            let r = 2.0;
            let mut ode = make_ode(jit_backend, MatrixType::NalgebraDense, OdeSolverType::Bdf);
            ode.set_rtol(1e-8).unwrap();
            ode.set_atol(1e-8).unwrap();

            let final_time = 1.0;
            let solution = ode.solve(vector_host(&[r]), final_time, None).unwrap();

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
            assert_current_state(&solution, &[logistic_state(x0, r, final_time)], 5e-4);
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_forward_sensitivities_match_logistic_derivative_from_diffsl() {
        let mut ode = make_ode(
            JitBackendType::Llvm,
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_fwd_sens(vector_host(&[2.0]), vector_host(&t_eval), None)
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
    fn bdf_sum_squares_adjoint_matches_logistic_diffsl_model() {
        let mut ode = make_ode(
            JitBackendType::Llvm,
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let data_values: Vec<f64> = t_eval
            .iter()
            .map(|&t| logistic_integral(LOGISTIC_X0, 2.0, t))
            .collect();
        let data = crate::test_support::matrix_host(1, t_eval.len(), &data_values);
        let (value, sens) = ode
            .solve_sum_squares_adj(vector_host(&[2.0]), data, vector_host(&t_eval))
            .unwrap();
        let grad = Vec::<f64>::from_host_array(sens).unwrap();

        assert_close(value, 0.0, ASSERT_TOL, "jit sum_squares objective");
        assert_eq!(grad.len(), 1);
        assert!(
            grad[0].is_finite(),
            "jit sum_squares gradient should be finite"
        );
    }
}
