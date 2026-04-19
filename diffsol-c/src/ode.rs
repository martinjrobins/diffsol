use std::sync::{Arc, Mutex};

use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};

use crate::jit::JitBackendType;
use crate::{
    error::DiffsolRtError,
    host_array::HostArray,
    initial_condition_options::{
        InitialConditionSolverOptions, InitialConditionSolverOptionsSnapshot,
    },
    linear_solver_type::LinearSolverType,
    matrix_type::MatrixType,
    ode_options::{OdeSolverOptions, OdeSolverOptionsSnapshot},
    ode_solver_type::OdeSolverType,
    scalar_type::ScalarType,
    solution_wrapper::SolutionWrapper,
    solve::Solve,
};

pub struct Ode {
    pub(crate) solve: Box<dyn Solve>,
    code: String,
    scalar_type: ScalarType,
    jit_backend: Option<JitBackendType>,
    linear_solver: LinearSolverType,
    ode_solver: OdeSolverType,
}

unsafe impl Send for Ode {}
unsafe impl Sync for Ode {}

#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OdeWrapperSnapshot {
    code: String,
    equation: Vec<u8>,
    jit_backend: JitBackendType,
    scalar_type: ScalarType,
    matrix_type: MatrixType,
    linear_solver: LinearSolverType,
    ode_solver: OdeSolverType,
    rtol: f64,
    atol: f64,
    ic_options: InitialConditionSolverOptionsSnapshot,
    ode_options: OdeSolverOptionsSnapshot,
}

impl OdeWrapper {
    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolRtError> {
        self.0.lock().map_err(|_| {
            DiffsolRtError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on ODE solver".to_string(),
            ))
        })
    }
}

impl OdeWrapper {
    fn snapshot(&self) -> Result<OdeWrapperSnapshot, DiffsolRtError> {
        let ode = self.guard()?;
        let jit_backend = ode.jit_backend.ok_or_else(|| {
            DiffsolRtError::from(diffsol::error::DiffsolError::Other(
                "OdeWrapper serialization is only supported for JIT-backed solvers".to_string(),
            ))
        })?;
        Ok(OdeWrapperSnapshot {
            code: ode.code.clone(),
            equation: ode.solve.serialized_diffsl()?,
            jit_backend,
            scalar_type: ode.scalar_type,
            matrix_type: ode.solve.matrix_type(),
            linear_solver: ode.linear_solver,
            ode_solver: ode.ode_solver,
            rtol: ode.solve.rtol(),
            atol: ode.solve.atol(),
            ic_options: InitialConditionSolverOptionsSnapshot::from_solve(ode.solve.as_ref()),
            ode_options: OdeSolverOptionsSnapshot::from_solve(ode.solve.as_ref()),
        })
    }

    fn build(
        code: String,
        scalar_type: ScalarType,
        solve: Box<dyn Solve>,
        jit_backend: Option<JitBackendType>,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        solve.check(linear_solver)?;
        Ok(OdeWrapper(Arc::new(Mutex::new(Ode {
            code,
            scalar_type,
            solve,
            jit_backend,
            linear_solver,
            ode_solver,
        }))))
    }

    fn from_snapshot(snapshot: OdeWrapperSnapshot) -> Result<Self, DiffsolRtError> {
        let solve = crate::solve::solve_factory_from_serialized_diffsl(
            snapshot.equation.as_slice(),
            snapshot.matrix_type,
            snapshot.scalar_type,
        )?;
        let wrapper = Self::build(
            snapshot.code,
            snapshot.scalar_type,
            solve,
            Some(snapshot.jit_backend),
            snapshot.linear_solver,
            snapshot.ode_solver,
        )?;
        {
            let mut ode = wrapper.guard()?;
            ode.solve.set_rtol(snapshot.rtol);
            ode.solve.set_atol(snapshot.atol);
            snapshot.ic_options.apply_to_solve(ode.solve.as_mut());
            snapshot.ode_options.apply_to_solve(ode.solve.as_mut());
        }
        Ok(wrapper)
    }

    /// Construct an ODE solver backed by externally-provided DiffSL symbols.
    #[cfg(feature = "external")]
    pub fn new_external(
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        let solve = crate::solve::solve_factory_external(
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            matrix_type,
            scalar_type,
        )?;
        Self::build(
            String::new(),
            scalar_type,
            solve,
            None,
            linear_solver,
            ode_solver,
        )
    }

    /// Construct an ODE solver backed by DiffSL symbols loaded from a dynamic library.
    #[cfg(feature = "diffsl-external-dynamic")]
    pub fn new_external_dynamic(
        path: impl Into<std::path::PathBuf>,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        let solve = crate::solve::solve_factory_external_dynamic(
            path.into(),
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            matrix_type,
            scalar_type,
        )?;
        Self::build(
            String::new(),
            scalar_type,
            solve,
            None,
            linear_solver,
            ode_solver,
        )
    }

    /// Construct an ODE solver by JIT-compiling DiffSL code immediately.
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    pub fn new_jit(
        code: &str,
        jit_backend: JitBackendType,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        let solve = crate::solve::solve_factory_jit(code, jit_backend, matrix_type, scalar_type)?;
        Self::build(
            code.to_owned(),
            scalar_type,
            solve,
            Some(jit_backend),
            linear_solver,
            ode_solver,
        )
    }

    /// Matrix type used in the ODE solver. This is fixed after construction.
    pub fn get_matrix_type(&self) -> Result<MatrixType, DiffsolRtError> {
        Ok(self.guard()?.solve.matrix_type())
    }

    pub fn get_nstates(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.nstates())
    }

    pub fn get_nparams(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.nparams())
    }

    pub fn get_nout(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.nout())
    }

    pub fn has_stop(&self) -> Result<bool, DiffsolRtError> {
        Ok(self.guard()?.solve.has_stop())
    }

    /// Ode solver method, default Bdf (backward differentiation formula).
    pub fn get_ode_solver(&self) -> Result<OdeSolverType, DiffsolRtError> {
        Ok(self.guard()?.ode_solver)
    }

    pub fn set_ode_solver(&self, value: OdeSolverType) -> Result<(), DiffsolRtError> {
        self.guard()?.ode_solver = value;
        Ok(())
    }

    /// Linear solver type used in the ODE solver. Set to default to use the
    /// solver's default choice, which is typically an LU solver.
    pub fn get_linear_solver(&self) -> Result<LinearSolverType, DiffsolRtError> {
        Ok(self.guard()?.linear_solver)
    }

    pub fn set_linear_solver(&self, value: LinearSolverType) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.check(value)?;
        self.guard()?.linear_solver = value;
        Ok(())
    }

    /// Relative tolerance for the solver, default 1e-6. Governs the error relative to the solution size.
    pub fn get_rtol(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.rtol())
    }

    pub fn set_rtol(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_rtol(value);
        Ok(())
    }

    /// Absolute tolerance for the solver, default 1e-6. Governs the error as the solution goes to zero.
    pub fn get_atol(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.atol())
    }

    pub fn set_atol(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_atol(value);
        Ok(())
    }

    pub fn get_code(&self) -> Result<String, DiffsolRtError> {
        Ok(self.guard()?.code.clone())
    }

    pub fn get_scalar_type(&self) -> Result<ScalarType, DiffsolRtError> {
        Ok(self.guard()?.scalar_type)
    }

    pub fn get_jit_backend(&self) -> Result<Option<JitBackendType>, DiffsolRtError> {
        Ok(self.guard()?.jit_backend)
    }

    pub fn get_ic_options(&self) -> InitialConditionSolverOptions {
        InitialConditionSolverOptions::new(self.0.clone())
    }

    pub fn get_options(&self) -> OdeSolverOptions {
        OdeSolverOptions::new(self.0.clone())
    }

    /// Get the initial condition vector y0 as a 1D numpy array.
    pub fn y0(&self, params: HostArray) -> Result<HostArray, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        self_guard.solve.y0(params.as_slice()?)
    }

    /// evaluate the right-hand side function at time `t` and state `y`.
    pub fn rhs(
        &self,
        params: HostArray,
        t: f64,
        y: HostArray,
    ) -> Result<HostArray, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        self_guard.solve.rhs(params.as_slice()?, t, y.as_slice()?)
    }

    /// evaluate the right-hand side Jacobian-vector product `Jv`` at time `t` and state `y`.
    pub fn rhs_jac_mul(
        &self,
        params: HostArray,
        t: f64,
        y: HostArray,
        v: HostArray,
    ) -> Result<HostArray, DiffsolRtError> {
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
    pub fn solve(
        &self,
        params: HostArray,
        final_time: f64,
    ) -> Result<SolutionWrapper, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let solution = self_guard
            .solve
            .solve(method, linear_solver, params, final_time)?;
        Ok(SolutionWrapper::new(solution))
    }

    /// Solve a hybrid ODE up to `final_time`, automatically applying reset
    /// functions and continuing after root events until the solution completes.
    pub fn solve_hybrid(
        &self,
        params: HostArray,
        final_time: f64,
    ) -> Result<SolutionWrapper, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let solution = self_guard
            .solve
            .solve_hybrid(method, linear_solver, params, final_time)?;
        Ok(SolutionWrapper::new(solution))
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
    pub fn solve_dense(
        &self,
        params: HostArray,
        t_eval: HostArray,
    ) -> Result<SolutionWrapper, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let solution = self_guard
            .solve
            .solve_dense(method, linear_solver, params, t_eval)?;
        Ok(SolutionWrapper::new(solution))
    }

    /// Solve a hybrid ODE at dense evaluation times, automatically applying
    /// reset functions and continuing after root events until all requested
    /// output points are filled.
    pub fn solve_hybrid_dense(
        &self,
        params: HostArray,
        t_eval: HostArray,
    ) -> Result<SolutionWrapper, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let solution =
            self_guard
                .solve
                .solve_hybrid_dense(method, linear_solver, params, t_eval)?;
        Ok(SolutionWrapper::new(solution))
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
    pub fn solve_fwd_sens(
        &self,
        params: HostArray,
        t_eval: HostArray,
    ) -> Result<SolutionWrapper, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let solution = self_guard
            .solve
            .solve_fwd_sens(method, linear_solver, params, t_eval)?;
        Ok(SolutionWrapper::new(solution))
    }

    /// Solve a hybrid ODE with forward sensitivities at dense evaluation times,
    /// automatically applying sensitivity-aware reset functions and continuing
    /// after root events until all requested output points are filled.
    #[allow(clippy::type_complexity)]
    pub fn solve_hybrid_fwd_sens(
        &self,
        params: HostArray,
        t_eval: HostArray,
    ) -> Result<SolutionWrapper, DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let solution =
            self_guard
                .solve
                .solve_hybrid_fwd_sens(method, linear_solver, params, t_eval)?;
        Ok(SolutionWrapper::new(solution))
    }

    /// Using the provided state, solve the adjoint problem for the sum of squares
    /// objective given data at timepoints `t_eval`.
    /// Returns the objective value and a list of 1D arrays of adjoint sensitivities
    /// for each parameter.
    #[allow(clippy::type_complexity)]
    pub fn solve_sum_squares_adj(
        &self,
        params: HostArray,
        data: HostArray,
        t_eval: HostArray,
    ) -> Result<(f64, HostArray), DiffsolRtError> {
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

impl Serialize for OdeWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.snapshot()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OdeWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let snapshot = OdeWrapperSnapshot::deserialize(deserializer)?;
        Self::from_snapshot(snapshot).map_err(DeError::custom)
    }
}

#[cfg(all(test, feature = "diffsl-external-f64"))]
mod tests {
    use super::*;
    use crate::host_array::FromHostArray;
    use crate::linear_solver_type::LinearSolverType;
    use crate::scalar_type::ScalarType;
    use crate::test_support::{
        assert_close, assert_solution_tail, logistic_state, logistic_state_dr, mass_state_deps,
        rhs_input_deps, rhs_state_deps, vector_host, ASSERT_TOL, LOGISTIC_X0,
    };

    fn all_ode_solvers() -> [OdeSolverType; 4] {
        [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ]
    }

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

    fn assert_runtime_dispatch(matrix_type: MatrixType) {
        let ode = make_ode(matrix_type, OdeSolverType::Bdf);
        assert_eq!(ode.get_matrix_type().unwrap(), matrix_type);
        assert_eq!(ode.get_nstates().unwrap(), 1);
        assert_eq!(ode.get_nparams().unwrap(), 1);
        assert_eq!(ode.get_nout().unwrap(), 1);
        assert!(ode.has_stop().unwrap());

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

    fn assert_solver_dense_solution(matrix_type: MatrixType, ode_solver: OdeSolverType) {
        let ode = make_ode(matrix_type, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    }

    fn hybrid_root_time() -> f64 {
        0.5 * 9.0_f64.ln()
    }

    fn assert_hybrid_solution_applies_reset_after_root(ode_solver: OdeSolverType) {
        let ode = make_ode(MatrixType::NalgebraDense, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let final_time = 2.0;
        let solution = ode.solve_hybrid(vector_host(&[2.0]), final_time).unwrap();
        let ys = solution.get_ys().unwrap();
        let ys = ys.as_array::<f64>().unwrap();
        let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();
        let root_time = hybrid_root_time();

        assert_eq!(ys.nrows(), 1);
        assert_eq!(ys.ncols(), ts.len());
        assert!(!ts.is_empty(), "expected hybrid solve to produce output");
        assert_close(
            *ts.last().unwrap(),
            final_time,
            ASSERT_TOL,
            "hybrid final time",
        );
        assert_close(ys[(0, ys.ncols() - 1)], 1.0, 5e-4, "hybrid final value");
        assert!(
            ts.iter().any(|&t| t < root_time),
            "expected pre-root samples"
        );
        assert!(
            ts.iter().any(|&t| t > root_time),
            "expected post-root samples after reset"
        );
    }

    fn assert_hybrid_dense_solution_continues_after_reset(ode_solver: OdeSolverType) {
        let ode = make_ode(MatrixType::NalgebraDense, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];
        let solution = ode
            .solve_hybrid_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();
        let ys = solution.get_ys().unwrap();
        let ys = ys.as_array::<f64>().unwrap();

        assert_eq!(ys.nrows(), 1);
        assert_eq!(ys.ncols(), t_eval.len());
        assert_close(
            ys[(0, 0)],
            logistic_state(LOGISTIC_X0, 2.0, t_eval[0]),
            5e-4,
            "hybrid dense pre-root value",
        );
        assert_close(
            ys[(0, 1)],
            logistic_state(LOGISTIC_X0, 2.0, t_eval[1]),
            5e-4,
            "hybrid dense near-root value",
        );
        for col in 2..t_eval.len() {
            assert_close(ys[(0, col)], 1.0, 5e-4, "hybrid dense post-root value");
        }
    }

    fn assert_hybrid_forward_sensitivities_complete_across_reset(ode_solver: OdeSolverType) {
        let ode = make_ode(MatrixType::NalgebraDense, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];
        let solution = ode
            .solve_hybrid_fwd_sens(vector_host(&[2.0]), vector_host(&t_eval))
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
        assert_close(
            ys[(0, 0)],
            logistic_state(LOGISTIC_X0, 2.0, t_eval[0]),
            5e-4,
            "hybrid sens pre-root value",
        );
        for col in 2..t_eval.len() {
            assert_close(ys[(0, col)], 1.0, 5e-4, "hybrid sens post-root value");
            assert!(
                sens_values[(0, col)].is_finite(),
                "expected finite post-root sensitivity at column {col}"
            );
        }
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
    fn bdf_dense_solution_matches_logistic_solution() {
        let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    }

    #[test]
    fn esdirk34_dense_solution_matches_logistic_solution() {
        assert_solver_dense_solution(MatrixType::FaerDense, OdeSolverType::Esdirk34);
    }

    #[test]
    fn tr_bdf2_sparse_solution_matches_logistic_solution() {
        assert_solver_dense_solution(MatrixType::FaerSparse, OdeSolverType::TrBdf2);
    }

    #[test]
    fn tsit45_dense_solution_matches_logistic_solution() {
        assert_solver_dense_solution(MatrixType::NalgebraDense, OdeSolverType::Tsit45);
    }

    #[test]
    fn bdf_forward_sensitivities_match_logistic_derivative() {
        let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
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
                &format!("sensitivity[{i}]"),
            );
        }
    }

    #[test]
    fn bdf_sum_squares_adjoint_matches_external_logistic_model() {
        let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let data_values: Vec<f64> = t_eval
            .iter()
            .map(|&t| logistic_state(LOGISTIC_X0, 2.0, t))
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

    #[test]
    fn hybrid_solution_applies_reset_after_root_for_all_solvers() {
        for ode_solver in all_ode_solvers() {
            assert_hybrid_solution_applies_reset_after_root(ode_solver);
        }
    }

    #[test]
    fn hybrid_dense_solution_continues_after_reset_for_all_solvers() {
        for ode_solver in all_ode_solvers() {
            assert_hybrid_dense_solution_continues_after_reset(ode_solver);
        }
    }

    #[test]
    fn hybrid_forward_sensitivities_complete_across_reset_for_all_solvers() {
        for ode_solver in all_ode_solvers() {
            assert_hybrid_forward_sensitivities_complete_across_reset(ode_solver);
        }
    }
}

#[cfg(all(test, feature = "diffsl-external-dynamic"))]
mod dynamic_tests {
    use crate::host_array::FromHostArray;
    use crate::linear_solver_type::LinearSolverType;
    use crate::scalar_type::ScalarType;
    use crate::test_support::{
        assert_close, assert_solution_tail, external_dynamic_fixture_path, mass_state_deps,
        rhs_input_deps, rhs_state_deps, vector_host, ASSERT_TOL, LOGISTIC_X0,
    };

    use super::*;

    fn make_ode(matrix_type: MatrixType, ode_solver: OdeSolverType) -> OdeWrapper {
        OdeWrapper::new_external_dynamic(
            external_dynamic_fixture_path(),
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

    #[test]
    fn runtime_dispatch_matches_requested_matrix_type() {
        for matrix_type in [
            MatrixType::NalgebraDense,
            MatrixType::FaerDense,
            MatrixType::FaerSparse,
        ] {
            let ode = make_ode(matrix_type, OdeSolverType::Bdf);
            assert_eq!(ode.get_matrix_type().unwrap(), matrix_type);
            assert_eq!(ode.get_code().unwrap(), "");
            assert_eq!(ode.get_jit_backend().unwrap(), None);
            assert_eq!(ode.get_nstates().unwrap(), 1);
            assert_eq!(ode.get_nparams().unwrap(), 1);
            assert_eq!(ode.get_nout().unwrap(), 1);
            assert!(ode.has_stop().unwrap());

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
    }

    #[test]
    fn dense_solution_matches_logistic_solution() {
        let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    }

    #[test]
    fn non_jit_serialization_is_rejected() {
        let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
        let err = serde_json::to_string(&ode).unwrap_err().to_string();
        assert!(err.contains("JIT-backed"));
    }
}

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod jit_tests {
    use crate::host_array::FromHostArray;
    use crate::jit::JitBackendType;
    use crate::linear_solver_type::LinearSolverType;
    use crate::scalar_type::ScalarType;
    use crate::test_support::{
        assert_close, assert_solution_tail, available_jit_backends, hybrid_logistic_diffsl_code,
        hybrid_logistic_period, hybrid_logistic_state, logistic_diffsl_code, logistic_state,
        vector_host, ASSERT_TOL, LOGISTIC_X0,
    };
    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::{hybrid_logistic_state_dr, logistic_state_dr};
    #[cfg(any(
        all(feature = "diffsl-llvm", not(feature = "diffsl-cranelift")),
        all(feature = "diffsl-cranelift", not(feature = "diffsl-llvm"))
    ))]
    use serde_json::Value;
    use serde_json::{self};

    use super::*;

    fn all_ode_solvers() -> [OdeSolverType; 4] {
        [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ]
    }

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

    fn assert_serialization_roundtrip(jit_backend: JitBackendType, matrix_type: MatrixType) {
        let ode = make_ode(jit_backend, matrix_type, OdeSolverType::Bdf);
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
        assert_eq!(decoded.get_scalar_type().unwrap(), ScalarType::F64);
        assert_eq!(decoded.get_matrix_type().unwrap(), matrix_type);
        assert_eq!(
            decoded.get_linear_solver().unwrap(),
            serialized_linear_solver(matrix_type)
        );
        assert_eq!(decoded.get_ode_solver().unwrap(), OdeSolverType::TrBdf2);
        assert_eq!(decoded.get_rtol().unwrap(), 1e-7);
        assert_eq!(decoded.get_atol().unwrap(), 1e-9);

        let ic_options = decoded.get_ic_options();
        assert!(ic_options.get_use_linesearch().unwrap());
        assert_eq!(ic_options.get_max_linesearch_iterations().unwrap(), 13);
        assert_eq!(ic_options.get_max_newton_iterations().unwrap(), 17);
        assert_eq!(ic_options.get_max_linear_solver_setups().unwrap(), 19);
        assert_eq!(ic_options.get_step_reduction_factor().unwrap(), 0.5);
        assert_eq!(ic_options.get_armijo_constant().unwrap(), 1e-4);

        let options = decoded.get_options();
        assert_eq!(options.get_max_nonlinear_solver_iterations().unwrap(), 23);
        assert_eq!(options.get_max_error_test_failures().unwrap(), 29);
        assert_eq!(options.get_update_jacobian_after_steps().unwrap(), 31);
        assert_eq!(options.get_update_rhs_jacobian_after_steps().unwrap(), 37);
        assert_eq!(options.get_threshold_to_update_jacobian().unwrap(), 1e-3);
        assert_eq!(
            options.get_threshold_to_update_rhs_jacobian().unwrap(),
            2e-3
        );
        assert_eq!(options.get_min_timestep().unwrap(), 1e-4);

        let y0_after =
            Vec::<f64>::from_host_array(decoded.y0(vector_host(&[2.0])).unwrap()).unwrap();
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
        let t_eval = [0.25, 0.5, 1.0];
        let solution = decoded
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();
        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    }

    fn assert_runtime_dispatch(jit_backend: JitBackendType, matrix_type: MatrixType) {
        let ode = make_ode(jit_backend, matrix_type, OdeSolverType::Bdf);
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
        matrix_type: MatrixType,
        ode_solver: OdeSolverType,
    ) {
        let ode = make_ode(jit_backend, matrix_type, ode_solver);
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
            .unwrap();

        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
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

        let solution = ode.solve_hybrid(vector_host(&[r]), final_time).unwrap();
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
            .solve_hybrid_dense(vector_host(&[r]), vector_host(&t_eval))
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
            .solve_hybrid_fwd_sens(vector_host(&[r]), vector_host(&t_eval))
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
            for (matrix_type, solver) in [
                (MatrixType::FaerDense, OdeSolverType::Esdirk34),
                (MatrixType::FaerSparse, OdeSolverType::TrBdf2),
                (MatrixType::NalgebraDense, OdeSolverType::Tsit45),
            ] {
                assert_solver_dense_solution(jit_backend, matrix_type, solver);
            }
        }
    }

    #[test]
    fn bdf_dense_solution_matches_logistic_diffsl_model() {
        for jit_backend in available_jit_backends() {
            let ode = make_ode(jit_backend, MatrixType::NalgebraDense, OdeSolverType::Bdf);
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
            let ode = make_ode(jit_backend, MatrixType::NalgebraDense, OdeSolverType::Bdf);
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

    #[test]
    fn serialization_roundtrip_restores_full_solver_state() {
        for jit_backend in available_jit_backends() {
            for matrix_type in [MatrixType::NalgebraDense, MatrixType::FaerSparse] {
                assert_serialization_roundtrip(jit_backend, matrix_type);
            }
        }
    }

    #[cfg(all(feature = "diffsl-llvm", not(feature = "diffsl-cranelift")))]
    #[test]
    fn deserialization_rejects_unavailable_jit_backend() {
        let ode = make_ode(
            JitBackendType::Llvm,
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        let mut value = serde_json::to_value(&ode).unwrap();
        value["jit_backend"] = Value::String("cranelift".to_string());
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
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        let mut value = serde_json::to_value(&ode).unwrap();
        value["jit_backend"] = Value::String("llvm".to_string());
        let err = serde_json::from_value::<OdeWrapper>(value)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("unknown variant"));
    }

    #[test]
    fn hybrid_solution_matches_piecewise_logistic_diffsl_model() {
        for jit_backend in available_jit_backends() {
            for ode_solver in all_ode_solvers() {
                assert_hybrid_solution_matches_piecewise_logistic_diffsl_model(
                    jit_backend,
                    ode_solver,
                );
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
    fn bdf_forward_sensitivities_match_logistic_derivative_from_diffsl() {
        let ode = make_ode(
            JitBackendType::Llvm,
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
    fn hybrid_forward_sensitivities_match_piecewise_logistic_diffsl_model() {
        for ode_solver in all_ode_solvers() {
            assert_hybrid_forward_sensitivities_match_piecewise_logistic_diffsl_model(ode_solver);
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_sum_squares_adjoint_matches_logistic_diffsl_model() {
        let ode = make_ode(
            JitBackendType::Llvm,
            MatrixType::NalgebraDense,
            OdeSolverType::Bdf,
        );
        ode.set_rtol(1e-8).unwrap();
        ode.set_atol(1e-8).unwrap();

        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let data_values: Vec<f64> = t_eval
            .iter()
            .map(|&t| logistic_state(LOGISTIC_X0, 2.0, t))
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

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_sum_squares_adjoint_matches_finite_difference_gradient_for_logistic_model() {
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

        let data = crate::test_support::matrix_host(1, t_eval.len(), &data_values);
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

        let (objective_adj, sens) = ode_adj
            .solve_sum_squares_adj(vector_host(&fit_params), data, vector_host(&t_eval))
            .unwrap();
        let adjoint_gradient = Vec::<f64>::from_host_array(sens).unwrap();

        assert_eq!(adjoint_gradient.len(), 3);
        assert_close(
            objective_adj,
            objective_fd,
            1e-5,
            "sum_squares objective from dense finite differences",
        );
        for i in 0..adjoint_gradient.len() {
            assert_close(
                adjoint_gradient[i],
                finite_difference_gradient[i],
                5e-4,
                &format!("sum_squares gradient component {i}"),
            );
        }
    }
}
