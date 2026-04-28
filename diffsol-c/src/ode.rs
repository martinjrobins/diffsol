use std::sync::{Arc, Mutex};

use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};

use crate::jit::JitBackendType;
use crate::{
    adjoint_checkpoint::AdjointCheckpointWrapper,
    error::DiffsolRtError,
    host_array::{FromHostArray, HostArray},
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
    t0: f64,
    h0: f64,
    integrate_out: bool,
    sens_rtol: Option<f64>,
    sens_atol: Option<f64>,
    out_rtol: Option<f64>,
    out_atol: Option<f64>,
    param_rtol: Option<f64>,
    param_atol: Option<f64>,
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
            t0: ode.solve.t0(),
            h0: ode.solve.h0(),
            integrate_out: ode.solve.integrate_out(),
            sens_rtol: ode.solve.sens_rtol(),
            sens_atol: ode.solve.sens_atol(),
            out_rtol: ode.solve.out_rtol(),
            out_atol: ode.solve.out_atol(),
            param_rtol: ode.solve.param_rtol(),
            param_atol: ode.solve.param_atol(),
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
            ode.solve.set_t0(snapshot.t0);
            ode.solve.set_h0(snapshot.h0);
            ode.solve.set_integrate_out(snapshot.integrate_out);
            ode.solve.set_sens_rtol(snapshot.sens_rtol);
            ode.solve.set_sens_atol(snapshot.sens_atol);
            ode.solve.set_out_rtol(snapshot.out_rtol);
            ode.solve.set_out_atol(snapshot.out_atol);
            ode.solve.set_param_rtol(snapshot.param_rtol);
            ode.solve.set_param_atol(snapshot.param_atol);
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
    #[allow(clippy::too_many_arguments)]
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

    /// Initial time for the ODE solve, default 0.0.
    pub fn get_t0(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.t0())
    }

    pub fn set_t0(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_t0(value);
        Ok(())
    }

    /// Initial step size for the ODE solver, default 1.0.
    pub fn get_h0(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.h0())
    }

    pub fn set_h0(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_h0(value);
        Ok(())
    }

    /// Whether to integrate output equations alongside state equations.
    pub fn get_integrate_out(&self) -> Result<bool, DiffsolRtError> {
        Ok(self.guard()?.solve.integrate_out())
    }

    pub fn set_integrate_out(&self, value: bool) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_integrate_out(value);
        Ok(())
    }

    /// Relative tolerance for forward sensitivity or adjoint equations.
    pub fn get_sens_rtol(&self) -> Result<Option<f64>, DiffsolRtError> {
        Ok(self.guard()?.solve.sens_rtol())
    }

    pub fn set_sens_rtol(&self, value: Option<f64>) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_sens_rtol(value);
        Ok(())
    }

    /// Absolute tolerance for forward sensitivity or adjoint equations.
    pub fn get_sens_atol(&self) -> Result<Option<f64>, DiffsolRtError> {
        Ok(self.guard()?.solve.sens_atol())
    }

    pub fn set_sens_atol(&self, value: Option<f64>) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_sens_atol(value);
        Ok(())
    }

    /// Relative tolerance for integrated output equations.
    pub fn get_out_rtol(&self) -> Result<Option<f64>, DiffsolRtError> {
        Ok(self.guard()?.solve.out_rtol())
    }

    pub fn set_out_rtol(&self, value: Option<f64>) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_out_rtol(value);
        Ok(())
    }

    /// Absolute tolerance for integrated output equations.
    pub fn get_out_atol(&self) -> Result<Option<f64>, DiffsolRtError> {
        Ok(self.guard()?.solve.out_atol())
    }

    pub fn set_out_atol(&self, value: Option<f64>) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_out_atol(value);
        Ok(())
    }

    /// Relative tolerance for adjoint parameter gradient equations.
    pub fn get_param_rtol(&self) -> Result<Option<f64>, DiffsolRtError> {
        Ok(self.guard()?.solve.param_rtol())
    }

    pub fn set_param_rtol(&self, value: Option<f64>) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_param_rtol(value);
        Ok(())
    }

    /// Absolute tolerance for adjoint parameter gradient equations.
    pub fn get_param_atol(&self) -> Result<Option<f64>, DiffsolRtError> {
        Ok(self.guard()?.solve.param_atol())
    }

    pub fn set_param_atol(&self, value: Option<f64>) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_param_atol(value);
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

    /// Solve the continuous adjoint problem for the integral of the model output
    /// from the initial time to `final_time`.
    ///
    /// Returns `(integral, gradient)`, where `integral` is a vector of length
    /// `nout` and `gradient` is an `(nparams, nout)` matrix.
    pub fn solve_continuous_adjoint(
        &self,
        params: HostArray,
        final_time: f64,
    ) -> Result<(HostArray, HostArray), DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let linear_solver = self_guard.linear_solver;
        let ode_solver = self_guard.ode_solver;
        self_guard.solve.solve_continuous_adjoint(
            ode_solver,
            linear_solver,
            params.as_slice()?,
            final_time,
        )
    }

    /// Solve the forward problem at `t_eval` and retain checkpoint data for a
    /// later discrete adjoint backward pass.
    pub fn solve_adjoint_fwd(
        &self,
        params: HostArray,
        t_eval: HostArray,
    ) -> Result<(SolutionWrapper, AdjointCheckpointWrapper), DiffsolRtError> {
        let mut self_guard = self.guard()?;
        let params = params.as_slice()?;
        let t_eval = t_eval.as_slice()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        let (solution, checkpoint) =
            self_guard
                .solve
                .solve_adjoint_fwd(method, linear_solver, params, t_eval)?;
        Ok((SolutionWrapper::new(solution), checkpoint))
    }

    /// Solve the discrete adjoint backward pass using a prior forward adjoint
    /// checkpoint and the gradient of a scalar objective with respect to model
    /// outputs at each saved evaluation time.
    ///
    /// Returns an `(nparams, 1)` gradient matrix.
    pub fn solve_adjoint_bkwd(
        &self,
        solution: &SolutionWrapper,
        checkpoint: &AdjointCheckpointWrapper,
        dgdu_eval: HostArray,
    ) -> Result<HostArray, DiffsolRtError> {
        let t_eval_host = solution.get_ts()?;
        let t_eval = Vec::<f64>::from_host_array(t_eval_host)?;
        let mut self_guard = self.guard()?;
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.ode_solver;
        self_guard
            .solve
            .solve_adjoint_bkwd(method, linear_solver, checkpoint, &t_eval, dgdu_eval)
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
        let solution = ode.solve(vector_host(&[2.0]), final_time).unwrap();
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
            .solve_dense(vector_host(&[2.0]), vector_host(&t_eval))
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
            .solve_fwd_sens(vector_host(&[2.0]), vector_host(&t_eval))
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
