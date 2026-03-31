// Delegate solver types selected at runtime in Host to concrete solver types
// in Rust.

use diffsol::{
    CodegenModule, ConstantOp, DefaultDenseMatrix, DefaultSolver, DiffSl, MatrixCommon,
    NonLinearOp, NonLinearOpJacobian, OdeBuilder, OdeEquations, OdeSolverProblem, OdeSolverState,
    Op, Vector, VectorCommon, VectorHost, VectorRef,
    error::DiffsolError,
    matrix::{MatrixHost, MatrixRef},
};
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use diffsol::{CodegenModuleCompile, CodegenModuleJit};
use num_traits::{FromPrimitive, ToPrimitive}; // for from_f64 and to_f64
use paste::paste;

use crate::error::DiffsolJsError;
use crate::host_array::{HostArray, ToHostArray};
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use crate::jit::JitBackendType;
#[cfg(feature = "external")]
use crate::scalar_type::ExternalScalar;
use crate::scalar_type::{Scalar, ScalarType};
use crate::{
    generate_ic_option_accessors, generate_ode_option_accessors, generate_option_accessors,
    generate_trait_ic_option_accessors, generate_trait_ode_option_accessors,
    option_value_from_store, option_value_to_store,
};

use crate::{
    linear_solver_type::LinearSolverType,
    matrix_type::{MatrixKind, MatrixType},
    ode_solver_type::OdeSolverType,
    valid_linear_solver::{KluValidator, LuValidator, validate_linear_solver},
};

// Each matrix type implements PySolve as bridge between diffsol and Host

use crate::solution::{GenericSolution, GenericState, Solution};
pub(crate) type SolveError = (DiffsolJsError, Option<Box<dyn Solution>>);
pub(crate) type SolveResult = Result<Box<dyn Solution>, SolveError>;

pub(crate) trait Solve {
    fn matrix_type(&self) -> MatrixType;

    fn rhs(&mut self, params: &[f64], t: f64, y: &[f64]) -> Result<HostArray, DiffsolJsError>;

    fn rhs_jac_mul(
        &mut self,
        params: &[f64],
        t: f64,
        y: &[f64],
        v: &[f64],
    ) -> Result<HostArray, DiffsolJsError>;

    fn y0(&mut self, params: &[f64]) -> Result<HostArray, DiffsolJsError>;

    fn check(&self, linear_solver: LinearSolverType) -> Result<(), DiffsolJsError>;
    fn set_rtol(&mut self, rtol: f64);
    fn rtol(&self) -> f64;
    fn set_atol(&mut self, atol: f64);
    fn atol(&self) -> f64;

    // New API: solution object support
    fn solve(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
        solution: Option<Box<dyn Solution>>,
    ) -> SolveResult;

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
        solution: Option<Box<dyn Solution>>,
    ) -> SolveResult;

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
        solution: Option<Box<dyn Solution>>,
    ) -> SolveResult;

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn solve_sum_squares_adj(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        backwards_method: OdeSolverType,
        backwards_linear_solver: LinearSolverType,
        params: &[f64],
        data: HostArray,
        t_eval: &[f64],
    ) -> Result<(f64, HostArray), DiffsolJsError>;

    generate_trait_ic_option_accessors! {
        use_linesearch: bool,
        max_linesearch_iterations: usize,
        max_newton_iterations: usize,
        max_linear_solver_setups: usize,
        step_reduction_factor: f64,
        armijo_constant: f64
    }
    generate_trait_ode_option_accessors! {
        max_nonlinear_solver_iterations: usize,
        max_error_test_failures: usize,
        min_timestep: f64,
        update_jacobian_after_steps: usize,
        update_rhs_jacobian_after_steps: usize,
        threshold_to_update_jacobian: f64,
        threshold_to_update_rhs_jacobian: f64
    }
}
// Public factory method for generating an instance based on matrix type
#[cfg(feature = "external")]
pub(crate) fn solve_factory_external(
    rhs_state_deps: Vec<(usize, usize)>,
    rhs_input_deps: Vec<(usize, usize)>,
    mass_state_deps: Vec<(usize, usize)>,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolJsError> {
    let solve: Box<dyn Solve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            #[cfg(feature = "diffsl-external-f32")]
            ScalarType::F32 => Box::new(GenericSolve::<
                diffsol::NalgebraMat<f32>,
                diffsl::ExternalModule<f32>,
            >::from_external(
                rhs_state_deps, rhs_input_deps, mass_state_deps, false
            )?),
            #[cfg(feature = "diffsl-external-f64")]
            ScalarType::F64 => Box::new(GenericSolve::<
                diffsol::NalgebraMat<f64>,
                diffsl::ExternalModule<f64>,
            >::from_external(
                rhs_state_deps, rhs_input_deps, mass_state_deps, false
            )?),
            _ => {
                return Err(DiffsolJsError::from(DiffsolError::Other(
                    "Unsupported scalar type for NalgebraDense".to_string(),
                )));
            }
        },
        MatrixType::FaerDense => match scalar_type {
            #[cfg(feature = "diffsl-external-f32")]
            ScalarType::F32 => Box::new(GenericSolve::<
                diffsol::FaerMat<f32>,
                diffsl::ExternalModule<f32>,
            >::from_external(
                rhs_state_deps, rhs_input_deps, mass_state_deps, false
            )?),
            #[cfg(feature = "diffsl-external-f64")]
            ScalarType::F64 => Box::new(GenericSolve::<
                diffsol::FaerMat<f64>,
                diffsl::ExternalModule<f64>,
            >::from_external(
                rhs_state_deps, rhs_input_deps, mass_state_deps, false
            )?),
            _ => {
                return Err(DiffsolJsError::from(DiffsolError::Other(
                    "Unsupported scalar type for FaerDense".to_string(),
                )));
            }
        },
        MatrixType::FaerSparse => match scalar_type {
            #[cfg(feature = "diffsl-external-f32")]
            ScalarType::F32 => Box::new(GenericSolve::<
                diffsol::FaerSparseMat<f32>,
                diffsl::ExternalModule<f32>,
            >::from_external(
                rhs_state_deps, rhs_input_deps, mass_state_deps, false
            )?),
            #[cfg(feature = "diffsl-external-f64")]
            ScalarType::F64 => Box::new(GenericSolve::<
                diffsol::FaerSparseMat<f64>,
                diffsl::ExternalModule<f64>,
            >::from_external(
                rhs_state_deps, rhs_input_deps, mass_state_deps, false
            )?),
            _ => {
                return Err(DiffsolJsError::from(DiffsolError::Other(
                    "Unsupported scalar type for FaerSparse".to_string(),
                )));
            }
        },
    };
    Ok(solve)
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn solve_factory_jit(
    code: &str,
    jit_backend: JitBackendType,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolJsError> {
    match jit_backend {
        #[cfg(feature = "diffsl-cranelift")]
        JitBackendType::Cranelift => solve_factory_with_jit_backend::<diffsol::CraneliftJitModule>(
            code,
            matrix_type,
            scalar_type,
        ),
        #[cfg(feature = "diffsl-llvm")]
        JitBackendType::Llvm => {
            solve_factory_with_jit_backend::<diffsol::LlvmModule>(code, matrix_type, scalar_type)
        }
    }
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
fn solve_factory_with_jit_backend<CG>(
    code: &str,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolJsError>
where
    CG: CodegenModule + CodegenModuleJit + CodegenModuleCompile,
{
    let solve: Box<dyn Solve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            ScalarType::F32 => {
                let problem =
                    OdeBuilder::<diffsol::NalgebraMat<f32>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
            ScalarType::F64 => {
                let problem =
                    OdeBuilder::<diffsol::NalgebraMat<f64>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
        },
        MatrixType::FaerDense => match scalar_type {
            ScalarType::F32 => {
                let problem =
                    OdeBuilder::<diffsol::FaerMat<f32>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
            ScalarType::F64 => {
                let problem =
                    OdeBuilder::<diffsol::FaerMat<f64>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
        },
        MatrixType::FaerSparse => match scalar_type {
            ScalarType::F32 => {
                let problem = OdeBuilder::<diffsol::FaerSparseMat<f32>>::new()
                    .build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
            ScalarType::F64 => {
                let problem = OdeBuilder::<diffsol::FaerSparseMat<f64>>::new()
                    .build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
        },
    };
    Ok(solve)
}

pub(crate) struct GenericSolve<M, CG>
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost,
    CG: CodegenModule,
{
    problem: OdeSolverProblem<DiffSl<M, CG>>,
}

impl<M, CG> GenericSolve<M, CG>
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
    CG: CodegenModule,
{
    pub(crate) fn setup_problem(&mut self, params: &[f64]) -> Result<(), DiffsolJsError> {
        let params: Vec<M::T> = params.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let params = M::V::from_slice(&params, M::C::default());

        // Attempt to set problem from params and config
        let nparams = self.problem.eqn.nparams();
        if params.len() == nparams {
            self.problem.eqn.set_params(&params);
            Ok(())
        } else {
            Err(DiffsolError::Other(format!(
                "Expecting {} params but got {}",
                nparams,
                params.len()
            ))
            .into())
        }
    }

    fn get_initial_state(
        &self,
        solution: &dyn Solution,
    ) -> Result<GenericState<M::V>, DiffsolJsError>
    where
        M::V: 'static,
    {
        solution.downcast_typed_solution::<M::V>()?.state_clone()
    }

    fn create_or_append_solution(
        &self,
        solution: Option<Box<dyn Solution>>,
        state: GenericState<M::V>,
        ys: <M::V as DefaultDenseMatrix>::M,
        ts: Vec<M::T>,
        sens: Vec<<M::V as DefaultDenseMatrix>::M>,
    ) -> SolveResult
    where
        M::V: Send + Sync + 'static,
        <M::V as DefaultDenseMatrix>::M: Send + Sync,
        <M::V as VectorCommon>::Inner: ToHostArray<M::T> + Clone,
        <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: ToHostArray<M::T> + Clone,
    {
        if let Some(mut solution) = solution {
            let solution_typed = match solution.downcast_typed_solution_mut::<M::V>() {
                Ok(s) => s,
                Err(err) => return Err((err, Some(solution))),
            };
            if let Err(err) = solution_typed.append(state, ys, ts, sens) {
                return Err((DiffsolError::Other(err).into(), Some(solution)));
            }
            Ok(solution)
        } else {
            Ok(Box::new(GenericSolution::<M::V>::new(state, ys, ts, sens)))
        }
    }
}

#[cfg(feature = "external")]
impl<M> GenericSolve<M, diffsl::ExternalModule<M::T>>
where
    M: MatrixHost<T: ExternalScalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    pub fn from_external(
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        include_sensitivities: bool,
    ) -> Result<Self, DiffsolJsError> {
        let eqn = DiffSl::<M, diffsl::ExternalModule<M::T>>::from_external(
            M::C::default(),
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            include_sensitivities,
        )?;
        let default_p = vec![0.0; eqn.nparams()];
        let problem = OdeBuilder::<M>::new().p(default_p).build_from_eqn(eqn)?;
        Ok(GenericSolve { problem })
    }
}

impl<M, CG> Solve for GenericSolve<M, CG>
where
    M: MatrixHost<T: Scalar + ToPrimitive>
        + DefaultSolver
        + LuValidator<M>
        + KluValidator<M>
        + MatrixKind,
    CG: CodegenModule,
    for<'b> <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: ToHostArray<M::T> + Clone,
    for<'b> <M::V as VectorCommon>::Inner: ToHostArray<M::T> + Clone,
    M::V: VectorHost + DefaultDenseMatrix + Send + Sync + 'static,
    <M::V as DefaultDenseMatrix>::M: Send + Sync,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    fn matrix_type(&self) -> MatrixType {
        MatrixType::from_diffsol::<M>()
    }

    fn check(&self, linear_solver: LinearSolverType) -> Result<(), DiffsolJsError> {
        validate_linear_solver::<M>(linear_solver)
    }

    fn set_atol(&mut self, atol: f64) {
        self.problem.atol.fill(M::T::from_f64(atol).unwrap());
    }

    fn atol(&self) -> f64 {
        self.problem.atol[0].to_f64().unwrap()
    }

    fn set_rtol(&mut self, rtol: f64) {
        self.problem.rtol = M::T::from_f64(rtol).unwrap();
    }

    fn rtol(&self) -> f64 {
        self.problem.rtol.to_f64().unwrap()
    }

    generate_ic_option_accessors! {
        use_linesearch: bool,
        max_linesearch_iterations: usize,
        max_newton_iterations: usize,
        max_linear_solver_setups: usize,
        step_reduction_factor: f64,
        armijo_constant: f64
    }

    generate_ode_option_accessors! {
        max_nonlinear_solver_iterations: usize,
        max_error_test_failures: usize,
        min_timestep: f64,
        update_jacobian_after_steps: usize,
        update_rhs_jacobian_after_steps: usize,
        threshold_to_update_jacobian: f64,
        threshold_to_update_rhs_jacobian: f64
    }

    fn y0(&mut self, params: &[f64]) -> Result<HostArray, DiffsolJsError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let mut y0 = M::V::zeros(n, M::C::default());
        let t0 = self.problem.t0;
        self.problem.eqn.init().call_inplace(t0, &mut y0);
        Ok((*y0.inner()).clone().to_host_array())
    }

    fn rhs(&mut self, params: &[f64], t: f64, y: &[f64]) -> Result<HostArray, DiffsolJsError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let y = y
            .iter()
            .map(|&x| M::T::from_f64(x).unwrap())
            .collect::<Vec<_>>();
        let y_vec = M::V::from_slice(&y, M::C::default());
        let mut dydt = M::V::zeros(n, M::C::default());
        self.problem
            .eqn
            .rhs()
            .call_inplace(&y_vec, M::T::from_f64(t).unwrap(), &mut dydt);
        Ok((*dydt.inner()).clone().to_host_array())
    }

    fn rhs_jac_mul(
        &mut self,

        params: &[f64],
        t: f64,
        y: &[f64],
        v: &[f64],
    ) -> Result<HostArray, DiffsolJsError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let y = y
            .iter()
            .map(|&x| M::T::from_f64(x).unwrap())
            .collect::<Vec<_>>();
        let v = v
            .iter()
            .map(|&x| M::T::from_f64(x).unwrap())
            .collect::<Vec<_>>();
        let y_vec = M::V::from_slice(&y, M::C::default());
        let v_vec = M::V::from_slice(&v, M::C::default());
        let mut dydt = M::V::zeros(n, M::C::default());
        self.problem.eqn.rhs().jac_mul_inplace(
            &y_vec,
            M::T::from_f64(t).unwrap(),
            &v_vec,
            &mut dydt,
        );
        Ok((*dydt.inner()).clone().to_host_array())
    }

    fn solve(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
        solution: Option<Box<dyn Solution>>,
    ) -> SolveResult {
        if let Err(err) = self.check(linear_solver) {
            return Err((err, solution));
        }
        if let Err(err) = self.setup_problem(params) {
            return Err((err, solution));
        }
        let final_time = M::T::from_f64(final_time).unwrap();
        let initial_state = match solution
            .as_deref()
            .map(|solution| self.get_initial_state(solution))
            .transpose()
        {
            Ok(state) => state,
            Err(err) => return Err((err, solution)),
        };
        let solve_result = match linear_solver {
            LinearSolverType::Default => method.solve::<M, CG, <M as DefaultSolver>::LS>(
                &mut self.problem,
                final_time,
                initial_state,
            ),
            LinearSolverType::Lu => method.solve::<M, CG, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                final_time,
                initial_state,
            ),
            LinearSolverType::Klu => method.solve::<M, CG, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                final_time,
                initial_state,
            ),
        };
        let (ys, ts, state) = match solve_result {
            Ok(res) => res,
            Err(err) => return Err((err.into(), solution)),
        };

        self.create_or_append_solution(solution, state, ys, ts, Vec::new())
    }

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
        solution: Option<Box<dyn Solution>>,
    ) -> SolveResult {
        if let Err(err) = self.check(linear_solver) {
            return Err((err, solution));
        }
        if let Err(err) = self.setup_problem(params) {
            return Err((err, solution));
        }

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let initial_state = match solution
            .as_deref()
            .map(|solution| self.get_initial_state(solution))
            .transpose()
        {
            Ok(state) => state,
            Err(err) => return Err((err, solution)),
        };
        let solve_result = match linear_solver {
            LinearSolverType::Default => method.solve_dense::<M, CG, <M as DefaultSolver>::LS>(
                &mut self.problem,
                &t_eval,
                initial_state,
            ),
            LinearSolverType::Lu => method.solve_dense::<M, CG, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                &t_eval,
                initial_state,
            ),
            LinearSolverType::Klu => method.solve_dense::<M, CG, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                &t_eval,
                initial_state,
            ),
        };
        let (ys, state) = match solve_result {
            Ok(res) => res,
            Err(err) => return Err((err.into(), solution)),
        };

        let ncols = ys.ncols();
        let ts = if ncols == t_eval.len() {
            t_eval
        } else {
            let mut ts: Vec<M::T> = t_eval
                .iter()
                .copied()
                .take(ncols.saturating_sub(1))
                .collect();
            if ncols > 0 {
                let final_t = match &state {
                    GenericState::Bdf(s) => s.as_ref().t,
                    GenericState::Rk(s) => s.as_ref().t,
                };
                ts.push(final_t);
            }
            ts
        };
        self.create_or_append_solution(solution, state, ys, ts, Vec::new())
    }

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
        solution: Option<Box<dyn Solution>>,
    ) -> SolveResult {
        if let Err(err) = self.check(linear_solver) {
            return Err((err, solution));
        }
        if let Err(err) = self.setup_problem(params) {
            return Err((err, solution));
        }

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let initial_state = match solution
            .as_deref()
            .map(|solution| self.get_initial_state(solution))
            .transpose()
        {
            Ok(state) => state,
            Err(err) => return Err((err, solution)),
        };

        let solve_result = match linear_solver {
            LinearSolverType::Default => method.solve_fwd_sens::<M, CG, <M as DefaultSolver>::LS>(
                &mut self.problem,
                &t_eval,
                initial_state,
            ),
            LinearSolverType::Lu => method.solve_fwd_sens::<M, CG, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                &t_eval,
                initial_state,
            ),
            LinearSolverType::Klu => method.solve_fwd_sens::<M, CG, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                &t_eval,
                initial_state,
            ),
        };
        let (ys, sens, state) = match solve_result {
            Ok(res) => res,
            Err(err) => return Err((err.into(), solution)),
        };

        let ncols = ys.ncols();
        let ts = if ncols == t_eval.len() {
            t_eval
        } else {
            let mut ts: Vec<M::T> = t_eval
                .iter()
                .copied()
                .take(ncols.saturating_sub(1))
                .collect();
            if ncols > 0 {
                let final_t = match &state {
                    GenericState::Bdf(s) => s.as_ref().t,
                    GenericState::Rk(s) => s.as_ref().t,
                };
                ts.push(final_t);
            }
            ts
        };
        self.create_or_append_solution(solution, state, ys, ts, sens)
    }

    fn solve_sum_squares_adj(
        &mut self,

        method: OdeSolverType,
        linear_solver: LinearSolverType,
        backwards_method: OdeSolverType,
        backwards_linear_solver: LinearSolverType,
        params: &[f64],
        data: HostArray,
        t_eval: &[f64],
    ) -> Result<(f64, HostArray), DiffsolJsError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let data = data.as_array()?;
        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();

        let previous_integrate_out = self.problem.integrate_out;
        self.problem.integrate_out = true;
        let result = match linear_solver {
            LinearSolverType::Default => method
                .solve_sum_squares_adj::<M, CG, <M as DefaultSolver>::LS>(
                    &mut self.problem,
                    data,
                    &t_eval,
                    backwards_method,
                    backwards_linear_solver,
                ),
            LinearSolverType::Lu => method
                .solve_sum_squares_adj::<M, CG, <M as LuValidator<M>>::LS>(
                    &mut self.problem,
                    data,
                    &t_eval,
                    backwards_method,
                    backwards_linear_solver,
                ),
            LinearSolverType::Klu => method
                .solve_sum_squares_adj::<M, CG, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    data,
                    &t_eval,
                    backwards_method,
                    backwards_linear_solver,
                ),
        };
        self.problem.integrate_out = previous_integrate_out;
        let (y, y_sens) = result?;

        Ok((
            y.to_f64().unwrap(),
            (*y_sens.inner()).clone().to_host_array(),
        ))
    }
}
