// Delegate solver types selected at runtime in Host to concrete solver types
// in Rust.

use diffsol::{
    error::DiffsolError,
    matrix::{MatrixHost, MatrixRef},
    CodegenModule, ConstantOp, DefaultDenseMatrix, DefaultSolver, DiffSl, MatrixCommon,
    NonLinearOp, NonLinearOpJacobian, OdeBuilder, OdeEquations, OdeSolverProblem, Op, Vector,
    VectorCommon, VectorHost, VectorRef,
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
    valid_linear_solver::{validate_linear_solver, KluValidator, LuValidator},
};

// Each matrix type implements PySolve as bridge between diffsol and Host

use crate::solution::Solution;
pub(crate) type SolveResult = Result<Box<dyn Solution>, DiffsolJsError>;

pub(crate) trait Solve {
    fn matrix_type(&self) -> MatrixType;
    fn nstates(&self) -> usize;
    fn nparams(&self) -> usize;
    fn nout(&self) -> usize;
    fn has_stop(&self) -> bool;

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
    ) -> SolveResult;

    fn solve_hybrid(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
    ) -> SolveResult;

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult;

    fn solve_hybrid_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult;

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult;

    fn solve_hybrid_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
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

    fn nstates(&self) -> usize {
        self.problem.eqn.nstates()
    }

    fn nparams(&self) -> usize {
        self.problem.eqn.nparams()
    }

    fn nout(&self) -> usize {
        self.problem.eqn.nout()
    }

    fn has_stop(&self) -> bool {
        self.problem.eqn.root().is_some()
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
    ) -> SolveResult {
        self.check(linear_solver)?;
        self.setup_problem(params)?;
        let final_time = M::T::from_f64(final_time).unwrap();
        let soln = match linear_solver {
            LinearSolverType::Default => {
                method.solve::<M, CG, <M as DefaultSolver>::LS>(&mut self.problem, final_time)
            }
            LinearSolverType::Lu => {
                method.solve::<M, CG, <M as LuValidator<M>>::LS>(&mut self.problem, final_time)
            }
            LinearSolverType::Klu => {
                method.solve::<M, CG, <M as KluValidator<M>>::LS>(&mut self.problem, final_time)
            }
        };
        Ok(Box::new(soln?))
    }

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let soln =
            match linear_solver {
                LinearSolverType::Default => method
                    .solve_dense::<M, CG, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval),
                LinearSolverType::Lu => method
                    .solve_dense::<M, CG, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
                LinearSolverType::Klu => method
                    .solve_dense::<M, CG, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
            };
        Ok(Box::new(soln?))
    }

    fn solve_hybrid(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
    ) -> SolveResult {
        self.check(linear_solver)?;
        self.setup_problem(params)?;
        let final_time = M::T::from_f64(final_time).unwrap();
        let soln = match linear_solver {
            LinearSolverType::Default => method
                .solve_hybrid::<M, CG, <M as DefaultSolver>::LS>(&mut self.problem, final_time),
            LinearSolverType::Lu => method
                .solve_hybrid::<M, CG, <M as LuValidator<M>>::LS>(&mut self.problem, final_time),
            LinearSolverType::Klu => method
                .solve_hybrid::<M, CG, <M as KluValidator<M>>::LS>(&mut self.problem, final_time),
        };
        Ok(Box::new(soln?))
    }

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let soln = match linear_solver {
            LinearSolverType::Default => {
                method.solve_fwd_sens::<M, CG, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval)
            }
            LinearSolverType::Lu => method
                .solve_fwd_sens::<M, CG, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Klu => method
                .solve_fwd_sens::<M, CG, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
        };
        Ok(Box::new(soln?))
    }

    fn solve_hybrid_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let soln = match linear_solver {
            LinearSolverType::Default => method
                .solve_hybrid_dense::<M, CG, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Lu => method
                .solve_hybrid_dense::<M, CG, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Klu => method
                .solve_hybrid_dense::<M, CG, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    &t_eval,
                ),
        };
        Ok(Box::new(soln?))
    }

    fn solve_hybrid_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let soln = match linear_solver {
            LinearSolverType::Default => method
                .solve_hybrid_fwd_sens::<M, CG, <M as DefaultSolver>::LS>(
                    &mut self.problem,
                    &t_eval,
                ),
            LinearSolverType::Lu => method
                .solve_hybrid_fwd_sens::<M, CG, <M as LuValidator<M>>::LS>(
                    &mut self.problem,
                    &t_eval,
                ),
            LinearSolverType::Klu => method
                .solve_hybrid_fwd_sens::<M, CG, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    &t_eval,
                ),
        };
        Ok(Box::new(soln?))
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

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod tests {
    use diffsol::{
        CodegenModuleCompile, CodegenModuleJit, Context, OdeBuilder, OdeEquations, Vector,
    };

    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::{
        hybrid_logistic_state_dr, logistic_integral, logistic_state_dr, matrix_host,
    };
    use crate::{
        host_array::FromHostArray,
        linear_solver_type::LinearSolverType,
        matrix_type::MatrixType,
        ode_solver_type::OdeSolverType,
        scalar_type::ScalarType,
        test_support::{
            assert_close, hybrid_logistic_diffsl_code, hybrid_logistic_state, logistic_diffsl_code,
            logistic_state, LOGISTIC_X0,
        },
    };

    use super::{solve_factory_with_jit_backend, GenericSolve, Solve};

    fn make_generic_solve<CG>() -> GenericSolve<diffsol::NalgebraMat<f64>, CG>
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let problem = OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
            .build_from_diffsl::<CG>(logistic_diffsl_code())
            .unwrap();
        GenericSolve { problem }
    }

    fn assert_factory_supports_all_matrix_and_scalar_types<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        for matrix_type in [
            MatrixType::NalgebraDense,
            MatrixType::FaerDense,
            MatrixType::FaerSparse,
        ] {
            for scalar_type in [ScalarType::F32, ScalarType::F64] {
                assert!(solve_factory_with_jit_backend::<CG>(
                    logistic_diffsl_code(),
                    matrix_type,
                    scalar_type,
                )
                .is_ok());
            }
        }
    }

    fn assert_solve_metadata_and_helpers<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let mut solve = make_generic_solve::<CG>();
        assert_eq!(solve.matrix_type(), MatrixType::NalgebraDense);
        assert_eq!(solve.nstates(), 1);
        assert_eq!(solve.nparams(), 1);
        assert_eq!(solve.nout(), 1);
        assert!(!solve.has_stop());
        assert!(solve.check(LinearSolverType::Default).is_ok());
        assert!(solve.check(LinearSolverType::Lu).is_ok());
        assert!(solve.check(LinearSolverType::Klu).is_err());

        solve.set_atol(1e-5);
        solve.set_rtol(1e-4);
        assert_close(solve.atol(), 1e-5, 1e-12, "solve atol");
        assert_close(solve.rtol(), 1e-4, 1e-12, "solve rtol");

        let y0 = Vec::<f64>::from_host_array(solve.y0(&[2.0]).unwrap()).unwrap();
        assert_eq!(y0, vec![LOGISTIC_X0]);

        let rhs = Vec::<f64>::from_host_array(solve.rhs(&[2.0], 0.0, &[0.25]).unwrap()).unwrap();
        assert_eq!(rhs.len(), 1);
        assert_close(rhs[0], 0.375, 1e-12, "solve rhs");

        let jac_mul =
            Vec::<f64>::from_host_array(solve.rhs_jac_mul(&[2.0], 0.0, &[0.25], &[3.0]).unwrap())
                .unwrap();
        assert_eq!(jac_mul.len(), 1);
        assert_close(jac_mul[0], 3.0, 1e-12, "solve rhs jac mul");
    }

    fn assert_solve_runtime_paths<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let mut solve = make_generic_solve::<CG>();
        let soln = solve
            .solve(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], 1.0)
            .unwrap();
        let ts = Vec::<f64>::from_host_array(soln.get_ts()).unwrap();
        let ys = Vec::<Vec<f64>>::from_host_array(soln.get_ys()).unwrap();
        assert_close(*ts.last().unwrap(), 1.0, 5e-4, "solve final time");
        assert_close(
            ys[0][ts.len() - 1],
            logistic_state(LOGISTIC_X0, 2.0, 1.0),
            5e-4,
            "solve final value",
        );

        let mut solve = make_generic_solve::<CG>();
        let dense = solve
            .solve_dense(
                OdeSolverType::Tsit45,
                LinearSolverType::Lu,
                &[2.0],
                &[0.25, 0.5, 1.0],
            )
            .unwrap();
        let ts = Vec::<f64>::from_host_array(dense.get_ts()).unwrap();
        let ys = Vec::<Vec<f64>>::from_host_array(dense.get_ys()).unwrap();
        assert_eq!(ts, vec![0.25, 0.5, 1.0]);
        for (i, &t) in ts.iter().enumerate() {
            assert_close(
                ys[0][i],
                logistic_state(LOGISTIC_X0, 2.0, t),
                5e-4,
                &format!("solve_dense[{i}]"),
            );
        }

        let mut solve = make_generic_solve::<CG>();
        let err = match solve.solve(OdeSolverType::Bdf, LinearSolverType::Default, &[], 1.0) {
            Ok(_) => panic!("expected parameter count mismatch"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Expecting 1 params but got 0"));

        let hybrid_problem = OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
            .build_from_diffsl::<CG>(hybrid_logistic_diffsl_code())
            .unwrap();
        let mut hybrid_solve = GenericSolve {
            problem: hybrid_problem,
        };
        let hybrid = hybrid_solve
            .solve_hybrid(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], 2.0)
            .unwrap();
        let hybrid_ts = Vec::<f64>::from_host_array(hybrid.get_ts()).unwrap();
        let hybrid_ys = Vec::<Vec<f64>>::from_host_array(hybrid.get_ys()).unwrap();
        assert_close(
            *hybrid_ts.last().unwrap(),
            2.0,
            5e-4,
            "solve_hybrid final time",
        );
        assert_close(
            hybrid_ys[0][hybrid_ts.len() - 1],
            hybrid_logistic_state(2.0, 2.0),
            5e-4,
            "solve_hybrid final value",
        );

        let hybrid_problem = OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
            .build_from_diffsl::<CG>(hybrid_logistic_diffsl_code())
            .unwrap();
        let mut hybrid_solve = GenericSolve {
            problem: hybrid_problem,
        };
        let hybrid_dense = hybrid_solve
            .solve_hybrid_dense(
                OdeSolverType::Tsit45,
                LinearSolverType::Lu,
                &[2.0],
                &[0.5, 1.0, 1.5, 2.0],
            )
            .unwrap();
        let hybrid_dense_ts = Vec::<f64>::from_host_array(hybrid_dense.get_ts()).unwrap();
        let hybrid_dense_ys = Vec::<Vec<f64>>::from_host_array(hybrid_dense.get_ys()).unwrap();
        assert_eq!(hybrid_dense_ts, vec![0.5, 1.0, 1.5, 2.0]);
        for (i, &t) in hybrid_dense_ts.iter().enumerate() {
            assert_close(
                hybrid_dense_ys[0][i],
                hybrid_logistic_state(2.0, t),
                5e-4,
                &format!("solve_hybrid_dense[{i}]"),
            );
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    fn assert_solve_sensitivity_paths() {
        let t_eval = [0.25, 0.5, 1.0];

        let mut solve = make_generic_solve::<diffsol::LlvmModule>();
        let sens = solve
            .solve_fwd_sens(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], &t_eval)
            .unwrap();
        let sens_values = sens.get_sens();
        assert_eq!(sens_values.len(), 1);
        let sens_matrix =
            Vec::<Vec<f64>>::from_host_array(sens_values.into_iter().next().unwrap()).unwrap();
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                sens_matrix[0][i],
                logistic_state_dr(LOGISTIC_X0, 2.0, t),
                5e-4,
                &format!("solve_fwd_sens[{i}]"),
            );
        }

        let hybrid_problem = OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
            .build_from_diffsl::<diffsol::LlvmModule>(hybrid_logistic_diffsl_code())
            .unwrap();
        let mut solve = GenericSolve {
            problem: hybrid_problem,
        };
        let hybrid_sens = solve
            .solve_hybrid_fwd_sens(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], &t_eval)
            .unwrap();
        let sens_values = hybrid_sens.get_sens();
        let sens_matrix =
            Vec::<Vec<f64>>::from_host_array(sens_values.into_iter().next().unwrap()).unwrap();
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                sens_matrix[0][i],
                hybrid_logistic_state_dr(2.0, t),
                5e-4,
                &format!("solve_hybrid_fwd_sens[{i}]"),
            );
        }

        let adjoint_t_eval = [0.0, 0.25, 0.5, 1.0];
        let adjoint_data: Vec<f64> = adjoint_t_eval
            .iter()
            .map(|&t| logistic_integral(LOGISTIC_X0, 2.0, t))
            .collect();
        let mut solve = make_generic_solve::<diffsol::LlvmModule>();
        let (objective, gradient) = solve
            .solve_sum_squares_adj(
                OdeSolverType::Bdf,
                LinearSolverType::Lu,
                OdeSolverType::TrBdf2,
                LinearSolverType::Lu,
                &[2.0],
                matrix_host(1, adjoint_t_eval.len(), &adjoint_data),
                &adjoint_t_eval,
            )
            .unwrap();
        assert!(objective.is_finite());
        let gradient = Vec::<f64>::from_host_array(gradient).unwrap();
        assert_eq!(gradient.len(), 1);
        assert!(gradient[0].is_finite());
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn solve_factory_supports_all_jit_matrix_and_scalar_types_for_cranelift() {
        assert_factory_supports_all_matrix_and_scalar_types::<diffsol::CraneliftJitModule>();
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn solve_trait_helpers_and_runtime_paths_for_cranelift() {
        assert_solve_metadata_and_helpers::<diffsol::CraneliftJitModule>();
        assert_solve_runtime_paths::<diffsol::CraneliftJitModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn solve_factory_supports_all_jit_matrix_and_scalar_types_for_llvm() {
        assert_factory_supports_all_matrix_and_scalar_types::<diffsol::LlvmModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn solve_trait_helpers_and_runtime_paths_for_llvm() {
        assert_solve_metadata_and_helpers::<diffsol::LlvmModule>();
        assert_solve_runtime_paths::<diffsol::LlvmModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn solve_trait_sensitivity_paths_for_llvm() {
        assert_solve_sensitivity_paths();
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn setup_problem_validates_parameter_count_for_cranelift() {
        let mut solve = make_generic_solve::<diffsol::CraneliftJitModule>();
        let err = solve.setup_problem(&[]).unwrap_err();
        assert!(err.to_string().contains("Expecting 1 params but got 0"));

        solve.setup_problem(&[2.0]).unwrap();
        let mut params = solve.problem.context().vector_zeros(1);
        solve.problem.eqn.get_params(&mut params);
        assert_eq!(params.get_index(0), 2.0);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn setup_problem_validates_parameter_count_for_llvm() {
        let mut solve = make_generic_solve::<diffsol::LlvmModule>();
        let err = solve.setup_problem(&[]).unwrap_err();
        assert!(err.to_string().contains("Expecting 1 params but got 0"));

        solve.setup_problem(&[2.0]).unwrap();
        let mut params = solve.problem.context().vector_zeros(1);
        solve.problem.eqn.get_params(&mut params);
        assert_eq!(params.get_index(0), 2.0);
    }
}
