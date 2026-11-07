// Delegate solver types selected at runtime in Host to concrete solver types
// in Rust.

use crate::adjoint_checkpoint::AdjointCheckpointWrapper;
use crate::error::DiffsolRtError;
use crate::host_array::HostArray;
use crate::host_array::ToHostArray;
use crate::scalar_type::Scalar;
use crate::{
    generate_ic_option_accessors, generate_ode_option_accessors, generate_option_accessors,
    option_value_from_store, option_value_to_store,
};
use crate::{generate_trait_ic_option_accessors, generate_trait_ode_option_accessors};
use diffsol::OdeBuilder;
use diffsol::{
    error::DiffsolError,
    matrix::{Matrix, MatrixRef},
    ConstantOp, Context, DefaultDenseMatrix, DefaultSolver, DenseMatrix, MatrixCommon, NonLinearOp,
    NonLinearOpJacobian, OdeEquations, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens,
    OdeSolverProblem, Vector, VectorCommon, VectorRef, VectorView,
};
use num_traits::{FromPrimitive, ToPrimitive}; // for from_f64 and to_f64
use paste::paste;

use crate::solve_eqn::{EqnRequirement, SolveEqn};
use crate::{
    linear_solver_type::LinearSolverType, matrix_type::MatrixType, ode_solver_type::OdeSolverType,
};
use crate::{
    matrix_type::MatrixKind,
    valid_linear_solver::{validate_linear_solver, KluValidator, LuValidator},
};

// Each matrix type implements PySolve as bridge between diffsol and Host

use crate::solution::Solution;
pub(crate) type SolveResult = Result<Box<dyn Solution>, DiffsolRtError>;

pub(crate) trait Solve: Send {
    fn matrix_type(&self) -> MatrixType;
    fn nstates(&self) -> usize;
    fn nparams(&self) -> usize;
    fn nout(&self) -> usize;
    fn has_stop(&self) -> bool;

    fn rhs(&mut self, params: &[f64], t: f64, y: &[f64]) -> Result<HostArray, DiffsolRtError>;

    fn rhs_jac_mul(
        &mut self,
        params: &[f64],
        t: f64,
        y: &[f64],
        v: &[f64],
    ) -> Result<HostArray, DiffsolRtError>;

    fn y0(&mut self, params: &[f64]) -> Result<HostArray, DiffsolRtError>;

    fn check(&self, linear_solver: LinearSolverType) -> Result<(), DiffsolRtError>;
    fn set_rtol(&mut self, rtol: f64);
    fn rtol(&self) -> f64;
    fn set_atol(&mut self, atol: f64);
    fn atol(&self) -> f64;
    fn set_t0(&mut self, t0: f64);
    fn t0(&self) -> f64;
    fn set_h0(&mut self, h0: f64);
    fn h0(&self) -> f64;
    fn set_integrate_out(&mut self, integrate_out: bool);
    fn integrate_out(&self) -> bool;
    fn set_sens_rtol(&mut self, sens_rtol: Option<f64>);
    fn sens_rtol(&self) -> Option<f64>;
    fn set_sens_atol(&mut self, sens_atol: Option<f64>);
    fn sens_atol(&self) -> Option<f64>;
    fn set_out_rtol(&mut self, out_rtol: Option<f64>);
    fn out_rtol(&self) -> Option<f64>;
    fn set_out_atol(&mut self, out_atol: Option<f64>);
    fn out_atol(&self) -> Option<f64>;
    fn set_param_rtol(&mut self, param_rtol: Option<f64>);
    fn param_rtol(&self) -> Option<f64>;
    fn set_param_atol(&mut self, param_atol: Option<f64>);
    fn param_atol(&self) -> Option<f64>;
    fn serialized_diffsl(&self) -> Result<Vec<u8>, DiffsolRtError>;

    // New API: solution object support
    fn solve(
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

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult;

    fn solve_continuous_adjoint(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
    ) -> Result<(HostArray, HostArray), DiffsolRtError>;

    fn solve_adjoint_fwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> Result<(Box<dyn Solution>, AdjointCheckpointWrapper), DiffsolRtError>;

    fn solve_adjoint_bkwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        checkpoint: &AdjointCheckpointWrapper,
        t_eval: &[f64],
        dgdu_eval: HostArray,
    ) -> Result<HostArray, DiffsolRtError>;

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
        max_nonlinear_solver_failures: usize,
        min_timestep: f64,
        update_jacobian_after_steps: usize,
        update_rhs_jacobian_after_steps: usize,
        threshold_to_update_jacobian: f64,
        threshold_to_update_rhs_jacobian: f64,
        derivative_solve_tolerance: f64,
        derivative_solve_max_iter: usize
    }
}
pub(crate) struct GenericSolve<M, Eqn>
where
    M: Matrix<T: Scalar>,
    M::V: Vector,
    Eqn: OdeEquations<M = M, T = M::T, V = M::V, C = M::C>,
{
    pub(crate) problem: OdeSolverProblem<Eqn>,
}

impl<M, Eqn> GenericSolve<M, Eqn>
where
    M: Matrix<T: Scalar>,
    M::V: Vector + DefaultDenseMatrix,
    Eqn: OdeEquations<M = M, T = M::T, V = M::V, C = M::C> + SolveEqn,
{
    pub(crate) fn from_eqn(eqn: Eqn) -> Result<Self, DiffsolRtError> {
        let default_p = vec![0.0; eqn.nparams()];
        let problem = OdeBuilder::<M>::new().p(default_p).build_from_eqn(eqn)?;
        Ok(GenericSolve { problem })
    }

    /// Finish a solve with `result`.
    ///
    /// If the solve succeeded, discard the errors that the equations recorded. The solver
    /// recovered from them, for example by rejecting a step.
    ///
    /// If the solve failed, return the error of the solver. If the equations recorded an error
    /// during the solve, add it to the message. It is often the cause of the failure, but it can
    /// also be an earlier failure that the solver recovered from.
    fn finish<R, E: Into<DiffsolRtError>>(
        &self,
        result: Result<R, E>,
    ) -> Result<R, DiffsolRtError> {
        let recorded = self.problem.eqn.take_error();
        result.map_err(|err| {
            let err = err.into();
            match recorded {
                Some(recorded) => DiffsolError::Other(format!(
                    "{err} (a callback failed during the solve: {recorded})"
                ))
                .into(),
                None => err,
            }
        })
    }

    /// Return an error if `values` (the argument `name`) does not have one value per state.
    fn check_state_len(&self, name: &str, values: &[f64]) -> Result<(), DiffsolRtError> {
        let n = self.problem.eqn.nstates();
        if values.len() == n {
            Ok(())
        } else {
            Err(DiffsolError::Other(format!(
                "Expecting {n} values in {name} but got {}",
                values.len()
            ))
            .into())
        }
    }

    /// Return the error recorded by the equations during the last evaluation, if any.
    fn take_eqn_error(&self) -> Result<(), DiffsolRtError> {
        match self.problem.eqn.take_error() {
            Some(message) => Err(DiffsolError::Other(message).into()),
            None => Ok(()),
        }
    }

    /// Check, before a solve, that `linear_solver` is valid and that the equations provide
    /// everything `requirement` needs.
    fn precheck(
        &self,
        linear_solver: LinearSolverType,
        requirement: EqnRequirement,
    ) -> Result<(), DiffsolRtError>
    where
        Self: Solve,
    {
        self.check(linear_solver)?;
        self.problem.eqn.check_requirement(requirement)
    }

    pub(crate) fn setup_problem(&mut self, params: &[f64]) -> Result<(), DiffsolRtError> {
        // discard any error left over from an earlier evaluation
        let _ = self.problem.eqn.take_error();
        let params: Vec<M::T> = params.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let params = M::V::from_slice(&params, M::C::default());

        // Attempt to set problem from params and config
        let nparams = self.problem.eqn.nparams();
        if params.len() == nparams {
            self.problem.eqn.set_params(&params);
            // Reset model index to 0 for hybrid solves
            self.problem.eqn.set_model_index(0);
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

impl<M, Eqn> Solve for GenericSolve<M, Eqn>
where
    M: Matrix<T: Scalar + ToPrimitive>
        + DefaultSolver
        + LuValidator<M>
        + KluValidator<M>
        + MatrixKind,
    Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>
        + OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>
        + SolveEqn
        + Send
        + 'static,
    for<'b> <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: ToHostArray<M::T> + Clone,
    for<'b> <M::V as VectorCommon>::Inner: ToHostArray<M::T> + Clone,
    M::V: DefaultDenseMatrix + Send + Sync + 'static,
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

    fn check(&self, linear_solver: LinearSolverType) -> Result<(), DiffsolRtError> {
        validate_linear_solver::<M>(linear_solver)
    }

    fn set_atol(&mut self, atol: f64) {
        self.problem.atol.fill(M::T::from_f64(atol).unwrap());
    }

    fn atol(&self) -> f64 {
        self.problem.atol.get_index(0).to_f64().unwrap()
    }

    fn serialized_diffsl(&self) -> Result<Vec<u8>, DiffsolRtError> {
        self.problem.eqn.serialized()
    }

    fn set_rtol(&mut self, rtol: f64) {
        self.problem.rtol = M::T::from_f64(rtol).unwrap();
    }

    fn rtol(&self) -> f64 {
        self.problem.rtol.to_f64().unwrap()
    }

    fn set_t0(&mut self, t0: f64) {
        self.problem.t0 = M::T::from_f64(t0).unwrap();
    }

    fn t0(&self) -> f64 {
        self.problem.t0.to_f64().unwrap()
    }

    fn set_h0(&mut self, h0: f64) {
        self.problem.h0 = M::T::from_f64(h0).unwrap();
    }

    fn h0(&self) -> f64 {
        self.problem.h0.to_f64().unwrap()
    }

    fn set_integrate_out(&mut self, integrate_out: bool) {
        self.problem.integrate_out = integrate_out;
    }

    fn integrate_out(&self) -> bool {
        self.problem.integrate_out
    }

    fn set_sens_rtol(&mut self, sens_rtol: Option<f64>) {
        self.problem.sens_rtol = sens_rtol.map(|value| M::T::from_f64(value).unwrap());
    }

    fn sens_rtol(&self) -> Option<f64> {
        self.problem.sens_rtol.map(|value| value.to_f64().unwrap())
    }

    fn set_sens_atol(&mut self, sens_atol: Option<f64>) {
        // one batch lane per (problem batch, parameter), as the augmented state is laid out
        self.problem.sens_atol = sens_atol.map(|value| {
            let nparams = self.problem.eqn.nparams();
            let ctx = self.problem.context();
            let aug_ctx = ctx
                .clone_with_nbatch(ctx.nbatch() * nparams)
                .expect("failed to create the augmented context");
            M::V::from_element(
                self.problem.eqn.nstates(),
                M::T::from_f64(value).unwrap(),
                aug_ctx,
            )
        });
    }

    fn sens_atol(&self) -> Option<f64> {
        // every lane holds the same value unless parameter scales were set, which this API
        // cannot express, so reporting the first is exact for anything it can have set
        self.problem.sens_atol.as_ref().and_then(|value| {
            (value.len() > 0).then(|| value.get_batch(0).get_index(0).to_f64().unwrap())
        })
    }

    fn set_out_rtol(&mut self, out_rtol: Option<f64>) {
        self.problem.out_rtol = out_rtol.map(|value| M::T::from_f64(value).unwrap());
    }

    fn out_rtol(&self) -> Option<f64> {
        self.problem.out_rtol.map(|value| value.to_f64().unwrap())
    }

    fn set_out_atol(&mut self, out_atol: Option<f64>) {
        self.problem.out_atol = out_atol.map(|value| {
            let len = if self.problem.eqn.out().is_some() {
                self.problem.eqn.nout()
            } else {
                self.problem.eqn.nstates()
            };
            M::V::from_element(len, M::T::from_f64(value).unwrap(), M::C::default())
        });
    }

    fn out_atol(&self) -> Option<f64> {
        self.problem
            .out_atol
            .as_ref()
            .and_then(|value| (value.len() > 0).then(|| value.get_index(0).to_f64().unwrap()))
    }

    fn set_param_rtol(&mut self, param_rtol: Option<f64>) {
        self.problem.param_rtol = param_rtol.map(|value| M::T::from_f64(value).unwrap());
    }

    fn param_rtol(&self) -> Option<f64> {
        self.problem.param_rtol.map(|value| value.to_f64().unwrap())
    }

    fn set_param_atol(&mut self, param_atol: Option<f64>) {
        self.problem.param_atol = param_atol.map(|value| {
            M::V::from_element(
                self.problem.eqn.nparams(),
                M::T::from_f64(value).unwrap(),
                M::C::default(),
            )
        });
    }

    fn param_atol(&self) -> Option<f64> {
        self.problem
            .param_atol
            .as_ref()
            .and_then(|value| (value.len() > 0).then(|| value.get_index(0).to_f64().unwrap()))
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
        max_nonlinear_solver_failures: usize,
        min_timestep: f64,
        update_jacobian_after_steps: usize,
        update_rhs_jacobian_after_steps: usize,
        threshold_to_update_jacobian: f64,
        threshold_to_update_rhs_jacobian: f64,
        derivative_solve_tolerance: f64,
        derivative_solve_max_iter: usize
    }

    fn y0(&mut self, params: &[f64]) -> Result<HostArray, DiffsolRtError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let mut y0 = M::V::zeros(n, M::C::default());
        let t0 = self.problem.t0;
        self.problem.eqn.init().call_inplace(t0, &mut y0);
        self.take_eqn_error()?;
        Ok(y0.clone_as_vec().to_host_array())
    }

    fn rhs(&mut self, params: &[f64], t: f64, y: &[f64]) -> Result<HostArray, DiffsolRtError> {
        self.check_state_len("y", y)?;
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
        self.take_eqn_error()?;
        Ok(dydt.clone_as_vec().to_host_array())
    }

    fn rhs_jac_mul(
        &mut self,

        params: &[f64],
        t: f64,
        y: &[f64],
        v: &[f64],
    ) -> Result<HostArray, DiffsolRtError> {
        self.check_state_len("y", y)?;
        self.check_state_len("v", v)?;
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
        self.take_eqn_error()?;
        Ok(dydt.clone_as_vec().to_host_array())
    }

    fn solve(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
    ) -> SolveResult {
        self.precheck(linear_solver, EqnRequirement::Solve(method))?;
        self.setup_problem(params)?;
        let final_time = M::T::from_f64(final_time).unwrap();
        let soln = match linear_solver {
            LinearSolverType::Default => {
                method.solve::<M, Eqn, <M as DefaultSolver>::LS>(&mut self.problem, final_time)
            }
            LinearSolverType::Lu => {
                method.solve::<M, Eqn, <M as LuValidator<M>>::LS>(&mut self.problem, final_time)
            }
            LinearSolverType::Klu => {
                method.solve::<M, Eqn, <M as KluValidator<M>>::LS>(&mut self.problem, final_time)
            }
        };
        Ok(Box::new(self.finish(soln)?))
    }

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult {
        self.precheck(linear_solver, EqnRequirement::Solve(method))?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let soln =
            match linear_solver {
                LinearSolverType::Default => method
                    .solve_dense::<M, Eqn, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval),
                LinearSolverType::Lu => method
                    .solve_dense::<M, Eqn, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
                LinearSolverType::Klu => method
                    .solve_dense::<M, Eqn, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
            };
        Ok(Box::new(self.finish(soln)?))
    }

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> SolveResult {
        self.precheck(linear_solver, EqnRequirement::FwdSens)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let soln = match linear_solver {
            LinearSolverType::Default => method
                .solve_fwd_sens::<M, Eqn, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Lu => method
                .solve_fwd_sens::<M, Eqn, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Klu => method
                .solve_fwd_sens::<M, Eqn, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
        };
        Ok(Box::new(self.finish(soln)?))
    }

    fn solve_continuous_adjoint(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: f64,
    ) -> Result<(HostArray, HostArray), DiffsolRtError> {
        self.precheck(linear_solver, EqnRequirement::Adjoint(method))?;
        self.setup_problem(params)?;

        let final_time = M::T::from_f64(final_time).unwrap();
        let integrate_out = self.problem.integrate_out;
        self.problem.integrate_out = true;
        let result = match linear_solver {
            LinearSolverType::Default => method
                .solve_continuous_adjoint::<M, Eqn, <M as DefaultSolver>::LS>(
                    &mut self.problem,
                    final_time,
                ),
            LinearSolverType::Lu => method
                .solve_continuous_adjoint::<M, Eqn, <M as LuValidator<M>>::LS>(
                    &mut self.problem,
                    final_time,
                ),
            LinearSolverType::Klu => method
                .solve_continuous_adjoint::<M, Eqn, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    final_time,
                ),
        };
        self.problem.integrate_out = integrate_out;
        let (integral, gradient) = self.finish(result)?;
        Ok((
            integral.clone_as_vec().to_host_array(),
            gradient.to_host_array(),
        ))
    }

    fn solve_adjoint_fwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> Result<(Box<dyn Solution>, AdjointCheckpointWrapper), DiffsolRtError> {
        self.precheck(linear_solver, EqnRequirement::Adjoint(method))?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let result = match linear_solver {
            LinearSolverType::Default => method
                .solve_adjoint_fwd::<M, Eqn, <M as DefaultSolver>::LS>(
                    &mut self.problem,
                    &t_eval,
                    params,
                    linear_solver,
                ),
            LinearSolverType::Lu => method.solve_adjoint_fwd::<M, Eqn, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                &t_eval,
                params,
                linear_solver,
            ),
            LinearSolverType::Klu => method
                .solve_adjoint_fwd::<M, Eqn, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    &t_eval,
                    params,
                    linear_solver,
                ),
        };
        let (soln, checkpoint) = self.finish(result)?;
        Ok((Box::new(soln), AdjointCheckpointWrapper::new(checkpoint)))
    }

    fn solve_adjoint_bkwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        checkpoint: &AdjointCheckpointWrapper,
        t_eval: &[f64],
        dgdu_eval: HostArray,
    ) -> Result<HostArray, DiffsolRtError> {
        self.precheck(linear_solver, EqnRequirement::Adjoint(method))?;
        let checkpoint = checkpoint.guard()?;
        self.setup_problem(checkpoint.params())?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let dgdu_eval = host_array_to_dense_matrix::<M>(dgdu_eval)?;
        if dgdu_eval.nrows() != self.problem.eqn.nout() {
            return Err(DiffsolError::Other(format!(
                "Expected dgdu_eval to have {} rows, got {}",
                self.problem.eqn.nout(),
                dgdu_eval.nrows()
            ))
            .into());
        }
        if dgdu_eval.ncols() != t_eval.len() {
            return Err(DiffsolError::Other(format!(
                "Expected dgdu_eval to have {} columns, got {}",
                t_eval.len(),
                dgdu_eval.ncols()
            ))
            .into());
        }

        let result = match linear_solver {
            LinearSolverType::Default => method
                .solve_adjoint_bkwd::<M, Eqn, <M as DefaultSolver>::LS>(
                    &mut self.problem,
                    checkpoint.as_ref(),
                    &dgdu_eval,
                    &t_eval,
                ),
            LinearSolverType::Lu => method.solve_adjoint_bkwd::<M, Eqn, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                checkpoint.as_ref(),
                &dgdu_eval,
                &t_eval,
            ),
            LinearSolverType::Klu => method
                .solve_adjoint_bkwd::<M, Eqn, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    checkpoint.as_ref(),
                    &dgdu_eval,
                    &t_eval,
                ),
        };
        let gradient = self.finish(result)?;
        Ok(gradient.to_host_array())
    }
}

fn host_array_to_dense_matrix<M>(
    array: HostArray,
) -> Result<<M::V as DefaultDenseMatrix>::M, DiffsolRtError>
where
    M: Matrix<T: Scalar>,
    M::V: DefaultDenseMatrix,
{
    let view = array.as_array::<M::T>()?;
    let mut values = Vec::with_capacity(view.nrows() * view.ncols());
    for col in 0..view.ncols() {
        for row in 0..view.nrows() {
            values.push(view[(row, col)]);
        }
    }
    Ok(<M::V as DefaultDenseMatrix>::M::from_vec(
        view.nrows(),
        view.ncols(),
        values,
        M::C::default(),
    ))
}
