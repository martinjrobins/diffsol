// Delegate solver types selected at runtime in Host to concrete solver types
// in Rust.

use std::sync::Arc;

use crate::adjoint_checkpoint::AdjointCheckpointWrapper;
use crate::error::DiffsolRtError;
use crate::scalar_type::Scalar;
use crate::tensor::{column_major_values, FromTensor, Tensor, TensorLayout, TensorView, ToTensor};
use crate::{
    generate_ic_option_accessors, generate_ode_option_accessors, generate_option_accessors,
    option_value_from_store, option_value_to_store,
};
use crate::{generate_trait_ic_option_accessors, generate_trait_ode_option_accessors};
use diffsol::OdeBuilder;
use diffsol::{
    error::DiffsolError,
    matrix::{Matrix, MatrixRef},
    ConstantOp, Context, DefaultDenseMatrix, DefaultSolver, NonLinearOp, NonLinearOpJacobian,
    OdeEquations, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverProblem, Vector,
    VectorRef, VectorView,
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
pub(crate) type SolveResult = Result<Arc<dyn Solution>, DiffsolRtError>;

pub(crate) trait Solve: Send {
    fn matrix_type(&self) -> MatrixType;
    fn nstates(&self) -> usize;
    fn nparams(&self) -> usize;
    fn nout(&self) -> usize;
    fn has_stop(&self) -> bool;
    /// The number of lanes (independent copies of the system) that the solver integrates
    /// together.
    fn nbatch(&self) -> usize;

    fn rhs(
        &mut self,
        params: TensorView<'_>,
        t: f64,
        y: TensorView<'_>,
    ) -> Result<Tensor, DiffsolRtError>;

    fn rhs_jac_mul(
        &mut self,
        params: TensorView<'_>,
        t: f64,
        y: TensorView<'_>,
        v: TensorView<'_>,
    ) -> Result<Tensor, DiffsolRtError>;

    fn y0(&mut self, params: TensorView<'_>) -> Result<Tensor, DiffsolRtError>;

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
        params: TensorView<'_>,
        final_time: f64,
    ) -> SolveResult;

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        t_eval: TensorView<'_>,
    ) -> SolveResult;

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        t_eval: TensorView<'_>,
    ) -> SolveResult;

    fn solve_continuous_adjoint(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        final_time: f64,
    ) -> Result<(Tensor, Tensor), DiffsolRtError>;

    fn solve_adjoint_fwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        t_eval: TensorView<'_>,
    ) -> Result<(Arc<dyn Solution>, AdjointCheckpointWrapper), DiffsolRtError>;

    fn solve_adjoint_bkwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        checkpoint: &AdjointCheckpointWrapper,
        t_eval: TensorView<'_>,
        dgdu_eval: TensorView<'_>,
    ) -> Result<Tensor, DiffsolRtError>;

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

/// Read the times `t_eval` as `T` values. All the lanes use the same times, so `t_eval` is a 1D
/// `[nt]` array, or a 2D `[nt, 1]` array with one lane.
fn read_times<T: Scalar>(times: TensorView<'_>) -> Result<Vec<T>, DiffsolRtError> {
    match times.shape() {
        [_] | [_, 1] => column_major_values(times).map_err(|e| e.named("t_eval")),
        shape => {
            let message = format!("Expected an [nt] or an [nt, 1] array, got the shape {shape:?}");
            Err(DiffsolRtError::from(DiffsolError::Other(message)).named("t_eval"))
        }
    }
}

/// The first value of lane 0 of a tolerance vector, or `None` if it has no values.
///
/// This API sets all values of a tolerance vector to one value, in all lanes, so the first
/// value is the value that was set.
fn first_lane_value<V: Vector<T: Scalar>>(v: &V) -> Option<f64> {
    (v.len() > 0).then(|| v.get_batch(0).get_index(0).to_f64().unwrap())
}

impl<M, Eqn> GenericSolve<M, Eqn>
where
    M: Matrix<T: Scalar>,
    M::V: Vector + DefaultDenseMatrix,
    Eqn: OdeEquations<M = M, T = M::T, V = M::V, C = M::C> + SolveEqn,
{
    pub(crate) fn from_eqn(eqn: Eqn) -> Result<Self, DiffsolRtError> {
        // the builder needs one block of parameters for each lane
        let ctx = eqn.context().clone();
        let default_p = vec![0.0; eqn.nparams() * ctx.nbatch()];
        let problem = OdeBuilder::<M>::new()
            .context(ctx)
            .p(default_p)
            .build_from_eqn(eqn)?;
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

    /// Convert `tensor` (the argument `name`) to a diffsol value with the lanes of the problem.
    /// Each lane must have the shape `lane_shape`. The errors start with `name`.
    fn read_input<T: FromTensor<Context = M::C>>(
        &self,
        tensor: TensorView<'_>,
        lane_shape: &[usize],
        name: &str,
    ) -> Result<T, DiffsolRtError> {
        // the conversion checks the number of axes and the number of lanes
        let value = T::from_tensor(tensor, self.problem.context()).map_err(|e| e.named(name))?;
        // the last axis is the lane axis
        let got = &tensor.shape()[..tensor.shape().len() - 1];
        if got != lane_shape {
            let message = match (lane_shape, got) {
                ([n], [m]) => format!("Expected {n} values in each lane, got {m}"),
                ([r, c], [gr, gc]) => format!(
                    "Expected {r} rows and {c} columns in each lane, got {gr} rows and {gc} columns"
                ),
                _ => format!("Expected the shape {lane_shape:?} in each lane, got {got:?}"),
            };
            return Err(DiffsolRtError::from(DiffsolError::Other(message)).named(name));
        }
        Ok(value)
    }

    /// Convert an adjoint gradient to a Tensor with the shape `[nparams, nchannels, nbatch]`.
    ///
    /// The gradient has one lane for each (batch lane, output channel) pair, with lane
    /// `b * nchannels + j` for batch lane `b` and channel `j`.
    fn gradient_to_tensor(&self, gradient: M::V) -> Tensor
    where
        M::V: ToTensor,
    {
        // `[nparams, nbatch * nchannels]`, with the lane axis split into `[nchannels, nbatch]`
        let tensor = gradient.to_tensor();
        let nbatch = self.problem.context().nbatch();
        let nchannels = tensor.shape()[1] / nbatch;
        let shape = [tensor.shape()[0], nchannels, nbatch];
        let strides = [
            tensor.strides()[0],
            tensor.strides()[1],
            nchannels * tensor.strides()[1],
        ];
        tensor.with_layout(&shape, &strides)
    }

    /// Return the error recorded by the equations during the last evaluation, if any.
    fn take_eqn_error(&self) -> Result<(), DiffsolRtError> {
        match self.problem.eqn.take_error() {
            Some(message) => Err(DiffsolError::Other(message).into()),
            None => Ok(()),
        }
    }

    pub(crate) fn setup_problem(&mut self, params: TensorView<'_>) -> Result<(), DiffsolRtError>
    where
        M::V: FromTensor<Context = M::C>,
    {
        self.reset_eval_state();
        let nparams = self.problem.eqn.nparams();
        let params = self.read_input(params, &[nparams], "params")?;
        // a single lane of parameters broadcasts to all the lanes of the problem
        self.problem.eqn.set_params(&params);
        Ok(())
    }

    /// Discard any error left over from an earlier evaluation, and reset the model index to 0
    /// for hybrid solves.
    fn reset_eval_state(&mut self) {
        let _ = self.problem.eqn.take_error();
        self.problem.eqn.set_model_index(0);
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
    <M::V as DefaultDenseMatrix>::M: TensorLayout + FromTensor<Context = M::C>,
    M::V: TensorLayout + FromTensor<Context = M::C>,
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

    fn nbatch(&self) -> usize {
        self.problem.context().nbatch()
    }

    fn check(&self, linear_solver: LinearSolverType) -> Result<(), DiffsolRtError> {
        validate_linear_solver::<M>(linear_solver)
    }

    fn set_atol(&mut self, atol: f64) {
        self.problem.atol.fill(M::T::from_f64(atol).unwrap());
    }

    fn atol(&self) -> f64 {
        first_lane_value(&self.problem.atol).expect("atol has no values")
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
        self.problem.sens_atol.as_ref().and_then(first_lane_value)
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
            M::V::from_element(
                len,
                M::T::from_f64(value).unwrap(),
                self.problem.context().clone(),
            )
        });
    }

    fn out_atol(&self) -> Option<f64> {
        self.problem.out_atol.as_ref().and_then(first_lane_value)
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
                self.problem.context().clone(),
            )
        });
    }

    fn param_atol(&self) -> Option<f64> {
        self.problem.param_atol.as_ref().and_then(first_lane_value)
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

    fn y0(&mut self, params: TensorView<'_>) -> Result<Tensor, DiffsolRtError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let mut y0 = M::V::zeros(n, self.problem.context().clone());
        let t0 = self.problem.t0;
        self.problem.eqn.init().call_inplace(t0, &mut y0);
        self.take_eqn_error()?;
        Ok(y0.to_tensor())
    }

    fn rhs(
        &mut self,
        params: TensorView<'_>,
        t: f64,
        y: TensorView<'_>,
    ) -> Result<Tensor, DiffsolRtError> {
        let n = self.problem.eqn.nstates();
        let y = self.read_input(y, &[n], "y")?;
        self.setup_problem(params)?;
        let mut dydt = M::V::zeros(n, self.problem.context().clone());
        self.problem
            .eqn
            .rhs()
            .call_inplace(&y, M::T::from_f64(t).unwrap(), &mut dydt);
        self.take_eqn_error()?;
        Ok(dydt.to_tensor())
    }

    fn rhs_jac_mul(
        &mut self,
        params: TensorView<'_>,
        t: f64,
        y: TensorView<'_>,
        v: TensorView<'_>,
    ) -> Result<Tensor, DiffsolRtError> {
        let n = self.problem.eqn.nstates();
        let y = self.read_input(y, &[n], "y")?;
        let v = self.read_input(v, &[n], "v")?;
        self.setup_problem(params)?;
        let mut dydt = M::V::zeros(n, self.problem.context().clone());
        self.problem
            .eqn
            .rhs()
            .jac_mul_inplace(&y, M::T::from_f64(t).unwrap(), &v, &mut dydt);
        self.take_eqn_error()?;
        Ok(dydt.to_tensor())
    }

    fn solve(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        final_time: f64,
    ) -> SolveResult {
        self.problem
            .eqn
            .check_requirement(EqnRequirement::Solve(method))?;
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
        Ok(Arc::new(self.finish(soln)?))
    }

    fn solve_dense(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        t_eval: TensorView<'_>,
    ) -> SolveResult {
        self.problem
            .eqn
            .check_requirement(EqnRequirement::Solve(method))?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = read_times(t_eval)?;
        let soln =
            match linear_solver {
                LinearSolverType::Default => method
                    .solve_dense::<M, Eqn, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval),
                LinearSolverType::Lu => method
                    .solve_dense::<M, Eqn, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
                LinearSolverType::Klu => method
                    .solve_dense::<M, Eqn, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
            };
        Ok(Arc::new(self.finish(soln)?))
    }

    fn solve_fwd_sens(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        t_eval: TensorView<'_>,
    ) -> SolveResult {
        self.problem
            .eqn
            .check_requirement(EqnRequirement::FwdSens)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = read_times(t_eval)?;
        let soln = match linear_solver {
            LinearSolverType::Default => method
                .solve_fwd_sens::<M, Eqn, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Lu => method
                .solve_fwd_sens::<M, Eqn, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
            LinearSolverType::Klu => method
                .solve_fwd_sens::<M, Eqn, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
        };
        Ok(Arc::new(self.finish(soln)?))
    }

    fn solve_continuous_adjoint(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        final_time: f64,
    ) -> Result<(Tensor, Tensor), DiffsolRtError> {
        self.problem
            .eqn
            .check_requirement(EqnRequirement::Adjoint(method))?;
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
        Ok((integral.to_tensor(), self.gradient_to_tensor(gradient)))
    }

    fn solve_adjoint_fwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        params: TensorView<'_>,
        t_eval: TensorView<'_>,
    ) -> Result<(Arc<dyn Solution>, AdjointCheckpointWrapper), DiffsolRtError> {
        self.problem
            .eqn
            .check_requirement(EqnRequirement::Adjoint(method))?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = read_times(t_eval)?;
        let result = match linear_solver {
            LinearSolverType::Default => method
                .solve_adjoint_fwd::<M, Eqn, <M as DefaultSolver>::LS>(
                    &mut self.problem,
                    &t_eval,
                    linear_solver,
                ),
            LinearSolverType::Lu => method.solve_adjoint_fwd::<M, Eqn, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                &t_eval,
                linear_solver,
            ),
            LinearSolverType::Klu => method
                .solve_adjoint_fwd::<M, Eqn, <M as KluValidator<M>>::LS>(
                    &mut self.problem,
                    &t_eval,
                    linear_solver,
                ),
        };
        let (soln, checkpoint) = self.finish(result)?;
        Ok((Arc::new(soln), AdjointCheckpointWrapper::new(checkpoint)))
    }

    fn solve_adjoint_bkwd(
        &mut self,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
        checkpoint: &AdjointCheckpointWrapper,
        t_eval: TensorView<'_>,
        dgdu_eval: TensorView<'_>,
    ) -> Result<Tensor, DiffsolRtError> {
        self.problem
            .eqn
            .check_requirement(EqnRequirement::Adjoint(method))?;
        let checkpoint = checkpoint.guard()?;
        // the backward pass sets the parameters that the checkpoint holds
        self.reset_eval_state();

        let t_eval: Vec<M::T> = read_times(t_eval)?;
        let lane_shape = [self.problem.eqn.nout(), t_eval.len()];
        let dgdu_eval: <M::V as DefaultDenseMatrix>::M =
            self.read_input(dgdu_eval, &lane_shape, "dgdu_eval")?;

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
        Ok(self.gradient_to_tensor(gradient))
    }
}
