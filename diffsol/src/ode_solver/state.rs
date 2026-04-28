use log::debug;
use num_traits::FromPrimitive;
use num_traits::{One, Pow, Signed, Zero};

use crate::error::NonLinearSolverError;
use crate::Scalar;
use crate::{
    error::{DiffsolError, OdeSolverError},
    nonlinear_solver::{convergence::Convergence, NonLinearSolver},
    ode_solver_error, scale, AdjointEquations, AugmentedOdeEquations,
    AugmentedOdeEquationsImplicit, ConstantOp, InitOp, LinearOp, LinearSolver, Matrix,
    NewtonNonlinearSolver, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
    NonLinearOpSensAdjoint, NonLinearOpTimePartial, OdeEquations, OdeEquationsAdjoint,
    OdeEquationsImplicit, OdeEquationsImplicitSens, OdeSolverMethod, OdeSolverProblem, Op,
    SensEquations, Vector, VectorIndex,
};
use crate::{non_linear_solver_error, BacktrackingLineSearch, NoLineSearch};

/// A state holding those variables that are common to all ODE solver states,
/// can be used to create a new state for a specific solver.
pub struct StateCommon<V: Vector> {
    pub y: V,
    pub dy: V,
    pub g: V,
    pub dg: V,
    pub s: Vec<V>,
    pub ds: Vec<V>,
    pub sg: Vec<V>,
    pub dsg: Vec<V>,
    pub t: V::T,
    pub h: V::T,
}

/// A reference to the state of the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
/// - the sensitivity vectors of the output function `sg`
/// - the derivative of the sensitivity vectors of the output function wrt time `dsg`
pub struct StateRef<'a, V: Vector> {
    pub y: &'a V,
    pub dy: &'a V,
    pub g: &'a V,
    pub dg: &'a V,
    pub s: &'a [V],
    pub ds: &'a [V],
    pub sg: &'a [V],
    pub dsg: &'a [V],
    pub t: V::T,
    pub h: V::T,
}

/// A mutable reference to the state of the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
/// - the sensitivity vectors of the output function `sg`
/// - the derivative of the sensitivity vectors of the output function wrt time `dsg`
pub struct StateRefMut<'a, V: Vector> {
    pub y: &'a mut V,
    pub dy: &'a mut V,
    pub g: &'a mut V,
    pub dg: &'a mut V,
    pub s: &'a mut [V],
    pub ds: &'a mut [V],
    pub sg: &'a mut [V],
    pub dsg: &'a mut [V],
    pub t: &'a mut V::T,
    pub h: &'a mut V::T,
}

fn refresh_augmented_state_ref_mut<V, Eqn, AugmentedEqn>(
    state: &mut StateRefMut<'_, V>,
    augmented_eqn: &mut AugmentedEqn,
) -> Result<(), DiffsolError>
where
    V: Vector,
    Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    augmented_eqn.update_rhs_out_state(state.y, state.dy, *state.t);
    let naug = augmented_eqn.max_index();
    for i in 0..naug {
        augmented_eqn.set_index(i);
        augmented_eqn
            .rhs()
            .call_inplace(&state.s[i], *state.t, &mut state.ds[i]);
        if let Some(out) = augmented_eqn.out() {
            out.call_inplace(&state.s[i], *state.t, &mut state.dsg[i]);
        }
    }
    Ok(())
}

impl<V: Vector> StateRefMut<'_, V> {
    /// Apply a non-linear operator to the current state in-place, replacing `state.y` with
    /// `op(state.y, state.t)` and recomputing `state.dy = rhs(state.y, state.t)`.
    ///
    /// Note: mass matrix equations are not supported for this operation.
    pub fn state_mut_op<Rhs, O>(
        &mut self,
        rhs: &Rhs,
        has_mass: bool,
        op: &O,
    ) -> Result<(), DiffsolError>
    where
        Rhs: NonLinearOp<T = V::T, V = V, C = V::C>,
        O: NonLinearOp<T = V::T, V = V, M = Rhs::M, C = V::C>,
    {
        if has_mass {
            return Err(ode_solver_error!(MassMatrixNotSupported));
        }

        let nstates = rhs.nstates();
        let mut y_out = V::zeros(nstates, rhs.context().clone());
        op.call_inplace(self.y, *self.t, &mut y_out);

        self.y.copy_from(&y_out);
        rhs.call_inplace(self.y, *self.t, &mut y_out);
        self.dy.copy_from(&y_out);
        Ok(())
    }

    /// Apply a reset operator to the current state and propagate sensitivities through a
    /// time-dependent root-triggered event correction.
    ///
    /// If the pre-event state is `x^-` and the reset map is `g(x, t, p)`, this method updates
    /// the state to `x^+ = g(x^-, t, p)`, then recomputes the post-event vector field
    /// `f^+ = rhs(x^+, t, p)`.
    ///
    /// Note: mass matrix equations are not supported for this operation.
    pub fn state_mut_op_with_sens_and_reset<Rhs, G, R>(
        &mut self,
        rhs: &Rhs,
        has_mass: bool,
        reset_op: &G,
        root_op: &R,
        root_idx: usize,
    ) -> Result<(), DiffsolError>
    where
        Rhs: NonLinearOp<T = V::T, V = V, C = V::C>,
        G: NonLinearOpJacobian<T = V::T, V = V, M = Rhs::M, C = V::C>
            + NonLinearOpSens<T = V::T, V = V, M = Rhs::M, C = V::C>
            + NonLinearOpTimePartial<T = V::T, V = V, M = Rhs::M, C = V::C>,
        R: NonLinearOpJacobian<T = V::T, V = V, M = Rhs::M, C = V::C>
            + NonLinearOpSens<T = V::T, V = V, M = Rhs::M, C = V::C>
            + NonLinearOpTimePartial<T = V::T, V = V, M = Rhs::M, C = V::C>,
    {
        if has_mass {
            return Err(ode_solver_error!(MassMatrixNotSupported));
        }

        let nstates = rhs.nstates();
        let nroots = root_op.nout();
        if root_idx >= nroots {
            return Err(ode_solver_error!(
                Other,
                format!(
                    "root index {root_idx} out of bounds for root function with {nroots} outputs"
                )
            ));
        }

        let ctx = rhs.context().clone();
        let t = *self.t;
        let y_before = self.y.clone();
        let f_minus = self.dy.clone();
        let s_before = self.s.to_vec();
        let nparams = s_before.len();
        let reset_t = reset_op.time_derive(&y_before, t);
        let root_t = root_op.time_derive(&y_before, t);

        let mut y_plus = V::zeros(nstates, ctx.clone());
        reset_op.call_inplace(&y_before, t, &mut y_plus);

        let mut f_plus = V::zeros(nstates, ctx.clone());
        rhs.call_inplace(&y_plus, t, &mut f_plus);

        let mut correction_dir = V::zeros(nstates, ctx.clone());
        reset_op.jac_mul_inplace(&y_before, t, &f_minus, &mut correction_dir);
        correction_dir += &reset_t;
        correction_dir -= &f_plus;

        let mut root_flow = V::zeros(nroots, ctx.clone());
        root_op.jac_mul_inplace(&y_before, t, &f_minus, &mut root_flow);
        let denom = root_flow.get_index(root_idx) + root_t.get_index(root_idx);
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        if denom.abs() <= denom_tol {
            return Err(ode_solver_error!(
                Other,
                "reset sensitivity correction undefined: active root derivative along flow is zero"
            ));
        }

        let mut basis = V::zeros(nparams, ctx.clone());
        let mut reset_jac_s = V::zeros(nstates, ctx.clone());
        let mut reset_sens = V::zeros(nstates, ctx.clone());
        let mut root_jac_s = V::zeros(nroots, ctx.clone());
        let mut root_sens = V::zeros(nroots, ctx);
        let mut s_plus = Vec::with_capacity(nparams);
        for (j, s_j_before) in s_before.iter().enumerate() {
            basis.set_index(j, V::T::one());

            reset_op.jac_mul_inplace(&y_before, t, s_j_before, &mut reset_jac_s);
            reset_op.sens_mul_inplace(&y_before, t, &basis, &mut reset_sens);

            root_op.jac_mul_inplace(&y_before, t, s_j_before, &mut root_jac_s);
            root_op.sens_mul_inplace(&y_before, t, &basis, &mut root_sens);

            let numerator = root_jac_s.get_index(root_idx) + root_sens.get_index(root_idx);
            let tau_p = -numerator / denom;

            let mut s_j_plus = reset_jac_s.clone();
            s_j_plus += &reset_sens;
            s_j_plus.axpy(tau_p, &correction_dir, V::T::one());
            s_plus.push(s_j_plus);

            basis.set_index(j, V::T::zero());
        }

        self.y.copy_from(&y_plus);
        self.dy.copy_from(&f_plus);
        for (dst, src) in self.s.iter_mut().zip(s_plus.iter()) {
            dst.copy_from(src);
        }
        Ok(())
    }

    /// Propagate adjoint variables through a time-dependent root-triggered reset.
    ///
    /// Note: mass matrix equations are not supported for this operation.
    pub fn state_mut_op_with_adjoint_and_reset<'a, Eqn, Method, G, R, State>(
        &mut self,
        adj_eqn: &mut AdjointEquations<'a, Eqn, Method>,
        reset_op: &G,
        root_op: &R,
        root_idx: usize,
        fwd_state_minus: &State,
        fwd_state_plus: &State,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsAdjoint<T = V::T, V = V, C = V::C>,
        Method: OdeSolverMethod<'a, Eqn>,
        State: OdeSolverState<V>,
        G: NonLinearOpJacobian<T = V::T, V = V, M = Eqn::M, C = V::C>
            + NonLinearOpAdjoint<T = V::T, V = V, M = Eqn::M, C = V::C>
            + NonLinearOpSensAdjoint<T = V::T, V = V, M = Eqn::M, C = V::C>
            + NonLinearOpTimePartial<T = V::T, V = V, M = Eqn::M, C = V::C>,
        R: NonLinearOpJacobian<T = V::T, V = V, M = Eqn::M, C = V::C>
            + NonLinearOpAdjoint<T = V::T, V = V, M = Eqn::M, C = V::C>
            + NonLinearOpSensAdjoint<T = V::T, V = V, M = Eqn::M, C = V::C>
            + NonLinearOpTimePartial<T = V::T, V = V, M = Eqn::M, C = V::C>,
    {
        let eqn = adj_eqn.eqn();
        if eqn.mass().is_some() {
            return Err(ode_solver_error!(MassMatrixNotSupported));
        }

        let nroots = root_op.nout();
        if root_idx >= nroots {
            return Err(ode_solver_error!(
                Other,
                format!(
                    "root index {root_idx} out of bounds for root function with {nroots} outputs"
                )
            ));
        }

        let ctx = eqn.context().clone();
        let t_event = fwd_state_minus.as_ref().t;
        let y_minus = fwd_state_minus.as_ref().y;
        let y_plus = fwd_state_plus.as_ref().y;
        let f_minus = fwd_state_minus.as_ref().dy;
        let f_plus = fwd_state_plus.as_ref().dy;
        let nchannels = self.s.len();
        let nstates = y_minus.len();
        let nparams = eqn.rhs().nparams();

        let reset_t = reset_op.time_derive(y_minus, t_event);
        let root_t = root_op.time_derive(y_minus, t_event);

        let mut correction_dir = V::zeros(nstates, ctx.clone());
        reset_op.jac_mul_inplace(y_minus, t_event, f_minus, &mut correction_dir);
        correction_dir += reset_t;
        correction_dir -= f_plus;

        let mut root_flow = V::zeros(nroots, ctx.clone());
        root_op.jac_mul_inplace(y_minus, t_event, f_minus, &mut root_flow);
        let denom = root_flow.get_index(root_idx) + root_t.get_index(root_idx);
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        if denom.abs() <= denom_tol {
            return Err(ode_solver_error!(
                Other,
                "reset adjoint correction undefined: active root derivative along flow is zero"
            ));
        }

        let (l_minus, l_plus) = if adj_eqn.with_out() {
            if let Some(out_op) = eqn.out() {
                (
                    Some(out_op.call(y_minus, t_event)),
                    Some(out_op.call(y_plus, t_event)),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let mut root_basis = V::zeros(nroots, ctx.clone());
        let mut reset_adj = V::zeros(nstates, ctx.clone());
        let mut root_adj = V::zeros(nstates, ctx.clone());
        let mut reset_sens_adj = V::zeros(nparams, ctx.clone());
        let mut root_sens_adj = V::zeros(nparams, ctx.clone());

        for i in 0..nchannels {
            let alpha = {
                let lambda_i = &self.s[i];
                let mut alpha_num = V::T::zero();
                for j in 0..nstates {
                    alpha_num += lambda_i.get_index(j) * correction_dir.get_index(j);
                }
                if let (Some(l_minus), Some(l_plus)) = (&l_minus, &l_plus) {
                    alpha_num += l_minus.get_index(i) - l_plus.get_index(i);
                }
                alpha_num / denom
            };

            {
                let lambda_i = &self.s[i];
                reset_op.jac_transpose_mul_inplace(y_minus, t_event, lambda_i, &mut reset_adj);
                reset_op.sens_transpose_mul_inplace(
                    y_minus,
                    t_event,
                    lambda_i,
                    &mut reset_sens_adj,
                );
            }

            root_basis.set_index(root_idx, alpha);
            root_op.jac_transpose_mul_inplace(y_minus, t_event, &root_basis, &mut root_adj);
            root_op.sens_transpose_mul_inplace(y_minus, t_event, &root_basis, &mut root_sens_adj);
            root_basis.set_index(root_idx, V::T::zero());

            self.s[i].copy_from(&root_adj);
            self.s[i].axpy(-V::T::one(), &reset_adj, V::T::one());
            self.sg[i] -= &reset_sens_adj;
            self.sg[i] += &root_sens_adj;
        }
        refresh_augmented_state_ref_mut::<V, Eqn, _>(self, adj_eqn)
    }

    /// Add the terminal-root adjoint correction for a root-defined final time.
    pub fn state_mut_adjoint_terminal_root<'a, Eqn, Method, State>(
        &mut self,
        adj_eqn: &mut AdjointEquations<'a, Eqn, Method>,
        root_idx: usize,
        forward: &State,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsAdjoint<
            T = V::T,
            V = V,
            C = V::C,
            Root: NonLinearOpJacobian<T = V::T, V = V, M = Eqn::M, C = V::C>
                      + NonLinearOpAdjoint<T = V::T, V = V, M = Eqn::M, C = V::C>
                      + NonLinearOpSensAdjoint<T = V::T, V = V, M = Eqn::M, C = V::C>
                      + NonLinearOpTimePartial<T = V::T, V = V, M = Eqn::M, C = V::C>,
            Out: NonLinearOp<T = V::T, V = V, M = Eqn::M, C = V::C>,
        >,
        Method: OdeSolverMethod<'a, Eqn>,
        State: OdeSolverState<V>,
    {
        let eqn = adj_eqn.eqn();

        if eqn.mass().is_some() {
            return Err(ode_solver_error!(MassMatrixNotSupported));
        }

        if !adj_eqn.with_out() {
            return Ok(());
        }

        let Some(out_op) = eqn.out() else {
            return Ok(());
        };
        let Some(root_op) = eqn.root() else {
            return Ok(());
        };
        let forward = forward.as_ref();

        let nout = out_op.nout();
        if self.s.len() != nout || self.sg.len() != nout || self.dsg.len() != nout {
            return Ok(());
        }

        let nroots = root_op.nout();
        if root_idx >= nroots {
            return Err(ode_solver_error!(
                Other,
                format!(
                    "root index {root_idx} out of bounds for root function with {nroots} outputs"
                )
            ));
        }

        let ctx = eqn.context().clone();
        let out = out_op.call(forward.y, forward.t);
        let root_t = root_op.time_derive(forward.y, forward.t);
        let mut root_flow = V::zeros(nroots, ctx.clone());
        root_op.jac_mul_inplace(forward.y, forward.t, forward.dy, &mut root_flow);
        let denom = root_flow.get_index(root_idx) + root_t.get_index(root_idx);
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        if denom.abs() <= denom_tol {
            return Err(ode_solver_error!(
                Other,
                "terminal root adjoint correction undefined: active root derivative along flow is zero"
            ));
        }

        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let mut root_basis = V::zeros(nroots, ctx.clone());
        let mut lambda_corr = V::zeros(nstates, ctx.clone());
        let mut q_corr = V::zeros(nparams, ctx.clone());
        for i in 0..nout {
            root_basis.set_index(root_idx, out.get_index(i) / denom);
            root_op.jac_transpose_mul_inplace(forward.y, forward.t, &root_basis, &mut lambda_corr);
            root_op.sens_transpose_mul_inplace(forward.y, forward.t, &root_basis, &mut q_corr);
            root_basis.set_index(root_idx, V::T::zero());
            self.s[i] += &lambda_corr;
            self.sg[i] += &q_corr;
        }
        refresh_augmented_state_ref_mut::<V, Eqn, _>(self, adj_eqn)
    }
}

/// State for the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`,
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
///
pub trait OdeSolverState<V: Vector>: Clone + Sized + Send {
    /// Get an immutable reference to the state.
    fn as_ref(&self) -> StateRef<'_, V>;
    /// Get a mutable reference to the state.
    fn as_mut(&mut self) -> StateRefMut<'_, V>;
    /// Convert the state into a common state representation.
    fn into_common(self) -> StateCommon<V>;
    /// Create a new state from a common state representation.
    fn new_from_common(state: StateCommon<V>) -> Self;

    /// Set the ODE problem for the state, allocating any necessary data structures.
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError>;

    /// Set the augmented ODE problem (for sensitivities) for the state.
    fn set_augmented_problem<Eqn: OdeEquations, AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError>;

    /// Check that the state is consistent with the given ODE problem.
    fn check_consistent_with_problem<Eqn: OdeEquations>(
        &self,
        problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        if self.as_ref().y.len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if self.as_ref().dy.len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    /// Check that the sensitivity vectors in the state are consistent with the given ODE problem.
    fn check_sens_consistent_with_problem<
        Eqn: OdeEquations,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    >(
        &self,
        problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError> {
        let state = self.as_ref();
        if state.s.len() != augmented_eqn.max_index() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !state.s.is_empty() && state.s[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if state.ds.len() != augmented_eqn.max_index() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !state.ds.is_empty() && state.ds[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    /// Create a new solver state from an ODE problem.
    /// This function will set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    fn new<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let mut ret = Self::new_without_initialise(ode_problem)?;
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem.
    /// This function will make the state consistent with any algebraic constraints using a default nonlinear solver.
    /// It will also set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    fn new_and_consistent<LS, Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        LS: LinearSolver<Eqn::M>,
    {
        let mut ret = Self::new_without_initialise(ode_problem)?;
        if ode_problem.ic_options.use_linesearch {
            let mut ls = BacktrackingLineSearch::default();
            ls.c = ode_problem.ic_options.armijo_constant;
            ls.max_iter = ode_problem.ic_options.max_linesearch_iterations;
            ls.tau = ode_problem.ic_options.step_reduction_factor;
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), ls);
            ret.set_consistent(ode_problem, &mut root_solver)?;
        } else {
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            ret.set_consistent(ode_problem, &mut root_solver)?;
        }
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem with sensitivity equations.
    /// This will initialize the sensitivity vectors but will not make them consistent with algebraic constraints.
    fn new_with_sensitivities<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens<T = V::T, V = V, C = V::C>,
    {
        let mut augmented_eqn = SensEquations::new(ode_problem);
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;

        // eval the rhs since we're not calling set_consistent_augmented
        let state = ret.as_mut();
        augmented_eqn.update_rhs_out_state(state.y, state.dy, *state.t);
        let naug = augmented_eqn.max_index();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            augmented_eqn
                .rhs()
                .call_inplace(&state.s[i], *state.t, &mut state.ds[i]);
        }
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem with sensitivity equations, making both the main state and sensitivities consistent with algebraic constraints.
    fn new_with_sensitivities_and_consistent<LS, Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens<T = V::T, V = V, C = V::C>,
        LS: LinearSolver<Eqn::M>,
    {
        let mut augmented_eqn = SensEquations::new(ode_problem);
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;
        if ode_problem.ic_options.use_linesearch {
            let mut ls = BacktrackingLineSearch::default();
            ls.c = ode_problem.ic_options.armijo_constant;
            ls.max_iter = ode_problem.ic_options.max_linesearch_iterations;
            ls.tau = ode_problem.ic_options.step_reduction_factor;
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), ls);
            ret.set_consistent(ode_problem, &mut root_solver)?;
        } else {
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            ret.set_consistent(ode_problem, &mut root_solver)?;
        }
        if ode_problem.ic_options.use_linesearch {
            let mut ls = BacktrackingLineSearch::default();
            ls.c = ode_problem.ic_options.armijo_constant;
            ls.max_iter = ode_problem.ic_options.max_linesearch_iterations;
            ls.tau = ode_problem.ic_options.step_reduction_factor;
            let mut root_solver_sens = NewtonNonlinearSolver::new(LS::default(), ls);
            ret.set_consistent_augmented(ode_problem, &mut augmented_eqn, &mut root_solver_sens)?;
        } else {
            let mut root_solver_sens = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            ret.set_consistent_augmented(ode_problem, &mut augmented_eqn, &mut root_solver_sens)?;
        }
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem, without any initialisation apart from setting the initial time state vector y,
    /// the initial time derivative dy and if applicable the sensitivity vectors s.
    /// This is useful if you want to set up the state yourself, or if you want to use a different nonlinear solver to make the state consistent,
    /// or if you want to set the step size yourself or based on the exact order of the solver.
    fn new_without_initialise<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init().call(t);
        let dy = ode_problem.eqn.rhs().call(&y, t);
        let (s, ds) = (vec![], vec![]);
        let (dg, g) = if ode_problem.integrate_out {
            if let Some(out) = ode_problem.eqn.out() {
                (out.call(&y, t), V::zeros(out.nout(), y.context().clone()))
            } else {
                // If no explicit output is defined, default output is identity on state.
                (y.clone(), V::zeros(y.len(), y.context().clone()))
            }
        } else {
            (
                V::zeros(0, y.context().clone()),
                V::zeros(0, y.context().clone()),
            )
        };
        let (sg, dsg) = (vec![], vec![]);
        let state = StateCommon {
            y,
            dy,
            g,
            dg,
            s,
            ds,
            sg,
            dsg,
            t,
            h,
        };
        Ok(Self::new_from_common(state))
    }

    /// Create a new solver state with augmented equations (sensitivities) from an ODE problem, without making the augmented state consistent.
    fn new_without_initialise_augmented<Eqn, AugmentedEqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &mut AugmentedEqn,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let mut state = Self::new_without_initialise(ode_problem)?.into_common();
        Self::initialise_augmented_state(augmented_eqn, ode_problem, &mut state)?;
        Ok(Self::new_from_common(state))
    }

    /// Create a new solver state with augmented equations from an ODE problem, evaluating the
    /// augmented initial/output operators at a caller-supplied time while leaving the base state
    /// allocation behavior unchanged.
    fn new_without_initialise_augmented_at<Eqn, AugmentedEqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &mut AugmentedEqn,
        t: V::T,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let mut state = Self::new_without_initialise(ode_problem)?.into_common();
        state.t = t;
        Self::initialise_augmented_state(augmented_eqn, ode_problem, &mut state)?;
        Ok(Self::new_from_common(state))
    }

    fn initialise_augmented_state<Eqn, AugmentedEqn>(
        augmented_eqn: &mut AugmentedEqn,
        ode_problem: &OdeSolverProblem<Eqn>,
        state: &mut StateCommon<V>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let naug = augmented_eqn.max_index();
        let mut s = Vec::with_capacity(naug);
        let mut ds = Vec::with_capacity(naug);
        let nstates = augmented_eqn.rhs().nstates();
        let ctx = ode_problem.context();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            let si = augmented_eqn.init().call(state.t);
            let dsi = V::zeros(nstates, ctx.clone());
            s.push(si);
            ds.push(dsi);
        }
        state.s = s;
        state.ds = ds;
        let (dsg, sg) = if augmented_eqn.out().is_some() {
            let mut sg = Vec::with_capacity(naug);
            let mut dsg = Vec::with_capacity(naug);
            for i in 0..naug {
                augmented_eqn.set_index(i);
                let dsgi = if let Some(out) = augmented_eqn.out() {
                    out.call(&state.s[i], state.t)
                } else {
                    state.s[i].clone()
                };
                let sgi = V::zeros(dsgi.len(), ctx.clone());
                sg.push(sgi);
                dsg.push(dsgi);
            }
            (dsg, sg)
        } else {
            (vec![], vec![])
        };
        state.sg = sg;
        state.dsg = dsg;
        Ok(())
    }

    /// Calculate a consistent state and time derivative of the state, based on the equations of the problem.
    fn set_consistent<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        S: NonLinearSolver<Eqn::M>,
    {
        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }
        let state = self.as_mut();
        let (algebraic_indices, _) = ode_problem
            .eqn
            .mass()
            .unwrap()
            .matrix(ode_problem.t0)
            .partition_indices_by_zero_diagonal();
        if algebraic_indices.is_empty() {
            return Ok(());
        }

        // equations are:
        // h(t, u, v, du) = 0
        // g(t, u, v) = 0
        // first we solve for du, v
        let f = InitOp::new(
            &ode_problem.eqn,
            ode_problem.t0,
            state.y,
            algebraic_indices.clone(),
        );
        let rtol = ode_problem.rtol;
        let atol = &ode_problem.atol;
        root_solver.set_problem(&f);
        let mut y_tmp = state.dy.clone();
        y_tmp.copy_from_indices(state.y, &f.algebraic_indices);
        let mut yerr = y_tmp.clone();
        let mut convergence = Convergence::with_tolerance(
            rtol,
            atol,
            ode_problem.ode_options.nonlinear_solver_tolerance,
        );
        convergence.set_max_iter(ode_problem.ic_options.max_newton_iterations);
        let mut result = Ok(());
        debug!("Setting consistent initial conditions at t = {}", state.t);
        for _ in 0..ode_problem.ic_options.max_linear_solver_setups {
            root_solver.reset_jacobian(&f, &y_tmp, *state.t);
            result = root_solver.solve_in_place(&f, &mut y_tmp, *state.t, &yerr, &mut convergence);
            match &result {
                Ok(()) => break,
                Err(DiffsolError::NonLinearSolverError(
                    NonLinearSolverError::NewtonMaxIterations,
                )) => (),
                e => e.clone()?,
            }
            yerr.copy_from(&y_tmp);
        }
        if result.is_err() {
            return Err(non_linear_solver_error!(InitialConditionDidNotConverge));
        }
        f.scatter_soln(&y_tmp, state.y, state.dy);
        // dv is not solved for, so we set it to zero, it will be solved for in the first step of the solver
        state
            .dy
            .assign_at_indices(&algebraic_indices, Eqn::T::zero());
        Ok(())
    }

    /// Calculate the initial sensitivity vectors and their time derivatives, based on the equations of the problem.
    /// Note that this function assumes that the state is already consistent with the algebraic constraints
    /// (either via [Self::set_consistent] or by setting the state up manually).
    fn set_consistent_augmented<Eqn, AugmentedEqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &mut AugmentedEqn,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn> + std::fmt::Debug,
        S: NonLinearSolver<AugmentedEqn::M>,
    {
        let state = self.as_mut();
        augmented_eqn.update_rhs_out_state(state.y, state.dy, *state.t);
        let naug = augmented_eqn.max_index();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            augmented_eqn
                .rhs()
                .call_inplace(&state.s[i], *state.t, &mut state.ds[i]);
        }

        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }

        let mut convergence = Convergence::with_tolerance(
            ode_problem.rtol,
            &ode_problem.atol,
            ode_problem.ode_options.nonlinear_solver_tolerance,
        );
        convergence.set_max_iter(ode_problem.ic_options.max_newton_iterations);
        let (algebraic_indices, _) = ode_problem
            .eqn
            .mass()
            .unwrap()
            .matrix(ode_problem.t0)
            .partition_indices_by_zero_diagonal();
        if algebraic_indices.is_empty() {
            return Ok(());
        }

        for i in 0..naug {
            augmented_eqn.set_index(i);
            let f = InitOp::new(
                augmented_eqn,
                *state.t,
                &state.s[i],
                algebraic_indices.clone(),
            );
            root_solver.set_problem(&f);

            let mut y = state.ds[i].clone();
            y.copy_from_indices(&state.s[i], &f.algebraic_indices);
            let mut yerr = y.clone();
            let mut result = Ok(());
            for _ in 0..ode_problem.ic_options.max_linear_solver_setups {
                root_solver.reset_jacobian(&f, &y, *state.t);
                result = root_solver.solve_in_place(&f, &mut y, *state.t, &yerr, &mut convergence);
                match &result {
                    Ok(()) => break,
                    Err(DiffsolError::NonLinearSolverError(
                        NonLinearSolverError::NewtonMaxIterations,
                    )) => (),
                    e => e.clone()?,
                }
                yerr.copy_from(&y);
            }
            if result.is_err() {
                return Err(non_linear_solver_error!(InitialConditionDidNotConverge));
            }
            f.scatter_soln(&y, &mut state.s[i], &mut state.ds[i]);
        }
        Ok(())
    }

    /// compute size of first step based on alg in Hairer, Norsett, Wanner
    /// Solving Ordinary Differential Equations I, Nonstiff Problems
    /// Section II.4.2
    /// Note: this assumes that the state is already consistent with the algebraic constraints
    /// and y and dy are already set appropriately
    fn set_step_size<Eqn>(
        &mut self,
        h0: Eqn::T,
        atol: &Eqn::V,
        rtol: Eqn::T,
        eqn: &Eqn,
        solver_order: usize,
    ) where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let is_neg_h = h0 < Eqn::T::zero();
        let (h0, h1) = {
            let state = self.as_ref();
            let y0 = state.y;
            let t0 = state.t;
            let f0 = state.dy;

            let d0 = y0.squared_norm(y0, atol, rtol).sqrt();
            let d1 = f0.squared_norm(y0, atol, rtol).sqrt();

            let h0 = if d0 < Eqn::T::from_f64(1e-5).unwrap() || d1 < Eqn::T::from_f64(1e-5).unwrap()
            {
                Eqn::T::from_f64(1e-6).unwrap()
            } else {
                Eqn::T::from_f64(0.01).unwrap() * (d0 / d1)
            };

            // make sure we preserve the sign of h0
            let f1 = if is_neg_h {
                let y1 = f0.clone() * scale(-h0) + y0;
                let t1 = t0 - h0;
                eqn.rhs().call(&y1, t1)
            } else {
                let y1 = f0.clone() * scale(h0) + y0;
                let t1 = t0 + h0;
                eqn.rhs().call(&y1, t1)
            };

            let df = f1 - f0;
            let d2 = df.squared_norm(y0, atol, rtol).sqrt() / h0.abs();

            let mut max_d = d2;
            if max_d < d1 {
                max_d = d1;
            }
            let h1 = if max_d < Eqn::T::from_f64(1e-15).unwrap() {
                let h1 = h0 * Eqn::T::from_f64(1e-3).unwrap();
                if h1 < Eqn::T::from_f64(1e-6).unwrap() {
                    Eqn::T::from_f64(1e-6).unwrap()
                } else {
                    h1
                }
            } else {
                (Eqn::T::from_f64(0.01).unwrap() / max_d)
                    .pow(Eqn::T::one() / Eqn::T::from_f64(1.0 + solver_order as f64).unwrap())
            };
            (h0, h1)
        };

        let state = self.as_mut();
        *state.h = Eqn::T::from_f64(100.0).unwrap() * h0;
        if *state.h > h1 {
            *state.h = h1;
        }

        if is_neg_h {
            *state.h = -*state.h;
        }
    }
}

#[cfg(test)]
mod test {
    use super::StateCommon;
    use crate::{
        error::{DiffsolError, OdeSolverError},
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            exponential_decay::exponential_decay_problem,
            exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_sens,
        },
        op::closure_with_adjoint::ClosureWithAdjoint,
        op::closure_with_sens::ClosureWithSens,
        BdfState, LinearSolver, Matrix, NonLinearOp, NonLinearOpTimePartial, OdeBuilder,
        OdeEquations, OdeSolverMethod, OdeSolverState, ParameterisedOp, Vector, VectorHost,
    };
    use num_traits::FromPrimitive;

    #[test]
    fn test_init_bdf_nalgebra() {
        type M = crate::NalgebraMat<f64>;
        type V = crate::NalgebraVec<f64>;
        type LS = crate::NalgebraLU<f64>;
        test_consistent_initialisation::<M, crate::BdfState<V>, LS>();
    }

    #[test]
    fn test_init_rk_nalgebra() {
        type M = crate::NalgebraMat<f64>;
        type V = crate::NalgebraVec<f64>;
        type LS = crate::NalgebraLU<f64>;
        test_consistent_initialisation::<M, crate::RkState<V>, LS>();
    }

    #[test]
    fn test_init_bdf_faer_sparse() {
        type M = crate::FaerSparseMat<f64>;
        type V = crate::FaerVec<f64>;
        type LS = crate::FaerSparseLU<f64>;
        test_consistent_initialisation::<M, crate::BdfState<V>, LS>();
    }

    #[test]
    fn test_init_rk_faer_sparse() {
        type M = crate::FaerSparseMat<f64>;
        type V = crate::FaerVec<f64>;
        type LS = crate::FaerSparseLU<f64>;
        test_consistent_initialisation::<M, crate::RkState<V>, LS>();
    }

    fn test_consistent_initialisation<
        M: Matrix<V: VectorHost>,
        S: OdeSolverState<M::V>,
        LS: LinearSolver<M>,
    >() {
        let (mut problem, soln) = exponential_decay_with_algebraic_problem_sens::<M>();

        for line_search in [false, true] {
            problem.ic_options.use_linesearch = line_search;

            let s = S::new_and_consistent::<LS, _>(&problem, 1).unwrap();
            s.as_ref().y.assert_eq_norm(
                &soln.solution_points[0].state,
                &problem.atol,
                problem.rtol,
                M::T::from_f64(10.).unwrap(),
            );

            let s = S::new_with_sensitivities_and_consistent::<LS, _>(&problem, 1).unwrap();
            s.as_ref().y.assert_eq_norm(
                &soln.solution_points[0].state,
                &problem.atol,
                problem.rtol,
                M::T::from_f64(10.).unwrap(),
            );
            let sens_soln = soln.sens_solution_points.as_ref().unwrap();
            for (i, ssoln) in sens_soln.iter().enumerate() {
                s.as_ref().s[i].assert_eq_norm(
                    &ssoln[0].state,
                    &problem.atol,
                    problem.rtol,
                    M::T::from_f64(10.).unwrap(),
                );
            }
        }
    }

    #[test]
    fn step_size_preserves_negative_direction() {
        type M = crate::NalgebraMat<f64>;
        type V = crate::NalgebraVec<f64>;

        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        problem.h0 = -problem.h0.abs();

        let mut state = BdfState::<V>::new_without_initialise(&problem).unwrap();
        state.set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);

        assert!(state.as_ref().h < 0.0);
    }

    #[test]
    fn step_size_clamps_tiny_initial_conditions() {
        type M = crate::NalgebraMat<f64>;
        type V = crate::NalgebraVec<f64>;

        let problem = OdeBuilder::<M>::new()
            .rhs(|_x, _p, _t, y| y[0] = 0.0)
            .init(|_p, _t, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut state = BdfState::<V>::new_without_initialise(&problem).unwrap();

        state.set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);

        assert!((state.as_ref().h - 1e-6).abs() < 1e-12);
    }

    type TestMat = NalgebraMat<f64>;
    type TestVec = crate::NalgebraVec<f64>;
    type TestState = BdfState<TestVec>;

    fn scalar_problem(
        lambda: f64,
    ) -> crate::OdeSolverProblem<
        impl crate::OdeEquationsImplicitSens<
            M = TestMat,
            V = TestVec,
            T = f64,
            C = crate::NalgebraContext,
        >,
    > {
        OdeBuilder::<TestMat>::new()
            .p([1.0, -2.0])
            .rhs_sens_implicit(
                move |x, _p, _t, y| y[0] = lambda * x[0],
                move |_x, _p, _t, v, y| y[0] = lambda * v[0],
                |_x, _p, _t, _v, y| y[0] = 0.0,
            )
            .init_sens(|_p, _t, y| y[0] = 0.0, |_p, _t, _v, y| y[0] = 0.0, 1)
            .build()
            .unwrap()
    }

    fn scalar_problem_adjoint(
        lambda: f64,
    ) -> crate::OdeSolverProblem<
        impl crate::OdeEquationsImplicitAdjoint<
            M = TestMat,
            V = TestVec,
            T = f64,
            C = crate::NalgebraContext,
        >,
    > {
        OdeBuilder::<TestMat>::new()
            .p([1.0, -2.0])
            .integrate_out(true)
            .rhs_adjoint_implicit(
                move |x, _p, _t, y| y[0] = lambda * x[0],
                move |_x, _p, _t, v, y| y[0] = lambda * v[0],
                move |_x, _p, _t, v, y| y[0] = -lambda * v[0],
                |_x, _p, _t, _v, y| y.fill(0.0),
            )
            .init_adjoint(|_p, _t, y| y[0] = 0.0, |_p, _t, _v, y| y.fill(0.0), 1)
            .out_adjoint_implicit(
                |x, _p, _t, y| {
                    y[0] = x[0];
                    y[1] = 2.0 * x[0];
                },
                |_x, _p, _t, v, y| {
                    y[0] = v[0];
                    y[1] = 2.0 * v[0];
                },
                |_x, _p, _t, v, y| y[0] = -(v[0] + 2.0 * v[1]),
                |_x, _p, _t, v, y| {
                    y[0] = 0.5 * v[0] - 0.25 * v[1];
                    y[1] = -0.75 * v[0] + 0.5 * v[1];
                },
                2,
            )
            .build()
            .unwrap()
    }

    fn scalar_problem_with_mass(
        lambda: f64,
    ) -> crate::OdeSolverProblem<
        impl crate::OdeEquationsImplicit<M = TestMat, V = TestVec, T = f64, C = crate::NalgebraContext>,
    > {
        OdeBuilder::<TestMat>::new()
            .p([1.0, -2.0])
            .rhs_implicit(
                move |x, _p, _t, y| y[0] = lambda * x[0],
                move |_x, _p, _t, v, y| y[0] = lambda * v[0],
            )
            .mass(|v, _p, _t, beta, y| y.axpy(1.0, v, beta))
            .init(|_p, _t, y| y[0] = 0.0, 1)
            .build()
            .unwrap()
    }

    fn scalar_problem_with_mass_adjoint(
        lambda: f64,
    ) -> crate::OdeSolverProblem<
        impl crate::OdeEquationsImplicitAdjoint<
            M = TestMat,
            V = TestVec,
            T = f64,
            C = crate::NalgebraContext,
        >,
    > {
        OdeBuilder::<TestMat>::new()
            .p([1.0, -2.0])
            .integrate_out(true)
            .rhs_adjoint_implicit(
                move |x, _p, _t, y| y[0] = lambda * x[0],
                move |_x, _p, _t, v, y| y[0] = lambda * v[0],
                move |_x, _p, _t, v, y| y[0] = -lambda * v[0],
                |_x, _p, _t, _v, y| y.fill(0.0),
            )
            .mass_adjoint(
                |v, _p, _t, beta, y| y.axpy(1.0, v, beta),
                |v, _p, _t, beta, y| y.axpy(1.0, v, beta),
            )
            .init_adjoint(|_p, _t, y| y[0] = 0.0, |_p, _t, _v, y| y.fill(0.0), 1)
            .out_adjoint_implicit(
                |x, _p, _t, y| {
                    y[0] = x[0];
                    y[1] = 2.0 * x[0];
                },
                |_x, _p, _t, v, y| {
                    y[0] = v[0];
                    y[1] = 2.0 * v[0];
                },
                |_x, _p, _t, v, y| y[0] = -(v[0] + 2.0 * v[1]),
                |_x, _p, _t, v, y| {
                    y[0] = 0.5 * v[0] - 0.25 * v[1];
                    y[1] = -0.75 * v[0] + 0.5 * v[1];
                },
                2,
            )
            .build()
            .unwrap()
    }

    fn make_state(
        problem: &crate::OdeSolverProblem<
            impl crate::OdeEquationsImplicitSens<
                M = TestMat,
                V = TestVec,
                T = f64,
                C = crate::NalgebraContext,
            >,
        >,
        t: f64,
        y: f64,
        s: [f64; 2],
    ) -> TestState {
        let mut state = TestState::new_with_sensitivities(problem, 1).unwrap();
        let state_mut = state.as_mut();
        *state_mut.t = t;
        state_mut.y[0] = y;
        state_mut.dy[0] = problem.eqn.rhs().call(state_mut.y, t)[0];
        state_mut.s[0][0] = s[0];
        state_mut.s[1][0] = s[1];
        state
    }

    fn make_adjoint_state(
        problem: &crate::OdeSolverProblem<
            impl OdeEquations<M = TestMat, V = TestVec, T = f64, C = crate::NalgebraContext>,
        >,
        t: f64,
        y: f64,
        dy: f64,
        lambda: [f64; 2],
        q: [[f64; 2]; 2],
    ) -> TestState {
        let ctx = *problem.context();
        let s = lambda
            .iter()
            .map(|lambda_i| TestVec::from_vec(vec![*lambda_i], ctx))
            .collect::<Vec<_>>();
        let ds = vec![TestVec::zeros(1, ctx); s.len()];
        let sg = q
            .iter()
            .map(|q_i| TestVec::from_vec(q_i.to_vec(), ctx))
            .collect::<Vec<_>>();
        let dsg = vec![TestVec::zeros(2, ctx); sg.len()];
        TestState::new_from_common(StateCommon {
            y: TestVec::from_vec(vec![y], ctx),
            dy: TestVec::from_vec(vec![dy], ctx),
            g: TestVec::zeros(0, ctx),
            dg: TestVec::zeros(0, ctx),
            s,
            ds,
            sg,
            dsg,
            t,
            h: 0.0,
        })
    }

    fn assert_scalar_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_scalar_close_tol(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_other_error(err: DiffsolError, needle: &str) {
        match err {
            DiffsolError::OdeSolverError(OdeSolverError::Other(msg)) => {
                assert!(
                    msg.contains(needle),
                    "expected error containing {needle:?}, got {msg:?}"
                );
            }
            other => panic!("expected OdeSolverError::Other, got {other:?}"),
        }
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_matches_autonomous_formula() {
        let problem = scalar_problem(0.25);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.2, -0.7], crate::NalgebraContext);
        let mut state = make_state(&problem, 0.0, 2.0, [0.3, -0.4]);

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 1.5 * x[0] + 0.2 * p[0] - 0.1 * p[1]
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 0.2 * v[0] - 0.1 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 4.0 * v[0];
                y[1] = -2.0 * v[0];
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = v[0];
                y[1] = 0.5 * v[0] - 1.5 * v[1];
            },
            1,
            2,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        state
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 1)
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], 3.31);
        assert_scalar_close(state.as_ref().dy[0], 0.8275);
        assert_scalar_close(state.as_ref().s[0][0], 0.65775);
        assert_scalar_close(state.as_ref().s[1][0], -0.64575);
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_uses_selected_root_component() {
        let problem = scalar_problem(0.25);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.2, -0.7], crate::NalgebraContext);
        let mut state_root0 = make_state(&problem, 0.0, 2.0, [0.3, -0.4]);
        let mut state_root1 = state_root0.clone();

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 1.5 * x[0] + 0.2 * p[0] - 0.1 * p[1]
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 0.2 * v[0] - 0.1 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 4.0 * v[0];
                y[1] = -2.0 * v[0];
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = v[0];
                y[1] = 0.5 * v[0] - 1.5 * v[1];
            },
            1,
            2,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        state_root0
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 0)
            .unwrap();
        state_root1
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 1)
            .unwrap();

        assert!(
            (state_root0.as_ref().s[0][0] - state_root1.as_ref().s[0][0]).abs() > 1e-6,
            "different root components should produce different sensitivity updates"
        );
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_supports_root_time_dependence_without_state_dependence() {
        let problem = scalar_problem(0.5);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.0, 2.0], crate::NalgebraContext);
        let mut state = make_state(&problem, 1.5, 1.2, [0.1, -0.2]);

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, _t, y: &mut TestVec| y[0] = x[0] + 0.5 * p[0] + 0.25 * p[1],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 0.5 * v[0] + 0.25 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, t, y: &mut TestVec| y[0] = 4.0 * t,
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y[0] = 0.0,
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 2.0 * v[0] - 3.0 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);
        state
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 0)
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], 2.2);
        assert_scalar_close(state.as_ref().dy[0], 1.1);
        assert_scalar_close(state.as_ref().s[0][0], 0.85);
        assert_scalar_close(state.as_ref().s[1][0], -0.325);
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_matches_time_dependent_formula() {
        let problem = scalar_problem(0.1);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = make_state(&problem, 3.0, 2.0, [0.2, -0.1]);

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, t, y: &mut TestVec| {
                y[0] = 1.2 * x[0] + 0.4 * p[0] - 0.3 * p[1] + 0.8 * t
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.2 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 0.4 * v[0] - 0.3 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, t, y: &mut TestVec| {
                y[0] = 0.5 * x[0] - 0.7 * p[0] + 1.1 * p[1] - 0.2 * t
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 0.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -0.7 * v[0] + 1.1 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);
        state
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 0)
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], 5.8);
        assert_scalar_close(state.as_ref().dy[0], 0.58);
        assert_scalar_close_tol(state.as_ref().s[0][0], -2.12, 1e-8);
        assert_scalar_close_tol(state.as_ref().s[1][0], 4.41, 1e-8);
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_estimates_time_derivatives_via_finite_difference() {
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let x = TestVec::from_vec(vec![2.0], crate::NalgebraContext);
        let t = 3.0;

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, t, y: &mut TestVec| {
                y[0] = 1.2 * x[0] + 0.4 * p[0] - 0.3 * p[1] + 0.8 * t
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.2 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 0.4 * v[0] - 0.3 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, p: &TestVec, t, y: &mut TestVec| {
                y[0] = 0.5 * x[0] - 0.7 * p[0] + 1.1 * p[1] - 0.2 * t
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 0.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -0.7 * v[0] + 1.1 * v[1]
            },
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let reset_dt = reset.time_derive(&x, t);
        let root_dt = root.time_derive(&x, t);

        assert_scalar_close_tol(reset_dt[0], 0.8, 1e-8);
        assert_scalar_close_tol(root_dt[0], -0.2, 1e-8);
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_rejects_invalid_root_index() {
        let problem = scalar_problem(0.25);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = make_state(&problem, 0.0, 1.0, [0.0, 0.0]);

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = x[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y[0] = 0.0,
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);
        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| {
                y[0] = 1.0;
                y[1] = 1.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            1,
            2,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let err = state
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 2)
            .unwrap_err();
        assert_other_error(err, "root index 2 out of bounds");
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_rejects_zero_event_denominator() {
        let problem = scalar_problem(0.0);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = make_state(&problem, 0.0, 0.0, [0.0, 0.0]);

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = x[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y[0] = 0.0,
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);
        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = 0.0,
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let err = state
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 0)
            .unwrap_err();
        assert_other_error(err, "active root derivative along flow is zero");
    }

    #[test]
    fn state_mut_op_with_sens_and_reset_rejects_mass_matrix_equations() {
        let problem = scalar_problem_with_mass(0.25);
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = TestState::new_without_initialise(&problem).unwrap();

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = x[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y[0] = 0.0,
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);
        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = 0.0,
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y[0] = 0.0,
            1,
            2,
            1,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let err = state
            .as_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, 0)
            .unwrap_err();
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::MassMatrixNotSupported)
        ));
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_matches_autonomous_formula() {
        let problem = scalar_problem_adjoint(0.25);
        let forward_problem = scalar_problem(0.25);
        let p = TestVec::from_vec(vec![1.2, -0.7], crate::NalgebraContext);
        let mut state = make_adjoint_state(
            &problem,
            0.0,
            7.0,
            -3.0,
            [0.3, -0.4],
            [[0.1, -0.2], [0.5, 0.6]],
        );
        let mut fwd_state_minus = make_state(&forward_problem, 0.0, 2.0, [0.0, 0.0]);
        fwd_state_minus.as_mut().dy[0] = 0.5;
        let mut fwd_state_plus = fwd_state_minus.clone();
        fwd_state_plus.as_mut().dy[0] = 0.8275;
        let mut forward_solver = problem.bdf::<crate::NalgebraLU<f64>>().unwrap();
        let (checkpointer, _, _, _) = forward_solver.solve_with_checkpointing(1.0, None).unwrap();
        let mut adjoint_eqn =
            problem.adjoint_equations(checkpointer, Some(forward_solver.clone()), Some(2));

        let reset = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 1.5 * x[0] + 0.2 * p[0] - 0.1 * p[1]
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -1.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -0.2 * v[0];
                y[1] = 0.1 * v[0];
            },
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, t, y: &mut TestVec| {
                y[0] = 0.3 * t;
                y[1] = 0.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 3.0 * v[0];
                y[1] = -2.0 * v[0];
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -(3.0 * v[0] - 2.0 * v[1]);
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -(0.8 * v[0] + 0.5 * v[1]);
                y[1] = -(-1.5 * v[1]);
            },
            1,
            2,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let y_before = state.as_ref().y[0];
        let dy_before = state.as_ref().dy[0];

        state
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn,
                &reset,
                &root,
                1,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], y_before);
        assert_scalar_close(state.as_ref().dy[0], dy_before);
        assert_scalar_close(state.as_ref().s[0][0], 0.4965);
        assert_scalar_close(state.as_ref().s[1][0], -0.662);
        assert_scalar_close(state.as_ref().sg[0][0], 0.148375);
        assert_scalar_close(state.as_ref().sg[0][1], -0.195125);
        assert_scalar_close(state.as_ref().sg[1][0], 0.4355);
        assert_scalar_close(state.as_ref().sg[1][1], 0.5935);
        assert!(
            state.as_ref().ds.iter().any(|ds_i| ds_i[0].abs() > 1e-12),
            "expected adjoint reset to refresh ds"
        );
        assert!(
            state
                .as_ref()
                .dsg
                .iter()
                .flat_map(|dsg_i| (0..dsg_i.len()).map(|j| dsg_i[j]))
                .any(|value| value.abs() > 1e-12),
            "expected adjoint reset to refresh dsg"
        );
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_uses_selected_root_component() {
        let problem = scalar_problem_adjoint(0.25);
        let forward_problem = scalar_problem(0.25);
        let p = TestVec::from_vec(vec![1.2, -0.7], crate::NalgebraContext);
        let mut state_root0 = make_adjoint_state(
            &problem,
            0.0,
            7.0,
            -3.0,
            [0.3, -0.4],
            [[0.1, -0.2], [0.5, 0.6]],
        );
        let mut state_root1 = state_root0.clone();
        let mut fwd_state_minus = make_state(&forward_problem, 0.0, 2.0, [0.0, 0.0]);
        fwd_state_minus.as_mut().dy[0] = 0.5;
        let mut fwd_state_plus = fwd_state_minus.clone();
        fwd_state_plus.as_mut().dy[0] = 0.8275;
        let mut forward_solver = problem.bdf::<crate::NalgebraLU<f64>>().unwrap();
        let (checkpointer, _, _, _) = forward_solver.solve_with_checkpointing(1.0, None).unwrap();
        let mut adjoint_eqn_root0 =
            problem.adjoint_equations(checkpointer.clone(), Some(forward_solver.clone()), Some(2));
        let mut adjoint_eqn_root1 =
            problem.adjoint_equations(checkpointer, Some(forward_solver.clone()), Some(2));

        let reset = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 1.5 * x[0] + 0.2 * p[0] - 0.1 * p[1]
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -1.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -0.2 * v[0];
                y[1] = 0.1 * v[0];
            },
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, t, y: &mut TestVec| {
                y[0] = 0.3 * t;
                y[1] = 0.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 3.0 * v[0];
                y[1] = -2.0 * v[0];
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -(3.0 * v[0] - 2.0 * v[1]);
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -(0.8 * v[0] + 0.5 * v[1]);
                y[1] = -(-1.5 * v[1]);
            },
            1,
            2,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        state_root0
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn_root0,
                &reset,
                &root,
                0,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap();
        state_root1
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn_root1,
                &reset,
                &root,
                1,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap();

        assert!(
            (state_root0.as_ref().s[0][0] - state_root1.as_ref().s[0][0]).abs() > 1e-6,
            "different root components should produce different adjoint updates"
        );
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_matches_time_dependent_formula() {
        let problem = scalar_problem_adjoint(0.1);
        let forward_problem = scalar_problem(0.1);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = make_adjoint_state(
            &problem,
            3.0,
            7.0,
            -3.0,
            [0.2, -0.1],
            [[0.3, -0.4], [0.5, 0.2]],
        );
        let mut fwd_state_minus = make_state(&forward_problem, 3.0, 2.0, [0.0, 0.0]);
        fwd_state_minus.as_mut().dy[0] = 0.2;
        let mut fwd_state_plus = fwd_state_minus.clone();
        fwd_state_plus.as_mut().dy[0] = 0.58;
        let mut forward_solver = problem.bdf::<crate::NalgebraLU<f64>>().unwrap();
        let (checkpointer, _, _, _) = forward_solver.solve_with_checkpointing(4.0, None).unwrap();
        let mut adjoint_eqn =
            problem.adjoint_equations(checkpointer, Some(forward_solver.clone()), Some(2));

        let reset = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, p: &TestVec, t, y: &mut TestVec| {
                y[0] = 1.2 * x[0] + 0.4 * p[0] - 0.3 * p[1] + 0.8 * t
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 1.2 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -1.2 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -0.4 * v[0];
                y[1] = 0.3 * v[0];
            },
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, p: &TestVec, t, y: &mut TestVec| {
                y[0] = 0.5 * x[0] - 0.7 * p[0] + 1.1 * p[1] - 0.2 * t
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = 0.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -0.5 * v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = 0.7 * v[0];
                y[1] = -1.1 * v[0];
            },
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let y_before = state.as_ref().y[0];
        let dy_before = state.as_ref().dy[0];

        state
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn,
                &reset,
                &root,
                0,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], y_before);
        assert_scalar_close(state.as_ref().dy[0], dy_before);
        assert_scalar_close_tol(state.as_ref().s[0][0], 0.7, 1e-8);
        assert_scalar_close_tol(state.as_ref().s[1][0], -0.35, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg[0][0], -0.264, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg[0][1], 0.552, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg[1][0], 0.782, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg[1][1], -0.276, 1e-8);
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_rejects_invalid_root_index() {
        let problem = scalar_problem_adjoint(0.25);
        let forward_problem = scalar_problem(0.25);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = make_adjoint_state(
            &problem,
            0.0,
            7.0,
            -3.0,
            [0.0, 0.0],
            [[0.0, 0.0], [0.0, 0.0]],
        );
        let mut fwd_state_minus = make_state(&forward_problem, 0.0, 1.0, [0.0, 0.0]);
        fwd_state_minus.as_mut().dy[0] = 1.0;
        let fwd_state_plus = fwd_state_minus.clone();
        let mut forward_solver = problem.bdf::<crate::NalgebraLU<f64>>().unwrap();
        let (checkpointer, _, _, _) = forward_solver.solve_with_checkpointing(1.0, None).unwrap();
        let mut adjoint_eqn =
            problem.adjoint_equations(checkpointer, Some(forward_solver.clone()), Some(2));

        let reset = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = x[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y.fill(0.0),
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);
        let root = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| {
                y[0] = 1.0;
                y[1] = 1.0;
            },
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| {
                y[0] = -(v[0] + v[1]);
            },
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y.fill(0.0),
            1,
            2,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let err = state
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn,
                &reset,
                &root,
                2,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap_err();
        assert_other_error(err, "root index 2 out of bounds");
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_rejects_zero_event_denominator() {
        let problem = scalar_problem_adjoint(0.0);
        let forward_problem = scalar_problem(0.0);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let mut state = make_adjoint_state(
            &problem,
            0.0,
            7.0,
            -3.0,
            [0.0, 0.0],
            [[0.0, 0.0], [0.0, 0.0]],
        );
        let mut fwd_state_minus = make_state(&forward_problem, 0.0, 0.0, [0.0, 0.0]);
        fwd_state_minus.as_mut().dy[0] = 0.0;
        let fwd_state_plus = fwd_state_minus.clone();
        let mut forward_solver = problem.bdf::<crate::NalgebraLU<f64>>().unwrap();
        let (checkpointer, _, _, _) = forward_solver.solve_with_checkpointing(1.0, None).unwrap();
        let mut adjoint_eqn =
            problem.adjoint_equations(checkpointer, Some(forward_solver.clone()), Some(2));

        let reset = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = x[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y.fill(0.0),
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);
        let root = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = 0.0,
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y.fill(0.0),
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let err = state
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn,
                &reset,
                &root,
                0,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap_err();
        assert_other_error(err, "active root derivative along flow is zero");
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_rejects_mass_matrix_equations() {
        let problem = scalar_problem_with_mass_adjoint(0.25);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext);
        let common = StateCommon {
            y: TestVec::zeros(1, crate::NalgebraContext),
            dy: TestVec::zeros(1, crate::NalgebraContext),
            g: TestVec::zeros(0, crate::NalgebraContext),
            dg: TestVec::zeros(0, crate::NalgebraContext),
            s: vec![TestVec::zeros(1, crate::NalgebraContext)],
            ds: vec![TestVec::zeros(1, crate::NalgebraContext)],
            sg: vec![TestVec::zeros(2, crate::NalgebraContext)],
            dsg: vec![TestVec::zeros(2, crate::NalgebraContext)],
            t: 0.0,
            h: 0.0,
        };
        let mut state = TestState::new_from_common(common);
        let fwd_state_minus = state.clone();
        let fwd_state_plus = state.clone();
        let mut forward_solver = problem.bdf::<crate::NalgebraLU<f64>>().unwrap();
        let (checkpointer, _, _, _) = forward_solver.solve_with_checkpointing(1.0, None).unwrap();
        let mut adjoint_eqn =
            problem.adjoint_equations(checkpointer, Some(forward_solver.clone()), Some(2));

        let reset = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = x[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y.fill(0.0),
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let reset = ParameterisedOp::new(&reset, &p);
        let root = ClosureWithAdjoint::<TestMat, _, _, _, _>::new(
            |_x: &TestVec, _p: &TestVec, _t, y: &mut TestVec| y[0] = 0.0,
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = v[0],
            |_x: &TestVec, _p: &TestVec, _t, v: &TestVec, y: &mut TestVec| y[0] = -v[0],
            |_x: &TestVec, _p: &TestVec, _t, _v: &TestVec, y: &mut TestVec| y.fill(0.0),
            1,
            1,
            2,
            crate::NalgebraContext,
        );
        let root = ParameterisedOp::new(&root, &p);

        let err = state
            .as_mut()
            .state_mut_op_with_adjoint_and_reset(
                &mut adjoint_eqn,
                &reset,
                &root,
                0,
                &fwd_state_minus,
                &fwd_state_plus,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::MassMatrixNotSupported)
        ));
    }
}
