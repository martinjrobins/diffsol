use log::debug;
use num_traits::FromPrimitive;
use num_traits::{One, Pow, Signed, Zero};

use crate::error::NonLinearSolverError;
use crate::Scalar;
use crate::{
    error::DiffsolError,
    nonlinear_solver::{convergence::Convergence, NonLinearSolver},
    ode_solver_error, scale, AugmentedOdeEquations, AugmentedOdeEquationsImplicit, ConstantOp,
    Context, InitOp, LinearOp, LinearSolver, Matrix, NewtonNonlinearSolver, NonLinearOp,
    NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint,
    NonLinearOpTimePartial, OdeEquations, OdeEquationsAdjoint, OdeEquationsImplicit,
    OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverProblem, Op, SensEquations,
    Vector, VectorIndex, VectorView,
};
use crate::{non_linear_solver_error, BacktrackingLineSearch, NoLineSearch};

/// A state holding those variables that are common to all ODE solver states,
/// can be used to create a new state for a specific solver.
pub struct StateCommon<V: Vector> {
    pub y: V,
    pub dy: V,
    pub g: V,
    pub dg: V,
    pub s: V,
    pub ds: V,
    pub sg: V,
    pub dsg: V,
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
/// - the sensitivity vectors `s`, one per augmented channel in its batch lanes (see
///   [`crate::AugmentedOdeEquations`])
/// - the derivative of the sensitivity vectors wrt time `ds`
/// - the sensitivity vectors of the output function `sg`
/// - the derivative of the sensitivity vectors of the output function wrt time `dsg`
pub struct StateRef<'a, V: Vector> {
    pub y: &'a V,
    pub dy: &'a V,
    pub g: &'a V,
    pub dg: &'a V,
    pub s: &'a V,
    pub ds: &'a V,
    pub sg: &'a V,
    pub dsg: &'a V,
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
/// - the sensitivity vectors `s`, one per augmented channel in its batch lanes (see
///   [`crate::AugmentedOdeEquations`])
/// - the derivative of the sensitivity vectors wrt time `ds`
/// - the sensitivity vectors of the output function `sg`
/// - the derivative of the sensitivity vectors of the output function wrt time `dsg`
pub struct StateRefMut<'a, V: Vector> {
    pub y: &'a mut V,
    pub dy: &'a mut V,
    pub g: &'a mut V,
    pub dg: &'a mut V,
    pub s: &'a mut V,
    pub ds: &'a mut V,
    pub sg: &'a mut V,
    pub dsg: &'a mut V,
    pub t: &'a mut V::T,
    pub h: &'a mut V::T,
}

impl<V: Vector> StateRefMut<'_, V> {
    /// Calculate a consistent state and time derivative of the state, based on the equations of the problem.
    pub fn set_consistent<Eqn, S>(
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

        debug!(
            "Setting consistent initial conditions: checking mass matrix for algebraic constraints"
        );

        let f = InitOp::new(
            &ode_problem.eqn,
            ode_problem.t0,
            self.y,
            algebraic_indices.clone(),
        );

        debug!(
            "Found {} algebraic variables (zero diagonal in mass matrix) out of {} total states",
            algebraic_indices.len(),
            self.y.len()
        );

        let rtol = ode_problem.rtol;
        let atol = &ode_problem.atol;
        root_solver.set_problem(&f);
        let mut y_tmp = self.dy.clone();
        y_tmp.copy_from_indices(self.y, &f.algebraic_indices);
        let mut yerr = y_tmp.clone();
        let mut convergence = Convergence::with_tolerance(
            rtol,
            atol,
            ode_problem.ode_options.nonlinear_solver_tolerance,
        );
        convergence.set_max_iter(ode_problem.ic_options.max_newton_iterations);
        let mut result = Ok(());
        debug!("Setting consistent initial conditions at t = {}", self.t);
        for _ in 0..ode_problem.ic_options.max_linear_solver_setups {
            root_solver.reset_jacobian(&f, &y_tmp, *self.t);
            result = root_solver.solve_in_place(&f, &mut y_tmp, *self.t, &yerr, &mut convergence);
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
        f.scatter_soln(&y_tmp, self.y, self.dy);
        // dv is not solved for, so we set it to zero, it will be solved for in the first step of the solver
        self.dy
            .assign_at_indices(&algebraic_indices, Eqn::T::zero());
        Ok(())
    }

    /// Calculate the initial sensitivity vectors and their time derivatives, based on the equations of the problem.
    /// Note that this function assumes that the state is already consistent with the algebraic constraints
    /// (either via [Self::set_consistent] or by setting the state up manually).
    pub fn set_consistent_augmented<Eqn, AugmentedEqn, S>(
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
        augmented_eqn.update_rhs_out_state(self.y, self.dy, *self.t);
        augmented_eqn.rhs().call_inplace(self.s, *self.t, self.ds);

        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }

        let (algebraic_indices, _) = ode_problem
            .eqn
            .mass()
            .unwrap()
            .matrix(ode_problem.t0)
            .partition_indices_by_zero_diagonal();
        if algebraic_indices.is_empty() {
            return Ok(());
        }

        let mut convergence = Convergence::with_tolerance(
            ode_problem.rtol,
            &ode_problem.atol,
            ode_problem.ode_options.nonlinear_solver_tolerance,
        );
        convergence.set_max_iter(ode_problem.ic_options.max_newton_iterations);

        // every augmented channel at once: the Jacobian is shared by all channels and the
        // solution holds one channel per batch lane
        let f = InitOp::new(augmented_eqn, *self.t, self.s, algebraic_indices.clone());
        root_solver.set_problem(&f);
        let mut y = self.ds.clone();
        y.copy_from_indices(self.s, &f.algebraic_indices);
        let mut yerr = y.clone();
        let mut result = Ok(());
        for _ in 0..ode_problem.ic_options.max_linear_solver_setups {
            root_solver.reset_jacobian(&f, &y, *self.t);
            result = root_solver.solve_in_place(&f, &mut y, *self.t, &yerr, &mut convergence);
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
        f.scatter_soln(&y, self.s, self.ds);
        Ok(())
    }

    /// Apply the equation set's reset operator to the current state in-place (no mass matrix).
    ///
    /// Applies `op(y, t)` to update `y`, then recomputes `dy = rhs(y, t)`.
    /// Only valid for equations without a mass matrix. If a mass matrix is present use
    /// [`Self::apply_reset_with_mass`] instead.
    pub fn apply_reset<Eqn>(&mut self, problem: &OdeSolverProblem<Eqn>) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let eqn = &problem.eqn;
        if eqn.mass().is_some() {
            return Err(ode_solver_error!(
                Other,
                "apply_reset cannot be used with a mass matrix; use apply_reset_with_mass instead"
            ));
        }
        let rhs = eqn.rhs();
        let reset = eqn.reset().ok_or_else(|| {
            ode_solver_error!(Other, "No reset operator configured for this problem")
        })?;

        let nstates = rhs.nstates();
        let mut y_out = V::zeros(nstates, rhs.context().clone());
        reset.call_inplace(self.y, *self.t, &mut y_out);
        self.y.copy_from(&y_out);

        rhs.call_inplace(self.y, *self.t, &mut y_out);
        self.dy.copy_from(&y_out);
        Ok(())
    }

    /// Apply the equation set's reset operator to the current state in-place (with mass matrix
    /// support).
    ///
    /// Applies `op(y, t)` to update `y`, then recomputes `dy`. If the equations have no mass
    /// matrix, `dy` is set directly from `rhs(y, t)`. If a mass matrix is present,
    /// [`Self::set_consistent`] is called after the reset to ensure `y` and `dy` satisfy the
    /// algebraic constraints.
    pub fn apply_reset_with_mass<LS, Eqn>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        LS: LinearSolver<Eqn::M>,
    {
        let eqn = &problem.eqn;
        let rhs = eqn.rhs();
        let reset = eqn.reset().ok_or_else(|| {
            ode_solver_error!(Other, "No reset operator configured for this problem")
        })?;

        let nstates = rhs.nstates();
        let mut y_out = V::zeros(nstates, rhs.context().clone());
        reset.call_inplace(self.y, *self.t, &mut y_out);
        self.y.copy_from(&y_out);

        if eqn.mass().is_some() {
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            self.set_consistent(problem, &mut root_solver)?;
        } else {
            rhs.call_inplace(self.y, *self.t, &mut y_out);
            self.dy.copy_from(&y_out);
        }
        Ok(())
    }

    pub fn apply_reset_with_sens<Eqn>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        root_idx: usize,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens<T = V::T, V = V, C = V::C>,
    {
        let eqn = &problem.eqn;
        if eqn.mass().is_some() {
            return Err(ode_solver_error!(
                Other,
                "apply_reset_with_sens cannot be used with a mass matrix; use apply_reset_with_sens_mass instead"
            ));
        }
        let rhs = eqn.rhs();
        let reset_op = eqn.reset().ok_or_else(|| {
            ode_solver_error!(Other, "No reset operator configured for this problem")
        })?;
        let root_op = eqn.root().ok_or_else(|| {
            ode_solver_error!(Other, "No root operator configured for this problem")
        })?;

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
        let s_before = self.s.clone();
        let nparams = rhs.nparams();
        let reset_t = reset_op.time_derive(&y_before, t);
        let root_t = root_op.time_derive(&y_before, t);

        // Delegate reset of self.y and self.dy (no mass matrix path).
        self.apply_reset::<Eqn>(problem)?;

        let mut correction_dir = V::zeros(nstates, ctx.clone());
        reset_op.jac_mul_inplace(&y_before, t, &f_minus, &mut correction_dir);
        correction_dir += &reset_t;
        correction_dir -= &*self.dy;

        let mut root_flow = V::zeros(nroots, ctx.clone());
        root_op.jac_mul_inplace(&y_before, t, &f_minus, &mut root_flow);
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        let nbatch_denom = root_flow.context().nbatch();
        for b in 0..nbatch_denom {
            let denom = root_flow.get_batch(b).get_index(root_idx)
                + root_t.get_batch(b).get_index(root_idx);
            if denom.abs() <= denom_tol {
                return Err(ode_solver_error!(
                    Other,
                    "reset sensitivity correction undefined: active root derivative along flow is zero"
                ));
            }
        }

        // apply reset to all sensitivitiy channels (shared reset/root Jacobians and `correction_dir`).
        let aug_ctx = s_before.context().clone();
        let mut reset_sens = V::zeros(nstates, aug_ctx.clone());
        let mut root_jac_s = V::zeros(nroots, aug_ctx.clone());
        let mut root_sens = V::zeros(nroots, aug_ctx.clone());
        let mut tau_p = V::zeros(1, aug_ctx.clone());
        let mut s_plus = V::zeros(nstates, aug_ctx);
        let mut reset_sens_mat =
            Eqn::M::new_from_sparsity(nstates, nparams, reset_op.sens_sparsity(), ctx.clone());
        let mut root_sens_mat =
            Eqn::M::new_from_sparsity(nroots, nparams, root_op.sens_sparsity(), ctx);

        reset_op.jac_mul_inplace(&y_before, t, &s_before, &mut s_plus);
        // parameter Jacobians are evaluated as matrices at the problem's own batch count and
        // reshaped onto the lanes: column `p` belongs in lane `b * nparams + p`.
        reset_op.sens_inplace(&y_before, t, &mut reset_sens_mat);
        reset_sens_mat.add_columns_to_batched_vector(&mut reset_sens);
        s_plus += &reset_sens;

        root_op.jac_mul_inplace(&y_before, t, &s_before, &mut root_jac_s);
        root_op.sens_inplace(&y_before, t, &mut root_sens_mat);
        root_sens_mat.add_columns_to_batched_vector(&mut root_sens);

        // tau = -(r_x s + r_p) / d, one scalar per parameter lane
        tau_p.for_each_batch(
            [&root_flow, &root_t, &root_jac_s, &root_sens],
            |tau, [rflow, rt, rjac, rsens], _| {
                tau[0] = -(rjac[root_idx] + rsens[root_idx]) / (rflow[root_idx] + rt[root_idx]);
            },
        );
        s_plus.batched_axpy(&tau_p, &correction_dir, V::T::one());
        self.s.copy_from(&s_plus);

        Ok(())
    }

    /// Apply a reset operator to the current state and propagate sensitivities through a
    /// time-dependent root-triggered event correction, with mass-matrix support.
    ///
    /// Like [`Self::apply_reset_with_sens`] but also handles DAE problems with a mass matrix by
    /// calling [`Self::set_consistent_augmented`] after the reset.
    pub fn apply_reset_with_sens_mass<LS, Eqn>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        root_idx: usize,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens<T = V::T, V = V, C = V::C>,
        LS: LinearSolver<Eqn::M>,
    {
        let eqn = &problem.eqn;
        let rhs = eqn.rhs();
        let reset_op = eqn.reset().ok_or_else(|| {
            ode_solver_error!(Other, "No reset operator configured for this problem")
        })?;
        let root_op = eqn.root().ok_or_else(|| {
            ode_solver_error!(Other, "No root operator configured for this problem")
        })?;

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
        let s_before = self.s.clone();
        let nparams = rhs.nparams();
        let reset_t = reset_op.time_derive(&y_before, t);
        let root_t = root_op.time_derive(&y_before, t);

        // Delegate reset of self.y and self.dy (including set_consistent for mass matrices).
        self.apply_reset_with_mass::<LS, Eqn>(problem)?;

        let mut correction_dir = V::zeros(nstates, ctx.clone());
        reset_op.jac_mul_inplace(&y_before, t, &f_minus, &mut correction_dir);
        correction_dir += &reset_t;
        correction_dir -= &*self.dy;

        let mut root_flow = V::zeros(nroots, ctx.clone());
        root_op.jac_mul_inplace(&y_before, t, &f_minus, &mut root_flow);
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        let nbatch_denom = root_flow.context().nbatch();
        for b in 0..nbatch_denom {
            let denom = root_flow.get_batch(b).get_index(root_idx)
                + root_t.get_batch(b).get_index(root_idx);
            if denom.abs() <= denom_tol {
                return Err(ode_solver_error!(
                    Other,
                    "reset sensitivity correction undefined: active root derivative along flow is zero"
                ));
            }
        }

        // apply reset to all sensitivity channels (shared reset/root Jacobians and `correction_dir`).
        let aug_ctx = s_before.context().clone();
        let mut reset_sens = V::zeros(nstates, aug_ctx.clone());
        let mut root_jac_s = V::zeros(nroots, aug_ctx.clone());
        let mut root_sens = V::zeros(nroots, aug_ctx.clone());
        let mut tau_p = V::zeros(1, aug_ctx.clone());
        let mut s_plus = V::zeros(nstates, aug_ctx);
        let mut reset_sens_mat =
            Eqn::M::new_from_sparsity(nstates, nparams, reset_op.sens_sparsity(), ctx.clone());
        let mut root_sens_mat =
            Eqn::M::new_from_sparsity(nroots, nparams, root_op.sens_sparsity(), ctx);

        reset_op.jac_mul_inplace(&y_before, t, &s_before, &mut s_plus);
        // parameter Jacobians are evaluated as matrices at the problem's own batch count and
        // reshaped onto the lanes: column `p` belongs in lane `b * nparams + p`.
        reset_op.sens_inplace(&y_before, t, &mut reset_sens_mat);
        reset_sens_mat.add_columns_to_batched_vector(&mut reset_sens);
        s_plus += &reset_sens;

        root_op.jac_mul_inplace(&y_before, t, &s_before, &mut root_jac_s);
        root_op.sens_inplace(&y_before, t, &mut root_sens_mat);
        root_sens_mat.add_columns_to_batched_vector(&mut root_sens);

        // tau = -(r_x s + r_p) / d, one scalar per parameter lane
        tau_p.for_each_batch(
            [&root_flow, &root_t, &root_jac_s, &root_sens],
            |tau, [rflow, rt, rjac, rsens], _| {
                tau[0] = -(rjac[root_idx] + rsens[root_idx]) / (rflow[root_idx] + rt[root_idx]);
            },
        );
        s_plus.batched_axpy(&tau_p, &correction_dir, V::T::one());
        self.s.copy_from(&s_plus);

        if eqn.mass().is_some() {
            let mut augmented_eqn = SensEquations::new(problem)?;
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            self.set_consistent_augmented(problem, &mut augmented_eqn, &mut root_solver)?;
        }

        Ok(())
    }

    /// Propagate adjoint variables through a time-dependent root-triggered reset.
    ///
    /// This helper pulls the reset and root operators from `eqn`, checks that both are
    /// configured, then applies the adjoint event correction.
    ///
    /// This method assumes `self` stores the post-event adjoint state `(lambda^+, q^+)`,
    /// while `fwd_state_minus` and `fwd_state_plus` provide the forward state derivatives
    /// immediately before and after the reset. The terminal-root contribution for a root-defined
    /// segment end must already have been applied with
    /// [`Self::state_mut_adjoint_terminal_root`].
    ///
    /// If the pre-event forward state is `x^-`, the post-event state is `x^+ = g(x^-, t, p)`,
    /// and the active root is `r_k(x^-, t, p) = 0`, define
    /// `f^- = dx^-/dt`,
    /// `f^+ = dx^+/dt`,
    /// `c = g_x f^- + g_t - f^+`,
    /// `d = [r_x f^-]_k + [r_t]_k`,
    /// and, when continuous outputs are being integrated, `l^- = out(x^-, t)` and
    /// `l^+ = out(x^+, t)`.
    ///
    /// For each adjoint channel with post-event adjoint `lambda^+` and post-event parameter
    /// gradient `q^+`, the pre-event values are
    /// `alpha = (lambda^+ · c + l^- - l^+) / d`,
    /// `lambda^- = g_x^T lambda^+ - r_{x,k}^T alpha`,
    /// and
    /// `q^- = q^+ + g_p^T lambda^+ - r_{p,k}^T alpha`.
    ///
    /// Here `g_x = ∂g/∂x`, `g_t = ∂g/∂t`, `g_p = ∂g/∂p`, `r_x = ∂r/∂x`,
    /// `r_t = ∂r/∂t`, and `r_p = ∂r/∂p`, all evaluated at `(x^-, t, p)`.
    /// The `l^- - l^+` term is the running-cost jump contribution at the interior event.
    /// The forward state vectors in `self` are left untouched; `self.s` and `self.sg`
    /// are updated to the pre-event values.
    ///
    /// Note: mass matrix equations are not supported for this operation.
    pub fn apply_reset_with_adjoint<Eqn>(
        &mut self,
        eqn: &Eqn,
        root_idx: usize,
        fwd_state_minus: StateRef<'_, V>,
        fwd_state_plus: StateRef<'_, V>,
        integrate_out: bool,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitAdjoint<T = V::T, V = V, C = V::C>,
    {
        let reset_op = eqn.reset().ok_or_else(|| {
            ode_solver_error!(Other, "No reset operator configured for this problem")
        })?;
        let root_op = eqn.root().ok_or_else(|| {
            ode_solver_error!(Other, "No root operator configured for this problem")
        })?;

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
        let t_event = fwd_state_minus.t;
        let y_minus = fwd_state_minus.y;
        let y_plus = fwd_state_plus.y;
        let f_minus = fwd_state_minus.dy;
        let f_plus = fwd_state_plus.dy;
        let aug_ctx = self.s.context().clone();
        let nlanes = aug_ctx.nbatch();
        let nchannels = nlanes / ctx.nbatch();
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
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        let nbatch_denom = root_flow.context().nbatch();
        for b in 0..nbatch_denom {
            let denom = root_flow.get_batch(b).get_index(root_idx)
                + root_t.get_batch(b).get_index(root_idx);
            if denom.abs() <= denom_tol {
                return Err(ode_solver_error!(
                    Other,
                    "reset adjoint correction undefined: active root derivative along flow is zero"
                ));
            }
        }

        let (l_minus, l_plus) = if integrate_out {
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

        let mut root_basis = V::zeros(nroots, aug_ctx.clone());
        let mut reset_adj = V::zeros(nstates, aug_ctx.clone());
        let mut root_adj = V::zeros(nstates, aug_ctx.clone());
        let mut reset_sens_adj = V::zeros(nparams, aug_ctx.clone());
        let mut root_sens_adj = V::zeros(nparams, aug_ctx);

        // alpha = (lambda^+ · c + l^- - l^+) / d. The running-cost jump is only there when outputs
        // are being integrated.
        match (&l_minus, &l_plus) {
            (Some(l_minus), Some(l_plus)) => root_basis.for_each_batch(
                [
                    self.s,
                    &correction_dir,
                    &root_flow,
                    &root_t,
                    l_minus,
                    l_plus,
                ],
                |basis, [lambda, cdir, rflow, rt, l_minus, l_plus], lane| {
                    //`lane % nchannels` is this lane's output channel.
                    let i = lane % nchannels;
                    let alpha_num = lambda
                        .iter()
                        .zip(cdir.iter())
                        .fold(V::T::zero(), |acc, (l, c)| acc + *l * *c)
                        + l_minus[i]
                        - l_plus[i];
                    basis[root_idx] = alpha_num / (rflow[root_idx] + rt[root_idx]);
                },
            ),
            _ => root_basis.for_each_batch(
                [self.s, &correction_dir, &root_flow, &root_t],
                |basis, [lambda, cdir, rflow, rt], _| {
                    let alpha_num = lambda
                        .iter()
                        .zip(cdir.iter())
                        .fold(V::T::zero(), |acc, (l, c)| acc + *l * *c);
                    basis[root_idx] = alpha_num / (rflow[root_idx] + rt[root_idx]);
                },
            ),
        }

        reset_op.jac_transpose_mul_inplace(y_minus, t_event, self.s, &mut reset_adj);
        reset_op.sens_transpose_mul_inplace(y_minus, t_event, self.s, &mut reset_sens_adj);
        root_op.jac_transpose_mul_inplace(y_minus, t_event, &root_basis, &mut root_adj);
        root_op.sens_transpose_mul_inplace(y_minus, t_event, &root_basis, &mut root_sens_adj);

        self.s.copy_from(&root_adj);
        self.s.axpy(-V::T::one(), &reset_adj, V::T::one());
        *self.sg -= &reset_sens_adj;
        *self.sg += &root_sens_adj;
        Ok(())
    }

    ///
    /// Given a forward terminal state satisfying the active root condition
    /// `r_k(y_f, t_f, p) = 0`, this method adds the terminal contribution
    /// `lambda_f += r_{x,k}^T * (u_k / d)` and `q_f += r_{p,k}^T * (u_k / d)` to each adjoint
    /// channel, where `u_k` is the corresponding model output component and
    /// `d = [r_x f_f]_k + [r_t]_k`.
    ///
    pub fn state_mut_adjoint_terminal_root<Eqn, State>(
        &mut self,
        eqn: &Eqn,
        root_idx: usize,
        forward: &State,
        integrate_out: bool,
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
        State: OdeSolverState<V>,
    {
        if eqn.mass().is_some() {
            return Err(ode_solver_error!(MassMatrixNotSupported));
        }

        if !integrate_out {
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
        let nbatch = eqn.context().nbatch();
        if self.s.context().nbatch() != nbatch * nout
            || self.sg.context().nbatch() != nbatch * nout
            || self.dsg.context().nbatch() != nbatch * nout
        {
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
        let denom_tol = V::T::from_f64(100.0).unwrap() * V::T::EPSILON;
        let nbatch_denom = root_flow.context().nbatch();
        for b in 0..nbatch_denom {
            let denom = root_flow.get_batch(b).get_index(root_idx)
                + root_t.get_batch(b).get_index(root_idx);
            if denom.abs() <= denom_tol {
                return Err(ode_solver_error!(
                    Other,
                    "terminal root adjoint correction undefined: active root derivative along flow is zero"
                ));
            }
        }

        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let aug_ctx = self.s.context().clone();
        let mut root_basis = V::zeros(nroots, aug_ctx.clone());
        let mut lambda_corr = V::zeros(nstates, aug_ctx.clone());
        let mut q_corr = V::zeros(nparams, aug_ctx);
        // one output component per lane, so the whole terminal contribution is two batched
        // transpose products with the root basis vector
        root_basis.for_each_batch(
            [&out, &root_flow, &root_t],
            |basis, [out, rflow, rt], lane| {
                basis[root_idx] = out[lane % nout] / (rflow[root_idx] + rt[root_idx]);
            },
        );
        root_op.jac_transpose_mul_inplace(forward.y, forward.t, &root_basis, &mut lambda_corr);
        root_op.sens_transpose_mul_inplace(forward.y, forward.t, &root_basis, &mut q_corr);
        *self.s += &lambda_corr;
        *self.sg += &q_corr;
        Ok(())
    }

    /// compute size of first step based on alg in Hairer, Norsett, Wanner
    /// Solving Ordinary Differential Equations I, Nonstiff Problems
    /// Section II.4.2
    /// Note: this assumes that the state is already consistent with the algebraic constraints
    /// and y and dy are already set appropriately
    pub fn set_step_size<Eqn>(
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
            let y0 = &*self.y;
            let t0 = *self.t;
            let f0 = &*self.dy;

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

        *self.h = Eqn::T::from_f64(100.0).unwrap() * h0;
        if *self.h > h1 {
            *self.h = h1;
        }

        if is_neg_h {
            *self.h = -*self.h;
        }
    }
}

/// State for the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`,
/// - the sensitivity vectors `s`, one per augmented channel in its batch lanes (see
///   [`crate::AugmentedOdeEquations`])
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
        let nlanes = problem.context().nbatch() * augmented_eqn.max_index();
        let nstates = problem.eqn.rhs().nstates();
        for v in [state.s, state.ds] {
            if v.context().nbatch() != nlanes || v.len() != nstates {
                return Err(ode_solver_error!(StateProblemMismatch));
            }
        }
        Ok(())
    }

    /// Create a new solver state from an ODE problem.
    /// This function will set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [`StateRefMut::set_consistent`] and [StateRefMut::set_step_size] to set the state up if you need to.
    fn new<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let mut ret = Self::new_without_initialise(ode_problem)?;
        ret.as_mut().set_step_size(
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
    /// You can then use [`StateRefMut::set_consistent`] and [StateRefMut::set_step_size] to set the state up if you need to.
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
            ret.as_mut().set_consistent(ode_problem, &mut root_solver)?;
        } else {
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            ret.as_mut().set_consistent(ode_problem, &mut root_solver)?;
        }
        ret.as_mut().set_step_size(
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
        let mut augmented_eqn = SensEquations::new(ode_problem)?;
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;

        // eval the rhs since we're not calling set_consistent_augmented
        let state = ret.as_mut();
        augmented_eqn.update_rhs_out_state(state.y, state.dy, *state.t);
        augmented_eqn
            .rhs()
            .call_inplace(state.s, *state.t, state.ds);
        ret.as_mut().set_step_size(
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
        let mut augmented_eqn = SensEquations::new(ode_problem)?;
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;
        if ode_problem.ic_options.use_linesearch {
            let mut ls = BacktrackingLineSearch::default();
            ls.c = ode_problem.ic_options.armijo_constant;
            ls.max_iter = ode_problem.ic_options.max_linesearch_iterations;
            ls.tau = ode_problem.ic_options.step_reduction_factor;
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), ls);
            ret.as_mut().set_consistent(ode_problem, &mut root_solver)?;
        } else {
            let mut root_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            ret.as_mut().set_consistent(ode_problem, &mut root_solver)?;
        }
        if ode_problem.ic_options.use_linesearch {
            let mut ls = BacktrackingLineSearch::default();
            ls.c = ode_problem.ic_options.armijo_constant;
            ls.max_iter = ode_problem.ic_options.max_linesearch_iterations;
            ls.tau = ode_problem.ic_options.step_reduction_factor;
            let mut root_solver_sens = NewtonNonlinearSolver::new(LS::default(), ls);
            ret.as_mut().set_consistent_augmented(
                ode_problem,
                &mut augmented_eqn,
                &mut root_solver_sens,
            )?;
        } else {
            let mut root_solver_sens = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
            ret.as_mut().set_consistent_augmented(
                ode_problem,
                &mut augmented_eqn,
                &mut root_solver_sens,
            )?;
        }
        ret.as_mut().set_step_size(
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
        let empty = || V::zeros(0, y.context().clone());
        let (s, ds) = (empty(), empty());
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
        let (sg, dsg) = (empty(), empty());
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
        _ode_problem: &OdeSolverProblem<Eqn>,
        state: &mut StateCommon<V>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let nstates = augmented_eqn.rhs().nstates();
        let aug_ctx = augmented_eqn.aug_context().clone();
        // all channels at once: the augmented init operator writes every batch lane
        let mut s = V::zeros(nstates, aug_ctx.clone());
        augmented_eqn.init().call_inplace(state.t, &mut s);
        state.ds = V::zeros(nstates, aug_ctx.clone());
        let (dsg, sg) = if let Some(out) = augmented_eqn.out() {
            let mut dsg = V::zeros(out.nout(), aug_ctx.clone());
            out.call_inplace(&s, state.t, &mut dsg);
            let sg = V::zeros(out.nout(), aug_ctx.clone());
            (dsg, sg)
        } else {
            (V::zeros(0, aug_ctx.clone()), V::zeros(0, aug_ctx))
        };
        state.s = s;
        state.sg = sg;
        state.dsg = dsg;
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
            exponential_decay::exponential_decay_with_constant_reset_problem_sens,
            exponential_decay::exponential_decay_with_reset_problem,
            exponential_decay::exponential_decay_with_reset_problem_sens,
            exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_sens,
        },
        op::closure_with_sens::ClosureWithSens,
        BdfState, Context, LinearSolver, Matrix, NalgebraLU, NonLinearOp, NonLinearOpTimePartial,
        OdeBuilder, OdeEquations, OdeSolverState, ParameterisedOp, Vector, VectorView,
        VectorViewMut,
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

    fn test_consistent_initialisation<M: Matrix, S: OdeSolverState<M::V>, LS: LinearSolver<M>>() {
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
            let nchannels = sens_soln.len();
            let mut s_i = M::V::zeros(s.as_ref().s.len(), problem.context().clone());
            for (i, ssoln) in sens_soln.iter().enumerate() {
                crate::ode_equations::augmented_channel(s.as_ref().s, nchannels, i, &mut s_i);
                s_i.assert_eq_norm(
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
        state
            .as_mut()
            .set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);

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

        state
            .as_mut()
            .set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);

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

    #[allow(dead_code)]
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
            .mass(|v: &[f64], _p: &[f64], _t, beta: f64, y: &mut [f64]| {
                for (y, v) in y.iter_mut().zip(v.iter()) {
                    *y = *v + beta * *y;
                }
            })
            .init(|_p, _t, y| y[0] = 0.0, 1)
            .build()
            .unwrap()
    }

    #[allow(dead_code)]
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
                |v: &[f64], _p: &[f64], _t, beta: f64, y: &mut [f64]| {
                    for (y, v) in y.iter_mut().zip(v.iter()) {
                        *y = *v + beta * *y;
                    }
                },
                |v: &[f64], _p: &[f64], _t, beta: f64, y: &mut [f64]| {
                    for (y, v) in y.iter_mut().zip(v.iter()) {
                        *y = *v + beta * *y;
                    }
                },
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

    /// Apply a per-lane closure over every batch lane of `y`, reading `v`'s corresponding
    /// (broadcast) lane. Test operators are written scalar-wise, but the augmented state holds
    /// one channel per batch lane, so they have to run once per lane.
    #[allow(clippy::too_many_arguments)]
    fn scalar_problem_adjoint_with_reset_root<RF, RJ, RA, RSA, GF, GJ, GA, GSA>(
        lambda: f64,
        reset_fn: RF,
        reset_jac_fn: RJ,
        reset_adj_fn: RA,
        reset_sens_adj_fn: RSA,
        root_fn: GF,
        root_jac_fn: GJ,
        root_adj_fn: GA,
        root_sens_adj_fn: GSA,
        nroots: usize,
    ) -> crate::OdeSolverProblem<
        impl crate::OdeEquationsImplicitAdjoint<
            M = TestMat,
            V = TestVec,
            T = f64,
            C = crate::NalgebraContext,
        >,
    >
    where
        RF: Fn(&[f64], &[f64], f64, &mut [f64]),
        RJ: Fn(&[f64], &[f64], f64, &[f64], &mut [f64]),
        RA: Fn(&[f64], &[f64], f64, &[f64], &mut [f64]),
        RSA: Fn(&[f64], &[f64], f64, &[f64], &mut [f64]),
        GF: Fn(&[f64], &[f64], f64, &mut [f64]),
        GJ: Fn(&[f64], &[f64], f64, &[f64], &mut [f64]),
        GA: Fn(&[f64], &[f64], f64, &[f64], &mut [f64]),
        GSA: Fn(&[f64], &[f64], f64, &[f64], &mut [f64]),
    {
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
            .root_adjoint_implicit(root_fn, root_jac_fn, root_adj_fn, root_sens_adj_fn, nroots)
            .reset_adjoint_implicit(reset_fn, reset_jac_fn, reset_adj_fn, reset_sens_adj_fn)
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
        // one parameter per batch lane
        for (p, s_p) in s.iter().enumerate() {
            state_mut.s.get_batch_mut(p).set_index(0, *s_p);
        }
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
        // one adjoint channel per batch lane
        let aug_ctx = ctx.clone_with_nbatch(lambda.len()).unwrap();
        let mut s = TestVec::zeros(1, aug_ctx);
        for (i, lambda_i) in lambda.iter().enumerate() {
            s.get_batch_mut(i).set_index(0, *lambda_i);
        }
        let ds = TestVec::zeros(1, aug_ctx);
        let mut sg = TestVec::zeros(2, aug_ctx);
        for (i, q_i) in q.iter().enumerate() {
            let mut sg_i = sg.get_batch_mut(i);
            for (j, q_ij) in q_i.iter().enumerate() {
                sg_i.set_index(j, *q_ij);
            }
        }
        let dsg = TestVec::zeros(2, aug_ctx);
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
    fn state_ref_mut_apply_reset_uses_equation_reset() {
        let (problem, _soln) = exponential_decay_with_reset_problem::<TestMat>();
        let mut state = TestState::new_without_initialise(&problem).unwrap();
        {
            let mut state_mut = state.as_mut();
            state_mut.y.fill(0.6);
            state_mut.dy.fill(-0.06);
            state_mut
                .apply_reset_with_mass::<NalgebraLU<f64>, _>(&problem)
                .unwrap();
        }

        assert_scalar_close(state.as_ref().y[0], 0.4);
        assert_scalar_close(state.as_ref().y[1], 0.4);
        assert_scalar_close(state.as_ref().dy[0], -0.04);
        assert_scalar_close(state.as_ref().dy[1], -0.04);
    }

    #[test]
    fn state_ref_mut_apply_reset_rejects_missing_reset() {
        let problem = scalar_problem(0.25);
        let mut state = TestState::new_without_initialise(&problem).unwrap();

        let err = state
            .as_mut()
            .apply_reset_with_mass::<NalgebraLU<f64>, _>(&problem)
            .unwrap_err();
        assert_other_error(err, "No reset operator configured");
    }

    #[test]
    fn apply_reset_rejects_mass_matrix_problem() {
        let problem = scalar_problem_with_mass(0.25);
        let mut state = TestState::new_without_initialise(&problem).unwrap();

        let err = state.as_mut().apply_reset::<_>(&problem).unwrap_err();
        assert_other_error(err, "apply_reset cannot be used with a mass matrix");
    }

    #[test]
    fn apply_reset_with_sens_rejects_mass_matrix_problem() {
        // Build a minimal sens problem that also has a mass matrix.
        let problem = OdeBuilder::<TestMat>::new()
            .p([1.0])
            .rhs_sens_implicit(
                |x: &[f64], _p, _t, y: &mut [f64]| y[0] = -x[0],
                |_x, _p, _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
                |_x, _p, _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
            )
            .mass(|v: &[f64], _p: &[f64], _t, beta: f64, y: &mut [f64]| {
                for (y, v) in y.iter_mut().zip(v.iter()) {
                    *y = *v + beta * *y;
                }
            })
            .init_sens(
                |_p, _t, y: &mut [f64]| y[0] = 1.0,
                |_p, _t, _v, y: &mut [f64]| y[0] = 0.0,
                1,
            )
            .root_sens_implicit(
                |x: &[f64], _p, _t, y: &mut [f64]| y[0] = x[0] - 0.5,
                |_x, _p, _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
                |_x, _p, _t, _v, y: &mut [f64]| y[0] = 0.0,
                1,
            )
            .reset_sens_implicit(
                |x: &[f64], _p, _t, y: &mut [f64]| y[0] = x[0],
                |_x, _p, _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
                |_x, _p, _t, _v, y: &mut [f64]| y[0] = 0.0,
            )
            .build()
            .unwrap();
        let mut state = TestState::new_with_sensitivities(&problem, 1).unwrap();

        let err = state
            .as_mut()
            .apply_reset_with_sens::<_>(&problem, 0)
            .unwrap_err();
        assert_other_error(
            err,
            "apply_reset_with_sens cannot be used with a mass matrix",
        );
    }

    #[test]
    fn state_ref_mut_apply_reset_with_sens_updates_state_and_sensitivities() {
        let (problem, _soln) = exponential_decay_with_constant_reset_problem_sens::<TestMat>();
        let mut state = TestState::new_with_sensitivities(&problem, 1).unwrap();

        let t_root = 10.0 * f64::ln(5.0 / 3.0);
        {
            let state_mut = state.as_mut();
            *state_mut.t = t_root;
            state_mut.y[0] = 0.6;
            state_mut.y[1] = 0.6;
            state_mut.dy[0] = -0.06;
            state_mut.dy[1] = -0.06;
            state_mut.s.get_batch_mut(0).set_index(0, -t_root * 0.6);
            state_mut.s.get_batch_mut(0).set_index(1, -t_root * 0.6);
            state_mut.s.get_batch_mut(1).set_index(0, 0.6);
            state_mut.s.get_batch_mut(1).set_index(1, 0.6);
        }

        state
            .as_mut()
            .apply_reset_with_sens_mass::<NalgebraLU<f64>, _>(&problem, 0)
            .unwrap();

        for i in 0..2 {
            assert_scalar_close(state.as_ref().y[i], 0.4);
            assert_scalar_close(state.as_ref().dy[i], -0.04);
            assert_scalar_close(
                state.as_ref().s.get_batch(0).get_index(i),
                -4.0 * f64::ln(5.0 / 3.0),
            );
            assert_scalar_close(state.as_ref().s.get_batch(1).get_index(i), 0.4);
        }
    }

    #[test]
    fn apply_reset_with_sens_rejects_invalid_root_index() {
        let (problem, _soln) = exponential_decay_with_reset_problem_sens::<TestMat>();
        let mut state = TestState::new_with_sensitivities(&problem, 1).unwrap();

        let err = state
            .as_mut()
            .apply_reset_with_sens_mass::<NalgebraLU<f64>, _>(&problem, 2)
            .unwrap_err();
        assert_other_error(err, "root index 2 out of bounds");
    }

    #[test]
    fn apply_reset_with_sens_rejects_zero_event_denominator() {
        // Build a problem with lambda=0 (dy=0) and root g(x)=x[0], so root derivative
        // along flow = dg/dx * f + dg/dt = 1*0 + 0 = 0 => zero denominator.
        let problem = OdeBuilder::<TestMat>::new()
            .p([1.0, -2.0])
            .rhs_sens_implicit(
                |x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = 0.0 * x[0],
                |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y[0] = 0.0,
                |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y[0] = 0.0,
            )
            .init_sens(
                |_p: &[f64], _t, y: &mut [f64]| y[0] = 0.0,
                |_p: &[f64], _t, _v: &[f64], y: &mut [f64]| y[0] = 0.0,
                1,
            )
            .root_sens_implicit(
                |x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = x[0],
                |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
                |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y[0] = 0.0,
                1,
            )
            .reset_sens_implicit(
                |x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = x[0],
                |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
                |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y[0] = 0.0,
            )
            .build()
            .unwrap();
        let mut state = TestState::new_with_sensitivities(&problem, 1).unwrap();

        let err = state
            .as_mut()
            .apply_reset_with_sens_mass::<NalgebraLU<f64>, _>(&problem, 0)
            .unwrap_err();
        assert_other_error(err, "active root derivative along flow is zero");
    }

    #[test]
    fn parameterised_op_time_derive_uses_finite_difference() {
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext::default());
        let x = TestVec::from_vec(vec![2.0], crate::NalgebraContext::default());
        let t = 3.0;

        let reset = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &[f64], p: &[f64], t, y: &mut [f64]| {
                y[0] = 1.2 * x[0] + 0.4 * p[0] - 0.3 * p[1] + 0.8 * t
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 1.2 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 0.4 * v[0] - 0.3 * v[1],
            1,
            2,
            1,
            crate::NalgebraContext::default(),
        );
        let reset = ParameterisedOp::new(&reset, &p);

        let root = ClosureWithSens::<TestMat, _, _, _>::new(
            |x: &[f64], p: &[f64], t, y: &mut [f64]| {
                y[0] = 0.5 * x[0] - 0.7 * p[0] + 1.1 * p[1] - 0.2 * t
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 0.5 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -0.7 * v[0] + 1.1 * v[1],
            1,
            2,
            1,
            crate::NalgebraContext::default(),
        );
        let root = ParameterisedOp::new(&root, &p);

        let reset_dt = reset.time_derive(&x, t);
        let root_dt = root.time_derive(&x, t);

        assert_scalar_close_tol(reset_dt[0], 0.8, 1e-8);
        assert_scalar_close_tol(root_dt[0], -0.2, 1e-8);
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_matches_autonomous_formula() {
        let mut problem = scalar_problem_adjoint_with_reset_root(
            0.25,
            |x: &[f64], p: &[f64], _t, y: &mut [f64]| y[0] = 1.5 * x[0] + 0.2 * p[0] - 0.1 * p[1],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 1.5 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -1.5 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -0.2 * v[0];
                y[1] = 0.1 * v[0];
            },
            |_x: &[f64], _p: &[f64], t, y: &mut [f64]| {
                y[0] = 0.3 * t;
                y[1] = 0.0;
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = 3.0 * v[0];
                y[1] = -2.0 * v[0];
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -(3.0 * v[0] - 2.0 * v[1])
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -(0.8 * v[0] + 0.5 * v[1]);
                y[1] = -(-1.5 * v[1]);
            },
            2,
        );
        let forward_problem = scalar_problem(0.25);
        let p = TestVec::from_vec(vec![1.2, -0.7], crate::NalgebraContext::default());
        problem.eqn.set_params(&p);
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

        let y_before = state.as_ref().y[0];
        let dy_before = state.as_ref().dy[0];

        state
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                1,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], y_before);
        assert_scalar_close(state.as_ref().dy[0], dy_before);
        assert_scalar_close(state.as_ref().s.get_batch(0).get_index(0), 0.4965);
        assert_scalar_close(state.as_ref().s.get_batch(1).get_index(0), -0.662);
        assert_scalar_close(state.as_ref().sg.get_batch(0).get_index(0), 0.148375);
        assert_scalar_close(state.as_ref().sg.get_batch(0).get_index(1), -0.195125);
        assert_scalar_close(state.as_ref().sg.get_batch(1).get_index(0), 0.4355);
        assert_scalar_close(state.as_ref().sg.get_batch(1).get_index(1), 0.5935);
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_uses_selected_root_component() {
        let mut problem = scalar_problem_adjoint_with_reset_root(
            0.25,
            |x: &[f64], p: &[f64], _t, y: &mut [f64]| y[0] = 1.5 * x[0] + 0.2 * p[0] - 0.1 * p[1],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 1.5 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -1.5 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -0.2 * v[0];
                y[1] = 0.1 * v[0];
            },
            |_x: &[f64], _p: &[f64], t, y: &mut [f64]| {
                y[0] = 0.3 * t;
                y[1] = 0.0;
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = 3.0 * v[0];
                y[1] = -2.0 * v[0];
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -(3.0 * v[0] - 2.0 * v[1])
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -(0.8 * v[0] + 0.5 * v[1]);
                y[1] = -(-1.5 * v[1]);
            },
            2,
        );
        let forward_problem = scalar_problem(0.25);
        let p = TestVec::from_vec(vec![1.2, -0.7], crate::NalgebraContext::default());
        problem.eqn.set_params(&p);
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

        state_root0
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                0,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap();
        state_root1
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                1,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap();

        assert!(
            (state_root0.as_ref().s.get_batch(0).get_index(0)
                - state_root1.as_ref().s.get_batch(0).get_index(0))
            .abs()
                > 1e-6,
            "different root components should produce different adjoint updates"
        );
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_matches_time_dependent_formula() {
        let mut problem = scalar_problem_adjoint_with_reset_root(
            0.1,
            |x: &[f64], p: &[f64], t, y: &mut [f64]| {
                y[0] = 1.2 * x[0] + 0.4 * p[0] - 0.3 * p[1] + 0.8 * t
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 1.2 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -1.2 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -0.4 * v[0];
                y[1] = 0.3 * v[0];
            },
            |x: &[f64], p: &[f64], t, y: &mut [f64]| {
                y[0] = 0.5 * x[0] - 0.7 * p[0] + 1.1 * p[1] - 0.2 * t
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = 0.5 * v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -(0.5 * v[0]),
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = 0.7 * v[0];
                y[1] = -1.1 * v[0];
            },
            1,
        );
        let forward_problem = scalar_problem(0.1);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext::default());
        problem.eqn.set_params(&p);
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

        let y_before = state.as_ref().y[0];
        let dy_before = state.as_ref().dy[0];

        state
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                0,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap();

        assert_scalar_close(state.as_ref().y[0], y_before);
        assert_scalar_close(state.as_ref().dy[0], dy_before);
        assert_scalar_close_tol(state.as_ref().s.get_batch(0).get_index(0), 0.7, 1e-8);
        assert_scalar_close_tol(state.as_ref().s.get_batch(1).get_index(0), -0.35, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg.get_batch(0).get_index(0), -0.264, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg.get_batch(0).get_index(1), 0.552, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg.get_batch(1).get_index(0), 0.782, 1e-8);
        assert_scalar_close_tol(state.as_ref().sg.get_batch(1).get_index(1), -0.276, 1e-8);
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_rejects_invalid_root_index() {
        let mut problem = scalar_problem_adjoint_with_reset_root(
            0.25,
            |x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = x[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
            |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y.fill(0.0),
            |_x: &[f64], _p: &[f64], _t, y: &mut [f64]| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| {
                y[0] = 1.0;
                y[1] = 1.0;
            },
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| {
                y[0] = -(v[0] + v[1]);
            },
            |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y.fill(0.0),
            2,
        );
        let forward_problem = scalar_problem(0.25);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext::default());
        problem.eqn.set_params(&p);
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

        let err = state
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                2,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap_err();
        assert_other_error(err, "root index 2 out of bounds");
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_rejects_zero_event_denominator() {
        let mut problem = scalar_problem_adjoint_with_reset_root(
            0.0,
            |x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = x[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
            |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y.fill(0.0),
            |_x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = 0.0,
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
            |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
            |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y.fill(0.0),
            1,
        );
        let forward_problem = scalar_problem(0.0);
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext::default());
        problem.eqn.set_params(&p);
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

        let err = state
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                0,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap_err();
        assert_other_error(err, "active root derivative along flow is zero");
    }

    #[test]
    fn state_mut_op_with_adjoint_and_reset_rejects_mass_matrix_equations() {
        let mut problem = OdeBuilder::<TestMat>::new()
            .p([1.0, -2.0])
            .integrate_out(true)
            .rhs_adjoint_implicit(
                move |x, _p, _t, y| y[0] = 0.25 * x[0],
                move |_x, _p, _t, v, y| y[0] = 0.25 * v[0],
                move |_x, _p, _t, v, y| y[0] = -0.25 * v[0],
                |_x, _p, _t, _v, y| y.fill(0.0),
            )
            .mass_adjoint(
                |v: &[f64], _p: &[f64], _t, beta: f64, y: &mut [f64]| {
                    for (y, v) in y.iter_mut().zip(v.iter()) {
                        *y = *v + beta * *y;
                    }
                },
                |v: &[f64], _p: &[f64], _t, beta: f64, y: &mut [f64]| {
                    for (y, v) in y.iter_mut().zip(v.iter()) {
                        *y = *v + beta * *y;
                    }
                },
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
            .root_adjoint_implicit(
                |_x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = 0.0,
                |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
                |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
                |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y.fill(0.0),
                1,
            )
            .reset_adjoint_implicit(
                |x: &[f64], _p: &[f64], _t, y: &mut [f64]| y[0] = x[0],
                |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = v[0],
                |_x: &[f64], _p: &[f64], _t, v: &[f64], y: &mut [f64]| y[0] = -v[0],
                |_x: &[f64], _p: &[f64], _t, _v: &[f64], y: &mut [f64]| y.fill(0.0),
            )
            .build()
            .unwrap();
        let p = TestVec::from_vec(vec![1.0, -2.0], crate::NalgebraContext::default());
        problem.eqn.set_params(&p);
        let common = StateCommon {
            y: TestVec::zeros(1, crate::NalgebraContext::default()),
            dy: TestVec::zeros(1, crate::NalgebraContext::default()),
            g: TestVec::zeros(0, crate::NalgebraContext::default()),
            dg: TestVec::zeros(0, crate::NalgebraContext::default()),
            s: TestVec::zeros(1, crate::NalgebraContext::default()),
            ds: TestVec::zeros(1, crate::NalgebraContext::default()),
            sg: TestVec::zeros(2, crate::NalgebraContext::default()),
            dsg: TestVec::zeros(2, crate::NalgebraContext::default()),
            t: 0.0,
            h: 0.0,
        };
        let mut state = TestState::new_from_common(common);
        let fwd_state_minus = state.clone();
        let fwd_state_plus = state.clone();

        let err = state
            .as_mut()
            .apply_reset_with_adjoint(
                &problem.eqn,
                0,
                fwd_state_minus.as_ref(),
                fwd_state_plus.as_ref(),
                problem.integrate_out,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::MassMatrixNotSupported)
        ));
    }
}
