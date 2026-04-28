use std::{cell::Ref, marker::PhantomData};

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_equations::OdeEquationsImplicitSens,
    ode_solver::solution::{SolutionMode, INITIAL_NCOLS},
    ode_solver_error,
    scalar::Scalar,
    AugmentedOdeEquations, Checkpointing, CheckpointingPath, Context, DefaultDenseMatrix,
    DenseMatrix, HermiteInterpolator, MatrixCommon, NonLinearOp, OdeEquations, OdeSolverConfig,
    OdeSolverProblem, OdeSolverState, Op, Solution, StateRef, StateRefMut, Vector, VectorViewMut,
};
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum OdeSolverStopReason<T: Scalar> {
    InternalTimestep,
    RootFound(T, usize),
    TstopReached,
}

/// Trait for ODE solver methods. This is the main user interface for the ODE solvers.
///
/// The solver is responsible for stepping the solution (given in the `OdeSolverState`), and interpolating the solution at a given time.
/// However, the solver does not own the state, so the user is responsible for creating and managing the state. If the user
/// wants to change the state, they should call `set_problem` again.
///
/// # Example
///
/// ```
/// use diffsol::{ OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeEquationsImplicit, DefaultSolver };
///
/// fn solve_ode<'a, Eqn>(solver: &mut impl OdeSolverMethod<'a, Eqn>, t: Eqn::T) -> Eqn::V
/// where
///    Eqn: OdeEquationsImplicit + 'a,
///    Eqn::M: DefaultSolver,
/// {
///     while solver.state().t <= t {
///         solver.step().unwrap();
///     }
///     solver.interpolate(t).unwrap()
/// }
/// ```
pub trait OdeSolverMethod<'a, Eqn: OdeEquations>: Clone
where
    Self: Sized,
    Eqn: 'a,
{
    /// The state type used by the solver
    type State: OdeSolverState<Eqn::V>;

    /// The configuration type used by the solver
    type Config: OdeSolverConfig<Eqn::T>;

    /// Get the current problem
    fn problem(&self) -> &'a OdeSolverProblem<Eqn>;

    /// Take a checkpoint of the current state of the solver, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem but want to keep this solver active. If you don't need to use this solver again, you can use `take_state` instead.
    /// Note that this will force a reinitialisation of the internal Jacobian for the solver, if it has one.
    fn checkpoint(&mut self) -> Self::State;

    /// Clone the current state of the solver without triggering any internal Jacobian reset.
    fn state_clone(&self) -> Self::State;

    /// Replace the current state of the solver with a new state.
    fn set_state(&mut self, state: Self::State);

    /// Take the current state of the solver, if it exists, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem. Note that this will unset the current problem and solver state, so you will need to call
    /// `set_problem` again before calling `step` or `solve`.
    fn into_state(self) -> Self::State;

    /// Get the current state of the solver
    fn state(&self) -> StateRef<'_, Eqn::V>;

    /// Get a mutable reference to the current state of the solver
    /// Note that calling this will cause the next call to `step` to perform some reinitialisation to take into
    /// account the mutated state, this could be expensive for multi-step methods.
    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V>;

    /// Get a reference to the current configuration of the solver
    fn config(&self) -> &Self::Config;

    /// Get a mutable reference to the current configuration of the solver
    fn config_mut(&mut self) -> &mut Self::Config;

    /// Returns the current jacobian matrix of the solver, if it has one
    /// Note that this will force a full recalculation of the Jacobian.
    fn jacobian(&self) -> Option<Ref<'_, Eqn::M>>;

    /// Returns the current mass matrix of the solver, if it has one
    /// Note that this will force a full recalculation of the mass matrix.
    fn mass(&self) -> Option<Ref<'_, Eqn::M>>;

    /// Step the solution forward by one step, altering the internal state of the solver.
    /// The return value is a `Result` containing the reason for stopping the solver, possible reasons are:
    /// - `InternalTimestep`: The solver has taken a step forward in time, the internal state of the solver is at time self.state().t
    /// - `RootFound(t_root)`: The solver has found a root at time `t_root`. Note that the internal state of the solver is at the internal time step `self.state().t`, *not* at time `t_root`.
    /// - `TstopReached`: The solver has reached the stop time set by [Self::set_stop_time], the internal state of the solver is at time `tstop`, which is the same as `self.state().t`
    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError>;

    /// Set a stop time for the solver. The solver will stop when the internal time reaches this time.
    /// Once it stops, the stop time is unset. If `tstop` is at or before the current internal time, an error is returned.
    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError>;

    /// Interpolate the solution at a given time. This time should be between the current time and the last solver time step
    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        let nstates = self.problem().eqn.rhs().nstates();
        let mut y = Eqn::V::zeros(nstates, self.problem().context().clone());
        self.interpolate_inplace(t, &mut y)?;
        Ok(y)
    }

    /// Interpolate the solution at a given time and place in `y`. This time should be between the current time and the last solver time step
    fn interpolate_inplace(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError>;

    /// Interpolate the time derivative dy/dt at a given time. This time should be between the current time and the last solver time step
    fn interpolate_dy(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        let nstates = self.problem().eqn.rhs().nstates();
        let mut dy = Eqn::V::zeros(nstates, self.problem().context().clone());
        self.interpolate_dy_inplace(t, &mut dy)?;
        Ok(dy)
    }

    /// Interpolate the time derivative dy/dt at a given time and place in `dy`. This time should be between the current time and the last solver time step
    fn interpolate_dy_inplace(&self, t: Eqn::T, dy: &mut Eqn::V) -> Result<(), DiffsolError>;

    /// Interpolate the integral of the output function at a given time. This time should be between the current time and the last solver time step
    fn interpolate_out(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        let nout = if let Some(out) = self.problem().eqn.out() {
            out.nout()
        } else {
            self.problem().eqn.rhs().nstates()
        };
        let mut g = Eqn::V::zeros(nout, self.problem().context().clone());
        self.interpolate_out_inplace(t, &mut g)?;
        Ok(g)
    }

    /// Interpolate the integral of the output function at a given time and place in `g`. This time should be between the current time and the last solver time step
    fn interpolate_out_inplace(&self, t: Eqn::T, g: &mut Eqn::V) -> Result<(), DiffsolError>;

    /// Interpolate the sensitivity vectors at a given time. This time should be between the current time and the last solver time step
    fn interpolate_sens(&self, t: Eqn::T) -> Result<Vec<Eqn::V>, DiffsolError> {
        let nsens = self.state().s.len();
        if nsens == 0 {
            return Ok(Vec::new());
        }
        let mut sens = Vec::with_capacity(nsens);
        for _ in 0..nsens {
            sens.push(Eqn::V::zeros(
                self.problem().eqn.rhs().nstates(),
                self.problem().context().clone(),
            ));
        }
        self.interpolate_sens_inplace(t, &mut sens)?;
        Ok(sens)
    }

    /// Interpolate the sensitivity vectors at a given time and place in `sens`. This time should be between the current time and the last solver time step
    fn interpolate_sens_inplace(&self, t: Eqn::T, sens: &mut [Eqn::V]) -> Result<(), DiffsolError>;

    /// Move the solver state back to time `t` by interpolating `y`, `dy`, and (if
    /// `integrate_out` is set) `g` to that time and writing them into the current state.
    /// If the state contains sensitivity vectors they are also interpolated to time `t`.
    /// This is typically called after a root is found to pin the state to the root time.
    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError>;

    /// Apply the problem's configured reset operator to the current state.
    ///
    /// This is typically used after [`Self::state_mut_back`] has moved the solver to a root time.
    /// The helper recomputes `dy` from the problem RHS after updating the state vector.
    fn apply_reset(&mut self) -> Result<(), DiffsolError> {
        let (rhs, has_mass, reset) = {
            let eqn = &self.problem().eqn;
            (
                eqn.rhs(),
                eqn.mass().is_some(),
                eqn.reset().ok_or_else(|| {
                    ode_solver_error!(Other, "No reset operator configured for this problem")
                })?,
            )
        };
        self.state_mut().state_mut_op(&rhs, has_mass, &reset)
    }

    /// Apply the problem's configured reset operator to the current state and
    /// propagate forward sensitivities through the active root event.
    ///
    /// This is typically used after [`Self::state_mut_back`] has moved the solver to a root time.
    /// The helper recomputes `dy` from the problem RHS after updating the state vector and
    /// applies the root-time correction to each sensitivity vector.
    fn apply_reset_with_sens(&mut self, root_idx: usize) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let (rhs, has_mass, reset, root) = {
            let eqn = &self.problem().eqn;
            (
                eqn.rhs(),
                eqn.mass().is_some(),
                eqn.reset().ok_or_else(|| {
                    ode_solver_error!(Other, "No reset operator configured for this problem")
                })?,
                eqn.root().ok_or_else(|| {
                    ode_solver_error!(Other, "No root operator configured for this problem")
                })?,
            )
        };
        self.state_mut()
            .state_mut_op_with_sens_and_reset(&rhs, has_mass, &reset, &root, root_idx)
    }

    /// Get the current order of accuracy of the solver (e.g. explict euler method is first-order)
    fn order(&self) -> usize;

    /// Solve the ODE from the current time to `final_time`.
    ///
    /// This method integrates the system and returns the solution at adaptive timepoints chosen by the solver's
    /// internal error control mechanism. This is useful when you want the minimal number of timepoints for a given accuracy.
    ///
    /// If a root function is provided, the solver will stop if any of the root function elements change sign.
    /// The internal state of the solver is set to the time that the zero-crossing occured.
    /// If both a root function and a reset operator are configured, roots are handled internally by
    /// applying the reset and continuing the integration to `final_time`.
    ///
    /// # Arguments
    /// - `final_time`: The time to integrate to
    ///
    /// # Returns
    /// A tuple of `(solution_matrix, times, stop_reason)` where:
    /// - `solution_matrix` is a dense matrix with one column per solution time and one row per state variable
    /// - `times` is a vector of times at which the solution was evaluated
    /// - `stop_reason` indicates whether the solve reached `final_time` or stopped on a root
    ///
    /// # Post-condition
    /// After the solver finishes, the internal state of the solver is at time `final_time`.
    /// If a root is found and no reset operator is configured, the solver stops early. The
    /// internal state is moved to the root time, and the root time/value are returned as the
    /// last entry.
    /// If a reset operator is configured, the solver writes out the reset state at the root time
    /// and continues integrating.
    #[allow(clippy::type_complexity)]
    fn solve(
        &mut self,
        final_time: Eqn::T,
    ) -> Result<
        (
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<Eqn::T>,
            OdeSolverStopReason<Eqn::T>,
        ),
        DiffsolError,
    >
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let mut ret_t = Vec::new();
        let (mut ret_y, mut tmp_nout) = allocate_return(self)?;
        let (stop_reason, _) = solve(
            &mut ret_y,
            &mut ret_t,
            &mut tmp_nout,
            self,
            final_time,
            true,
            false,
            0,
        )?;
        let ntimes = ret_t.len();
        ret_y.resize_cols(ntimes);
        Ok((ret_y, ret_t, stop_reason))
    }

    /// Continue solving into an existing [`Solution`], appending newly computed output.
    ///
    /// This method is intended for multi-stage integrations where the caller may need to
    /// inspect [`Solution::stop_reason`], mutate the equations or state, and then resume the
    /// solve by calling `solve_soln` again with the returned solver state.
    ///
    /// The behavior depends on how the [`Solution`] was created:
    /// - If created with [`Solution::new`], this method behaves like [`Self::solve`], appending
    ///   solution values at the solver's adaptive internal timesteps until the target time is
    ///   reached or a root is found.
    /// - If created with [`Solution::new_dense`], this method behaves like [`Self::solve_dense`],
    ///   filling the remaining evaluation points and updating internal tracking to resume at the
    ///   next point on a subsequent call.
    ///
    /// On return, `soln.stop_reason` contains the reason this stage stopped. If a root is found,
    /// the solver state is moved back to the root time in the same way as [`Self::solve`] and
    /// [`Self::solve_dense`], so the caller can apply resets or parameter changes before resuming.
    ///
    /// # Example
    /// ```
    /// use diffsol::{
    ///     OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverStopReason, Solution,
    ///     NalgebraLU, NalgebraMat, NalgebraVec, Vector,
    /// };
    /// type M = NalgebraMat<f64>;
    /// type V = NalgebraVec<f64>;
    /// type LS = NalgebraLU<f64>;
    ///
    /// let mut problem = OdeBuilder::<M>::new()
    ///     .p([0.1])
    ///     .rhs_implicit(
    ///         |x, p, _t, y| { y[0] = -p[0] * x[0]; },
    ///         |_x, p, _t, v, y| { y[0] = -p[0] * v[0]; },
    ///     )
    ///     .root(|x, _p, _t, y| { y[0] = x[0] - 0.1; }, 1)
    ///     .init(|_p, _t, y| { y[0] = 1.0; }, 1)
    ///     .build()
    ///     .unwrap();
    ///
    /// let t_final = 10.0_f64;
    /// let mut state = problem.bdf_state::<LS>().unwrap();
    /// let mut soln = Solution::<V>::new(t_final);
    ///
    /// while !soln.is_complete() {
    ///     state = problem.bdf_solver::<LS>(state).unwrap()
    ///         .solve_soln(&mut soln).unwrap()
    ///         .into_state();
    ///     if let Some(OdeSolverStopReason::RootFound(_, _)) = &soln.stop_reason {
    ///         let p_new = V::from_vec(vec![0.2], *problem.context());
    ///         problem.eqn_mut().set_params(&p_new);
    ///     }
    /// }
    /// ```
    ///
    /// # Returns
    /// The updated solver, allowing ownership of the solver state to be threaded through repeated
    /// calls to `solve_soln`.
    fn solve_soln(mut self, soln: &mut Solution<Eqn::V>) -> Result<Self, DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let nrows = if let Some(out) = self.problem().eqn.out() {
            out.nout()
        } else {
            self.problem().eqn.rhs().nstates()
        };
        let nout = self.problem().eqn.out().map(|out| out.nout()).unwrap_or(0);
        let nstates = self.problem().eqn.rhs().nstates();
        soln.ensure_ode_allocation(self.problem().context(), nrows, nout, nstates)?;

        match soln.mode {
            SolutionMode::Tfinal(t_final) => {
                let (stop_reason, _) = solve(
                    &mut soln.ys,
                    &mut soln.ts,
                    &mut soln.tmp_nout,
                    &mut self,
                    t_final,
                    false,
                    false,
                    0,
                )?;
                soln.stop_reason = Some(stop_reason);
            }
            SolutionMode::Tevals(start_col) => {
                let (stop_reason, col, _) = solve_dense(
                    &mut soln.ys,
                    &soln.ts,
                    &mut soln.tmp_nout,
                    &mut soln.tmp_nstates,
                    &mut self,
                    start_col,
                    false,
                    false,
                    0,
                )?;
                soln.stop_reason = Some(stop_reason);
                soln.mode = SolutionMode::Tevals(col);
            }
        }
        Ok(self)
    }

    /// Continue solving into an existing [`Solution`], appending newly computed
    /// output and checkpointing segments.
    ///
    /// This has the same staged-solve behavior as [`Self::solve_soln`], but also
    /// records a checkpointing segment for the call and appends it to the
    /// caller-owned checkpointing path. A segment with
    /// `terminal_reset_root_idx() == Some(root_idx)` ended at a root. A segment
    /// with `None` ended at the configured stop time.
    fn solve_soln_with_checkpointing(
        mut self,
        soln: &mut Solution<Eqn::V>,
        checkpointing: &mut CheckpointingPath<Eqn, Self::State>,
        max_steps_between_checkpoints: Option<usize>,
    ) -> Result<Self, DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let nrows = if let Some(out) = self.problem().eqn.out() {
            out.nout()
        } else {
            self.problem().eqn.rhs().nstates()
        };
        let nout = self.problem().eqn.out().map(|out| out.nout()).unwrap_or(0);
        let nstates = self.problem().eqn.rhs().nstates();
        soln.ensure_ode_allocation(self.problem().context(), nrows, nout, nstates)?;
        let max_steps_between_checkpoints = max_steps_between_checkpoints.unwrap_or(500);

        let mut new_checkpointing = match soln.mode {
            SolutionMode::Tfinal(t_final) => {
                let (stop_reason, checkpointers) = solve(
                    &mut soln.ys,
                    &mut soln.ts,
                    &mut soln.tmp_nout,
                    &mut self,
                    t_final,
                    false,
                    true,
                    max_steps_between_checkpoints,
                )?;
                soln.stop_reason = Some(stop_reason);
                checkpointers.unwrap()
            }
            SolutionMode::Tevals(start_col) => {
                let (stop_reason, col, checkpointers) = solve_dense(
                    &mut soln.ys,
                    &soln.ts,
                    &mut soln.tmp_nout,
                    &mut soln.tmp_nstates,
                    &mut self,
                    start_col,
                    false,
                    true,
                    max_steps_between_checkpoints,
                )?;
                soln.stop_reason = Some(stop_reason);
                soln.mode = SolutionMode::Tevals(col);
                checkpointers.unwrap()
            }
        };
        checkpointing.append(&mut new_checkpointing);
        Ok(self)
    }

    /// Solve the ODE from the current time to `t_eval[t_eval.len()-1]`, evaluating at specified times.
    ///
    /// This method integrates the system and returns the solution interpolated at the specified times.
    /// The solver uses its own internal timesteps for accuracy, but the output is interpolated to the
    /// requested evaluation times. This is useful when you need the solution at specific timepoints
    /// and want the solver's adaptive stepping for accuracy.
    ///
    /// If a root function is provided, the solver will stop if any of the root function elements change sign.
    /// The internal state of the solver is set to the time that the zero-crossing occured.
    /// If both a root function and a reset operator are configured, roots are handled internally by
    /// applying the reset and continuing the integration until `t_eval[t_eval.len()-1]`.
    ///
    /// # Arguments
    /// - `t_eval`: A slice of times at which to evaluate the solution. Times should be in increasing order.
    ///
    /// # Returns
    /// A tuple of `(solution_matrix, stop_reason)` where:
    /// - `solution_matrix` has one column per evaluation time (in the same order as `t_eval`) and one row per state variable,
    ///   plus one final column at the root time if a root fires before `t_eval` is exhausted and no reset operator is configured.
    /// - `stop_reason` indicates whether the solve reached `t_eval[t_eval.len()-1]` or stopped on a root.
    ///
    /// # Post-condition
    /// In the case that no roots are found that stop the solve early, the internal state is at time `t_eval[t_eval.len()-1]`.
    /// If a root is found and no reset operator is configured, the solver stops early. The internal
    /// state is moved to the root time, and the last column corresponds to the root time (which may
    /// not be in `t_eval`).
    /// If a reset operator is configured, any `t_eval` values exactly at the root time receive the
    /// reset state and the solve continues through the remaining evaluation times.
    #[allow(clippy::type_complexity)]
    fn solve_dense(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<
        (
            <Eqn::V as DefaultDenseMatrix>::M,
            OdeSolverStopReason<Eqn::T>,
        ),
        DiffsolError,
    >
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let (mut ret, mut tmp_nout, mut tmp_nstates) = dense_allocate_return(self, t_eval)?;
        let (stop_reason, col, _) = solve_dense(
            &mut ret,
            t_eval,
            &mut tmp_nout,
            &mut tmp_nstates,
            self,
            0,
            true,
            false,
            0,
        )?;

        // if we stopped on a root before exhausting t_eval, we need to write_out the solution at the root time to the last column of ret
        if let OdeSolverStopReason::RootFound(_, _) = stop_reason {
            if col < t_eval.len() {
                write_state_out(self.problem(), &self.state(), &mut ret, col, &mut tmp_nout);
                if col + 1 < ret.ncols() {
                    ret.resize_cols(col + 1);
                }
            }
        }
        Ok((ret, stop_reason))
    }

    /// Solve the ODE from the current time to `final_time`, saving checkpoints at regular intervals.
    ///
    /// This method is useful for adjoint sensitivity analysis, where you need to store the solution at
    /// intermediate times to efficiently compute gradients.
    ///
    /// # Arguments
    /// - `final_time`: The time to integrate to
    /// - `max_steps_between_checkpoints`: The maximum number of solver steps to take between saving checkpoints (if `None`, defaults to 500)
    ///
    /// # Returns
    /// A tuple of `(checkpointers, output_matrix, output_times, stop_reason)` where:
    /// - `checkpointers` is a path of checkpointing segments, split at reset events, for adjoint integrations
    /// - `output_matrix` a dense matrix containing the solution at each output time
    /// - `output_times` a vector of timepoints corresponding to the columns of `output_matrix`
    /// - `stop_reason` is the reason the solve terminated
    #[allow(clippy::type_complexity)]
    fn solve_with_checkpointing(
        &mut self,
        final_time: Eqn::T,
        max_steps_between_checkpoints: Option<usize>,
    ) -> Result<
        (
            CheckpointingPath<Eqn, Self::State>,
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<Eqn::T>,
            OdeSolverStopReason<Eqn::T>,
        ),
        DiffsolError,
    >
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let mut ret_t = Vec::new();
        let (mut ret_y, mut tmp_nout) = allocate_return(self)?;
        let max_steps_between_checkpoints = max_steps_between_checkpoints.unwrap_or(500);
        let (stop_reason, checkpointers) = solve(
            &mut ret_y,
            &mut ret_t,
            &mut tmp_nout,
            self,
            final_time,
            true,
            true,
            max_steps_between_checkpoints,
        )?;
        let ntimes = ret_t.len();
        ret_y.resize_cols(ntimes);

        Ok((checkpointers.unwrap(), ret_y, ret_t, stop_reason))
    }

    /// Solve the ODE from the current time to `t_eval[t_eval.len()-1]` with checkpointing, evaluating at specified times.
    ///
    /// This method is similar to [Self::solve_dense] but additionally saves checkpoints of the solver state
    /// at regular intervals. Checkpointing enables efficient adjoint sensitivity analysis by storing the
    /// forward integration state, allowing backward integration to compute gradients without recomputing
    /// the entire forward solution.
    ///
    /// # Arguments
    /// - `t_eval`: A slice of times at which to evaluate the solution. Times should be in increasing order.
    /// - `max_steps_between_checkpoints`: The maximum number of solver steps to take between saving checkpoints.
    ///   If `None`, defaults to 500.
    ///
    /// # Returns
    /// A tuple of `(checkpointers, solution_matrix, stop_reason)` where:
    /// - `checkpointers` is a path of checkpointing segments, split at reset events, storing the forward integration state for use in adjoint integrations
    /// - `solution_matrix` is a dense matrix with one column per evaluation time and one row per state variable
    /// - `stop_reason` is the reason the solve terminated
    ///
    /// # Post-condition
    /// After the solver finishes, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    ///
    /// # See also
    /// - [Self::solve_dense] for a similar method without checkpointing
    /// - [Self::solve_with_checkpointing] for checkpointing with adaptive output times
    #[allow(clippy::type_complexity)]
    fn solve_dense_with_checkpointing(
        &mut self,
        t_eval: &[Eqn::T],
        max_steps_between_checkpoints: Option<usize>,
    ) -> Result<
        (
            CheckpointingPath<Eqn, Self::State>,
            <Eqn::V as DefaultDenseMatrix>::M,
            OdeSolverStopReason<Eqn::T>,
        ),
        DiffsolError,
    >
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let (mut ret, mut tmp_nout, mut tmp_nstates) = dense_allocate_return(self, t_eval)?;
        let max_steps_between_checkpoints = max_steps_between_checkpoints.unwrap_or(500);
        let (stop_reason, _, checkpointers) = solve_dense(
            &mut ret,
            t_eval,
            &mut tmp_nout,
            &mut tmp_nstates,
            self,
            0,
            true,
            true,
            max_steps_between_checkpoints,
        )?;

        Ok((checkpointers.unwrap(), ret, stop_reason))
    }
}

struct CheckpointingRecorder<Eqn, State, Method>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
{
    checkpointers: CheckpointingPath<Eqn, State>,
    checkpoints: Vec<State>,
    ts: Vec<Eqn::T>,
    ys: Vec<Eqn::V>,
    ydots: Vec<Eqn::V>,
    phantom: PhantomData<Method>,
}

impl<'a, Eqn, Method> CheckpointingRecorder<Eqn, Method::State, Method>
where
    Eqn: OdeEquations + 'a,
    Eqn::V: DefaultDenseMatrix,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn new(solver: &Method) -> Self {
        Self {
            checkpointers: Vec::new(),
            checkpoints: vec![solver.state_clone()],
            ts: vec![solver.state().t],
            ys: vec![solver.state().y.clone()],
            ydots: vec![solver.state().dy.clone()],
            phantom: PhantomData,
        }
    }

    fn into_checkpointers(self) -> CheckpointingPath<Eqn, Method::State> {
        self.checkpointers
    }

    fn record_sample(&mut self, solver: &Method) {
        self.ts.push(solver.state().t);
        self.ys.push(solver.state().y.clone());
        self.ydots.push(solver.state().dy.clone());
    }

    fn reset_samples_from_current(&mut self, solver: &Method) {
        self.ts.clear();
        self.ys.clear();
        self.ydots.clear();
        self.record_sample(solver);
    }

    fn start_segment_from_current(&mut self, solver: &Method) {
        self.checkpoints.clear();
        self.checkpoints.push(solver.state_clone());
        self.reset_samples_from_current(solver);
    }

    fn checkpoint_and_restart_samples(&mut self, solver: &mut Method) {
        self.checkpoints.push(solver.checkpoint());
        self.reset_samples_from_current(solver);
    }

    fn record_terminal_state(&mut self, solver: &Method) {
        self.record_sample(solver);
        self.checkpoints.push(solver.state_clone());
    }

    fn finish_segment(&mut self, terminal_reset_root_idx: Option<usize>) {
        let last_segment = HermiteInterpolator::new(
            self.ys.split_off(0),
            self.ydots.split_off(0),
            self.ts.split_off(0),
        );
        let mut checkpointer = Checkpointing::new::<Method>(
            None,
            self.checkpoints.len() - 2,
            self.checkpoints.split_off(0),
            Some(last_segment),
        );
        if let Some(root_idx) = terminal_reset_root_idx {
            checkpointer.set_terminal_reset_root_idx(root_idx);
        }
        self.checkpointers.push(checkpointer);
    }

    fn finish_terminal_segment(&mut self, solver: &Method) {
        self.record_terminal_state(solver);
        self.finish_segment(None);
    }
}

pub trait AugmentedOdeSolverMethod<'a, Eqn, AugmentedEqn>: OdeSolverMethod<'a, Eqn>
where
    Eqn: OdeEquations + 'a,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugmentedEqn>);
    fn augmented_eqn(&self) -> Option<&AugmentedEqn>;
    fn augmented_eqn_mut(&mut self) -> Option<&mut AugmentedEqn>;
    fn state_and_augmented_eqn_mut(
        &mut self,
    ) -> Option<(StateRefMut<'_, Eqn::V>, &mut AugmentedEqn)>;
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn solve_dense<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    ret: &mut <Eqn::V as DefaultDenseMatrix>::M,
    t_eval: &[Eqn::T],
    tmp_nout: &mut Eqn::V,
    tmp_nstates: &mut Eqn::V,
    s: &mut S,
    start_col: usize,
    continue_after_reset: bool,
    checkpointing: bool,
    max_steps_between_checkpoints: usize,
) -> Result<
    (
        OdeSolverStopReason<Eqn::T>,
        usize,
        Option<CheckpointingPath<Eqn, S::State>>,
    ),
    DiffsolError,
>
where
    Eqn::V: DefaultDenseMatrix,
{
    let final_time = t_eval[t_eval.len() - 1];
    s.set_stop_time(final_time)?;
    let has_reset = continue_after_reset && s.problem().eqn.reset().is_some();
    let mut checkpointing = checkpointing.then(|| CheckpointingRecorder::new(&*s));
    let mut nsteps = 0;
    let mut stop_reason: OdeSolverStopReason<Eqn::T>;
    let mut col = start_col;
    loop {
        stop_reason = s.step()?;
        match stop_reason {
            OdeSolverStopReason::InternalTimestep => {
                if let Some(checkpointing) = checkpointing.as_mut() {
                    checkpointing.record_sample(&*s);
                    nsteps += 1;
                    if nsteps > max_steps_between_checkpoints {
                        checkpointing.checkpoint_and_restart_samples(s);
                        nsteps = 0;
                    }
                }
                while col < t_eval.len() && t_eval[col] <= s.state().t {
                    dense_write_out(s, ret, t_eval[col], col, tmp_nout, tmp_nstates)?;
                    col += 1;
                }
            }
            OdeSolverStopReason::TstopReached => {
                while col < t_eval.len() && t_eval[col] <= s.state().t {
                    dense_write_out(s, ret, t_eval[col], col, tmp_nout, tmp_nstates)?;
                    col += 1;
                }
                assert!(col == t_eval.len(), "Solver reached stop time before consuming all t_eval points, this should not happen");
                break;
            }
            OdeSolverStopReason::RootFound(t_root, root_idx) => {
                while col < t_eval.len() && t_eval[col] <= t_root {
                    dense_write_out(s, ret, t_eval[col], col, tmp_nout, tmp_nstates)?;
                    col += 1;
                }
                s.state_mut_back(t_root)?;
                if let Some(checkpointing) = checkpointing.as_mut() {
                    checkpointing.record_terminal_state(&*s);
                }
                if has_reset {
                    if let Some(checkpointing) = checkpointing.as_mut() {
                        checkpointing.finish_segment(Some(root_idx));
                    }
                    s.apply_reset()?;
                    if let Some(checkpointing) = checkpointing.as_mut() {
                        checkpointing.start_segment_from_current(&*s);
                        nsteps = 0;
                    }
                    if s.state().t < final_time {
                        s.set_stop_time(final_time)?;
                    } else {
                        stop_reason = OdeSolverStopReason::TstopReached;
                        break;
                    }
                } else {
                    if let Some(checkpointing) = checkpointing.as_mut() {
                        checkpointing.finish_segment(Some(root_idx));
                    }
                    stop_reason = OdeSolverStopReason::RootFound(t_root, root_idx);
                    break;
                }
            }
        }
    }
    if !matches!(stop_reason, OdeSolverStopReason::RootFound(_, _)) {
        if let Some(checkpointing) = checkpointing.as_mut() {
            checkpointing.finish_terminal_segment(&*s);
        }
    }
    Ok((
        stop_reason,
        col,
        checkpointing.map(CheckpointingRecorder::into_checkpointers),
    ))
}

/// Utility function to write out the solution at a given timepoint
/// This function is used by the `solve_dense` method to write out the solution at a given timepoint.
fn dense_write_out<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    s: &S,
    y_out: &mut <Eqn::V as DefaultDenseMatrix>::M,
    t: Eqn::T,
    i: usize,
    tmp_nout: &mut Eqn::V,
    tmp_nstates: &mut Eqn::V,
) -> Result<(), DiffsolError>
where
    Eqn::V: DefaultDenseMatrix,
{
    let mut y_out = y_out.column_mut(i);
    if s.problem().integrate_out {
        s.interpolate_out_inplace(t, tmp_nout)?;
        y_out.copy_from(tmp_nout);
    } else {
        s.interpolate_inplace(t, tmp_nstates)?;
        match s.problem().eqn.out() {
            Some(out) => {
                out.call_inplace(tmp_nstates, t, tmp_nout);
                y_out.copy_from(tmp_nout)
            }
            None => y_out.copy_from(tmp_nstates),
        }
    }
    Ok(())
}

pub(crate) fn write_state_out<Eqn>(
    problem: &OdeSolverProblem<Eqn>,
    state: &StateRef<'_, Eqn::V>,
    y_out: &mut <Eqn::V as DefaultDenseMatrix>::M,
    col: usize,
    tmp_nout: &mut Eqn::V,
) where
    Eqn: OdeEquations,
    Eqn::V: DefaultDenseMatrix,
{
    let mut y_out_col = y_out.column_mut(col);
    match problem.eqn.out() {
        Some(out) => {
            if problem.integrate_out {
                y_out_col.copy_from(state.g);
            } else {
                out.call_inplace(state.y, state.t, tmp_nout);
                y_out_col.copy_from(tmp_nout);
            }
        }
        None => {
            if problem.integrate_out {
                y_out_col.copy_from(state.g);
            } else {
                y_out_col.copy_from(state.y);
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn solve<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    ret_y: &mut <Eqn::V as DefaultDenseMatrix>::M,
    ret_t: &mut Vec<Eqn::T>,
    tmp_nout: &mut Eqn::V,
    s: &mut S,
    final_time: Eqn::T,
    continue_after_reset: bool,
    checkpointing: bool,
    max_steps_between_checkpoints: usize,
) -> Result<
    (
        OdeSolverStopReason<Eqn::T>,
        Option<CheckpointingPath<Eqn, S::State>>,
    ),
    DiffsolError,
>
where
    Eqn::V: DefaultDenseMatrix,
{
    // do the main loop
    write_out(s, ret_y, ret_t, tmp_nout);
    s.set_stop_time(final_time)?;
    let has_reset = continue_after_reset && s.problem().eqn.reset().is_some();
    let mut checkpointing = checkpointing.then(|| CheckpointingRecorder::new(&*s));
    let mut nsteps = 0;
    let stop_reason = loop {
        match s.step()? {
            OdeSolverStopReason::InternalTimestep => {
                write_out(s, ret_y, ret_t, tmp_nout);
                if let Some(checkpointing) = checkpointing.as_mut() {
                    checkpointing.record_sample(&*s);
                    nsteps += 1;
                    if nsteps > max_steps_between_checkpoints {
                        checkpointing.checkpoint_and_restart_samples(s);
                        nsteps = 0;
                    }
                }
            }
            OdeSolverStopReason::TstopReached => {
                write_out(s, ret_y, ret_t, tmp_nout);
                break OdeSolverStopReason::TstopReached;
            }
            OdeSolverStopReason::RootFound(t_root, root_idx) => {
                s.state_mut_back(t_root)?;
                if let Some(checkpointing) = checkpointing.as_mut() {
                    checkpointing.record_terminal_state(&*s);
                }
                if has_reset {
                    if let Some(checkpointing) = checkpointing.as_mut() {
                        checkpointing.finish_segment(Some(root_idx));
                    }
                    s.apply_reset()?;
                    write_out(s, ret_y, ret_t, tmp_nout);
                    if let Some(checkpointing) = checkpointing.as_mut() {
                        checkpointing.start_segment_from_current(&*s);
                        nsteps = 0;
                    }
                    if s.state().t < final_time {
                        s.set_stop_time(final_time)?;
                    } else {
                        break OdeSolverStopReason::TstopReached;
                    }
                } else {
                    write_out(s, ret_y, ret_t, tmp_nout);
                    if let Some(checkpointing) = checkpointing.as_mut() {
                        checkpointing.finish_segment(Some(root_idx));
                    }
                    break OdeSolverStopReason::RootFound(t_root, root_idx);
                }
            }
        }
    };
    if !matches!(stop_reason, OdeSolverStopReason::RootFound(_, _)) {
        if let Some(checkpointing) = checkpointing.as_mut() {
            checkpointing.finish_terminal_segment(&*s);
        }
    }
    Ok((
        stop_reason,
        checkpointing.map(CheckpointingRecorder::into_checkpointers),
    ))
}

/// utility function to write out the solution at a given timepoint
/// This function is used by the `solve` method to write out the solution at a given timepoint.
fn write_out<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    s: &S,
    ret_y: &mut <Eqn::V as DefaultDenseMatrix>::M,
    ret_t: &mut Vec<Eqn::T>,
    tmp_nout: &mut Eqn::V,
) where
    Eqn::V: DefaultDenseMatrix,
{
    let t = s.state().t;
    let y = s.state().y;
    ret_t.push(t);
    let i = ret_t.len() - 1;
    if i >= ret_y.ncols() {
        const GROWTH_FACTOR: usize = 2;
        ret_y.resize_cols(GROWTH_FACTOR * ret_y.ncols());
    }
    let mut ret_y_col = ret_y.column_mut(i);
    match s.problem().eqn.out() {
        Some(out) => {
            if s.problem().integrate_out {
                ret_y_col.copy_from(s.state().g);
            } else {
                out.call_inplace(y, t, tmp_nout);
                ret_y_col.copy_from(tmp_nout);
            }
        }
        None => {
            if s.problem().integrate_out {
                ret_y_col.copy_from(s.state().g);
            } else {
                ret_y_col.copy_from(y);
            }
        }
    }
}

/// Utility function to allocate the return matrix for the `solve`
/// method
fn allocate_return<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    s: &S,
) -> Result<(<Eqn::V as DefaultDenseMatrix>::M, Eqn::V), DiffsolError>
where
    Eqn::V: DefaultDenseMatrix,
{
    let nrows = s.problem().eqn.nout();
    let ret = s
        .problem()
        .context()
        .dense_mat_zeros::<Eqn::V>(nrows, INITIAL_NCOLS);

    let tmp_nout = Eqn::V::zeros(s.problem().eqn.nout(), s.problem().context().clone());
    Ok((ret, tmp_nout))
}

/// Utility function to allocate the return matrix for the `solve_dense`
/// and `solve_dense_sensitivities` methods.
#[allow(clippy::type_complexity)]
fn dense_allocate_return<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    s: &S,
    t_eval: &[Eqn::T],
) -> Result<(<Eqn::V as DefaultDenseMatrix>::M, Eqn::V, Eqn::V), DiffsolError>
where
    Eqn::V: DefaultDenseMatrix,
{
    let nrows = s.problem().eqn.nout();
    let ret = s
        .problem()
        .context()
        .dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());

    // check t_eval is increasing and all values are greater than or equal to the current time
    let t0 = s.state().t;
    if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
        return Err(ode_solver_error!(InvalidTEval));
    }
    let tmp_nout = Eqn::V::zeros(s.problem().eqn.nout(), s.problem().context().clone());
    let tmp_nstates = Eqn::V::zeros(
        s.problem().eqn.rhs().nstates(),
        s.problem().context().clone(),
    );
    Ok((ret, tmp_nout, tmp_nstates))
}

#[cfg(test)]
mod test {
    use crate::ConstantOp;
    use crate::{
        error::{DiffsolError, OdeSolverError},
        matrix::dense_nalgebra_serial::NalgebraMat,
        matrix::MatrixCommon,
        ode_equations::test_models::exponential_decay::{
            exponential_decay_problem, exponential_decay_problem_adjoint,
            exponential_decay_problem_sens, exponential_decay_problem_sens_with_out,
            exponential_decay_problem_with_root, exponential_decay_with_reset_problem,
            exponential_decay_with_reset_problem_sens,
        },
        scale, AdjointOdeSolverMethod, DenseMatrix, NalgebraLU, NalgebraVec, OdeBuilder,
        OdeEquations, OdeSolverMethod, OdeSolverStopReason, Op, SensitivitiesOdeSolverMethod,
        Solution, Vector, VectorView,
    };

    #[test]
    fn test_solve() {
        let (problem, _soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let k = 0.1;
        let y0 = NalgebraVec::from_vec(vec![1.0, 1.0], *problem.context());
        let expect = |t: f64| &y0 * scale(f64::exp(-k * t));
        let (y, t, stop_reason) = s.solve(10.0).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        assert!((t[0] - 0.0).abs() < 1e-10);
        assert!((t[t.len() - 1] - 10.0).abs() < 1e-10);
        for (i, t_i) in t.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&expect(*t_i), &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_solve_stops_on_root() {
        let (problem, _soln) =
            exponential_decay_problem_with_root::<NalgebraMat<f64>>(false, false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let (y, t, stop_reason) = s.solve(10.0).unwrap();
        assert!(matches!(stop_reason, OdeSolverStopReason::RootFound(_, _)));
        let t_root = -0.6_f64.ln() / 0.1;
        let t_last = *t.last().unwrap();
        assert!((t_last - t_root).abs() < 1e-3);
        assert!((s.state().t - t_root).abs() < 1e-3);

        let y_last = y.column(y.ncols() - 1).into_owned();
        let expected = NalgebraVec::from_vec(vec![0.6, 0.6], *problem.context());
        y_last.assert_eq_norm(&expected, &problem.atol, problem.rtol, 15.0);
    }

    #[test]
    fn test_solve_integrate_out() {
        let (problem, _soln) = exponential_decay_problem_adjoint::<NalgebraMat<f64>>(true);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let k = 0.1;
        let y0 = NalgebraVec::from_vec(vec![1.0, 1.0], *problem.context());
        let t0 = 0.0;
        let expect = |t: f64| {
            let g = &y0 * scale((f64::exp(-k * t0) - f64::exp(-k * t)) / k);
            NalgebraVec::from_vec(
                vec![1.0 * g[0] + 2.0 * g[1], 3.0 * g[0] + 4.0 * g[1]],
                *problem.context(),
            )
        };
        let (y, t, stop_reason) = s.solve(10.0).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        for (i, t_i) in t.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&expect(*t_i), &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_dense_solve() {
        let (problem, soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (y, stop_reason) = s.solve_dense(t_eval.as_slice()).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_dense_solve_stops_on_root() {
        let (problem, _soln) =
            exponential_decay_problem_with_root::<NalgebraMat<f64>>(false, false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = (0..=10).map(|i| i as f64).collect::<Vec<_>>();
        let (y, stop_reason) = s.solve_dense(t_eval.as_slice()).unwrap();
        assert!(matches!(stop_reason, OdeSolverStopReason::RootFound(_, _)));
        let t_root = -0.6_f64.ln() / 0.1;
        assert!((s.state().t - t_root).abs() < 1e-3);
        assert!(y.ncols() < t_eval.len());

        let t_root_minus_one = t_eval.iter().position(|x| x >= &t_root).unwrap() - 1;
        let y_root_minus_one = y.column(t_root_minus_one).into_owned();
        let expected_minus_one = NalgebraVec::from_vec(
            vec![f64::exp(-0.1 * t_eval[t_root_minus_one]); 2],
            *problem.context(),
        );
        y_root_minus_one.assert_eq_norm(&expected_minus_one, &problem.atol, problem.rtol, 15.0);

        let y_last = y.column(y.ncols() - 1).into_owned();
        let expected = NalgebraVec::from_vec(vec![0.6, 0.6], *problem.context());
        y_last.assert_eq_norm(&expected, &problem.atol, problem.rtol, 15.0);
    }

    #[test]
    fn test_dense_solve_integrate_out_stops_on_root() {
        let (problem, _soln) = exponential_decay_problem_with_root::<NalgebraMat<f64>>(false, true);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = (0..=10).map(|i| i as f64).collect::<Vec<_>>();
        let (y, stop_reason) = s.solve_dense(t_eval.as_slice()).unwrap();
        assert!(matches!(stop_reason, OdeSolverStopReason::RootFound(_, _)));
        let k = 0.1;
        let decay = 0.6_f64;
        let t_root = -decay.ln() / k;
        assert!((s.state().t - t_root).abs() < 1e-3);
        assert!(y.ncols() < t_eval.len());

        let y_last = y.column(y.ncols() - 1).into_owned();
        let y0 = problem.eqn.init().call(0.0);
        let integral = (1.0 - decay) / k;
        let expected = y0 * scale(integral);
        y_last.assert_eq_norm(&expected, &problem.atol, problem.rtol, 15.0);
    }

    #[test]
    fn test_dense_solve_integrate_out() {
        let (problem, soln) = exponential_decay_problem_adjoint::<NalgebraMat<f64>>(true);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (y, stop_reason) = s.solve_dense(t_eval.as_slice()).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_t_eval_errors() {
        let (problem, _soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();
        let t_eval = vec![0.0, 1.0, 0.5, 2.0];
        let err = s.solve_dense(t_eval.as_slice()).unwrap_err();
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::InvalidTEval)
        ));
    }

    #[test]
    fn test_dense_solve_sensitivities() {
        let (problem, soln) = exponential_decay_problem_sens::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf_sens::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (y, sens, stop_reason) = s.solve_dense_sensitivities(t_eval.as_slice()).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
        for (j, soln_pts) in soln.sens_solution_points.unwrap().iter().enumerate() {
            for (i, soln_pt) in soln_pts.iter().enumerate() {
                let sens_i = sens[j].column(i).into_owned();
                sens_i.assert_eq_norm(
                    &soln_pt.state,
                    problem.sens_atol.as_ref().unwrap(),
                    problem.sens_rtol.unwrap(),
                    15.0,
                );
            }
        }
    }

    #[test]
    fn test_dense_solve_sensitivities_with_out() {
        let (problem, soln) = exponential_decay_problem_sens_with_out::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf_sens::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (y, sens, stop_reason) = s.solve_dense_sensitivities(t_eval.as_slice()).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
        for (j, soln_pts) in soln.sens_solution_points.unwrap().iter().enumerate() {
            for (i, soln_pt) in soln_pts.iter().enumerate() {
                let sens_i = sens[j].column(i).into_owned();
                sens_i.assert_eq_norm(
                    &soln_pt.state,
                    problem.sens_atol.as_ref().unwrap(),
                    problem.sens_rtol.unwrap(),
                    15.0,
                );
            }
        }
    }

    #[test]
    fn test_solve_soln_sensitivities_dense() {
        type LS = NalgebraLU<f64>;

        let (problem, soln) = exponential_decay_problem_sens::<NalgebraMat<f64>>(false);
        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let mut state = problem.bdf_state_sens::<LS>().unwrap();
        let mut dense_soln = Solution::new_dense(t_eval.clone()).unwrap();

        while !dense_soln.is_complete() {
            state = problem
                .bdf_solver_sens::<LS>(state)
                .unwrap()
                .solve_soln_sensitivities(&mut dense_soln)
                .unwrap()
                .into_state();
        }

        assert_eq!(dense_soln.ts, t_eval);
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = dense_soln.ys.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
        for (j, soln_pts) in soln.sens_solution_points.unwrap().iter().enumerate() {
            for (i, soln_pt) in soln_pts.iter().enumerate() {
                let sens_i = dense_soln.y_sens[j].column(i).into_owned();
                sens_i.assert_eq_norm(
                    &soln_pt.state,
                    problem.sens_atol.as_ref().unwrap(),
                    problem.sens_rtol.unwrap(),
                    15.0,
                );
            }
        }
    }

    #[test]
    fn test_solve_soln_sensitivities_stops_on_root_without_writing_root_column() {
        type LS = NalgebraLU<f64>;

        let (problem, _soln) = exponential_decay_with_reset_problem_sens::<NalgebraMat<f64>>();
        let t_eval = vec![1.0_f64, 6.0_f64];
        let mut state = problem.bdf_state_sens::<LS>().unwrap();
        let mut dense_soln = Solution::new_dense(t_eval.clone()).unwrap();
        state = problem
            .bdf_solver_sens::<LS>(state)
            .unwrap()
            .solve_soln_sensitivities(&mut dense_soln)
            .unwrap()
            .into_state();

        assert!(matches!(
            dense_soln.stop_reason,
            Some(OdeSolverStopReason::RootFound(_, _))
        ));
        assert!(!dense_soln.is_complete());
        assert!(state.t < t_eval[1]);

        let decay = f64::exp(-0.1 * t_eval[0]);
        let expected = NalgebraVec::from_vec(vec![decay; 2], *problem.context());
        dense_soln.ys.column(0).into_owned().assert_eq_norm(
            &expected,
            &problem.atol,
            problem.rtol,
            15.0,
        );

        let s_k = NalgebraVec::from_vec(vec![-t_eval[0] * decay; 2], *problem.context());
        let s_y0 = NalgebraVec::from_vec(vec![decay; 2], *problem.context());
        dense_soln.y_sens[0].column(0).into_owned().assert_eq_norm(
            &s_k,
            problem.sens_atol.as_ref().unwrap(),
            problem.sens_rtol.unwrap(),
            15.0,
        );
        dense_soln.y_sens[1].column(0).into_owned().assert_eq_norm(
            &s_y0,
            problem.sens_atol.as_ref().unwrap(),
            problem.sens_rtol.unwrap(),
            15.0,
        );
    }

    #[test]
    fn test_solve_soln_multi_stage_matches_piecewise_analytical_solution() {
        type LS = NalgebraLU<f64>;

        let k = 0.1_f64;
        let x0 = 1.0_f64;
        let t_final = 10.0_f64;

        let problem = OdeBuilder::<NalgebraMat<f64>>::new()
            .p([k])
            .rhs_implicit(
                |x, p, _t, y| {
                    y[0] = -p[0] * x[0];
                },
                |_x, p, _t, v, y| {
                    y[0] = -p[0] * v[0];
                },
            )
            .root(|x, _p, _t, y| y[0] = x[0] - 0.1, 1)
            .init(|_p, _t, y| y[0] = x0, 1)
            .build()
            .unwrap();

        let mut state = problem.bdf_state::<LS>().unwrap();
        let mut soln = Solution::new(t_final);
        let mut stages = 0_usize;

        while !soln.is_complete() {
            stages += 1;
            assert!(
                stages < 32,
                "solve_soln did not converge in expected stages"
            );
            let stage_t0 = state.t;
            let stage_y0 = state.y.clone();
            let old_len = soln.ts.len();
            state = problem
                .bdf_solver::<LS>(state)
                .unwrap()
                .solve_soln(&mut soln)
                .unwrap()
                .into_state();

            for i in old_len..soln.ts.len() {
                let t = soln.ts[i];
                let expected = &stage_y0 * scale(f64::exp(-k * (t - stage_t0)));
                let got = soln.ys.column(i).into_owned();
                got.assert_eq_norm(&expected, &problem.atol, problem.rtol, 15.0);
            }
        }

        assert!(!soln.ts.is_empty());
        assert!((soln.ts.last().copied().unwrap() - t_final).abs() < 1e-8);
    }

    #[test]
    fn test_solve_adjoint() {
        let (problem, soln) = exponential_decay_problem_adjoint::<NalgebraMat<f64>>(true);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let final_time = soln.solution_points[soln.solution_points.len() - 1].t;
        let (checkpointers, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let g = s.state().g;
        g.assert_eq_norm(
            &soln.solution_points[soln.solution_points.len() - 1].state,
            problem.out_atol.as_ref().unwrap(),
            problem.out_rtol.unwrap(),
            15.0,
        );
        let adjoint_solver = problem
            .bdf_solver_adjoint::<NalgebraLU<f64>, _>(checkpointers, Some(s), None)
            .unwrap();
        let state = adjoint_solver
            .solve_adjoint_backwards_pass(&[], &[])
            .unwrap();

        let gs_adj = state.sg;
        for (j, soln_pts) in soln.sens_solution_points.unwrap().iter().enumerate() {
            gs_adj[j].assert_eq_norm(
                &soln_pts[0].state,
                problem.out_atol.as_ref().unwrap(),
                problem.out_rtol.unwrap(),
                15.0,
            );
        }
    }

    #[test]
    fn test_solve_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();
        let k = 0.1;
        let y0 = NalgebraVec::from_vec(vec![1.0, 1.0], *problem.context());
        let expect = |t: f64| &y0 * scale(f64::exp(-k * t));
        let final_time = soln.solution_points[soln.solution_points.len() - 1].t;
        let (checkpointers, y, t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        for (i, t_i) in t.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&expect(*t_i), &problem.atol, problem.rtol, 15.0);
        }
        let mut y = NalgebraVec::zeros(problem.eqn.rhs().nstates(), *problem.context());
        let mut replay_solver = s.clone();
        let checkpointer = checkpointers.last().unwrap();
        for point in soln.solution_points.iter() {
            checkpointer
                .interpolate(Some(&mut replay_solver), point.t, &mut y)
                .unwrap();
            y.assert_eq_norm(&point.state, &problem.atol, problem.rtol, 150.0);
        }
    }

    #[test]
    fn test_solve_with_checkpointing_continues_after_reset() {
        let (problem, soln) = exponential_decay_with_reset_problem::<NalgebraMat<f64>>();
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();
        let final_time = 100.0_f64;
        let (checkpointers, ys, ts, stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();

        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        assert!(checkpointers.len() > 1);
        let time_tol = soln.rtol * final_time.abs() + soln.atol.get_index(0);
        let t_last = *ts.last().unwrap();
        assert!((t_last - final_time).abs() < 30.0 * time_tol);
        assert!((s.state().t - final_time).abs() < 30.0 * time_tol);

        let reset_time = soln.solution_points[0].t;
        let reset_time_tol = 30.0 * (soln.rtol * reset_time.abs() + soln.atol.get_index(0));
        let reset_col = ts
            .iter()
            .position(|&t| (t - reset_time).abs() < reset_time_tol)
            .expect("expected checkpointed solve output to include the reset time");
        let reset_state = ys.column(reset_col).into_owned();
        let expected_reset = NalgebraVec::from_element(reset_state.len(), 0.4, *problem.context());
        reset_state.assert_eq_norm(&expected_reset, &problem.atol, problem.rtol, 20.0);
    }

    #[test]
    fn test_solve_dense_with_checkpointing_continues_after_reset() {
        let (problem, soln) = exponential_decay_with_reset_problem::<NalgebraMat<f64>>();
        let reset_time = soln.solution_points[0].t;
        let final_time = 2.0 * reset_time;
        let post_event_dt = 1.0_f64;
        let t_eval = vec![0.0, reset_time + post_event_dt, final_time];
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let (checkpointers, ys, stop_reason) = s
            .solve_dense_with_checkpointing(t_eval.as_slice(), None)
            .unwrap();

        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        assert!(checkpointers.len() > 1);
        assert_eq!(ys.ncols(), t_eval.len());
        let time_tol = soln.rtol * final_time.abs() + soln.atol.get_index(0);
        assert!((s.state().t - final_time).abs() < 30.0 * time_tol);

        let expected_post_reset_value = 0.4 * (-0.1 * post_event_dt).exp();
        let expected_post_reset = NalgebraVec::from_element(
            problem.eqn.rhs().nstates(),
            expected_post_reset_value,
            *problem.context(),
        );
        ys.column(1).into_owned().assert_eq_norm(
            &expected_post_reset,
            &problem.atol,
            problem.rtol,
            20.0,
        );
    }
}
