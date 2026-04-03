use crate::{
    error::DiffsolError,
    error::OdeSolverError,
    ode_solver::solution::{Solution, SolutionMode},
    ode_solver_error, AugmentedOdeSolverMethod, Context, DefaultDenseMatrix, DenseMatrix,
    MatrixCommon, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, OdeEquationsImplicitSens,
    OdeSolverStopReason, Op, SensEquations, Vector, VectorViewMut,
};
use num_traits::{One, Zero};
use std::ops::AddAssign;

pub trait SensitivitiesOdeSolverMethod<'a, Eqn>:
    AugmentedOdeSolverMethod<'a, Eqn, SensEquations<'a, Eqn>>
where
    Eqn: OdeEquationsImplicitSens + 'a,
{
    /// Apply the equations' reset function and propagate sensitivities through its Jacobian.
    fn reset_with_sens(&mut self) -> Result<(), DiffsolError>;

    /// Continue solving ODE and forward sensitivities into an existing dense [`Solution`].
    ///
    /// This method requires a [`Solution`] created with [`Solution::new_dense`]. It reuses the
    /// solution's storage so staged integrations can stop on roots, apply resets or parameter
    /// changes, and resume without reallocating the dense output buffers.
    fn solve_soln_sensitivities(mut self, soln: &mut Solution<Eqn::V>) -> Result<Self, DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        if self.problem().integrate_out {
            return Err(ode_solver_error!(
                Other,
                "Cannot integrate out when solving for sensitivities"
            ));
        }

        let start_col = match soln.mode {
            SolutionMode::Tevals(start_col) => start_col,
            SolutionMode::Tfinal(_) => {
                return Err(ode_solver_error!(
                    Other,
                    "solve_soln_sensitivities requires Solution::new_dense"
                ));
            }
        };

        let ctx = self.problem().context().clone();
        let nstates = self.problem().eqn.rhs().nstates();
        let nparams = self.problem().eqn.rhs().nparams();

        if soln.y_sens.len() != nparams {
            soln.y_sens =
                vec![ctx.dense_mat_zeros::<Eqn::V>(soln.ys.nrows(), soln.ys.ncols()); nparams];
        }
        if soln.tmp_nstates.len() != nstates {
            soln.tmp_nstates = Eqn::V::zeros(nstates, ctx.clone());
        }
        if soln.tmp_nsens.len() != nparams {
            soln.tmp_nsens = vec![Eqn::V::zeros(nstates, ctx.clone()); nparams];
        }
        let nout = self.problem().eqn.out().map(|out| out.nout()).unwrap_or(0);
        if soln.tmp_nout.len() != nout {
            soln.tmp_nout = Eqn::V::zeros(nout, ctx.clone());
        }
        let nout_params = self
            .problem()
            .eqn
            .out()
            .map(|out| out.nparams())
            .unwrap_or(0);
        if soln.tmp_nparams.len() != nout_params {
            soln.tmp_nparams = Eqn::V::zeros(nout_params, ctx);
        }

        let (stop_reason, col) = solve_dense_sensitivities(
            &mut soln.ys,
            &mut soln.y_sens,
            &soln.ts,
            &mut soln.tmp_nout,
            &mut soln.tmp_nparams,
            &mut soln.tmp_nstates,
            &mut soln.tmp_nsens,
            &mut self,
            start_col,
        )?;
        soln.stop_reason = Some(stop_reason);
        soln.mode = SolutionMode::Tevals(col);
        Ok(self)
    }

    /// Solve the ODE and the forward sensitivity equations from the current time to `t_eval[t_eval.len()-1]`,
    /// evaluating at specified times.
    ///
    /// This method integrates the system and returns the solution interpolated at the specified times.
    /// The solver uses its own internal timesteps for accuracy, but the output is interpolated to the
    /// requested evaluation times. This is useful when you need the solution at specific timepoints
    /// and want the solver's adaptive stepping for accuracy.
    ///
    /// If a root function is provided, the solver will stop if any of the root function elements change sign.
    /// The internal state of the solver is set to the time that the zero-crossing occured.
    ///
    /// # Arguments
    /// - `t_eval`: A slice of times at which to evaluate the solution. Times should be in increasing order.
    ///
    /// # Returns
    /// A tuple of `(ode_solution, sensitivities, stop_reason)` at the specified evaluation times.
    ///
    /// The ODE solution is a dense matrix with one column per evaluation time (in the same order as `t_eval`) and one row per state variable,
    /// plus one final column at the root time if a root fires before `t_eval` is exhausted.
    ///
    /// The sensitivities are returned as a Vec of dense matrices of identical shape as the ODE solution,
    /// where the ith element of the Vec corresponds to the sensitivities with respect to the ith parameter.
    /// `stop_reason` indicates whether the solve reached `t_eval[t_eval.len()-1]` or stopped on a root.
    ///
    /// # Post-condition
    /// In the case that no roots are found that stop the solve early, the internal state is at time `t_eval[t_eval.len()-1]`.
    /// If a root is found, the solver stops early. The internal state is moved to the root time,
    /// and the last column corresponds to the root time (which may not be in `t_eval`).
    #[allow(clippy::type_complexity)]
    fn solve_dense_sensitivities(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<
        (
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<<Eqn::V as DefaultDenseMatrix>::M>,
            OdeSolverStopReason<Eqn::T>,
        ),
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicitSens,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        if self.problem().integrate_out {
            return Err(ode_solver_error!(
                Other,
                "Cannot integrate out when solving for sensitivities"
            ));
        }
        let nrows = if let Some(out) = self.problem().eqn.out() {
            out.nout()
        } else {
            self.problem().eqn.rhs().nout()
        };
        let nstates = self.problem().eqn.rhs().nstates();
        let nparams = self.problem().eqn.rhs().nparams();
        let ctx = self.problem().context().clone();

        let mut ret = ctx.dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());
        let mut ret_sens = vec![ctx.dense_mat_zeros::<Eqn::V>(nrows, t_eval.len()); nparams];
        let mut tmp_nout = Eqn::V::zeros(
            self.problem().eqn.out().map(|out| out.nout()).unwrap_or(0),
            ctx.clone(),
        );
        let mut tmp_nparams = Eqn::V::zeros(
            self.problem()
                .eqn
                .out()
                .map(|out| out.nparams())
                .unwrap_or(0),
            ctx.clone(),
        );
        let mut tmp_nstates = Eqn::V::zeros(nstates, ctx.clone());
        let mut tmp_nsens = vec![Eqn::V::zeros(nstates, ctx); nparams];

        // check t_eval is increasing and all values are >= the current time
        let t0 = self.state().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        let (stop_reason, col) = solve_dense_sensitivities(
            &mut ret,
            &mut ret_sens,
            t_eval,
            &mut tmp_nout,
            &mut tmp_nparams,
            &mut tmp_nstates,
            &mut tmp_nsens,
            self,
            0,
        )?;

        if let OdeSolverStopReason::RootFound(_, _) = stop_reason {
            if col < t_eval.len() {
                let t = self.state().t;
                dense_write_out_sensitivities(
                    self,
                    &mut ret,
                    &mut ret_sens,
                    t,
                    col,
                    &mut tmp_nout,
                    &mut tmp_nparams,
                    &mut tmp_nstates,
                    &mut tmp_nsens,
                )?;
                if col + 1 < ret.ncols() {
                    ret.resize_cols(col + 1);
                    for rs in &mut ret_sens {
                        rs.resize_cols(col + 1);
                    }
                }
            }
        }
        Ok((ret, ret_sens, stop_reason))
    }
}

fn solve_dense_sensitivities<'a, Eqn, S>(
    ret: &mut <Eqn::V as DefaultDenseMatrix>::M,
    ret_sens: &mut [<Eqn::V as DefaultDenseMatrix>::M],
    t_eval: &[Eqn::T],
    tmp_nout: &mut Eqn::V,
    tmp_nparams: &mut Eqn::V,
    tmp_nstates: &mut Eqn::V,
    tmp_nsens: &mut [Eqn::V],
    s: &mut S,
    start_col: usize,
) -> Result<(OdeSolverStopReason<Eqn::T>, usize), DiffsolError>
where
    Eqn: OdeEquationsImplicitSens + 'a,
    Eqn::V: DefaultDenseMatrix,
    S: SensitivitiesOdeSolverMethod<'a, Eqn>,
{
    s.set_stop_time(t_eval[t_eval.len() - 1])?;
    let mut stop_reason: OdeSolverStopReason<Eqn::T>;
    let mut col = start_col;
    loop {
        stop_reason = s.step()?;
        let t_current = if let OdeSolverStopReason::RootFound(t, _) = stop_reason {
            t
        } else {
            s.state().t
        };
        while col < t_eval.len() && t_eval[col] <= t_current {
            dense_write_out_sensitivities(
                s,
                ret,
                ret_sens,
                t_eval[col],
                col,
                tmp_nout,
                tmp_nparams,
                tmp_nstates,
                tmp_nsens,
            )?;
            col += 1;
        }
        match stop_reason {
            OdeSolverStopReason::InternalTimestep => {}
            OdeSolverStopReason::TstopReached => {
                assert!(
                    col == t_eval.len(),
                    "Solver reached stop time before consuming all t_eval points, this should not happen"
                );
                break;
            }
            OdeSolverStopReason::RootFound(t_root, _) => {
                s.state_mut_back(t_root)?;
                break;
            }
        }
    }
    Ok((stop_reason, col))
}

fn dense_write_out_sensitivities<'a, Eqn, S>(
    s: &S,
    ret: &mut <Eqn::V as DefaultDenseMatrix>::M,
    ret_sens: &mut [<Eqn::V as DefaultDenseMatrix>::M],
    t: Eqn::T,
    col: usize,
    tmp_nout: &mut Eqn::V,
    tmp_nparams: &mut Eqn::V,
    tmp_nstates: &mut Eqn::V,
    tmp_nsens: &mut [Eqn::V],
) -> Result<(), DiffsolError>
where
    Eqn: OdeEquationsImplicitSens + 'a,
    Eqn::V: DefaultDenseMatrix,
    S: SensitivitiesOdeSolverMethod<'a, Eqn>,
{
    s.interpolate_inplace(t, tmp_nstates)?;
    s.interpolate_sens_inplace(t, tmp_nsens)?;
    if let Some(out) = s.problem().eqn.out() {
        out.call_inplace(tmp_nstates, t, tmp_nout);
        ret.column_mut(col).copy_from(tmp_nout);
        for (j, s_j) in tmp_nsens.iter().enumerate() {
            let mut col_v = ret_sens[j].column_mut(col);
            tmp_nparams.set_index(j, Eqn::T::one());
            out.jac_mul_inplace(tmp_nstates, t, s_j, tmp_nout);
            col_v.copy_from(&*tmp_nout);
            out.sens_mul_inplace(tmp_nstates, t, tmp_nparams, tmp_nout);
            col_v.add_assign(&*tmp_nout);
            tmp_nparams.set_index(j, Eqn::T::zero());
        }
    } else {
        ret.column_mut(col).copy_from(tmp_nstates);
        for (j, s_j) in tmp_nsens.iter().enumerate() {
            ret_sens[j].column_mut(col).copy_from(s_j);
        }
    }
    Ok(())
}
