use std::cell::Ref;

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    scalar::Scalar,
    AugmentedOdeEquations, Checkpointing, Context, DefaultDenseMatrix, DenseMatrix,
    HermiteInterpolator, MatrixCommon, NonLinearOp, OdeEquations, OdeSolverConfig,
    OdeSolverProblem, OdeSolverState, Op, StateRef, StateRefMut, Vector, VectorViewMut,
};

#[derive(Debug, PartialEq)]
pub enum OdeSolverStopReason<T: Scalar> {
    InternalTimestep,
    RootFound(T),
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
    type State: OdeSolverState<Eqn::V>;
    type Config: OdeSolverConfig<Eqn::T>;

    /// Get the current problem
    fn problem(&self) -> &'a OdeSolverProblem<Eqn>;

    /// Take a checkpoint of the current state of the solver, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem but want to keep this solver active. If you don't need to use this solver again, you can use `take_state` instead.
    /// Note that this will force a reinitialisation of the internal Jacobian for the solver, if it has one.
    fn checkpoint(&mut self) -> Self::State;

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

    /// Get the current order of accuracy of the solver (e.g. explict euler method is first-order)
    fn order(&self) -> usize;

    /// Using the provided state, solve the problem up to time `final_time`
    /// Returns a Vec of solution values at timepoints chosen by the solver.
    /// After the solver has finished, the internal state of the solver is at time `final_time`.
    #[allow(clippy::type_complexity)]
    fn solve(
        &mut self,
        final_time: Eqn::T,
    ) -> Result<(<Eqn::V as DefaultDenseMatrix>::M, Vec<Eqn::T>), DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let mut ret_t = Vec::new();
        let (mut ret_y, mut tmp_nout) = allocate_return(self)?;

        // do the main loop
        write_out(self, &mut ret_y, &mut ret_t, &mut tmp_nout);
        self.set_stop_time(final_time)?;
        while self.step()? != OdeSolverStopReason::TstopReached {
            write_out(self, &mut ret_y, &mut ret_t, &mut tmp_nout);
        }

        // store the final step
        write_out(self, &mut ret_y, &mut ret_t, &mut tmp_nout);
        let ntimes = ret_t.len();
        ret_y.resize_cols(ntimes);
        Ok((ret_y, ret_t))
    }

    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a Vec of solution values at timepoints given by `t_eval`.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    fn solve_dense(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<<Eqn::V as DefaultDenseMatrix>::M, DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let (mut ret, mut tmp_nout, mut tmp_nstates) = dense_allocate_return(self, t_eval)?;

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().enumerate() {
            while self.state().t < *t {
                step_reason = self.step()?;
            }
            dense_write_out(self, &mut ret, t_eval, i, &mut tmp_nout, &mut tmp_nstates)?;
        }
        assert_eq!(step_reason, OdeSolverStopReason::TstopReached);
        Ok(ret)
    }

    #[allow(clippy::type_complexity)]
    fn solve_with_checkpointing(
        &mut self,
        final_time: Eqn::T,
        max_steps_between_checkpoints: Option<usize>,
    ) -> Result<
        (
            Checkpointing<'a, Eqn, Self>,
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<Eqn::T>,
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

        // allocate checkpoint info
        let mut nsteps = 0;
        let t0 = self.state().t;
        let mut checkpoints = vec![self.checkpoint()];
        let mut ts = vec![t0];
        let mut ys = vec![self.state().y.clone()];
        let mut ydots = vec![self.state().dy.clone()];

        // do the main loop, saving checkpoints
        write_out(self, &mut ret_y, &mut ret_t, &mut tmp_nout);
        self.set_stop_time(final_time)?;
        while self.step()? != OdeSolverStopReason::TstopReached {
            write_out(self, &mut ret_y, &mut ret_t, &mut tmp_nout);
            ts.push(self.state().t);
            ys.push(self.state().y.clone());
            ydots.push(self.state().dy.clone());
            nsteps += 1;
            if nsteps > max_steps_between_checkpoints {
                checkpoints.push(self.checkpoint());
                nsteps = 0;
                ts.clear();
                ys.clear();
                ydots.clear();
            }
        }

        // store the final step
        write_out(self, &mut ret_y, &mut ret_t, &mut tmp_nout);
        let ntimes = ret_t.len();
        ret_y.resize_cols(ntimes);

        // add final checkpoint
        ts.push(self.state().t);
        ys.push(self.state().y.clone());
        ydots.push(self.state().dy.clone());
        checkpoints.push(self.checkpoint());

        // construct checkpointing
        let last_segment = HermiteInterpolator::new(ys, ydots, ts);
        let checkpointer = Checkpointing::new(
            self.clone(),
            checkpoints.len() - 2,
            checkpoints,
            Some(last_segment),
        );

        Ok((checkpointer, ret_y, ret_t))
    }

    /// Solve the problem and write out the solution at the given timepoints, using checkpointing so that
    /// the solution can be interpolated at any timepoint.
    /// See [Self::solve_dense] for a similar method that does not use checkpointing.
    #[allow(clippy::type_complexity)]
    fn solve_dense_with_checkpointing(
        &mut self,
        t_eval: &[Eqn::T],
        max_steps_between_checkpoints: Option<usize>,
    ) -> Result<
        (
            Checkpointing<'a, Eqn, Self>,
            <Eqn::V as DefaultDenseMatrix>::M,
        ),
        DiffsolError,
    >
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let (mut ret, mut tmp_nout, mut tmp_nstates) = dense_allocate_return(self, t_eval)?;
        let max_steps_between_checkpoints = max_steps_between_checkpoints.unwrap_or(500);

        // allocate checkpoint info
        let mut nsteps = 0;
        let t0 = self.state().t;
        let mut checkpoints = vec![self.checkpoint()];
        let mut ts = vec![t0];
        let mut ys = vec![self.state().y.clone()];
        let mut ydots = vec![self.state().dy.clone()];

        // do loop, saving checkpoints
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().enumerate() {
            while self.state().t < *t {
                step_reason = self.step()?;
                ts.push(self.state().t);
                ys.push(self.state().y.clone());
                ydots.push(self.state().dy.clone());
                nsteps += 1;
                if nsteps > max_steps_between_checkpoints
                    && step_reason != OdeSolverStopReason::TstopReached
                {
                    checkpoints.push(self.checkpoint());
                    nsteps = 0;
                    ts.clear();
                    ys.clear();
                    ydots.clear();
                }
            }
            dense_write_out(self, &mut ret, t_eval, i, &mut tmp_nout, &mut tmp_nstates)?;
        }
        assert_eq!(step_reason, OdeSolverStopReason::TstopReached);

        // add final checkpoint
        checkpoints.push(self.checkpoint());

        // construct the adjoint equations
        let last_segment = HermiteInterpolator::new(ys, ydots, ts);

        // construct checkpointing
        let checkpointer = Checkpointing::new(
            self.clone(),
            checkpoints.len() - 2,
            checkpoints,
            Some(last_segment),
        );

        Ok((checkpointer, ret))
    }
}

pub trait AugmentedOdeSolverMethod<'a, Eqn, AugmentedEqn>: OdeSolverMethod<'a, Eqn>
where
    Eqn: OdeEquations + 'a,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugmentedEqn>);
    fn augmented_eqn(&self) -> Option<&AugmentedEqn>;
}

/// Utility function to write out the solution at a given timepoint
/// This function is used by the `solve_dense` method to write out the solution at a given timepoint.
fn dense_write_out<'a, Eqn: OdeEquations + 'a, S: OdeSolverMethod<'a, Eqn>>(
    s: &S,
    y_out: &mut <Eqn::V as DefaultDenseMatrix>::M,
    t_eval: &[Eqn::T],
    i: usize,
    tmp_nout: &mut Eqn::V,
    tmp_nstates: &mut Eqn::V,
) -> Result<(), DiffsolError>
where
    Eqn::V: DefaultDenseMatrix,
{
    let mut y_out = y_out.column_mut(i);
    let t = t_eval[i];
    if s.problem().integrate_out {
        s.interpolate_out_inplace(t, tmp_nout)?;
        y_out.copy_from(tmp_nout);
    } else {
        s.interpolate_inplace(t, tmp_nstates)?;
        match s.problem().eqn.out() {
            Some(out) => {
                out.call_inplace(tmp_nstates, t_eval[i], tmp_nout);
                y_out.copy_from(tmp_nout)
            }
            None => y_out.copy_from(tmp_nstates),
        }
    }
    Ok(())
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
        None => ret_y_col.copy_from(y),
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
    let nrows = if s.problem().eqn.out().is_some() {
        s.problem().eqn.out().unwrap().nout()
    } else {
        s.problem().eqn.rhs().nstates()
    };
    const INITIAL_NCOLS: usize = 10;
    let ret = s
        .problem()
        .context()
        .dense_mat_zeros::<Eqn::V>(nrows, INITIAL_NCOLS);

    // check t_eval is increasing and all values are greater than or equal to the current time
    let tmp_nout = if let Some(out) = s.problem().eqn.out() {
        Eqn::V::zeros(out.nout(), s.problem().context().clone())
    } else {
        Eqn::V::zeros(0, s.problem().context().clone())
    };
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
    let nrows = if s.problem().eqn.out().is_some() {
        s.problem().eqn.out().unwrap().nout()
    } else {
        s.problem().eqn.rhs().nstates()
    };
    let ret = s
        .problem()
        .context()
        .dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());

    // check t_eval is increasing and all values are greater than or equal to the current time
    let t0 = s.state().t;
    if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
        return Err(ode_solver_error!(InvalidTEval));
    }
    let tmp_nout = if let Some(out) = s.problem().eqn.out() {
        Eqn::V::zeros(out.nout(), s.problem().context().clone())
    } else {
        Eqn::V::zeros(0, s.problem().context().clone())
    };
    let tmp_nstates = Eqn::V::zeros(
        s.problem().eqn.rhs().nstates(),
        s.problem().context().clone(),
    );
    Ok((ret, tmp_nout, tmp_nstates))
}

#[cfg(test)]
mod test {
    use crate::{
        error::{DiffsolError, OdeSolverError},
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::exponential_decay::{
            exponential_decay_problem, exponential_decay_problem_adjoint,
            exponential_decay_problem_sens, exponential_decay_problem_sens_with_out,
        },
        scale, AdjointOdeSolverMethod, DenseMatrix, NalgebraLU, NalgebraVec, OdeEquations,
        OdeSolverMethod, Op, SensitivitiesOdeSolverMethod, Vector, VectorView,
    };

    #[test]
    fn test_solve() {
        let (problem, _soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let k = 0.1;
        let y0 = NalgebraVec::from_vec(vec![1.0, 1.0], *problem.context());
        let expect = |t: f64| &y0 * scale(f64::exp(-k * t));
        let (y, t) = s.solve(10.0).unwrap();
        assert!((t[0] - 0.0).abs() < 1e-10);
        assert!((t[t.len() - 1] - 10.0).abs() < 1e-10);
        for (i, t_i) in t.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&expect(*t_i), &problem.atol, problem.rtol, 15.0);
        }
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
        let (y, t) = s.solve(10.0).unwrap();
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
        let y = s.solve_dense(t_eval.as_slice()).unwrap();
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_dense_solve_integrate_out() {
        let (problem, soln) = exponential_decay_problem_adjoint::<NalgebraMat<f64>>(true);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let y = s.solve_dense(t_eval.as_slice()).unwrap();
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
        let (y, sens) = s.solve_dense_sensitivities(t_eval.as_slice()).unwrap();
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
        let (y, sens) = s.solve_dense_sensitivities(t_eval.as_slice()).unwrap();
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
    fn test_solve_adjoint() {
        let (problem, soln) = exponential_decay_problem_adjoint::<NalgebraMat<f64>>(true);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let final_time = soln.solution_points[soln.solution_points.len() - 1].t;
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let g = s.state().g;
        g.assert_eq_norm(
            &soln.solution_points[soln.solution_points.len() - 1].state,
            problem.out_atol.as_ref().unwrap(),
            problem.out_rtol.unwrap(),
            15.0,
        );
        let adjoint_solver = problem
            .bdf_solver_adjoint::<NalgebraLU<f64>, _>(checkpointer, None)
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
        let final_time = soln.solution_points[soln.solution_points.len() - 1].t;
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let mut y = NalgebraVec::zeros(problem.eqn.rhs().nstates(), *problem.context());
        for point in soln.solution_points.iter() {
            checkpointer.interpolate(point.t, &mut y).unwrap();
            y.assert_eq_norm(&point.state, &problem.atol, problem.rtol, 100.0);
        }
    }
}
