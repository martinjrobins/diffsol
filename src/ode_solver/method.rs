use nalgebra::ComplexField;

use crate::{
    error::{DiffsolError, OdeSolverError},
    matrix::default_solver::DefaultSolver,
    ode_solver_error,
    scalar::Scalar,
    AdjointEquations, AugmentedOdeEquations, DefaultDenseMatrix, DenseMatrix, Matrix, MatrixCommon,
    NonLinearOp, OdeEquations, OdeEquationsAdjoint, OdeEquationsSens, OdeSolverProblem,
    OdeSolverState, Op, SensEquations, StateRef, StateRefMut, VectorViewMut,
};

use super::checkpointing::HermiteInterpolator;

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
/// fn solve_ode<Eqn>(solver: &mut impl OdeSolverMethod<Eqn>, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Eqn::V
/// where
///    Eqn: OdeEquationsImplicit,
///    Eqn::M: DefaultSolver,
/// {
///     let state = OdeSolverState::new(problem, solver).unwrap();
///     solver.set_problem(state, problem);
///     while solver.state().unwrap().t <= t {
///         solver.step().unwrap();
///     }
///     solver.interpolate(t).unwrap()
/// }
/// ```
pub trait OdeSolverMethod<Eqn: OdeEquations>
where
    Self: Sized,
{
    type State: OdeSolverState<Eqn::V>;

    /// Get the current problem if it has been set
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;

    /// Set the problem to solve, this performs any initialisation required by the solver. Call this before calling `step` or `solve`.
    /// The solver takes ownership of the initial state given by `state`, this is assumed to be consistent with any algebraic constraints,
    /// and the time step `h` is assumed to be set appropriately for the problem
    fn set_problem(
        &mut self,
        state: Self::State,
        problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError>;

    /// Take a checkpoint of the current state of the solver, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem but want to keep this solver active. If you don't need to use this solver again, you can use `take_state` instead.
    /// Note that this will force a reinitialisation of the internal Jacobian for the solver, if it has one.
    fn checkpoint(&mut self) -> Result<Self::State, DiffsolError>;

    /// Take the current state of the solver, if it exists, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem. Note that this will unset the current problem and solver state, so you will need to call
    /// `set_problem` again before calling `step` or `solve`.
    fn take_state(&mut self) -> Option<Self::State>;

    /// Get the current state of the solver, if it exists
    fn state(&self) -> Option<StateRef<Eqn::V>>;

    /// Get a mutable reference to the current state of the solver, if it exists
    /// Note that calling this will cause the next call to `step` to perform some reinitialisation to take into
    /// account the mutated state, this could be expensive for multi-step methods.
    fn state_mut(&mut self) -> Option<StateRefMut<Eqn::V>>;

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
    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError>;

    /// Interpolate the integral of the output function at a given time. This time should be between the current time and the last solver time step
    fn interpolate_out(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError>;

    /// Interpolate the sensitivity vectors at a given time. This time should be between the current time and the last solver time step
    fn interpolate_sens(&self, t: Eqn::T) -> Result<Vec<Eqn::V>, DiffsolError>;

    /// Get the current order of accuracy of the solver (e.g. explict euler method is first-order)
    fn order(&self) -> usize;

    /// Using the provided state, solve the problem up to time `final_time`
    /// Returns a Vec of solution values at timepoints chosen by the solver.
    /// After the solver has finished, the internal state of the solver is at time `final_time`.
    #[allow(clippy::type_complexity)]
    fn solve(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        state: Self::State,
        final_time: Eqn::T,
    ) -> Result<(<Eqn::V as DefaultDenseMatrix>::M, Vec<Eqn::T>), DiffsolError>
    where
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        self.set_problem(state, problem)?;
        let mut ret_t = vec![self.state().unwrap().t];
        let nstates = problem.eqn.rhs().nstates();
        let ntimes_guess = std::cmp::max(
            10,
            ((final_time - self.state().unwrap().t).abs() / self.state().unwrap().h)
                .into()
                .ceil() as usize,
        );
        let mut ret_y = <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nstates, ntimes_guess);
        {
            let mut y_i = ret_y.column_mut(0);
            match problem.eqn.out() {
                Some(out) => {
                    y_i.copy_from(&out.call(self.state().unwrap().y, self.state().unwrap().t))
                }
                None => y_i.copy_from(self.state().unwrap().y),
            }
        }
        self.set_stop_time(final_time)?;
        while self.step()? != OdeSolverStopReason::TstopReached {
            ret_t.push(self.state().unwrap().t);
            let mut y_i = {
                let max_i = ret_y.ncols();
                let curr_i = ret_t.len() - 1;
                if curr_i >= max_i {
                    ret_y =
                        <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nstates, max_i * 2);
                }
                ret_y.column_mut(curr_i)
            };
            match problem.eqn.out() {
                Some(out) => {
                    y_i.copy_from(&out.call(self.state().unwrap().y, self.state().unwrap().t))
                }
                None => y_i.copy_from(self.state().unwrap().y),
            }
        }

        // store the final step
        ret_t.push(self.state().unwrap().t);
        {
            let mut y_i = {
                let max_i = ret_y.ncols();
                let curr_i = ret_t.len() - 1;
                if curr_i >= max_i {
                    ret_y =
                        <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nstates, max_i + 1);
                }
                ret_y.column_mut(curr_i)
            };
            match problem.eqn.out() {
                Some(out) => {
                    y_i.copy_from(&out.call(self.state().unwrap().y, self.state().unwrap().t))
                }
                None => y_i.copy_from(self.state().unwrap().y),
            }
        }
        Ok((ret_y, ret_t))
    }

    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a Vec of solution values at timepoints given by `t_eval`.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    fn solve_dense(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        state: Self::State,
        t_eval: &[Eqn::T],
    ) -> Result<<Eqn::V as DefaultDenseMatrix>::M, DiffsolError>
    where
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        self.set_problem(state, problem)?;
        let nstates = problem.eqn.rhs().nstates();
        let mut ret = <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nstates, t_eval.len());

        // check t_eval is increasing and all values are greater than or equal to the current time
        let t0 = self.state().unwrap().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().take(t_eval.len() - 1).enumerate() {
            while self.state().unwrap().t < *t {
                step_reason = self.step()?;
            }
            let y = self.interpolate(*t)?;
            let mut y_out = ret.column_mut(i);
            match problem.eqn.out() {
                Some(out) => y_out.copy_from(&out.call(&y, *t)),
                None => y_out.copy_from(&y),
            }
        }

        // do final step
        while step_reason != OdeSolverStopReason::TstopReached {
            step_reason = self.step()?;
        }
        {
            let mut y_out = ret.column_mut(t_eval.len() - 1);
            match problem.eqn.out() {
                Some(out) => {
                    y_out.copy_from(&out.call(self.state().unwrap().y, self.state().unwrap().t))
                }
                None => y_out.copy_from(self.state().unwrap().y),
            }
        }
        Ok(ret)
    }
}

pub trait AugmentedOdeSolverMethod<Eqn, AugmentedEqn>: OdeSolverMethod<Eqn>
where
    Eqn: OdeEquations,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    fn set_augmented_problem(
        &mut self,
        state: Self::State,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: AugmentedEqn,
    ) -> Result<(), DiffsolError>;
}

pub trait SensitivitiesOdeSolverMethod<Eqn>:
    AugmentedOdeSolverMethod<Eqn, SensEquations<Eqn>>
where
    Eqn: OdeEquationsSens,
{
    fn set_problem_with_sensitivities(
        &mut self,
        state: Self::State,
        problem: &OdeSolverProblem<Eqn>,
        include_in_error_control: bool,
    ) -> Result<(), DiffsolError> {
        let mut augmented_eqn = SensEquations::new(&problem.eqn);
        augmented_eqn.set_include_in_error_control(include_in_error_control);
        self.set_augmented_problem(state, problem, augmented_eqn)
    }
}

pub trait AdjointOdeSolverMethod<Eqn>: OdeSolverMethod<Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    type AdjointSolver: AugmentedOdeSolverMethod<
        AdjointEquations<Eqn, Self>,
        AdjointEquations<Eqn, Self>,
    >;

    fn into_adjoint_solver(
        self,
        checkpoints: Vec<Self::State>,
        last_segment: HermiteInterpolator<Eqn::V>,
        include_in_error_control: bool,
    ) -> Result<Self::AdjointSolver, DiffsolError>;
}
