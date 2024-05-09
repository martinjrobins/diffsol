use anyhow::Result;
use num_traits::Zero;
use std::rc::Rc;

use crate::{
    op::filter::FilterCallable, scalar::Scalar, LinearOp, Matrix, NonLinearSolver, OdeEquations,
    OdeSolverProblem, SolverProblem, Vector, VectorIndex,
};

pub enum OdeSolverStopReason<T: Scalar> {
    InternalTimestep,
    RootFound(T),
    TstopReached,
}

/// Trait for ODE solver methods. This is the main user interface for the ODE solvers.
/// The solver is responsible for stepping the solution (given in the `OdeSolverState`), and interpolating the solution at a given time.
/// However, the solver does not own the state, so the user is responsible for creating and managing the state. If the user
/// wants to change the state, they should call `set_problem` again.
///
/// # Example
///
/// ```
/// use diffsol::{ OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeEquations };
///
/// fn solve_ode<Eqn: OdeEquations>(solver: &mut impl OdeSolverMethod<Eqn>, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Eqn::V {
///     let state = OdeSolverState::new(problem);
///     solver.set_problem(state, problem);
///     while solver.state().unwrap().t <= t {
///         solver.step().unwrap();
///     }
///     solver.interpolate(t).unwrap()
/// }
/// ```
pub trait OdeSolverMethod<Eqn: OdeEquations> {
    /// Get the current problem if it has been set
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;

    /// Set the problem to solve, this performs any initialisation required by the solver. Call this before calling `step` or `solve`.
    /// The solver takes ownership of the initial state given by `state`, this is assumed to be consistent with any algebraic constraints.
    fn set_problem(&mut self, state: OdeSolverState<Eqn::V>, problem: &OdeSolverProblem<Eqn>);

    /// Step the solution forward by one step, altering the internal state of the solver.
    /// The return value is a `Result` containing the reason for stopping the solver, possible reasons are:
    /// - `InternalTimestep`: The solver has taken a step forward in time, the internal state of the solver is at time self.state().t
    /// - `RootFound(t_root)`: The solver has found a root at time `t_root`. Note that the internal state of the solver is at the internal time step `self.state().t`, *not* at time `t_root`.
    /// - `TstopReached`: The solver has reached the stop time set by [Self::set_stop_time], the internal state of the solver is at time `tstop`, which is the same as `self.state().t`
    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>>;

    /// Set a stop time for the solver. The solver will stop when the internal time reaches this time.
    /// Once it stops, the stop time is unset. If `tstop` is at or before the current internal time, an error is returned.
    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<()>;

    /// Interpolate the solution at a given time. This time should be between the current time and the last solver time step
    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V>;

    /// Get the current state of the solver, if it exists
    fn state(&self) -> Option<&OdeSolverState<Eqn::V>>;

    /// Take the current state of the solver, if it exists, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem. Note that this will unset the current problem and solver state, so you will need to call
    /// `set_problem` again before calling `step` or `solve`.
    fn take_state(&mut self) -> Option<OdeSolverState<Eqn::V>>;

    /// Reinitialise the solver state and solve the problem up to time `t`
    fn solve(&mut self, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Result<Eqn::V> {
        let state = OdeSolverState::new(problem);
        self.set_problem(state, problem);
        self.set_stop_time(t)?;
        loop {
            if let OdeSolverStopReason::TstopReached = self.step()? {
                break;
            }
        }
        Ok(self.state().unwrap().y.clone())
    }

    /// Reinitialise the solver state making it consistent with the algebraic constraints and solve the problem up to time `t`
    fn make_consistent_and_solve<RS: NonLinearSolver<FilterCallable<Eqn::Rhs>>>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        t: Eqn::T,
        root_solver: &mut RS,
    ) -> Result<Eqn::V> {
        let state = OdeSolverState::new_consistent(problem, root_solver)?;
        self.set_problem(state, problem);
        self.set_stop_time(t)?;
        loop {
            if let OdeSolverStopReason::TstopReached = self.step()? {
                break;
            }
        }
        Ok(self.state().unwrap().y.clone())
    }
}

/// State for the ODE solver, containing the current solution `y`, the current time `t`, and the current step size `h`.
#[derive(Clone)]
pub struct OdeSolverState<V: Vector> {
    pub y: V,
    pub t: V::T,
    pub h: V::T,
}

impl<V: Vector> OdeSolverState<V> {
    /// Create a new solver state from an ODE problem. Note that this does not make the state consistent with any algebraic constraints.
    /// If you need to make the state consistent, use [Self::new_consistent].
    pub fn new<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init(t);
        Self { y, t, h }
    }

    /// Create a new solver state from an [OdeSolverProblem], making the state consistent with the algebraic constraints using a solver that implements [NonLinearSolver].
    /// If there are no algebraic constraints, please use [Self::new] instead.
    pub fn new_consistent<Eqn, S>(
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<Self>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<FilterCallable<Eqn::Rhs>> + ?Sized,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let mass_diagonal = ode_problem.eqn.mass().matrix(t).diagonal();
        let indices = mass_diagonal.filter_indices(|x| x == Eqn::T::zero());
        let mut y = ode_problem.eqn.init(t);
        if indices.len() == 0 {
            return Ok(Self { y, t, h });
        }
        let mut y_filtered = y.filter(&indices);
        let atol = Rc::new(ode_problem.atol.as_ref().filter(&indices));
        let rhs = ode_problem.eqn.rhs().clone();
        let f = Rc::new(FilterCallable::new(rhs, &y, indices));
        let rtol = ode_problem.rtol;
        let init_problem = SolverProblem::new(f, atol, rtol);
        root_solver.set_problem(&init_problem);
        root_solver.solve_in_place(&mut y_filtered, t)?;
        let indices = init_problem.f.indices();
        y.scatter_from(&y_filtered, indices);
        Ok(Self { y, t, h })
    }
}
