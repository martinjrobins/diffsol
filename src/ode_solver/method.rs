use anyhow::Result;
use num_traits::{One, Pow, Zero};
use std::rc::Rc;

use crate::{
    matrix::default_solver::DefaultSolver, op::filter::FilterCallable, scalar::Scalar, scale,
    LinearOp, Matrix, NewtonNonlinearSolver, NonLinearOp, NonLinearSolver, OdeEquations,
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
/// use diffsol::{ OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeEquations, DefaultSolver };
///
/// fn solve_ode<Eqn>(solver: &mut impl OdeSolverMethod<Eqn>, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Eqn::V
/// where
///    Eqn: OdeEquations,
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
pub trait OdeSolverMethod<Eqn: OdeEquations> {
    /// Get the current problem if it has been set
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;

    /// Set the problem to solve, this performs any initialisation required by the solver. Call this before calling `step` or `solve`.
    /// The solver takes ownership of the initial state given by `state`, this is assumed to be consistent with any algebraic constraints,
    /// and the time step `h` is assumed to be set appropriately for the problem
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

    /// Get a mutable reference to the current state of the solver, if it exists
    /// Note that calling this will cause the next call to `step` to perform some reinitialisation to take into
    /// account the mutated state, this could be expensive for multi-step methods.
    fn state_mut(&mut self) -> Option<&mut OdeSolverState<Eqn::V>>;

    /// Get the current order of accuracy of the solver (e.g. explict euler method is first-order)
    fn order(&self) -> usize;

    /// Take the current state of the solver, if it exists, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem. Note that this will unset the current problem and solver state, so you will need to call
    /// `set_problem` again before calling `step` or `solve`.
    fn take_state(&mut self) -> Option<OdeSolverState<Eqn::V>>;

    /// Reinitialise the solver state and solve the problem up to time `t`
    fn solve(&mut self, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Result<Eqn::V> {
        let mut state = OdeSolverState::new_without_initialise(problem);
        state.set_step_size(problem, self.order());
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
        let mut state = OdeSolverState::new_without_initialise(problem);
        state.set_consistent(problem, root_solver)?;
        state.set_step_size(problem, self.order());
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
    pub f: V,
    pub t: V::T,
    pub h: V::T,
}

impl<V: Vector> OdeSolverState<V> {
    /// Create a new solver state from an ODE problem.
    /// This function will make the state consistent with any algebraic constraints using a default nonlinear solver.
    /// It will also set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    pub fn new<Eqn, S>(ode_problem: &OdeSolverProblem<Eqn>, solver: &S) -> Result<Self>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        Eqn::M: DefaultSolver,
        S: OdeSolverMethod<Eqn>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init(t);
        let f = ode_problem.eqn.rhs().call(&y, t);

        let mut ret = Self { y, t, h, f };
        let ls = <Eqn::M as DefaultSolver>::default_solver();
        let mut root_solver = NewtonNonlinearSolver::new(ls);
        ret.set_consistent(ode_problem, &mut root_solver)?;
        ret.set_step_size(ode_problem, solver.order());
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem, without any initialisation.
    /// This is useful if you want to set up the state yourself, or if you want to use a different nonlinear solver to make the state consistent,
    /// or if you want to set the step size yourself or based on the exact order of the solver.
    pub fn new_without_initialise<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init(t);
        let f = ode_problem.eqn.rhs().call(&y, t);
        Self { y, t, h, f }
    }

    /// Set the state to be consistent with the algebraic constraints of the problem.
    pub fn set_consistent<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<()>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<FilterCallable<Eqn::Rhs>> + ?Sized,
    {
        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }
        let mass_diagonal = ode_problem.eqn.mass().unwrap().matrix(self.t).diagonal();
        let indices = mass_diagonal.filter_indices(|x| x == Eqn::T::zero());
        if indices.len() == 0 {
            return Ok(());
        }
        let mut y_filtered = self.y.filter(&indices);
        let atol = Rc::new(ode_problem.atol.as_ref().filter(&indices));
        let rhs = ode_problem.eqn.rhs().clone();
        let f = Rc::new(FilterCallable::new(rhs, &self.y, indices));
        let rtol = ode_problem.rtol;
        let init_problem = SolverProblem::new(f, atol, rtol);
        root_solver.set_problem(&init_problem);
        root_solver.solve_in_place(&mut y_filtered, self.t)?;
        let indices = init_problem.f.indices();
        self.y.scatter_from(&y_filtered, indices);
        Ok(())
    }

    /// compute size of first step based on alg in Hairer, Norsett, Wanner
    /// Solving Ordinary Differential Equations I, Nonstiff Problems
    /// Section II.4.2
    pub fn set_step_size<Eqn>(&mut self, ode_problem: &OdeSolverProblem<Eqn>, solver_order: usize)
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let y0 = &self.y;
        let t0 = self.t;
        let f0 = &self.f;

        let mut scale_factor = y0.abs();
        scale_factor *= scale(ode_problem.rtol);
        scale_factor += ode_problem.atol.as_ref();

        let mut tmp = f0.clone();
        tmp.component_div_assign(&scale_factor);
        let d0 = tmp.norm();

        tmp = y0.clone();
        tmp.component_div_assign(&scale_factor);
        let d1 = f0.norm();

        let h0 = if d0 < Eqn::T::from(1e-5) || d1 < Eqn::T::from(1e-5) {
            Eqn::T::from(1e-6)
        } else {
            Eqn::T::from(0.01) * (d0 / d1)
        };

        let y1 = f0.clone() * scale(h0) + y0;
        let t1 = t0 + h0;
        let f1 = ode_problem.eqn.rhs().call(&y1, t1);

        let mut df = f1 - f0;
        df *= scale(Eqn::T::one() / h0);
        df.component_div_assign(&scale_factor);
        let d2 = df.norm();

        let mut max_d = d2;
        if max_d < d1 {
            max_d = d1;
        }
        let h1 = if max_d < Eqn::T::from(1e-15) {
            let h1 = h0 * Eqn::T::from(1e-3);
            if h1 < Eqn::T::from(1e-6) {
                Eqn::T::from(1e-6)
            } else {
                h1
            }
        } else {
            (Eqn::T::from(0.01) / max_d)
                .pow(Eqn::T::one() / Eqn::T::from(1.0 + solver_order as f64))
        };

        self.h = Eqn::T::from(100.0) * h0;
        if self.h > h1 {
            self.h = h1;
        }

        // update initial step size based on function
        //let mut scale_factor = state.y.abs();
        //scale_factor *= scale(problem.rtol);
        //scale_factor += problem.atol.as_ref();

        //let f0 = problem.eqn.rhs().call(&state.y, state.t);
        //let hf0 = &f0 * scale(state.h);
        //let y1 = &state.y + &hf0;
        //let t1 = state.t + state.h;
        //let f1 = problem.eqn.rhs().call(&y1, t1);

        //// store f1 in diff[1] for use in step size control
        //self.diff.column_mut(1).copy_from(&hf0);

        //let mut df = f1 - f0;
        //df.component_div_assign(&scale_factor);
        //let d2 = df.norm();

        //let one_over_order_plus_one =
        //    Eqn::T::one() / (Eqn::T::from(self.order as f64) + Eqn::T::one());
        //let mut new_h = state.h * d2.pow(-one_over_order_plus_one);
        //if new_h > Eqn::T::from(100.0) * state.h {
        //    new_h = Eqn::T::from(100.0) * state.h;
        //}
        //state.h = new_h;
    }
}
