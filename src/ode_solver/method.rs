use nalgebra::ComplexField;
use num_traits::{One, Pow};
use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError}, matrix::default_solver::DefaultSolver, scalar::Scalar, scale, ConstantOp,
    InitOp, NewtonNonlinearSolver, NonLinearOp, NonLinearSolver, OdeEquations, OdeSolverProblem,
    Op, SensEquations, SolverProblem, Vector,
};

#[derive(Debug, PartialEq)]
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
    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError>;

    /// Set a stop time for the solver. The solver will stop when the internal time reaches this time.
    /// Once it stops, the stop time is unset. If `tstop` is at or before the current internal time, an error is returned.
    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError>;

    /// Interpolate the solution at a given time. This time should be between the current time and the last solver time step
    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError>;

    /// Interpolate the sensitivity vectors at a given time. This time should be between the current time and the last solver time step
    fn interpolate_sens(&self, t: Eqn::T) -> Result<Vec<Eqn::V>, DiffsolError>;

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

    /// Reinitialise the solver state and solve the problem up to time `final_time`
    /// Returns a Vec of solution values at timepoints chosen by the solver.
    /// After the solver has finished, the internal state of the solver is at time `final_time`.
    #[allow(clippy::type_complexity)]
    fn solve(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        final_time: Eqn::T,
    ) -> Result<(Vec<Eqn::V>, Vec<Eqn::T>), DiffsolError>
    where
        Eqn::M: DefaultSolver,
        Self: Sized,
    {
        let state = OdeSolverState::new(problem, self)?;
        self.set_problem(state, problem);
        let mut t = vec![self.state().unwrap().t];
        let mut y = vec![];
        match problem.eqn.out() {
            Some(out) => y.push(out.call(&self.state().unwrap().y, self.state().unwrap().t)),
            None => y.push(self.state().unwrap().y.clone()),
        }
        self.set_stop_time(final_time)?;
        while self.step()? != OdeSolverStopReason::TstopReached {
            t.push(self.state().unwrap().t);
            match problem.eqn.out() {
                Some(out) => y.push(out.call(&self.state().unwrap().y, self.state().unwrap().t)),
                None => y.push(self.state().unwrap().y.clone()),
            }
        }
        Ok((y, t))
    }

    /// Reinitialise the solver state and solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a Vec of solution values at timepoints given by `t_eval`.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    fn solve_dense(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        t_eval: &[Eqn::T],
    ) -> Result<Vec<Eqn::V>, DiffsolError>
    where
        Eqn::M: DefaultSolver,
        Self: Sized,
    {
        let state = OdeSolverState::new(problem, self)?;
        self.set_problem(state, problem);
        let mut ret = vec![];

        // check t_eval is increasing and all values are greater than or equal to the current time
        let t0 = self.state().unwrap().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(DiffsolError::from(OdeSolverError::InvalidTEval));
        }

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for t in t_eval.iter().take(t_eval.len() - 1) {
            while self.state().unwrap().t < *t {
                step_reason = self.step()?;
            }
            let y = self.interpolate(*t)?;
            match problem.eqn.out() {
                Some(out) => ret.push(out.call(&y, *t)),
                None => ret.push(y),
            }
        }

        // do final step
        while step_reason != OdeSolverStopReason::TstopReached {
            step_reason = self.step()?;
        }
        match problem.eqn.out() {
            Some(out) => ret.push(out.call(&self.state().unwrap().y, self.state().unwrap().t)),
            None => ret.push(self.state().unwrap().y.clone()),
        }
        Ok(ret)
    }
}

/// State for the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current time `t`
/// - the current step size `h`,
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
///
#[derive(Clone)]
pub struct OdeSolverState<V: Vector> {
    pub y: V,
    pub dy: V,
    pub s: Vec<V>,
    pub ds: Vec<V>,
    pub t: V::T,
    pub h: V::T,
}

impl<V: Vector> OdeSolverState<V> {
    /// Create a new solver state from an ODE problem.
    /// This function will make the state consistent with any algebraic constraints using a default nonlinear solver.
    /// It will also set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    pub fn new<Eqn, S>(ode_problem: &OdeSolverProblem<Eqn>, solver: &S) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        Eqn::M: DefaultSolver,
        S: OdeSolverMethod<Eqn>,
    {
        let mut ret = Self::new_without_initialise(ode_problem);
        let mut root_solver =
            NewtonNonlinearSolver::new(<Eqn::M as DefaultSolver>::default_solver());
        ret.set_consistent(ode_problem, &mut root_solver)?;
        let mut root_solver_sens =
            NewtonNonlinearSolver::new(<Eqn::M as DefaultSolver>::default_solver());
        ret.set_consistent_sens(ode_problem, &mut root_solver_sens)?;
        ret.set_step_size(ode_problem, solver.order());
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem, without any initialisation apart from setting the initial time state vector y,
    /// and if applicable the sensitivity vectors s.
    /// This is useful if you want to set up the state yourself, or if you want to use a different nonlinear solver to make the state consistent,
    /// or if you want to set the step size yourself or based on the exact order of the solver.
    pub fn new_without_initialise<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init().call(t);
        let dy = V::zeros(y.len());
        let nparams = ode_problem.eqn.rhs().nparams();
        let (s, ds) = if ode_problem.eqn_sens.is_none() {
            (vec![], vec![])
        } else {
            let eqn_sens = ode_problem.eqn_sens.as_ref().unwrap();
            eqn_sens.init().update_state(t);
            let mut s = Vec::with_capacity(nparams);
            let mut ds = Vec::with_capacity(nparams);
            for i in 0..nparams {
                eqn_sens.init().set_param_index(i);
                let si = eqn_sens.init().call(t);
                let dsi = V::zeros(y.len());
                s.push(si);
                ds.push(dsi);
            }
            (s, ds)
        };
        Self { y, t, h, dy, s, ds }
    }

    /// Calculate a consistent state and time derivative of the state, based on the equations of the problem.
    pub fn set_consistent<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<InitOp<Eqn>> + ?Sized,
    {
        ode_problem
            .eqn
            .rhs()
            .call_inplace(&self.y, self.t, &mut self.dy);
        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }
        let f = Rc::new(InitOp::new(&ode_problem.eqn, ode_problem.t0, &self.y));
        let rtol = ode_problem.rtol;
        let atol = ode_problem.atol.clone();
        let init_problem = SolverProblem::new(f.clone(), atol, rtol);
        root_solver.set_problem(&init_problem);
        let mut y = self.dy.clone();
        y.copy_from_indices(&self.y, &init_problem.f.algebraic_indices);
        let yerr = y.clone();
        root_solver.solve_in_place(&mut y, self.t, &yerr)?;
        f.scatter_soln(&y, &mut self.y, &mut self.dy);
        Ok(())
    }

    /// Calculate the initial sensitivity vectors and their time derivatives, based on the equations of the problem.
    /// Note that this function assumes that the state is already consistent with the algebraic constraints
    /// (either via [Self::set_consistent] or by setting the state up manually).
    pub fn set_consistent_sens<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<InitOp<SensEquations<Eqn>>> + ?Sized,
    {
        if ode_problem.eqn_sens.is_none() {
            return Ok(());
        }

        let eqn_sens = ode_problem.eqn_sens.as_ref().unwrap();
        eqn_sens.rhs().update_state(&self.y, &self.dy, self.t);
        for i in 0..ode_problem.eqn.rhs().nparams() {
            eqn_sens.init().set_param_index(i);
            eqn_sens.rhs().set_param_index(i);
            eqn_sens
                .rhs()
                .call_inplace(&self.s[i], self.t, &mut self.ds[i]);
        }

        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }

        for i in 0..ode_problem.eqn.rhs().nparams() {
            eqn_sens.init().set_param_index(i);
            eqn_sens.rhs().set_param_index(i);
            let f = Rc::new(InitOp::new(eqn_sens, ode_problem.t0, &self.s[i]));
            root_solver.set_problem(&SolverProblem::new(
                f.clone(),
                ode_problem.atol.clone(),
                ode_problem.rtol,
            ));

            let mut y = self.ds[i].clone();
            y.copy_from_indices(&self.y, &f.algebraic_indices);
            let yerr = y.clone();
            root_solver.solve_in_place(&mut y, self.t, &yerr)?;
            f.scatter_soln(&y, &mut self.s[i], &mut self.ds[i]);
        }
        Ok(())
    }

    /// compute size of first step based on alg in Hairer, Norsett, Wanner
    /// Solving Ordinary Differential Equations I, Nonstiff Problems
    /// Section II.4.2
    /// Note: this assumes that the state is already consistent with the algebraic constraints
    /// and y and dy are already set appropriately
    pub fn set_step_size<Eqn>(&mut self, ode_problem: &OdeSolverProblem<Eqn>, solver_order: usize)
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let y0 = &self.y;
        let t0 = self.t;
        let f0 = &self.dy;

        let rtol = ode_problem.rtol;
        let atol = ode_problem.atol.as_ref();

        let d0 = y0.squared_norm(y0, atol, rtol).sqrt();
        let d1 = f0.squared_norm(y0, atol, rtol).sqrt();

        let h0 = if d0 < Eqn::T::from(1e-5) || d1 < Eqn::T::from(1e-5) {
            Eqn::T::from(1e-6)
        } else {
            Eqn::T::from(0.01) * (d0 / d1)
        };

        let y1 = f0.clone() * scale(h0) + y0;
        let t1 = t0 + h0;
        let f1 = ode_problem.eqn.rhs().call(&y1, t1);

        let df = f1 - f0;
        let d2 = df.squared_norm(y0, atol, rtol).sqrt() / h0;

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
    }
}
