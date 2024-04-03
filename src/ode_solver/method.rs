use anyhow::Result;
use std::rc::Rc;

use crate::{
    op::{filter::FilterCallable, ode_rhs::OdeRhs},
    Matrix, NonLinearSolver, OdeEquations, OdeSolverProblem, SolverProblem, Vector, VectorIndex,
};

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
///     let mut state = OdeSolverState::new(problem);
///     solver.set_problem(&mut state, problem);
///     while state.t <= t {
///         solver.step(&mut state).unwrap();
///     }
///     solver.interpolate(&state, t)
/// }
/// ```
pub trait OdeSolverMethod<Eqn: OdeEquations> {
    /// Get the current problem if it has been set
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;

    /// Set the problem to solve, this performs any initialisation required by the solver.
    /// Call this before calling `step` or `solve`, and call it again if the state is changed manually (i.e. not by the solver)
    fn set_problem(&mut self, state: &mut OdeSolverState<Eqn::M>, problem: &OdeSolverProblem<Eqn>);

    /// Step the solution forward by one step, altering the state in place
    fn step(&mut self, state: &mut OdeSolverState<Eqn::M>) -> Result<()>;

    /// Interpolate the solution at a given time. This time should be between the current time and the last solver time step
    fn interpolate(&self, state: &OdeSolverState<Eqn::M>, t: Eqn::T) -> Eqn::V;

    /// Reinitialise the solver state and solve the problem up to time `t`
    fn solve(&mut self, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Result<Eqn::V> {
        let mut state = OdeSolverState::new(problem);
        self.set_problem(&mut state, problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }

    /// Reinitialise the solver state making it consistent with the algebraic constraints and solve the problem up to time `t`
    fn make_consistent_and_solve<RS: NonLinearSolver<FilterCallable<OdeRhs<Eqn>>>>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        t: Eqn::T,
        root_solver: &mut RS,
    ) -> Result<Eqn::V> {
        let mut state = OdeSolverState::new_consistent(problem, root_solver)?;
        self.set_problem(&mut state, problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
}

/// State for the ODE solver, containing the current solution `y`, the current time `t`, and the current step size `h`.
pub struct OdeSolverState<M: Matrix> {
    pub y: M::V,
    pub t: M::T,
    pub h: M::T,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix> OdeSolverState<M> {
    /// Create a new solver state from an ODE problem. Note that this does not make the state consistent with the algebraic constraints.
    /// If you need to make the state consistent, use `new_consistent` instead.
    pub fn new<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init(t);
        Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new solver state from an ODE problem, making the state consistent with the algebraic constraints.
    pub fn new_consistent<Eqn, S>(
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<Self>
    where
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
        S: NonLinearSolver<FilterCallable<OdeRhs<Eqn>>> + ?Sized,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let indices = ode_problem.eqn.algebraic_indices();
        let mut y = ode_problem.eqn.init(t);
        if indices.len() == 0 {
            return Ok(Self {
                y,
                t,
                h,
                _phantom: std::marker::PhantomData,
            });
        }
        let mut y_filtered = y.filter(&indices);
        let atol = Rc::new(ode_problem.atol.as_ref().filter(&indices));
        let rhs = Rc::new(OdeRhs::new(ode_problem.eqn.clone()));
        let f = Rc::new(FilterCallable::new(rhs, &y, indices));
        let rtol = ode_problem.rtol;
        let init_problem = SolverProblem::new(f, t, atol, rtol);
        root_solver.set_problem(init_problem);
        root_solver.solve_in_place(&mut y_filtered)?;
        let init_problem = root_solver.problem().unwrap();
        let indices = init_problem.f.indices();
        y.scatter_from(&y_filtered, indices);
        Ok(Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        })
    }
}
