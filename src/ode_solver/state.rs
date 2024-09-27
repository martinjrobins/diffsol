use nalgebra::ComplexField;
use num_traits::{One, Pow, Zero};
use std::rc::Rc;

use crate::{
    error::DiffsolError, error::OdeSolverError, nonlinear_solver::NonLinearSolver,
    ode_solver_error, scale, solver::SolverProblem, ConstantOp, DefaultSolver, InitOp,
    NewtonNonlinearSolver, NonLinearOp, OdeEquations, OdeSolverMethod, OdeSolverProblem, Op,
    SensEquations, Vector,
};

/// State for the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current time `t`
/// - the current step size `h`,
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
///
pub trait OdeSolverState<V: Vector>: Clone + Sized {
    fn y(&self) -> &V;
    fn y_mut(&mut self) -> &mut V;
    fn dy(&self) -> &V;
    fn dy_mut(&mut self) -> &mut V;
    fn y_dy_mut(&mut self) -> (&mut V, &mut V);
    fn s(&self) -> &[V];
    fn s_mut(&mut self) -> &mut [V];
    fn ds(&self) -> &[V];
    fn ds_mut(&mut self) -> &mut [V];
    fn s_ds_mut(&mut self) -> (&mut [V], &mut [V]);
    fn t(&self) -> V::T;
    fn t_mut(&mut self) -> &mut V::T;
    fn h(&self) -> V::T;
    fn h_mut(&mut self) -> &mut V::T;
    fn new_internal_state(y: V, dy: V, s: Vec<V>, ds: Vec<V>, t: <V>::T, h: <V>::T) -> Self;
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError>;

    fn check_consistent_with_problem<Eqn: OdeEquations>(
        &self,
        problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        if self.y().len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if self.dy().len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    fn check_sens_consistent_with_problem<Eqn: OdeEquations>(
        &self,
        problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        if self.s().len() != problem.eqn.rhs().nparams() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.s().is_empty() && self.s()[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if self.ds().len() != problem.eqn.rhs().nparams() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.ds().is_empty() && self.ds()[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    /// Create a new solver state from an ODE problem.
    /// This function will make the state consistent with any algebraic constraints using a default nonlinear solver.
    /// It will also set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    fn new<Eqn, S>(ode_problem: &OdeSolverProblem<Eqn>, solver: &S) -> Result<Self, DiffsolError>
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
    fn new_without_initialise<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init().call(t);
        let dy = V::zeros(y.len());
        let nparams = ode_problem.eqn.rhs().nparams();
        let (s, ds) = if !ode_problem.with_sensitivity {
            (vec![], vec![])
        } else {
            let mut eqn_sens = SensEquations::new_no_rhs(&ode_problem.eqn);
            eqn_sens.update_init_state(t);
            let mut s = Vec::with_capacity(nparams);
            let mut ds = Vec::with_capacity(nparams);
            for i in 0..nparams {
                eqn_sens.set_param_index(i);
                let si = eqn_sens.init().call(t);
                let dsi = V::zeros(y.len());
                s.push(si);
                ds.push(dsi);
            }
            (s, ds)
        };
        Self::new_internal_state(y, dy, s, ds, t, h)
    }

    /// Calculate a consistent state and time derivative of the state, based on the equations of the problem.
    fn set_consistent<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<InitOp<Eqn>> + ?Sized,
    {
        let t = self.t();
        let (y, dy) = self.y_dy_mut();
        ode_problem.eqn.rhs().call_inplace(y, t, dy);
        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }
        let f = Rc::new(InitOp::new(&ode_problem.eqn, ode_problem.t0, y));
        let rtol = ode_problem.rtol;
        let atol = ode_problem.atol.clone();
        let init_problem = SolverProblem::new(f.clone(), atol, rtol);
        root_solver.set_problem(&init_problem);
        let mut y_tmp = dy.clone();
        y_tmp.copy_from_indices(y, &init_problem.f.algebraic_indices);
        let yerr = y_tmp.clone();
        root_solver.solve_in_place(&mut y_tmp, t, &yerr)?;
        f.scatter_soln(&y_tmp, y, dy);
        Ok(())
    }

    /// Calculate the initial sensitivity vectors and their time derivatives, based on the equations of the problem.
    /// Note that this function assumes that the state is already consistent with the algebraic constraints
    /// (either via [Self::set_consistent] or by setting the state up manually).
    fn set_consistent_sens<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<InitOp<SensEquations<Eqn>>> + ?Sized,
    {
        if !ode_problem.with_sensitivity {
            return Ok(());
        }

        let mut eqn_sens = Rc::new(SensEquations::new(&ode_problem.eqn));
        Rc::get_mut(&mut eqn_sens).unwrap().update_rhs_state(self.y(), self.dy(), self.t());
        let t = self.t();
        let (s, ds) = self.s_ds_mut();
        for i in 0..ode_problem.eqn.rhs().nparams() {
            Rc::get_mut(&mut eqn_sens).unwrap().set_param_index(i);
            eqn_sens.rhs().call_inplace(&s[i], t, &mut ds[i]);
        }

        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }

        for i in 0..ode_problem.eqn.rhs().nparams() {
            Rc::get_mut(&mut eqn_sens).unwrap().set_param_index(i);
            let f = Rc::new(InitOp::new(&eqn_sens, ode_problem.t0, &self.s()[i]));
            root_solver.set_problem(&SolverProblem::new(
                f.clone(),
                ode_problem.atol.clone(),
                ode_problem.rtol,
            ));

            let mut y = self.ds()[i].clone();
            y.copy_from_indices(self.y(), &f.algebraic_indices);
            let yerr = y.clone();
            root_solver.solve_in_place(&mut y, self.t(), &yerr)?;
            let (s, ds) = self.s_ds_mut();
            f.scatter_soln(&y, &mut s[i], &mut ds[i]);
        }
        Ok(())
    }

    /// compute size of first step based on alg in Hairer, Norsett, Wanner
    /// Solving Ordinary Differential Equations I, Nonstiff Problems
    /// Section II.4.2
    /// Note: this assumes that the state is already consistent with the algebraic constraints
    /// and y and dy are already set appropriately
    fn set_step_size<Eqn>(&mut self, ode_problem: &OdeSolverProblem<Eqn>, solver_order: usize)
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let y0 = self.y();
        let t0 = self.t();
        let f0 = self.dy();

        let rtol = ode_problem.rtol;
        let atol = ode_problem.atol.as_ref();

        let d0 = y0.squared_norm(y0, atol, rtol).sqrt();
        let d1 = f0.squared_norm(y0, atol, rtol).sqrt();

        let h0 = if d0 < Eqn::T::from(1e-5) || d1 < Eqn::T::from(1e-5) {
            Eqn::T::from(1e-6)
        } else {
            Eqn::T::from(0.01) * (d0 / d1)
        };

        // make sure we preserve the sign of h0
        let is_neg_h = ode_problem.h0 < Eqn::T::zero();

        let f1 = if is_neg_h {
            let y1 = f0.clone() * scale(-h0) + y0;
            let t1 = t0 - h0;
            ode_problem.eqn.rhs().call(&y1, t1)
        } else {
            let y1 = f0.clone() * scale(h0) + y0;
            let t1 = t0 + h0;
            ode_problem.eqn.rhs().call(&y1, t1)
        };

        let df = f1 - f0;
        let d2 = df.squared_norm(y0, atol, rtol).sqrt() / h0.abs();

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

        *self.h_mut() = Eqn::T::from(100.0) * h0;
        if self.h() > h1 {
            *self.h_mut() = h1;
        }

        if is_neg_h {
            *self.h_mut() = -self.h();
        }
    }
}
