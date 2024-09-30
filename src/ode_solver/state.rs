use nalgebra::ComplexField;
use num_traits::{One, Pow, Zero};
use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError}, nonlinear_solver::NonLinearSolver, ode_solver_error, scale, solver::SolverProblem, AugmentedOdeEquations, ConstantOp, DefaultSolver, InitOp, NewtonNonlinearSolver, NonLinearOp, OdeEquations, OdeSolverMethod, OdeSolverProblem, Op, SensEquations, Vector
};

use super::method::SensitivitiesOdeSolverMethod;

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
    fn new_internal_state(y: V, dy: V, s: Vec<V>, ds: Vec<V>, t: <V>::T, h: <V>::T, naug: usize) -> Self;
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError>;
    
    fn set_augmented_problem<Eqn: OdeEquations, AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn,
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

    fn check_sens_consistent_with_problem<Eqn: OdeEquations, AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        &self,
        problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn
    ) -> Result<(), DiffsolError> {
        if self.s().len() != augmented_eqn.max_index() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.s().is_empty() && self.s()[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if self.ds().len() != augmented_eqn.max_index() {
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
        ret.set_step_size(ode_problem, solver.order());
        Ok(ret)
    }

    fn new_with_sensitivities<Eqn, S>(ode_problem: &OdeSolverProblem<Eqn>, solver: &S) -> Result<Self, DiffsolError> 
    where 
        Eqn: OdeEquations<T = V::T, V = V>,
        Eqn::M: DefaultSolver,
        S: SensitivitiesOdeSolverMethod<Eqn>,
    {
        let augmented_eqn = SensEquations::new(&ode_problem.eqn);
        Self::new_with_augmented(ode_problem, augmented_eqn, solver).map(|(state, _)| state)
    }
    
    fn new_with_augmented<Eqn, AugmentedEqn, S>(ode_problem: &OdeSolverProblem<Eqn>, mut augmented_eqn: AugmentedEqn, solver: &S) -> Result<(Self, AugmentedEqn), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        AugmentedEqn: AugmentedOdeEquations<Eqn> + std::fmt::Debug,
        Eqn::M: DefaultSolver,
        S: OdeSolverMethod<Eqn>,
    {
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn);
        let mut root_solver =
            NewtonNonlinearSolver::new(<Eqn::M as DefaultSolver>::default_solver());
        ret.set_consistent(ode_problem, &mut root_solver)?;
        let mut root_solver_sens =
            NewtonNonlinearSolver::new(<Eqn::M as DefaultSolver>::default_solver());
        let augmented_eqn = ret.set_consistent_augmented(ode_problem, augmented_eqn, &mut root_solver_sens)?;
        ret.set_step_size(ode_problem, solver.order());
        Ok((ret, augmented_eqn))
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
        let (s, ds) = (vec![], vec![]);
        Self::new_internal_state(y, dy, s, ds, t, h, 0)
    }
    
    fn new_without_initialise_augmented<Eqn, AugmentedEqn>(ode_problem: &OdeSolverProblem<Eqn>, augmented_eqn: &mut AugmentedEqn) -> Self
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init().call(t);
        let dy = V::zeros(y.len());
        let naug = augmented_eqn.max_index();
        let nstates = ode_problem.eqn.rhs().nstates();
        augmented_eqn.update_init_state(t);
        let mut s = Vec::with_capacity(naug);
        let mut ds = Vec::with_capacity(naug);
        for i in 0..naug {
            augmented_eqn.set_index(i);
            let si = augmented_eqn.init().call(t);
            let dsi = V::zeros(nstates);
            s.push(si);
            ds.push(dsi);
        }
        Self::new_internal_state(y, dy, s, ds, t, h, naug)
    }

    /// Calculate a consistent state and time derivative of the state, based on the equations of the problem.
    fn set_consistent<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        S: NonLinearSolver<InitOp<Eqn>>,
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
    fn set_consistent_augmented<Eqn, AugmentedEqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        mut augmented_eqn: AugmentedEqn,
        root_solver: &mut S,
    ) -> Result<AugmentedEqn, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        AugmentedEqn: AugmentedOdeEquations<Eqn> + std::fmt::Debug,
        S: NonLinearSolver<InitOp<AugmentedEqn>>,
    {
        augmented_eqn.update_rhs_state(self.y(), self.dy(), self.t());
        let naug = augmented_eqn.max_index();
        let t = self.t();
        let (s, ds) = self.s_ds_mut();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            augmented_eqn.rhs().call_inplace(&s[i], t,  &mut ds[i]);
        }

        if ode_problem.eqn.mass().is_none() {
            return Ok(augmented_eqn);
        }
        
        let mut augmented_eqn_rc = Rc::new(augmented_eqn);

        for i in 0..naug {
            Rc::get_mut(&mut augmented_eqn_rc).unwrap().set_index(i);
            let f = Rc::new(InitOp::new(&augmented_eqn_rc, ode_problem.t0, &self.s()[i]));
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
            root_solver.clear_problem();
        }
        Ok(Rc::try_unwrap(augmented_eqn_rc).unwrap())
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
