use nalgebra::ComplexField;
use num_traits::{One, Pow, Zero};
use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError},
    nonlinear_solver::NonLinearSolver,
    ode_solver_error, scale, AugmentedOdeEquations, AugmentedOdeEquationsImplicit, ConstantOp,
    DefaultSolver, InitOp, NewtonNonlinearSolver, NonLinearOp, OdeEquations, OdeEquationsImplicit,
    OdeEquationsSens, OdeSolverMethod, OdeSolverProblem, Op, SensEquations, Vector, LinearSolver,
};

use super::method::SensitivitiesOdeSolverMethod;

/// A state holding those variables that are common to all ODE solver states,
/// can be used to create a new state for a specific solver.
pub struct StateCommon<V: Vector> {
    pub y: V,
    pub dy: V,
    pub g: V,
    pub dg: V,
    pub s: Vec<V>,
    pub ds: Vec<V>,
    pub sg: Vec<V>,
    pub dsg: Vec<V>,
    pub t: V::T,
    pub h: V::T,
}

/// A reference to the state of the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
/// - the sensitivity vectors of the output function `sg`
/// - the derivative of the sensitivity vectors of the output function wrt time `dsg`
pub struct StateRef<'a, V: Vector> {
    pub y: &'a V,
    pub dy: &'a V,
    pub g: &'a V,
    pub dg: &'a V,
    pub s: &'a [V],
    pub ds: &'a [V],
    pub sg: &'a [V],
    pub dsg: &'a [V],
    pub t: V::T,
    pub h: V::T,
}

/// A mutable reference to the state of the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
/// - the sensitivity vectors of the output function `sg`
/// - the derivative of the sensitivity vectors of the output function wrt time `dsg`
pub struct StateRefMut<'a, V: Vector> {
    pub y: &'a mut V,
    pub dy: &'a mut V,
    pub g: &'a mut V,
    pub dg: &'a mut V,
    pub s: &'a mut [V],
    pub ds: &'a mut [V],
    pub sg: &'a mut [V],
    pub dsg: &'a mut [V],
    pub t: &'a mut V::T,
    pub h: &'a mut V::T,
}

/// State for the ODE solver, containing:
/// - the current solution `y`
/// - the derivative of the solution wrt time `dy`
/// - the current integral of the output function `g`
/// - the current derivative of the integral of the output function wrt time `dg`
/// - the current time `t`
/// - the current step size `h`,
/// - the sensitivity vectors `s`
/// - the derivative of the sensitivity vectors wrt time `ds`
///
pub trait OdeSolverState<V: Vector>: Clone + Sized {
    fn as_ref(&self) -> StateRef<V>;
    fn as_mut(&mut self) -> StateRefMut<V>;
    fn into_common(self) -> StateCommon<V>;
    fn new_from_common(state: StateCommon<V>) -> Self;

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
        if self.as_ref().y.len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if self.as_ref().dy.len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    fn check_sens_consistent_with_problem<
        Eqn: OdeEquations,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    >(
        &self,
        problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError> {
        let state = self.as_ref();
        if state.s.len() != augmented_eqn.max_index() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !state.s.is_empty() && state.s[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if state.ds.len() != augmented_eqn.max_index() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !state.ds.is_empty() && state.ds[0].len() != problem.eqn.rhs().nstates() {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    /// Create a new solver state from an ODE problem.
    /// This function will make the state consistent with any algebraic constraints using a default nonlinear solver.
    /// It will also set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    fn new<LS, Eqn>(ode_problem: &OdeSolverProblem<Eqn>, solver_order: usize) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V>,
        LS: LinearSolver<Eqn::M>,
    {
        let mut ret = Self::new_without_initialise(ode_problem)?;
        let mut root_solver =
            NewtonNonlinearSolver::new(LS::default());
        ret.set_consistent(ode_problem, &mut root_solver)?;
        ret.set_step_size(ode_problem, solver_order);
        Ok(ret)
    }

    fn new_with_sensitivities<LS, Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsSens<T = V::T, V = V>,
        LS: LinearSolver<Eqn::M>,
    {
        let augmented_eqn = SensEquations::new(ode_problem);
        Self::new_with_augmented::<LS, _, _>(ode_problem, augmented_eqn, solver_order).map(|(state, _)| state)
    }

    fn new_with_augmented<LS, Eqn, AugmentedEqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        mut augmented_eqn: AugmentedEqn,
        solver_order: usize,
    ) -> Result<(Self, AugmentedEqn), DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V>,
        AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn> + std::fmt::Debug,
        LS: LinearSolver<Eqn::M>,
    {
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;
        let mut root_solver =
            NewtonNonlinearSolver::new(LS::default());
        ret.set_consistent(ode_problem, &mut root_solver)?;
        let mut root_solver_sens =
            NewtonNonlinearSolver::new(<Eqn::M as DefaultSolver>::default_solver());
        let augmented_eqn =
            ret.set_consistent_augmented(ode_problem, augmented_eqn, &mut root_solver_sens)?;
        ret.set_step_size(ode_problem, solver_order);
        Ok((ret, augmented_eqn))
    }

    /// Create a new solver state from an ODE problem, without any initialisation apart from setting the initial time state vector y,
    /// and if applicable the sensitivity vectors s.
    /// This is useful if you want to set up the state yourself, or if you want to use a different nonlinear solver to make the state consistent,
    /// or if you want to set the step size yourself or based on the exact order of the solver.
    fn new_without_initialise<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init().call(t);
        let dy = V::zeros(y.len());
        let (s, ds) = (vec![], vec![]);
        let (dg, g) = if ode_problem.integrate_out {
            let out = ode_problem
                .eqn
                .out()
                .ok_or(ode_solver_error!(StateProblemMismatch))?;
            (out.call(&y, t), V::zeros(out.nout()))
        } else {
            (V::zeros(0), V::zeros(0))
        };
        let (sg, dsg) = (vec![], vec![]);
        let state = StateCommon {
            y,
            dy,
            g,
            dg,
            s,
            ds,
            sg,
            dsg,
            t,
            h,
        };
        Ok(Self::new_from_common(state))
    }

    fn new_without_initialise_augmented<Eqn, AugmentedEqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &mut AugmentedEqn,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let mut state = Self::new_without_initialise(ode_problem)?.into_common();
        let naug = augmented_eqn.max_index();
        let mut s = Vec::with_capacity(naug);
        let mut ds = Vec::with_capacity(naug);
        let nstates = augmented_eqn.rhs().nstates();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            let si = augmented_eqn.init().call(state.t);
            let dsi = V::zeros(nstates);
            s.push(si);
            ds.push(dsi);
        }
        state.s = s;
        state.ds = ds;
        let (dsg, sg) = if augmented_eqn.out().is_some() {
            let mut sg = Vec::with_capacity(naug);
            let mut dsg = Vec::with_capacity(naug);
            for i in 0..naug {
                augmented_eqn.set_index(i);
                let out = augmented_eqn
                    .out()
                    .ok_or(ode_solver_error!(StateProblemMismatch))?;
                let dsgi = out.call(&state.s[i], state.t);
                let sgi = V::zeros(out.nout());
                sg.push(sgi);
                dsg.push(dsgi);
            }
            (dsg, sg)
        } else {
            (vec![], vec![])
        };
        state.sg = sg;
        state.dsg = dsg;
        Ok(Self::new_from_common(state))
    }

    /// Calculate a consistent state and time derivative of the state, based on the equations of the problem.
    fn set_consistent<Eqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V>,
        S: NonLinearSolver<Eqn::M>,
    {
        let state = self.as_mut();
        ode_problem
            .eqn
            .rhs()
            .call_inplace(state.y, *state.t, state.dy);
        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }
        let f = InitOp::new(&ode_problem.eqn, ode_problem.t0, state.y);
        let rtol = ode_problem.rtol;
        let atol = ode_problem.atol.clone();
        root_solver.set_problem(&f, rtol, atol);
        let mut y_tmp = state.dy.clone();
        y_tmp.copy_from_indices(state.y, &f.algebraic_indices);
        let yerr = y_tmp.clone();
        root_solver.reset_jacobian(&f, &y_tmp, *state.t);
        root_solver.solve_in_place(&f, &mut y_tmp, *state.t, &yerr)?;
        f.scatter_soln(&y_tmp, state.y, state.dy);
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
        Eqn: OdeEquationsImplicit<T = V::T, V = V>,
        AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn> + std::fmt::Debug,
        S: NonLinearSolver<AugmentedEqn::M>,
    {
        let state = self.as_mut();
        augmented_eqn.update_rhs_out_state(state.y, state.dy, *state.t);
        let naug = augmented_eqn.max_index();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            augmented_eqn
                .rhs()
                .call_inplace(&state.s[i], *state.t, &mut state.ds[i]);
        }

        if ode_problem.eqn.mass().is_none() {
            return Ok(augmented_eqn);
        }

        let mut augmented_eqn_rc = Rc::new(augmented_eqn);

        for i in 0..naug {
            Rc::get_mut(&mut augmented_eqn_rc).unwrap().set_index(i);
            let f = InitOp::new(&augmented_eqn_rc, ode_problem.t0, &state.s[i]);
            root_solver.set_problem(&f, ode_problem.rtol, ode_problem.atol.clone());

            let mut y = state.ds[i].clone();
            y.copy_from_indices(state.y, &f.algebraic_indices);
            let yerr = y.clone();
            root_solver.reset_jacobian(&f, &y, *state.t);
            root_solver.solve_in_place(&f, &mut y, *state.t, &yerr)?;
            f.scatter_soln(&y, &mut state.s[i], &mut state.ds[i]);
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
        let is_neg_h = ode_problem.h0 < Eqn::T::zero();
        let (h0, h1) = {
            let state = self.as_ref();
            let y0 = state.y;
            let t0 = state.t;
            let f0 = state.dy;

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
            (h0, h1)
        };

        let state = self.as_mut();
        *state.h = Eqn::T::from(100.0) * h0;
        if *state.h > h1 {
            *state.h = h1;
        }

        if is_neg_h {
            *state.h = -*state.h;
        }
    }
}
