use nalgebra::ComplexField;
use num_traits::FromPrimitive;
use num_traits::{One, Pow, Zero};

use crate::{
    error::{DiffsolError, OdeSolverError},
    nonlinear_solver::{convergence::Convergence, NonLinearSolver},
    ode_solver_error, scale, AugmentedOdeEquations, AugmentedOdeEquationsImplicit, ConstantOp,
    InitOp, LinearOp, LinearSolver, Matrix, NewtonNonlinearSolver, NonLinearOp, OdeEquations,
    OdeEquationsImplicit, OdeEquationsImplicitSens, OdeSolverProblem, Op, SensEquations, Vector,
    VectorIndex,
};

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
    fn as_ref(&self) -> StateRef<'_, V>;
    fn as_mut(&mut self) -> StateRefMut<'_, V>;
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
    /// This function will set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    fn new<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let mut ret = Self::new_without_initialise(ode_problem)?;
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    /// Create a new solver state from an ODE problem.
    /// This function will make the state consistent with any algebraic constraints using a default nonlinear solver.
    /// It will also set the initial step size based on the given solver.
    /// If you want to create a state without this default initialisation, use [Self::new_without_initialise] instead.
    /// You can then use [Self::set_consistent] and [Self::set_step_size] to set the state up if you need to.
    fn new_and_consistent<LS, Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        LS: LinearSolver<Eqn::M>,
    {
        let mut ret = Self::new_without_initialise(ode_problem)?;
        let mut root_solver = NewtonNonlinearSolver::new(LS::default());
        ret.set_consistent(ode_problem, &mut root_solver)?;
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    fn new_with_sensitivities<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens<T = V::T, V = V, C = V::C>,
    {
        let mut augmented_eqn = SensEquations::new(ode_problem);
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;

        // eval the rhs since we're not calling set_consistent_augmented
        let state = ret.as_mut();
        augmented_eqn.update_rhs_out_state(state.y, state.dy, *state.t);
        let naug = augmented_eqn.max_index();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            augmented_eqn
                .rhs()
                .call_inplace(&state.s[i], *state.t, &mut state.ds[i]);
        }
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    fn new_with_sensitivities_and_consistent<LS, Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
        solver_order: usize,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens<T = V::T, V = V, C = V::C>,
        LS: LinearSolver<Eqn::M>,
    {
        let mut augmented_eqn = SensEquations::new(ode_problem);
        let mut ret = Self::new_without_initialise_augmented(ode_problem, &mut augmented_eqn)?;
        let mut root_solver = NewtonNonlinearSolver::new(LS::default());
        ret.set_consistent(ode_problem, &mut root_solver)?;
        let mut root_solver_sens = NewtonNonlinearSolver::new(LS::default());
        ret.set_consistent_augmented(ode_problem, &mut augmented_eqn, &mut root_solver_sens)?;
        ret.set_step_size(
            ode_problem.h0,
            &ode_problem.atol,
            ode_problem.rtol,
            &ode_problem.eqn,
            solver_order,
        );
        Ok(ret)
    }

    fn into_adjoint<LS, Eqn, AugmentedEqn>(
        self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &mut AugmentedEqn,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn> + std::fmt::Debug,
        LS: LinearSolver<AugmentedEqn::M>,
    {
        let mut state = self.into_common();
        state.h = -state.h;
        let naug = augmented_eqn.max_index();
        let mut s = Vec::with_capacity(naug);
        let mut ds = Vec::with_capacity(naug);
        let nstates = augmented_eqn.rhs().nstates();
        let ctx = ode_problem.context();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            let si = augmented_eqn.init().call(state.t);
            let dsi = V::zeros(nstates, ctx.clone());
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
                let sgi = V::zeros(out.nout(), ctx.clone());
                sg.push(sgi);
                dsg.push(dsgi);
            }
            (dsg, sg)
        } else {
            (vec![], vec![])
        };
        state.sg = sg;
        state.dsg = dsg;
        let mut state = Self::new_from_common(state);
        let mut root_solver_sens = NewtonNonlinearSolver::new(LS::default());
        state.set_consistent_augmented(ode_problem, augmented_eqn, &mut root_solver_sens)?;
        Ok(state)
    }

    /// Create a new solver state from an ODE problem, without any initialisation apart from setting the initial time state vector y,
    /// the initial time derivative dy and if applicable the sensitivity vectors s.
    /// This is useful if you want to set up the state yourself, or if you want to use a different nonlinear solver to make the state consistent,
    /// or if you want to set the step size yourself or based on the exact order of the solver.
    fn new_without_initialise<Eqn>(
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init().call(t);
        let dy = ode_problem.eqn.rhs().call(&y, t);
        let (s, ds) = (vec![], vec![]);
        let (dg, g) = if ode_problem.integrate_out {
            let out = ode_problem
                .eqn
                .out()
                .ok_or(ode_solver_error!(StateProblemMismatch))?;
            (out.call(&y, t), V::zeros(out.nout(), y.context().clone()))
        } else {
            (
                V::zeros(0, y.context().clone()),
                V::zeros(0, y.context().clone()),
            )
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
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
        AugmentedEqn: AugmentedOdeEquations<Eqn>,
    {
        let mut state = Self::new_without_initialise(ode_problem)?.into_common();
        let naug = augmented_eqn.max_index();
        let mut s = Vec::with_capacity(naug);
        let mut ds = Vec::with_capacity(naug);
        let nstates = augmented_eqn.rhs().nstates();
        let ctx = ode_problem.context();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            let si = augmented_eqn.init().call(state.t);
            let dsi = V::zeros(nstates, ctx.clone());
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
                let sgi = V::zeros(out.nout(), ctx.clone());
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
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
        S: NonLinearSolver<Eqn::M>,
    {
        if ode_problem.eqn.mass().is_none() {
            return Ok(());
        }
        let state = self.as_mut();
        let (algebraic_indices, _) = ode_problem
            .eqn
            .mass()
            .unwrap()
            .matrix(ode_problem.t0)
            .partition_indices_by_zero_diagonal();
        if algebraic_indices.is_empty() {
            return Ok(());
        }

        // equations are:
        // h(t, u, v, du) = 0
        // g(t, u, v) = 0
        // first we solve for du, v
        let f = InitOp::new(
            &ode_problem.eqn,
            ode_problem.t0,
            state.y,
            algebraic_indices.clone(),
        );
        let rtol = ode_problem.rtol;
        let atol = &ode_problem.atol;
        root_solver.set_problem(&f);
        let mut y_tmp = state.dy.clone();
        y_tmp.copy_from_indices(state.y, &f.algebraic_indices);
        let yerr = y_tmp.clone();
        root_solver.reset_jacobian(&f, &y_tmp, *state.t);
        let mut convergence = Convergence::new(rtol, atol);
        root_solver.solve_in_place(&f, &mut y_tmp, *state.t, &yerr, &mut convergence)?;
        f.scatter_soln(&y_tmp, state.y, state.dy);
        // dv is not solved for, so we set it to zero, it will be solved for in the first step of the solver
        state
            .dy
            .assign_at_indices(&algebraic_indices, Eqn::T::zero());
        Ok(())
    }

    /// Calculate the initial sensitivity vectors and their time derivatives, based on the equations of the problem.
    /// Note that this function assumes that the state is already consistent with the algebraic constraints
    /// (either via [Self::set_consistent] or by setting the state up manually).
    fn set_consistent_augmented<Eqn, AugmentedEqn, S>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &mut AugmentedEqn,
        root_solver: &mut S,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicit<T = V::T, V = V, C = V::C>,
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
            return Ok(());
        }

        let mut convergence = Convergence::new(ode_problem.rtol, &ode_problem.atol);
        let (algebraic_indices, _) = ode_problem
            .eqn
            .mass()
            .unwrap()
            .matrix(ode_problem.t0)
            .partition_indices_by_zero_diagonal();
        if algebraic_indices.is_empty() {
            return Ok(());
        }

        for i in 0..naug {
            augmented_eqn.set_index(i);
            let f = InitOp::new(
                augmented_eqn,
                *state.t,
                &state.s[i],
                algebraic_indices.clone(),
            );
            root_solver.set_problem(&f);

            let mut y = state.ds[i].clone();
            y.copy_from_indices(state.y, &f.algebraic_indices);
            let yerr = y.clone();
            root_solver.reset_jacobian(&f, &y, *state.t);
            root_solver.solve_in_place(&f, &mut y, *state.t, &yerr, &mut convergence)?;
            f.scatter_soln(&y, &mut state.s[i], &mut state.ds[i]);
        }
        Ok(())
    }

    /// compute size of first step based on alg in Hairer, Norsett, Wanner
    /// Solving Ordinary Differential Equations I, Nonstiff Problems
    /// Section II.4.2
    /// Note: this assumes that the state is already consistent with the algebraic constraints
    /// and y and dy are already set appropriately
    fn set_step_size<Eqn>(
        &mut self,
        h0: Eqn::T,
        atol: &Eqn::V,
        rtol: Eqn::T,
        eqn: &Eqn,
        solver_order: usize,
    ) where
        Eqn: OdeEquations<T = V::T, V = V, C = V::C>,
    {
        let is_neg_h = h0 < Eqn::T::zero();
        let (h0, h1) = {
            let state = self.as_ref();
            let y0 = state.y;
            let t0 = state.t;
            let f0 = state.dy;

            let d0 = y0.squared_norm(y0, atol, rtol).sqrt();
            let d1 = f0.squared_norm(y0, atol, rtol).sqrt();

            let h0 = if d0 < Eqn::T::from_f64(1e-5).unwrap() || d1 < Eqn::T::from_f64(1e-5).unwrap()
            {
                Eqn::T::from_f64(1e-6).unwrap()
            } else {
                Eqn::T::from_f64(0.01).unwrap() * (d0 / d1)
            };

            // make sure we preserve the sign of h0
            let f1 = if is_neg_h {
                let y1 = f0.clone() * scale(-h0) + y0;
                let t1 = t0 - h0;
                eqn.rhs().call(&y1, t1)
            } else {
                let y1 = f0.clone() * scale(h0) + y0;
                let t1 = t0 + h0;
                eqn.rhs().call(&y1, t1)
            };

            let df = f1 - f0;
            let d2 = df.squared_norm(y0, atol, rtol).sqrt() / h0.abs();

            let mut max_d = d2;
            if max_d < d1 {
                max_d = d1;
            }
            let h1 = if max_d < Eqn::T::from_f64(1e-15).unwrap() {
                let h1 = h0 * Eqn::T::from_f64(1e-3).unwrap();
                if h1 < Eqn::T::from_f64(1e-6).unwrap() {
                    Eqn::T::from_f64(1e-6).unwrap()
                } else {
                    h1
                }
            } else {
                (Eqn::T::from_f64(0.01).unwrap() / max_d)
                    .pow(Eqn::T::one() / Eqn::T::from_f64(1.0 + solver_order as f64).unwrap())
            };
            (h0, h1)
        };

        let state = self.as_mut();
        *state.h = Eqn::T::from_f64(100.0).unwrap() * h0;
        if *state.h > h1 {
            *state.h = h1;
        }

        if is_neg_h {
            *state.h = -*state.h;
        }
    }
}
