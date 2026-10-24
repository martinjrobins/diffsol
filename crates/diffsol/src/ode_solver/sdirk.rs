use crate::error::DiffsolError;
use crate::error::OdeSolverError;
use crate::matrix::MatrixRef;
use crate::ode_solver::runge_kutta::Rk;
use crate::ode_solver_error;
use crate::vector::VectorRef;
use crate::LinearSolver;
use crate::NewtonNonlinearSolver;
use crate::NoAug;
use crate::OdeSolverStopReason;
use crate::RkState;
use crate::Tableau;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, AugmentedOdeEquations,
    AugmentedOdeEquationsImplicit, Convergence, DefaultDenseMatrix, DenseMatrix, JacobianUpdate,
    Matrix, NoLineSearch, OdeEquationsImplicit, OdeEquationsImplicitAdjoint,
    OdeEquationsImplicitSens, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, SensEquations,
    StateRef, StateRefMut, Vector,
};
use log::debug;
use log::trace;
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

use super::adjoint::AdjointOdeSolverMethod;
use super::config::SdirkConfig;
use super::jacobian_update::{solve_with_jacobian_updates, SolverState};
use super::method::AugmentedOdeSolverMethod;
use super::sensitivities::SensitivitiesOdeSolverMethod;
use super::step_ad::StepVjp;
use super::OdeSolverStatistics;

impl<'a, M, Eqn, LS, AugEqn> AugmentedOdeSolverMethod<'a, Eqn, AugEqn>
    for Sdirk<'a, Eqn, LS, M, AugEqn>
where
    Eqn: OdeEquationsImplicit,
    AugEqn: AugmentedOdeEquationsImplicit<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugEqn>) {
        (self.rk.into_state(), self.s_op.map(|op| op.eqn))
    }
    fn augmented_eqn(&self) -> Option<&AugEqn> {
        self.s_op.as_ref().map(|op| op.eqn())
    }
    fn augmented_eqn_mut(&mut self) -> Option<&mut AugEqn> {
        self.s_op.as_mut().map(|op| &mut op.eqn)
    }
    fn state_and_augmented_eqn_mut(&mut self) -> Option<(StateRefMut<'_, Eqn::V>, &mut AugEqn)> {
        let state = self.rk.state_mut().as_mut();
        let augmented_eqn = self.s_op.as_mut().map(|op| &mut op.eqn)?;
        Some((state, augmented_eqn))
    }
}

impl<'a, M, Eqn, LS> SensitivitiesOdeSolverMethod<'a, Eqn>
    for Sdirk<'a, Eqn, LS, M, SensEquations<'a, Eqn>>
where
    Eqn: OdeEquationsImplicitSens + 'a,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
}

impl<'a, M, Eqn, LS, Solver> AdjointOdeSolverMethod<'a, Eqn, Solver>
    for Sdirk<'a, Eqn, LS, M, crate::AdjointEquations<'a, Eqn, Solver>>
where
    Eqn: crate::OdeEquationsImplicitAdjoint + 'a,
    Solver: OdeSolverMethod<'a, Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
}

/// A singly diagonally implicit Runge-Kutta method. Can optionally have an explicit first stage for ESDIRK methods.
///
/// The particular method is defined by the [Tableau] used to create the solver.
/// If the `beta` matrix of the [Tableau] is present this is used for interpolation, otherwise hermite interpolation is used.
///
/// Restrictions:
/// - The upper triangular part of the `a` matrix must be zero (i.e. not fully implicit).
/// - The diagonal of the `a` matrix must be the same non-zero value for all rows (i.e. an SDIRK method), except for the first row which can be zero for ESDIRK methods.
/// - The last row of the `a` matrix must be the same as the `b` vector, and the last element of the `c` vector must be 1 (i.e. a stiffly accurate method)
pub struct Sdirk<
    'a,
    Eqn,
    LS,
    M = <<Eqn as Op>::V as DefaultDenseMatrix>::M,
    AugmentedEqn = NoAug<Eqn>,
> where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    rk: Rk<'a, Eqn, M>,
    nonlinear_solver: NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
    convergence: Convergence<'a, Eqn::V>,
    /// Convergence of the sensitivity solves alongside the main equations
    s_convergence: Convergence<'a, Eqn::V>,
    vjp_convergence: Convergence<'a, Eqn::V>,
    /// Zero, the stage increment a [StepVjp::step_vjp] stage solve is linearised at: it sets the
    /// stage's known part to the whole stage value.
    vjp_zero: Eqn::V,
    op: Option<SdirkCallable<&'a Eqn>>,
    s_op: Option<SdirkCallable<AugmentedEqn>>,
    jacobian_update: JacobianUpdate<Eqn::T>,
    config: SdirkConfig<Eqn::T>,
}

impl<M, Eqn, LS, AugmentedEqn> Clone for Sdirk<'_, Eqn, LS, M, AugmentedEqn>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
{
    fn clone(&self) -> Self {
        let mut nonlinear_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
        let op = if let Some(op) = &self.op {
            let op = op.clone_state(&self.problem().eqn);
            nonlinear_solver.set_problem(&op);
            Some(op)
        } else {
            None
        };
        let s_op = self.s_op.as_ref().map(|op| {
            let op = op.clone_state(op.eqn().clone());
            op
        });
        let mut convergence = self.convergence.clone();
        convergence.reset_eta();
        let mut s_convergence = self.s_convergence.clone();
        s_convergence.reset_eta();
        let mut vjp_convergence = self.vjp_convergence.clone();
        vjp_convergence.reset_eta();
        Self {
            rk: self.rk.clone(),
            convergence,
            s_convergence,
            vjp_convergence,
            vjp_zero: self.vjp_zero.clone(),
            nonlinear_solver,
            op,
            s_op,
            jacobian_update: self.jacobian_update.clone(),
            config: self.config.clone(),
        }
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> Sdirk<'a, Eqn, LS, M, AugmentedEqn>
where
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
{
    fn gamma(&self) -> Eqn::T {
        self.rk.tableau().a(1, 1)
    }

    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V, M>,
        tableau: Tableau<Eqn::T>,
        linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        Rk::<Eqn, M>::check_sdirk_rk(&tableau)?;
        let rk = Rk::new(problem, state, tableau)?;
        let mut ret = Self::_new(
            rk,
            problem,
            linear_solver,
            true,
            SdirkConfig::new(&problem.ode_options),
        )?;
        ret.nonlinear_solver.set_problem(ret.op.as_ref().unwrap());
        Ok(ret)
    }

    fn _new(
        rk: Rk<'a, Eqn, M>,
        problem: &'a OdeSolverProblem<Eqn>,
        linear_solver: LS,
        integrate_main_eqn: bool,
        config: SdirkConfig<Eqn::T>,
    ) -> Result<Self, DiffsolError> {
        let state = rk.state();

        // setup linear solver for first step
        let mut jacobian_update = JacobianUpdate::new(&problem.ode_options);
        jacobian_update.update_jacobian(state.h);
        jacobian_update.update_rhs_jacobian(state.h);

        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver, NoLineSearch);

        // set max iterations for nonlinear solver
        let mut convergence = Convergence::with_tolerance(
            problem.rtol,
            &problem.atol,
            problem.ode_options.nonlinear_solver_tolerance,
        );
        convergence.set_max_iter(problem.ode_options.max_nonlinear_solver_iterations);
        // the derivative solves (the sensitivities alongside the main equations, and step_vjp)
        // have settings of their own
        let mut derivative_convergence = convergence.clone();
        derivative_convergence.set_tolerance(problem.ode_options.derivative_solve_tolerance);
        derivative_convergence.set_max_iter(problem.ode_options.derivative_solve_max_iter);
        let s_convergence = derivative_convergence.clone();
        // The first update of a transposed solve in StepVjp::step_vjp, w1 = P^-T b, can be
        // dominated by error components the preconditioner removes almost at once, so a rate
        // measured from it can be far faster than the one the rest of the error contracts at.
        let mut vjp_convergence = derivative_convergence;
        vjp_convergence.set_transient_updates(1);
        let vjp_zero =
            <Eqn::V as Vector>::zeros(problem.eqn.rhs().nstates(), problem.context().clone());

        let gamma = rk.tableau().a(1, 1);
        let op = if integrate_main_eqn {
            let callable = SdirkCallable::new(&problem.eqn, gamma, problem.context().clone());
            callable.set_h(state.h);
            Some(callable)
        } else {
            None
        };

        Ok(Self {
            rk,
            convergence,
            s_convergence,
            vjp_convergence,
            vjp_zero,
            nonlinear_solver,
            op,
            s_op: None,
            jacobian_update,
            config,
        })
    }

    pub fn new_augmented(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V, M>,
        tableau: Tableau<Eqn::T>,
        linear_solver: LS,
        augmented_eqn: AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        Rk::<Eqn, M>::check_sdirk_rk(&tableau)?;
        let integrate_main_eqn = augmented_eqn.integrate_main_eqn();
        let rk = Rk::new_augmented(problem, state, tableau, &augmented_eqn)?;
        let mut ret = Self::_new(
            rk,
            problem,
            linear_solver,
            integrate_main_eqn,
            SdirkConfig::new(&problem.ode_options),
        )?;

        let aug_ctx = augmented_eqn.aug_context().clone();
        ret.s_op = if integrate_main_eqn {
            ret.nonlinear_solver.set_problem(ret.op.as_ref().unwrap());
            let callable = SdirkCallable::new_no_jacobian(augmented_eqn, ret.gamma(), aug_ctx);
            callable.set_h(ret.rk.state().h);
            Some(callable)
        } else {
            let state = ret.rk.state();
            let callable = SdirkCallable::new(augmented_eqn, ret.gamma(), aug_ctx);
            callable.set_h(state.h);
            ret.nonlinear_solver.set_problem(&callable);
            Some(callable)
        };
        ret.jacobian_updates_at(ret.rk.state().h, SolverState::Checkpoint, None);
        Ok(ret)
    }

    /// Update the Newton matrix as the Jacobian update schedule says for `state`, with `h` the
    /// step size it is for. A refreshed Jacobian is evaluated at `at` (a state and time), or at
    /// the solver's state if `None`. Returns whether the matrix was updated.
    fn jacobian_updates_at(
        &mut self,
        h: Eqn::T,
        state: SolverState,
        at: Option<(&Eqn::V, Eqn::T)>,
    ) -> bool {
        if self.jacobian_update.check_rhs_jacobian_update(h, &state) {
            return self.refresh_jacobian_at(h, state, at);
        }
        let did_update = if self.jacobian_update.check_jacobian_update(h, &state) {
            // the rhs Jacobian is kept, so only h has changed and the point is not used
            let (y, t) = at.unwrap_or((&self.rk.state().y, self.rk.state().t));
            let did_reset = if let Some(op) = self.op.as_ref() {
                self.nonlinear_solver.reset_jacobian(op, y, t);
                true
            } else if let Some(s_op) = self.s_op.as_ref() {
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.rk.state().s, self.rk.state().t);
                true
            } else {
                false
            };
            self.jacobian_update.update_jacobian(h);
            self.reset_etas();
            did_reset
        } else {
            false
        };

        if did_update {
            self.rk.statistics_mut().record_linear_solver_setup(state);
        }
        did_update
    }

    /// Evaluate the rhs Jacobian at `at` (a state and time), or at the solver's state if `None`,
    /// and rebuild the Newton matrix from it for step size `h`, recorded as an update for
    /// `state`. Returns whether the matrix was updated.
    fn refresh_jacobian_at(
        &mut self,
        h: Eqn::T,
        state: SolverState,
        at: Option<(&Eqn::V, Eqn::T)>,
    ) -> bool {
        // a state, not a stage increment: the rhs Jacobian is evaluated there directly, and the
        // Newton matrix combined from it, whatever stage the op is set up for
        let (y, t) = at.unwrap_or((&self.rk.state().y, self.rk.state().t));
        let did_update = if let Some(op) = self.op.as_ref() {
            op.refresh_rhs_jacobian(y, t);
            self.nonlinear_solver.reset_jacobian(op, y, t);
            true
        } else if let Some(s_op) = self.s_op.as_mut() {
            s_op.set_jacobian_is_stale();
            self.nonlinear_solver
                .reset_jacobian(s_op, &self.rk.state().s, self.rk.state().t);
            true
        } else {
            false
        };
        self.jacobian_update.update_rhs_jacobian(h);
        self.jacobian_update.update_jacobian(h);
        self.reset_etas();
        if did_update {
            self.rk.statistics_mut().record_linear_solver_setup(state);
        }
        did_update
    }

    /// The Newton matrix has changed, so forget the convergence rates measured with the old one.
    fn reset_etas(&mut self) {
        self.convergence.reset_eta();
        self.s_convergence.reset_eta();
        self.vjp_convergence.reset_eta();
    }

    /// The step size changed, so the convergence rates measured at the old one are optimistic.
    fn reset_etas_timestep_change(&mut self) {
        self.convergence.reset_eta_timestep_change();
        self.s_convergence.reset_eta_timestep_change();
    }

    /// Solve stage `i` of the augmented equations, if any. With the main equations, these are
    /// the forward sensitivities, a linear solve with the stage's Jacobian, the derivative of
    /// the stage, converged to the derivative tolerance: if the Newton matrix is too poor a
    /// preconditioner for that, it is updated as for a failed Newton solve and the solve retried
    /// once, and only then does the step attempt fail. Without them (an adjoint solve backwards)
    /// it is the stage's own solve, and a failure fails the step attempt.
    fn do_stage_sens(&mut self, i: usize, h: Eqn::T) -> Result<(), DiffsolError> {
        if self.s_op.is_none() {
            return Ok(());
        }
        let retries: &[_] = if self.op.is_some() {
            &[SolverState::FirstConvergenceFail]
        } else {
            &[]
        };
        solve_with_jacobian_updates(
            self,
            retries,
            |s| {
                let convergence = if s.op.is_some() {
                    &mut s.s_convergence
                } else {
                    &mut s.convergence
                };
                s.rk.do_stage_sdirk_sens(
                    i,
                    h,
                    s.s_op.as_mut().unwrap(),
                    &mut s.nonlinear_solver,
                    convergence,
                )
            },
            |s, state| s.jacobian_updates_at(h, state, None),
        )
    }

    fn update_op_step_size(&mut self, h: Eqn::T) {
        // update h for new step size
        if let Some(op) = self.op.as_mut() {
            op.set_h(h);
        }
        if let Some(s_op) = self.s_op.as_mut() {
            s_op.set_h(h);
        }
    }
}

impl<'a, M, Eqn, AugmentedEqn, LS> OdeSolverMethod<'a, Eqn> for Sdirk<'a, Eqn, LS, M, AugmentedEqn>
where
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
{
    type State = RkState<Eqn::V, M>;
    type Config = SdirkConfig<Eqn::T>;

    fn config(&self) -> &SdirkConfig<Eqn::T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut SdirkConfig<Eqn::T> {
        &mut self.config
    }

    fn get_statistics(&self) -> &OdeSolverStatistics {
        self.rk.get_statistics()
    }

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.rk.problem()
    }

    fn jacobian(&self) -> Option<std::cell::Ref<'_, <Eqn>::M>> {
        let t = self.rk.state().t;
        if let Some(op) = self.op.as_ref() {
            let x = &self.rk.state().y;
            Some(op.rhs_jac(x, t))
        } else {
            let x = &self.rk.state().s;
            self.s_op.as_ref().map(|s_op| s_op.rhs_jac(x, t))
        }
    }

    fn mass(&self) -> Option<std::cell::Ref<'_, <Eqn>::M>> {
        let t = self.rk.state().t;
        if let Some(op) = self.op.as_ref() {
            Some(op.mass(t))
        } else {
            self.s_op.as_ref().map(|s_op| s_op.mass(t))
        }
    }

    fn order(&self) -> usize {
        self.rk.order()
    }

    fn apply_reset(&mut self) -> Result<(), DiffsolError> {
        let problem = self.problem();
        self.rk
            .state_mut()
            .as_mut()
            .apply_reset_with_mass::<LS, _>(problem)
    }

    fn apply_reset_with_sens(&mut self, root_idx: usize) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let problem = self.problem();
        self.rk
            .state_mut()
            .as_mut()
            .apply_reset_with_sens_mass::<LS, _>(problem, root_idx)
    }
    fn set_state(&mut self, state: Self::State) {
        let h = state.h;
        self.rk.set_state(state);

        self.update_op_step_size(h);

        // reinitialise jacobian updates as if a checkpoint was taken
        self.jacobian_updates_at(h, SolverState::Checkpoint, None);
    }

    fn into_state(self) -> RkState<Eqn::V, M> {
        self.rk.into_state()
    }

    fn checkpoint(&mut self) -> Self::State {
        self.jacobian_updates_at(self.rk.state().h, SolverState::Checkpoint, None);
        self.rk.state().clone()
    }

    fn state_clone(&self) -> Self::State {
        self.rk.state().clone()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        debug!(
            "Taking SDIRK step at time {}, step size {}",
            self.rk.state().t,
            self.rk.state().h
        );
        let mut h = self.rk.start_step()?;

        if h.abs() < self.config.minimum_timestep {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.rk.state().t.to_f64().unwrap(),
            }));
        }

        // setup the operators for the step
        self.update_op_step_size(h);

        // loop until step is accepted
        let mut nattempts = 0;
        let mut nonlinear_failures = 0;
        let mut updated_jacobian = false;
        let start = if self.rk.skip_first_stage() { 1 } else { 0 };
        let (factor, error_norm) = 'step: loop {
            // start a step attempt
            self.rk
                .start_step_attempt(h, self.s_op.as_mut().map(|s_op| s_op.eqn_mut()));
            for i in start..self.rk.tableau().s() {
                trace!("SDIRK Stage {}:", i);
                let failed = if self
                    .rk
                    .do_stage_sdirk(
                        i,
                        h,
                        self.op.as_ref(),
                        &mut self.nonlinear_solver,
                        &mut self.convergence,
                    )
                    .is_err()
                {
                    true
                } else if self.do_stage_sens(i, h).is_err() {
                    // with the main equations, the sensitivity solve has already updated the
                    // Newton matrix as for a first failure, and retried: cut the step
                    if self.op.is_some() {
                        updated_jacobian = true;
                    }
                    true
                } else {
                    false
                };
                if failed {
                    if !updated_jacobian {
                        // newton iteration did not converge, so update jacobian and try again
                        debug!("First convergence failure, updating Jacobian and trying again",);
                        updated_jacobian = true;
                        self.jacobian_updates_at(h, SolverState::FirstConvergenceFail, None);
                    } else {
                        // newton iteration did not converge and jacobian has been updated, so we reduce step size and try again
                        h *= Eqn::T::from_f64(0.3).unwrap();
                        debug!(
                            "Second convergence failure, reducing step size to {:.3e} and trying again",
                            h.to_f64().unwrap()
                        );
                        self.reset_etas_timestep_change();
                        self.update_op_step_size(h);
                        self.jacobian_updates_at(h, SolverState::SecondConvergenceFail, None);
                    }
                    self.rk.reset_prev_error();
                    nonlinear_failures += 1;
                    self.rk.solve_fail(
                        h,
                        nonlinear_failures,
                        self.config.minimum_timestep,
                        self.config.maximum_newton_fails,
                    )?;
                    // try again....
                    continue 'step;
                }
            }
            let error_norm = if self.problem().eqn().mass().is_some() && self.op.is_some() {
                let linear_solver = |x: &mut Eqn::V| {
                    let mass = self.op.as_ref().unwrap().current_mass();
                    let new_x = x.clone();
                    mass.gemv(Eqn::T::one(), &new_x, Eqn::T::zero(), x);
                    self.nonlinear_solver.solve_linearised_in_place(x)
                };
                self.rk.error_norm(
                    h,
                    self.s_op.as_mut().map(|s_op| s_op.eqn_mut()),
                    linear_solver,
                )?
            } else {
                let linear_solver =
                    |x: &mut Eqn::V| self.nonlinear_solver.solve_linearised_in_place(x);
                // let linear_solver = |_x: &mut Eqn::V| Ok(());
                self.rk.error_norm(
                    h,
                    self.s_op.as_mut().map(|s_op| s_op.eqn_mut()),
                    linear_solver,
                )?
            };
            //let error_norm = self.rk.error_norm(h, self.s_op.as_mut().map(|s_op| s_op.eqn_mut()), |_err: &mut Eqn::V| Ok(()))?;

            let maxiter = self.convergence.max_iter() as f64;
            let niter = self.convergence.niter() as f64;
            let safety_factor = (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter);
            let factor = self.rk.factor(
                error_norm,
                safety_factor,
                self.config.minimum_timestep_shrink,
                self.config.maximum_timestep_shrink,
                self.config.minimum_timestep_growth,
                self.config.maximum_timestep_growth,
            );
            if error_norm < Eqn::T::one() {
                debug!("Step accepted with error norm {}", error_norm,);
                break (factor, error_norm);
            }
            h *= factor;
            debug!(
                "Step rejected with error norm {}, reducing step size to {} and trying again",
                error_norm, h,
            );

            self.reset_etas_timestep_change();
            self.update_op_step_size(h);
            self.jacobian_updates_at(h, SolverState::ErrorTestFail, None);
            nattempts += 1;
            self.rk.reset_prev_error();
            self.rk.error_test_fail(
                h,
                nattempts,
                self.config.maximum_error_test_failures,
                self.config.minimum_timestep,
            )?;
        };

        // accept the step
        let new_h = h * factor;
        if factor != Eqn::T::one() {
            self.reset_etas_timestep_change();
        }

        self.update_op_step_size(new_h);
        self.rk.set_prev_error(error_norm);
        let result = self.rk.step_accepted(h, new_h, true);
        // update the Jacobian for the next step if due, at the new state, then advance the
        // schedule, so that a Jacobian evaluated here is a step old by the next one
        self.jacobian_updates_at(new_h, SolverState::StepSuccess, None);
        self.jacobian_update.step();
        result
    }

    fn set_stop_time(&mut self, tstop: <Eqn as Op>::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(tstop)
    }

    fn interpolate_sens_inplace(
        &self,
        t: <Eqn as Op>::T,
        sens: &mut Eqn::V,
    ) -> Result<(), DiffsolError> {
        self.rk.interpolate_sens_inplace(t, sens)
    }

    fn interpolate_inplace(&self, t: <Eqn>::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_inplace(t, y)
    }

    fn interpolate_dy_inplace(&self, t: <Eqn>::T, dy: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_dy_inplace(t, dy)
    }

    fn interpolate_out_inplace(&self, t: <Eqn>::T, g: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_out_inplace(t, g)
    }

    fn state(&self) -> StateRef<'_, Eqn::V> {
        self.rk.state().as_ref()
    }

    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        self.rk.state_mut().as_mut()
    }

    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        let integrate_out = self.rk.problem().integrate_out;
        self.rk.state_mut_back(t, integrate_out)
    }
}

impl<'a, M, Eqn, AugmentedEqn, LS> StepVjp<'a, Eqn> for Sdirk<'a, Eqn, LS, M, AugmentedEqn>
where
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
{
    /// The cotangent is an [RkState] whose `y`, `dy` and `diff` are read, as cotangents of the
    /// state, its derivative and the step's stage increments (which [Self::interpolate_vjp] adds
    /// to). On return `y` and `dy` hold the cotangents at the start of the step; its `diff` is
    /// zero. For a method whose first stage is solved rather than taken from `dy`, the
    /// cotangent of `dy` at the start is zero.
    fn step_vjp(
        &mut self,
        end: &Self::State,
        cotangent: &mut Self::State,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError> {
        let Some(op) = self.op.as_ref() else {
            return Err(ode_solver_error!(StepDifferentiationNotSupported));
        };
        let step = self.rk.vjp_check_inputs(end, cotangent, p_bar)?;
        let h = step.h;
        // differentiate with the step's own h, which is put back afterwards
        op.set_h(h);

        let result = 'stages: {
            self.rk.vjp_start(step, end, cotangent, p_bar);
            // whether the rhs Jacobian was last evaluated at the end of this step, when an update
            // at the end, after a first failure, would rebuild the same matrix
            let mut rhs_jacobian_at_end = !self.nonlinear_solver.is_jacobian_set();
            if rhs_jacobian_at_end {
                self.jacobian_updates_at(h, SolverState::Checkpoint, Some((&end.y, end.t)));
            }
            // the schedule's update at the end of a forward step, in reverse: at the start of the
            // step's pull back, update the matrix if due, at the step's end, and advance the
            // schedule
            rhs_jacobian_at_end |= self
                .jacobian_update
                .check_rhs_jacobian_update(h, &SolverState::StepSuccess);
            self.jacobian_updates_at(h, SolverState::StepSuccess, Some((&end.y, end.t)));
            self.jacobian_update.step();
            // the stages of step(), in reverse
            let start = if self.rk.skip_first_stage() { 1 } else { 0 };
            for i in (start..self.rk.tableau().s()).rev() {
                let t_i = self.rk.vjp_stage_time(i, step);
                // set the stage up as the forward solve did, at its value Y_i = phi + gamma K_i,
                // taken as phi with a zero increment
                let y_i = self.rk.vjp_stage_value(i, end);
                self.op.as_ref().unwrap().set_phi_direct(y_i);

                // Each stage's transposed equation (M - gamma h J_i)^T w = k_bar is solved with
                // the current Newton matrix as a preconditioner
                let result = solve_with_jacobian_updates(
                    self,
                    &[
                        SolverState::FirstConvergenceFail,
                        SolverState::SecondConvergenceFail,
                    ],
                    |s| {
                        // the solve overwrites w, so keep its right-hand side in k_bar
                        let (w, k_bar) = s.rk.vjp_stage_rhs(i, cotangent);
                        k_bar.copy_from(w);
                        s.nonlinear_solver.solve_jacobian_transpose_in_place(
                            s.op.as_ref().unwrap(),
                            &s.vjp_zero,
                            t_i,
                            k_bar,
                            w,
                            &mut s.vjp_convergence,
                        )
                    },
                    |s, state| match state {
                        SolverState::SecondConvergenceFail => {
                            rhs_jacobian_at_end = false;
                            let y_i = s.rk.vjp_stage_value(i, end).clone();
                            s.refresh_jacobian_at(h, state, Some((&y_i, t_i)))
                        }
                        _ if rhs_jacobian_at_end => false,
                        _ => {
                            rhs_jacobian_at_end =
                                s.jacobian_update.check_rhs_jacobian_update(h, &state);
                            s.jacobian_updates_at(h, state, Some((&end.y, end.t)))
                        }
                    },
                );
                if let Err(e) = result {
                    break 'stages Err(e);
                }
                self.rk.vjp_stage_inputs(i, step, cotangent, p_bar);
            }
            self.rk.vjp_finish(step, cotangent);
            Ok(())
        };
        // the derivative used the replayed step's h: put back the one of the solver's own state,
        // which the forward solve and its Jacobian schedule expect
        self.op.as_ref().unwrap().set_h(self.rk.state().h);
        result
    }

    fn interpolate_vjp(
        &self,
        end: &Self::State,
        t: Eqn::T,
        y_bar: &Eqn::V,
        cotangent: &mut Self::State,
    ) -> Result<(), DiffsolError> {
        // as for step_vjp, whose pull back through the step follows this
        if self.op.is_none() {
            return Err(ode_solver_error!(StepDifferentiationNotSupported));
        }
        self.rk.interpolate_vjp(end, t, y_bar, cotangent)
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches;

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    use crate::ode_equations::test_models::{
        exponential_decay::exponential_decay_problem_batched,
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_batched,
    };
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            exponential_decay::{
                exponential_decay_problem, exponential_decay_problem_adjoint,
                exponential_decay_problem_sens, exponential_decay_problem_with_mass,
                exponential_decay_problem_with_root, negative_exponential_decay_problem,
            },
            exponential_decay_with_algebraic::{
                exponential_decay_with_algebraic_adjoint_problem,
                exponential_decay_with_algebraic_problem,
            },
            heat2d::{head2d_problem, heat2d_elem_problem},
            nonlinear_failure::nonlinear_failure_problem,
            robertson::{robertson, robertson_sens},
            robertson_ode::robertson_ode,
        },
        ode_solver::tests::{
            setup_test_adjoint, setup_test_adjoint_sum_squares, test_adjoint,
            test_adjoint_sum_squares, test_checkpointing, test_config, test_interpolate,
            test_interpolate_dy, test_ode_solver, test_problem, test_state_mut,
            test_state_mut_on_problem,
        },
        scale, ConstantOp, Context, DenseMatrix, FaerSparseLU, FaerSparseMat, MatrixCommon,
        NalgebraLU, NalgebraVec, OdeEquations, OdeSolverMethod, Op, Vector, VectorView,
    };

    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn tr_bdf2_nonlinear_failures_stop_an_unrecoverable_step() {
        let problem = nonlinear_failure_problem::<M>();
        let mut solver = problem.tr_bdf2::<LS>().unwrap();
        let result = solver.step();
        assert_matches!(
            result,
            Err(crate::DiffsolError::OdeSolverError(
                crate::error::OdeSolverError::TooManyNonlinearSolverFailures { num_failures, .. },
            )) if num_failures == solver.get_statistics().number_of_nonlinear_solver_fails
        );
    }

    #[test]
    fn sdirk_state_mut() {
        test_state_mut(test_problem::<M>(false).tr_bdf2::<LS>().unwrap());
    }
    #[test]
    fn sdirk_config() {
        test_config(robertson_ode::<M>(false, 1).0.esdirk34::<LS>().unwrap());
    }

    #[test]
    fn tr_bdf2_test_fast_timescale() {
        let (mut problem, _) = exponential_decay_problem::<M>(false);
        let p = problem.eqn.context().vector_from_vec(vec![1e12, 1.0]);
        problem.eqn_mut().set_params(&p);
        problem.tr_bdf2::<LS>().unwrap().solve(5e-12).unwrap();
        problem.ode_options.min_timestep = 1e-13;
        assert!(matches!(
            problem.tr_bdf2::<LS>().unwrap().solve(5e-12),
            Err(crate::DiffsolError::OdeSolverError(
                crate::error::OdeSolverError::StepSizeTooSmall { .. }
            ))
        ));
    }

    #[test]
    fn sdirk_test_interpolate() {
        test_interpolate(test_problem::<M>(false).tr_bdf2::<LS>().unwrap());
    }

    #[test]
    fn sdirk_test_interpolate_out() {
        test_interpolate(test_problem::<M>(true).tr_bdf2::<LS>().unwrap());
    }

    #[test]
    fn sdirk_test_interpolate_sens() {
        test_interpolate(test_problem::<M>(false).tr_bdf2_sens::<LS>().unwrap());
    }

    #[test]
    fn sdirk_test_interpolate_dy() {
        test_interpolate_dy(test_problem::<M>(false).tr_bdf2::<LS>().unwrap());
    }

    #[test]
    fn sdirk_test_esdirk34_interpolate_dy() {
        test_interpolate_dy(test_problem::<M>(false).esdirk34::<LS>().unwrap());
    }

    #[test]
    fn sdirk_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let s1 = problem.tr_bdf2::<LS>().unwrap();
        let s2 = problem.tr_bdf2::<LS>().unwrap();
        test_checkpointing(soln, s1, s2);
    }

    #[test]
    fn sdirk_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = p.tr_bdf2::<LS>().unwrap();
        test_state_mut_on_problem(s, soln);
    }

    #[test]
    fn sdirk_test_nalgebra_negative_exponential_decay() {
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, Some(30.), false, false);
    }

    #[test]
    fn sdirk_test_exponential_decay_with_mass() {
        let (problem, soln) = exponential_decay_problem_with_mass::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay2() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 8
        number_of_steps: 46
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 110
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 7
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 112
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.tr_bdf2_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 10
        number_of_steps: 90
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 418
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 9
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 209
        number_of_jac_muls: 216
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 6
        number_of_steps: 18
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 66
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 5
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 68
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay_algebraic() {
        let (problem, soln) = exponential_decay_with_algebraic_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 6
        number_of_steps: 6
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 24
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 5
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 28
        number_of_jac_muls: 6
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.esdirk34_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 6
        number_of_steps: 33
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 236
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 5
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 119
        number_of_jac_muls: 122
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn sdirk_test_esdirk34_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true, true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer, Some(s), None)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 426
        number_of_jac_muls: 10
        number_of_matrix_evals: 5
        number_of_jac_adj_muls: 138
        "###);
    }

    #[test]
    fn sdirk_test_nalgebra_exponential_decay_algebraic_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, soln, _stop_reason) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 412
        number_of_jac_muls: 12
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 478
        "###);
    }

    #[test]
    fn sdirk_test_esdirk34_exponential_decay_algebraic_adjoint() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer, Some(s), None)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 373
        number_of_jac_muls: 30
        number_of_matrix_evals: 10
        number_of_jac_adj_muls: 117
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 56
        number_of_steps: 290
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 1606
        number_of_nonlinear_solver_fails: 7
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 7
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 48
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1609
        number_of_jac_muls: 33
        number_of_matrix_evals: 11
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_sens() {
        let (mut problem, soln) = robertson_sens::<M>();
        problem.ode_options.max_nonlinear_solver_iterations = 10;
        let mut s = problem.tr_bdf2_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 57
        number_of_steps: 288
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 3012
        number_of_nonlinear_solver_fails: 3
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 7
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 49
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1548
        number_of_jac_muls: 1508
        number_of_matrix_evals: 13
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 66
        number_of_steps: 232
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 2013
        number_of_nonlinear_solver_fails: 14
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 12
        number_of_linear_solver_setups_from_second_convergence_fail: 2
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 51
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 2016
        number_of_jac_muls: 45
        number_of_matrix_evals: 15
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson_sens() {
        let (problem, soln) = robertson_sens::<M>();
        let mut s = problem.esdirk34_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 66
        number_of_steps: 242
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 3849
        number_of_nonlinear_solver_fails: 10
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 15
        number_of_linear_solver_setups_from_second_convergence_fail: 2
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 48
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1939
        number_of_jac_muls: 1969
        number_of_matrix_evals: 18
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_ode() {
        let (problem, soln) = robertson_ode::<M>(false, 1);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 76
        number_of_steps: 378
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 2130
        number_of_nonlinear_solver_fails: 7
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 7
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 1
        number_of_linear_solver_setups_from_step_success: 67
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 2132
        number_of_jac_muls: 36
        number_of_matrix_evals: 12
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_faer_sparse_heat2d() {
        let (problem, soln) = head2d_problem::<FaerSparseMat<f64>, 10>();
        let mut s = problem.tr_bdf2::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tr_bdf2_faer_sparse_heat2d_elem() {
        let (problem, soln) = heat2d_elem_problem::<FaerSparseMat<f64>, 10>(1);
        let mut s = problem.tr_bdf2::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_tr_bdf2_cuda_oxide_heat2d_elem() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = heat2d_elem_problem::<OxideMat<f64>, 10>(1);
        let mut s = problem.tr_bdf2::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_tr_bdf2_cuda_oxide_heat2d_elem_batched() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = heat2d_elem_problem::<OxideMat<f64>, 10>(2);
        let mut s = problem.tr_bdf2::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tstop_tr_bdf2() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_tr_bdf2_cuda_exponential_decay_batched() {
        use crate::{CudaLU, CudaMat};
        let (problem, soln) = exponential_decay_problem_batched::<CudaMat<f64>>(2);
        let mut s = problem.tr_bdf2::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_tr_bdf2_cuda_exponential_decay_with_algebraic_batched() {
        use crate::{CudaLU, CudaMat};
        let (problem, soln) = exponential_decay_with_algebraic_problem_batched::<CudaMat<f64>>(2);
        let mut s = problem.tr_bdf2::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_root_finder_tr_bdf2() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false, true);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);

        let y0 = problem.eqn.init().call(0.0);
        let mut p = problem.context().vector_zeros(2);
        problem.eqn.get_params(&mut p);
        let k = p.get_index(0);
        let target = 0.6_f64;
        let t_root = -((target / y0.get_index(0)).ln()) / k;
        let expected = y0.clone() * scale(f64::exp(-k * t_root));
        y.assert_eq_norm(&expected, &problem.atol, problem.rtol, 15.0);

        let g = s.interpolate_out(t_root).unwrap();
        let integral = (1.0 - f64::exp(-k * t_root)) / k;
        let expected_g = y0 * scale(integral);
        g.assert_eq_norm(&expected_g, &problem.atol, problem.rtol, 15.0);
    }

    #[test]
    fn test_param_sweep_tr_bdf2() {
        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        let mut ps = Vec::new();
        for y0 in (1..10).map(f64::from) {
            ps.push(problem.context().vector_from_vec(vec![0.1, y0]));
        }

        let mut old_soln: Option<NalgebraVec<f64>> = None;
        for p in ps {
            problem.eqn_mut().set_params(&p);
            let mut s = problem.tr_bdf2::<LS>().unwrap();
            let (ys, _ts, _stop_reason) = s.solve(10.0).unwrap();
            // check that the new solution is different from the old one
            if let Some(old_soln) = &mut old_soln {
                let new_soln = ys.column(ys.ncols() - 1).into_owned();
                let error = new_soln - &*old_soln;
                let diff = error
                    .squared_norm(old_soln, &problem.atol, problem.rtol)
                    .sqrt();
                assert!(diff > 1.0e-6, "diff: {diff}");
            }
            old_soln = Some(ys.column(ys.ncols() - 1).into_owned());
        }
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn test_ball_bounce_tr_bdf2() {
        type M = crate::NalgebraMat<f64>;
        type LS = crate::NalgebraLU<f64>;
        let (x, v, t) = crate::ode_solver::tests::test_ball_bounce(
            crate::ode_solver::tests::test_ball_bounce_problem::<M>()
                .tr_bdf2::<LS>()
                .unwrap(),
        );
        let expected_x = [6.375884661615263];
        let expected_v = [0.6878538646461059];
        let expected_t = [2.5];
        for (i, ((x, v), t)) in x.iter().zip(v.iter()).zip(t.iter()).enumerate() {
            assert!((x - expected_x[i]).abs() < 1e-4);
            assert!((v - expected_v[i]).abs() < 1e-4);
            assert!((t - expected_t[i]).abs() < 1e-4);
        }
    }

    /// Test that `step()` includes the root index in `RootFound(t, index)`.
    #[test]
    fn test_root_found_index_tr_bdf2() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_two_roots_problem;
        use crate::ode_solver::tests::test_root_found_index;
        let (problem, soln) = exponential_decay_with_two_roots_problem::<M>();
        let solver = problem.tr_bdf2::<LS>().unwrap();
        test_root_found_index(solver, &soln, 0, 1e-4);
    }

    /// Test that `solve()` applies resets and continues to the final time.
    #[test]
    fn test_solve_with_reset_tr_bdf2() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_reset_problem;
        use crate::ode_solver::tests::test_solve_with_reset;
        let (problem, soln) = exponential_decay_with_reset_problem::<M>();
        let solver = problem.tr_bdf2::<LS>().unwrap();
        test_solve_with_reset(solver, &soln, 100.0);
    }

    /// Test that `solve_dense()` applies resets and continues to the final evaluation time.
    #[test]
    fn test_solve_dense_with_reset_tr_bdf2() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_reset_problem;
        use crate::ode_solver::tests::test_solve_dense_with_reset;
        let (problem, soln) = exponential_decay_with_reset_problem::<M>();
        let solver = problem.tr_bdf2::<LS>().unwrap();
        test_solve_dense_with_reset(solver, &soln);
    }

    /// Test that `solve_dense_sensitivities()` applies resets and continues for TR-BDF2.
    #[test]
    fn test_solve_dense_sensitivities_with_reset_tr_bdf2() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_reset_problem_sens;
        use crate::ode_solver::tests::test_solve_dense_sensitivities_with_reset;
        let (problem, soln) = exponential_decay_with_reset_problem_sens::<M>();
        let solver = problem.tr_bdf2_sens::<LS>().unwrap();
        test_solve_dense_sensitivities_with_reset(solver, &soln);
    }

    #[test]
    fn test_solve_adjoint_with_single_reset_root_tr_bdf2() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_single_reset_root_problem_adjoint;
        use crate::ode_solver::tests::test_solve_adjoint_with_single_reset_root;
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(true);
        test_solve_adjoint_with_single_reset_root(
            |state| match state {
                Some(state) => problem.tr_bdf2_solver::<LS>(state),
                None => problem.tr_bdf2::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.tr_bdf2_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| {
                problem.tr_bdf2_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
            },
            true,
        );
    }

    #[test]
    fn test_solve_adjoint_sum_squares_with_single_reset_root_tr_bdf2() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_single_reset_root_problem_adjoint;
        use crate::ode_solver::tests::{
            setup_test_adjoint_sum_squares_with_single_reset_root,
            single_reset_root_discrete_times,
            test_solve_adjoint_sum_squares_with_single_reset_root,
        };
        let (mut problem, soln) =
            exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        let times = single_reset_root_discrete_times(soln.solution_points[0].t);
        let (dgdp, data) = setup_test_adjoint_sum_squares_with_single_reset_root::<LS, _>(
            &mut problem,
            times.as_slice(),
        );
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        test_solve_adjoint_sum_squares_with_single_reset_root(
            |state| match state {
                Some(state) => problem.tr_bdf2_solver::<LS>(state),
                None => problem.tr_bdf2::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.tr_bdf2_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| {
                problem.tr_bdf2_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
            },
            true,
            dgdp,
            data,
            times.as_slice(),
        );
    }

    /// The step-AD test model, its derivative solves converged to `1e-6` (`max_iter` limits
    /// them, if given).
    fn ad_test_problem_for_sens(
        dae: bool,
        max_iter: Option<usize>,
    ) -> crate::OdeSolverProblem<crate::ode_equations::test_models::ad_test_model::AdTestEqn<M>>
    {
        let mut problem = crate::ode_equations::test_models::ad_test_model::ad_test_problem::<M>(
            [1.5, 3.0, 1.0],
            dae,
            1e-8,
        );
        problem.ode_options.derivative_solve_tolerance = 1e-6;
        if let Some(max_iter) = max_iter {
            problem.ode_options.derivative_solve_max_iter = max_iter;
        }
        problem
    }

    /// The step times of `solver` stepped to `t = 5`.
    fn step_times<'a>(
        solver: &mut impl OdeSolverMethod<
            'a,
            crate::ode_equations::test_models::ad_test_model::AdTestEqn<M>,
        >,
    ) -> Vec<f64> {
        solver.set_stop_time(5.0).unwrap();
        let mut times = Vec::new();
        while !matches!(
            solver.step().unwrap(),
            crate::OdeSolverStopReason::TstopReached
        ) {
            times.push(solver.state().t);
        }
        times.push(solver.state().t);
        times
    }

    /// A sensitivity solve that the Newton matrix is too poor a preconditioner for (forced with
    /// a small iteration limit) is retried after a Jacobian update, which converges, so no step
    /// attempt fails, and the run reaches the same sensitivities.
    #[test]
    fn sdirk_sens_solve_failure_updates_jacobian() {
        for dae in [false, true] {
            let plain_problem = ad_test_problem_for_sens(dae, None);
            let mut plain = plain_problem.tr_bdf2_sens::<LS>().unwrap();
            step_times(&mut plain);
            // enough iterations for an updated matrix to converge, but not a stale one: fewer
            // and the retries fail too, more and no solve fails
            let problem = ad_test_problem_for_sens(dae, Some(3));
            let mut solver = problem.tr_bdf2_sens::<LS>().unwrap();
            step_times(&mut solver);
            let stats = solver.get_statistics();
            assert!(
                stats.number_of_linear_solver_setups_from_first_convergence_fail > 0,
                "no sensitivity solve failed"
            );
            assert_eq!(stats.number_of_nonlinear_solver_fails, 0);
            let tol = <M as MatrixCommon>::V::from_element(3, 1e-6, Default::default());
            solver
                .state()
                .s
                .assert_eq_norm(plain.state().s, &tol, 1e-6, 1.0);
        }
    }

    /// With the sensitivities left out of error control, a solver with them takes exactly the
    /// steps of one without, as long as none of their solves fail: they keep their own
    /// convergence state, so they leave the Newton solves alone.
    #[test]
    fn sdirk_sens_takes_the_same_steps() {
        for dae in [false, true] {
            let problem = ad_test_problem_for_sens(dae, None);
            let mut plain = problem.tr_bdf2::<LS>().unwrap();
            let plain_times = step_times(&mut plain);
            let mut solver = problem.tr_bdf2_sens::<LS>().unwrap();
            let times = step_times(&mut solver);
            assert_eq!(solver.get_statistics().number_of_nonlinear_solver_fails, 0);
            assert_eq!(times, plain_times);
            // to within the Newton solves' error: the solver with sensitivities factorises its
            // Newton matrix at construction, the plain one at its first stage
            let tol = <M as MatrixCommon>::V::from_element(3, 1e-10, Default::default());
            solver
                .state()
                .y
                .assert_eq_norm(plain.state().y, &tol, 1e-10, 1.0);
        }
    }

    /// The solver's Jacobian is the rhs Jacobian at its state, which the adjoint equations of a
    /// DAE use, on a model whose Jacobian depends on the state; not at a point made from that
    /// state and the last stage the solver set up.
    #[test]
    fn sdirk_step_vjp_rejects_another_tableaus_state() {
        use crate::{error::OdeSolverError, DiffsolError, OdeSolverState, StepVjp};
        let problem = ad_test_problem_for_sens(false, None);
        let mut tr_bdf2 = problem.tr_bdf2::<LS>().unwrap();
        tr_bdf2.step().unwrap();
        let end = tr_bdf2.state_clone();
        let mut esdirk34 = problem.esdirk34::<LS>().unwrap();
        let nparams = problem.eqn.rhs().nparams();
        let mut p_bar = <M as MatrixCommon>::V::zeros(nparams, Default::default());
        let result = esdirk34.step_vjp(&end, &mut end.zeros_like(), &mut p_bar);
        assert!(
            matches!(
                result,
                Err(DiffsolError::OdeSolverError(
                    OdeSolverError::StepDifferentiationSizeMismatch { .. }
                ))
            ),
            "{result:?}"
        );
    }

    #[test]
    fn sdirk_jacobian_is_at_the_state() {
        use crate::NonLinearOpJacobian;
        for dae in [false, true] {
            let problem = ad_test_problem_for_sens(dae, None);
            let mut solver = problem.tr_bdf2::<LS>().unwrap();
            for _ in 0..5 {
                solver.step().unwrap();
            }
            let expected = problem
                .eqn
                .rhs()
                .jacobian(solver.state().y, solver.state().t);
            let jacobian = solver.jacobian().unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(
                        jacobian.get_index(i, j),
                        expected.get_index(i, j),
                        "dae = {dae}, entry ({i}, {j})"
                    );
                }
            }
        }
    }
}
