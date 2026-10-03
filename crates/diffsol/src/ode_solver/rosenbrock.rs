use super::{jacobian_update::SolverState, runge_kutta::Rk, OdeSolverStatistics};
use crate::{
    error::DiffsolError, error::OdeSolverError, ode_solver_error, op::sdirk::SdirkCallable,
    DefaultDenseMatrix, DenseMatrix, ExplicitRkConfig, LinearSolver, NoAug, NonLinearOpTimePartial,
    OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason,
    Op, RkState, StateRef, StateRefMut, Tableau, Vector,
};
use num_traits::{One, Signed, ToPrimitive, Zero};

/// A tableau-driven class of linearly implicit Rosenbrock-Wanner methods.
/// Each attempted step freezes the Jacobian and reuses one linear factorization.
/// A tableau's continuous extension supplies dense output; otherwise Hermite interpolation is used.
/// Constant mass matrices are supported for index-1 DAEs with a continuous extension.
/// Integrated outputs use quadrature along the state continuous extension, which is required.
/// Forward and adjoint sensitivities are not yet implemented for this class.
pub struct Rosenbrock<'a, Eqn, LS, M = <<Eqn as Op>::V as DefaultDenseMatrix>::M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    rk: Rk<'a, Eqn, M>,
    linear_solver: LS,
    op: SdirkCallable<&'a Eqn>,
    ft: Eqn::V,
    config: ExplicitRkConfig<Eqn::T>,
}
impl<Eqn, LS, M> Clone for Rosenbrock<'_, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    fn clone(&self) -> Self {
        let op = self.op.clone_state(self.rk.problem().eqn());
        let mut linear_solver = LS::default();
        linear_solver.set_problem(&op);
        Self {
            rk: self.rk.clone(),
            linear_solver,
            op,
            ft: self.ft.clone(),
            config: self.config.clone(),
        }
    }
}
impl<'a, Eqn, LS, M> Rosenbrock<'a, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<Eqn::T>,
        mut linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        let invalid = || {
            ode_solver_error!(InvalidTableau, "Expected finite, strictly lower triangular Rosenbrock coefficients with positive gamma and orders")
        };
        let row = tableau.rosenbrock().ok_or_else(invalid)?;
        let n = tableau.s();
        let finite = |x: Eqn::T| x.to_f64().is_some_and(f64::is_finite);
        if n == 0
            || tableau.order() == 0
            || row.error_order == 0
            || !finite(row.gamma)
            || row.gamma <= Eqn::T::zero()
            || row.time.len() != n
            || row.coupling.nrows() != n
            || row.coupling.ncols() != n
        {
            return Err(invalid());
        }
        for i in 0..n {
            if ![tableau.b()[i], tableau.c()[i], tableau.d()[i], row.time[i]]
                .into_iter()
                .all(finite)
            {
                return Err(invalid());
            }
            for j in 0..n {
                let a = tableau.a(i, j);
                let c = row.coupling[(i, j)];
                if !finite(a)
                    || !finite(c)
                    || (j >= i && (a != Eqn::T::zero() || c != Eqn::T::zero()))
                {
                    return Err(invalid());
                }
            }
        }
        if let Some(beta) = tableau.beta_t() {
            if beta.nrows() == 0 || beta.ncols() != n {
                return Err(invalid());
            }
            for i in 0..n {
                if !beta.as_col_slice(i).iter().copied().all(finite) {
                    return Err(invalid());
                }
            }
        } else if problem.eqn.mass().is_some() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Mass-matrix Rosenbrock methods require a continuous extension"
            ));
        }
        if !state.s.is_empty() {
            return Err(OdeSolverError::SensitivityNotSupported.into());
        }
        if problem.integrate_out && tableau.beta_t().is_none() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Integrated Rosenbrock outputs require a continuous extension"
            ));
        }
        let ft = Eqn::V::zeros(state.y.len(), problem.context().clone());
        let op = SdirkCallable::new(&problem.eqn, Eqn::T::one(), problem.context().clone());
        linear_solver.set_problem(&op);
        Ok(Self {
            rk: Rk::new(problem, state, tableau)?,
            linear_solver,
            op,
            ft,
            config: ExplicitRkConfig::new(&problem.ode_options),
        })
    }
    pub fn get_statistics(&self) -> &OdeSolverStatistics {
        self.rk.get_statistics()
    }
}
impl<'a, Eqn, LS, M> OdeSolverMethod<'a, Eqn> for Rosenbrock<'a, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    type State = RkState<Eqn::V>;
    type Config = ExplicitRkConfig<Eqn::T>;
    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn config_mut(&mut self) -> &mut Self::Config {
        &mut self.config
    }
    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.rk.problem()
    }
    fn state(&self) -> StateRef<'_, Eqn::V> {
        self.rk.state().as_ref()
    }
    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        self.rk.state_mut().as_mut()
    }
    fn state_clone(&self) -> Self::State {
        self.rk.state().clone()
    }
    fn checkpoint(&mut self) -> Self::State {
        self.rk.state().clone()
    }
    fn into_state(self) -> Self::State {
        self.rk.into_state()
    }
    fn set_state(&mut self, state: Self::State) {
        self.rk.set_state(state);
    }
    fn order(&self) -> usize {
        self.rk.order()
    }
    fn jacobian(&self) -> Option<std::cell::Ref<'_, Eqn::M>> {
        self.op.zero_phi();
        Some(self.op.rhs_jac(&self.rk.state().y, self.rk.state().t))
    }
    fn mass(&self) -> Option<std::cell::Ref<'_, Eqn::M>> {
        self.problem()
            .eqn
            .mass()
            .map(|_| self.op.mass(self.rk.state().t))
    }
    fn apply_reset(&mut self) -> Result<(), DiffsolError> {
        let problem = self.problem();
        self.rk
            .state_mut()
            .as_mut()
            .apply_reset_with_mass::<LS, _>(problem)
    }
    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut h = self.rk.start_step()?;
        if h.abs() < self.config.minimum_timestep {
            return Err(OdeSolverError::StepSizeTooSmall {
                time: self.rk.state().t.to_f64().unwrap(),
            }
            .into());
        }
        let gamma = self.rk.tableau().rosenbrock().unwrap().gamma;
        let mut attempts = 0;
        let (factor, error) = loop {
            let state = self.rk.state();
            self.op.zero_phi();
            self.op.set_h(gamma * h);
            self.op.set_jacobian_is_stale();
            LinearSolver::set_linearisation(&mut self.linear_solver, &self.op, &state.y, state.t);
            self.problem()
                .eqn
                .rhs()
                .time_derive_inplace(&state.y, state.t, &mut self.ft);
            self.rk
                .statistics_mut()
                .record_linear_solver_setup(if attempts == 0 {
                    SolverState::StepSuccess
                } else {
                    SolverState::ErrorTestFail
                });
            self.rk.start_step_attempt(h, None::<&mut NoAug<Eqn>>);
            for i in 0..self.rk.tableau().s() {
                self.rk
                    .do_stage_rosenbrock(i, h, &self.op, &mut self.linear_solver, &self.ft)?;
            }
            self.rk.finish_step_rosenbrock(h);
            if self.problem().integrate_out {
                self.rk.integrate_rosenbrock_outputs(h, &mut self.ft);
            }
            let error = self.rk.error_norm(h, None::<&mut NoAug<Eqn>>, |_| Ok(()))?;
            let factor = self.rk.factor(
                error,
                1.0,
                self.config.minimum_timestep_shrink,
                self.config.maximum_timestep_shrink,
                self.config.minimum_timestep_growth,
                self.config.maximum_timestep_growth,
            );
            if error < Eqn::T::one() {
                break (factor, error);
            }
            h *= factor;
            attempts += 1;
            self.rk.reset_prev_error();
            self.rk.error_test_fail(
                h,
                attempts,
                self.config.maximum_error_test_failures,
                self.config.minimum_timestep,
            )?;
        };
        self.rk.store_rosenbrock_hermite_derivatives(h);
        self.rk.set_prev_error(error);
        self.rk.step_accepted(h, h * factor, false)
    }
    fn set_stop_time(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(t)
    }
    fn interpolate_inplace(&self, t: Eqn::T, out: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_inplace(t, out)
    }
    fn interpolate_dy_inplace(&self, t: Eqn::T, out: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_dy_inplace(t, out)
    }
    fn interpolate_out_inplace(&self, t: Eqn::T, out: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_out_inplace(t, out)
    }
    fn interpolate_sens_inplace(&self, _t: Eqn::T, _out: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_sens_inplace(_t, _out)
    }
    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.state_mut_back(t, self.rk.problem().integrate_out)
    }
}
