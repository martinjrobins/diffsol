//! Adaptive Rosenbrock 2(3) integrator for ordinary differential equations.
//!
//! The three linear solves use a single matrix `I - gamma * h * J` per attempted
//! step. A one-sided, within-step finite difference supplies the explicit time
//! derivative. Declare known dose or covariate jumps with
//! `set_discontinuity_stop_time` to evaluate endpoint stages on the incoming side.

use std::cell::Ref;

use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

use crate::{
    error::{DiffsolError, OdeSolverError},
    op::sdirk::SdirkCallable,
    DefaultDenseMatrix, DenseMatrix, ExplicitRkConfig, LinearSolver, NonLinearOp,
    OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason,
    Op, RkState, Scalar, StateRef, StateRefMut, Tableau, TableauMat, TableauVec, Vector,
};

use super::{runge_kutta::Rk, OdeSolverStatistics};

type AttemptResult<T, V> = (T, V, V, V, V);

/// Rosenbrock 2(3), with a second-order accepted solution and a third-stage
/// local error estimate. Supports ODEs with an identity mass matrix.
///
/// The modified Rosenbrock pair is described by Shampine and Reichelt,
/// *The MATLAB ODE Suite*, SIAM J. Sci. Comput. 18 (1997), 1–22,
/// <https://doi.org/10.1137/S1064827594276424>. Its coefficients here are
/// `gamma = 1/(2 + sqrt(2))` and `c32 = 6 + sqrt(2)`; the endpoint is
/// `y0 + h*k2`, with error estimate `h*(k1 - 2*k2 + k3)/6`. The time
/// derivative enters both the first and third stages as `gamma*h*f_t`; any
/// other third-stage scale leaves an `O(h^2)` term in the estimate.
///
/// The method shares Diffsol's stop-time, root, reset, and quadratic interpolation
/// machinery. Forward sensitivities and integrated outputs are not yet exposed.
/// The right-hand side must provide a Jacobian action, as for BDF and SDIRK.
pub struct Rosenbrock23<'a, Eqn, LS, M = <<Eqn as Op>::V as DefaultDenseMatrix>::M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    rk: Rk<'a, Eqn, M>,
    linear_solver: LS,
    op: SdirkCallable<&'a Eqn>,
    config: ExplicitRkConfig<Eqn::T>,
    discontinuity_stop: Option<Eqn::T>,
}

impl<Eqn, LS, M> Clone for Rosenbrock23<'_, Eqn, LS, M>
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
            config: self.config.clone(),
            discontinuity_stop: self.discontinuity_stop,
        }
    }
}

impl<'a, Eqn, LS, M> Rosenbrock23<'a, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    fn tableau() -> Tableau<Eqn::T> {
        // The tableau is used only for shared order/step/interpolation storage.
        // Rosenbrock stages are computed below, rather than by RK tableau logic.
        // ode23s dense output is y(theta) = y0 + h*((theta-theta^2)*k1 + theta^2*k2).
        Tableau::new(
            TableauMat::zeros(3, 3),
            TableauVec::from_slice(&[Eqn::T::zero(), Eqn::T::one(), Eqn::T::zero()]),
            TableauVec::from_slice(&[
                Eqn::T::zero(),
                Eqn::T::from_f64(0.5).unwrap(),
                Eqn::T::one(),
            ]),
            TableauVec::zeros(3),
            2,
            Some(TableauMat::from_slice(
                3,
                2,
                &[
                    Eqn::T::one(),
                    Eqn::T::zero(),
                    Eqn::T::zero(),
                    -Eqn::T::one(),
                    Eqn::T::one(),
                    Eqn::T::zero(),
                ],
            )),
        )
    }

    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        mut linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        if problem.eqn.mass().is_some() {
            return Err(OdeSolverError::MassMatrixNotSupported.into());
        }
        if !state.s.is_empty() {
            return Err(OdeSolverError::SensitivityNotSupported.into());
        }
        if problem.integrate_out {
            return Err(OdeSolverError::IntegratedOutputNotSupported.into());
        }
        let rk = Rk::new(problem, state, Self::tableau())?;
        let op = SdirkCallable::new(&problem.eqn, Eqn::T::one(), problem.context().clone());
        linear_solver.set_problem(&op);
        Ok(Self {
            rk,
            linear_solver,
            op,
            config: ExplicitRkConfig::new(&problem.ode_options),
            discontinuity_stop: None,
        })
    }

    pub fn get_statistics(&self) -> &OdeSolverStatistics {
        self.rk.get_statistics()
    }

    /// Stop at a known time discontinuity in the right-hand side. Endpoint
    /// evaluation stays on the incoming side; the accepted time is `tstop`.
    /// Use `set_stop_time` for ordinary smooth output times.
    pub fn set_discontinuity_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(tstop)?;
        self.discontinuity_stop = Some(tstop);
        Ok(())
    }

    fn endpoint_eval_time(&self, h: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        let start = self.rk.state().t;
        let end = start + h;
        let Some(stop) = self.discontinuity_stop else {
            return Ok(end);
        };
        let roundoff = Eqn::T::from_f64(100.0).unwrap() * Eqn::T::EPSILON * (end.abs() + h.abs());
        if (end - stop).abs() > roundoff {
            return Ok(end);
        }
        let nominal = Eqn::T::EPSILON * (Eqn::T::one() + end.abs());
        let quarter = h.abs() / Eqn::T::from_f64(4.0).unwrap();
        let delta = if nominal < quarter { nominal } else { quarter };
        let shifted = if h >= Eqn::T::zero() {
            end - delta
        } else {
            end + delta
        };
        let inside = if h >= Eqn::T::zero() {
            start < shifted && shifted < end
        } else {
            end < shifted && shifted < start
        };
        if !inside {
            return Err(OdeSolverError::Other(
                "Rosenbrock23 discontinuity has no representable incoming endpoint".into(),
            )
            .into());
        }
        Ok(shifted)
    }

    fn attempt(
        &mut self,
        h: Eqn::T,
        retry: bool,
    ) -> Result<AttemptResult<Eqn::T, Eqn::V>, DiffsolError> {
        let state = self.rk.state();
        let y0 = state.y.clone();
        let t = state.t;
        let ctx = self.rk.problem().context().clone();
        let n = y0.len();
        let half = Eqn::T::from_f64(0.5).unwrap();
        let sixth = Eqn::T::from_f64(1.0 / 6.0).unwrap();
        let two = Eqn::T::from_f64(2.0).unwrap();
        let gamma = Eqn::T::from_f64(1.0 / (2.0 + 2.0_f64.sqrt())).unwrap();
        let c32 = Eqn::T::from_f64(6.0 + 2.0_f64.sqrt()).unwrap();

        let mut f0 = Eqn::V::zeros(n, ctx.clone());
        self.rk.problem().eqn.rhs().call_inplace(&y0, t, &mut f0);

        // One-sided derivative stays strictly inside the attempted step. This
        // matters when the solver is stopped exactly at a forcing discontinuity.
        let nominal = Eqn::T::EPSILON.sqrt() * (Eqn::T::one() + t.abs());
        let bounded = if nominal < h.abs() * half {
            nominal
        } else {
            h.abs() * half
        };
        let delta = if h >= Eqn::T::zero() {
            bounded
        } else {
            -bounded
        };
        if delta == Eqn::T::zero() || t + delta == t || t + delta == t + h {
            return Err(OdeSolverError::Other(
                "Rosenbrock23 time-derivative probe cannot advance within this step".into(),
            )
            .into());
        }
        let mut ft = Eqn::V::zeros(n, ctx.clone());
        self.rk
            .problem()
            .eqn
            .rhs()
            .call_inplace(&y0, t + delta, &mut ft);
        ft.axpy(-Eqn::T::one(), &f0, Eqn::T::one());
        ft *= crate::scale(Eqn::T::one() / delta);

        self.op.zero_phi();
        self.op.set_h(gamma * h);
        // Each Rosenbrock step linearizes at its current state, including after
        // an accepted step or a rejected attempt with a different step size.
        self.op.set_jacobian_is_stale();
        LinearSolver::set_linearisation(&mut self.linear_solver, &self.op, &y0, t);
        self.rk
            .statistics_mut()
            .record_linear_solver_setup(if retry {
                super::jacobian_update::SolverState::ErrorTestFail
            } else {
                super::jacobian_update::SolverState::StepSuccess
            });

        let mut k1 = f0.clone();
        k1.axpy(gamma * h, &ft, Eqn::T::one());
        self.linear_solver.solve_in_place(&mut k1)?;

        let mut y_mid = y0.clone();
        y_mid.axpy(h * half, &k1, Eqn::T::one());
        let mut f_mid = Eqn::V::zeros(n, ctx.clone());
        self.rk
            .problem()
            .eqn
            .rhs()
            .call_inplace(&y_mid, t + h * half, &mut f_mid);
        let mut k2 = f_mid.clone();
        k2.axpy(-Eqn::T::one(), &k1, Eqn::T::one());
        self.linear_solver.solve_in_place(&mut k2)?;
        k2.axpy(Eqn::T::one(), &k1, Eqn::T::one());

        let mut y1 = y0.clone();
        y1.axpy(h, &k2, Eqn::T::one());
        let mut f1 = Eqn::V::zeros(n, ctx.clone());
        self.rk
            .problem()
            .eqn
            .rhs()
            .call_inplace(&y1, self.endpoint_eval_time(h)?, &mut f1);

        let mut k3 = f1.clone();
        k3.axpy(-c32, &k2, Eqn::T::one());
        k3.axpy(c32, &f_mid, Eqn::T::one());
        k3.axpy(-two, &k1, Eqn::T::one());
        k3.axpy(two, &f0, Eqn::T::one());
        k3.axpy(gamma * h, &ft, Eqn::T::one());
        self.linear_solver.solve_in_place(&mut k3)?;

        let mut error = k1.clone();
        error.axpy(-two, &k2, Eqn::T::one());
        error.axpy(Eqn::T::one(), &k3, Eqn::T::one());
        error *= crate::scale(h * sixth);
        let error_norm = error.squared_norm(&y1, &self.rk.problem().atol, self.rk.problem().rtol);
        Ok((error_norm, y1, f1, k1, k2))
    }
}

impl<'a, Eqn, LS, M> OdeSolverMethod<'a, Eqn> for Rosenbrock23<'a, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    type State = RkState<Eqn::V>;
    type Config = ExplicitRkConfig<Eqn::T>;

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.rk.problem()
    }
    fn checkpoint(&mut self) -> Self::State {
        self.rk.checkpoint()
    }
    fn state_clone(&self) -> Self::State {
        self.rk.state().clone()
    }
    fn set_state(&mut self, state: Self::State) {
        self.rk.set_state(state);
    }
    fn into_state(self) -> Self::State {
        self.rk.into_state()
    }
    fn state(&self) -> StateRef<'_, Eqn::V> {
        self.rk.state().as_ref()
    }
    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        self.rk.state_mut().as_mut()
    }
    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn config_mut(&mut self) -> &mut Self::Config {
        &mut self.config
    }
    fn jacobian(&self) -> Option<Ref<'_, Eqn::M>> {
        Some(self.op.rhs_jac(&self.rk.state().y, self.rk.state().t))
    }
    fn mass(&self) -> Option<Ref<'_, Eqn::M>> {
        None
    }
    fn order(&self) -> usize {
        2
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut h = self.rk.start_step()?;
        if h.abs() < self.config.minimum_timestep {
            return Err(OdeSolverError::StepSizeTooSmall {
                time: self.rk.state().t.to_f64().unwrap(),
            }
            .into());
        }
        let mut attempts = 0;
        let (factor, error_norm, y1, f1, mut k1, mut k2) = loop {
            let (error_norm, y1, f1, k1, k2) = self.attempt(h, attempts > 0)?;
            let factor = self.rk.factor(
                error_norm,
                1.0,
                self.config.minimum_timestep_shrink,
                self.config.maximum_timestep_shrink,
                self.config.minimum_timestep_growth,
                self.config.maximum_timestep_growth,
            );
            if error_norm < Eqn::T::one() {
                break (factor, error_norm, y1, f1, k1, k2);
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
        k1 *= crate::scale(h);
        k2 *= crate::scale(h);
        self.rk.store_rosenbrock_stages(&y1, &f1, &[k1, k2]);
        self.rk.set_prev_error(error_norm);
        let stop_reason = self.rk.step_accepted(h, h * factor, false)?;
        if matches!(stop_reason, OdeSolverStopReason::TstopReached) {
            self.discontinuity_stop = None;
        }
        Ok(stop_reason)
    }
    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(tstop)?;
        self.discontinuity_stop = None;
        Ok(())
    }
    fn interpolate_inplace(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_inplace(t, y)
    }
    fn interpolate_dy_inplace(&self, t: Eqn::T, dy: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_dy_inplace(t, dy)
    }
    fn interpolate_out_inplace(&self, t: Eqn::T, g: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_out_inplace(t, g)
    }
    fn interpolate_sens_inplace(&self, t: Eqn::T, sens: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_sens_inplace(t, sens)
    }
    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.state_mut_back(t, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            exponential_decay::exponential_decay_problem, robertson_ode::robertson_ode,
        },
        ode_solver::tests::{
            test_config, test_interpolate, test_interpolate_dy, test_ode_solver, test_problem,
            test_state_mut,
        },
        NalgebraLU, OdeBuilder,
    };

    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn shared_solver_contract() {
        test_state_mut(test_problem::<M>(false).rosenbrock23::<LS>().unwrap());
        test_interpolate(test_problem::<M>(false).rosenbrock23::<LS>().unwrap());
        test_interpolate_dy(test_problem::<M>(false).rosenbrock23::<LS>().unwrap());
        test_config(robertson_ode::<M>(false, 1).0.rosenbrock23::<LS>().unwrap());

        let (problem, solution) = exponential_decay_problem::<M>(false);
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        // The shared fixture is calibrated for higher-order methods; ode23s
        // accepts a second-order endpoint and is covered by its own order test.
        test_ode_solver(&mut solver, solution, Some(1e-4), false, false);
    }

    #[test]
    fn checkpoint_resumes_the_same_state() {
        let (problem, _) = exponential_decay_problem::<M>(false);
        let mut original = problem.rosenbrock23::<LS>().unwrap();
        let mut resumed = problem.rosenbrock23::<LS>().unwrap();
        for _ in 0..8 {
            original.step().unwrap();
        }
        resumed.set_state(original.checkpoint());
        assert_eq!(original.state().t, resumed.state().t);
        assert!((original.state().y[0] - resumed.state().y[0]).abs() < 1e-12);
        let target = original.state().t + 0.2;
        original.set_stop_time(target).unwrap();
        resumed.set_stop_time(target).unwrap();
        while original.state().t < target {
            original.step().unwrap();
        }
        while resumed.state().t < target {
            resumed.step().unwrap();
        }
        assert!((original.state().y[0] - resumed.state().y[0]).abs() < 1e-5);
    }

    #[test]
    fn robertson_dense_output_matches_reference() {
        let (problem, solution) = robertson_ode::<M>(false, 1);
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        test_ode_solver(&mut solver, solution, None, false, false);
    }

    fn advance_to<'a, Eqn, Method>(solver: &mut Method, t: f64)
    where
        Eqn: OdeEquationsImplicit<T = f64> + 'a,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        solver.set_stop_time(t).unwrap();
        loop {
            if let OdeSolverStopReason::TstopReached = solver.step().unwrap() {
                break;
            }
        }
    }

    fn fixed_step_error(h: f64) -> f64 {
        let problem = OdeBuilder::<M>::new()
            .rtol(10.0)
            .atol([10.0])
            .rhs_implicit(
                |x, _, t, f| f[0] = t - 2.0 * x[0],
                |_, _, _, v, jv| jv[0] = -2.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        *solver.state_mut().h = h;
        let config = solver.config_mut();
        config.maximum_timestep_growth = 1.0;
        config.minimum_timestep_growth = 1.0;
        advance_to(&mut solver, 1.0);
        let exact = 0.25 + 1.25 * (-2.0f64).exp();
        (solver.state().y[0] - exact).abs()
    }

    #[test]
    fn cosine_forcing_paper_oracle() {
        // Steinebach (2023): y'=cos(t), y(0)=0, y(1)=sin(1).
        // This exercises nonautonomous stages; the paper studies Rodas5P.
        let error_at_tolerance = |rtol, atol| {
            let problem = OdeBuilder::<M>::new()
                .rtol(rtol)
                .atol([atol])
                .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
                .init(|_, _, y| y[0] = 0.0, 1)
                .build()
                .unwrap();
            let mut solver = problem.rosenbrock23::<LS>().unwrap();
            advance_to(&mut solver, 1.0);
            (solver.state().y[0] - 1.0f64.sin()).abs()
        };
        let coarse = error_at_tolerance(1e-7, 1e-9);
        let fine = error_at_tolerance(1e-9, 1e-11);
        assert!(coarse < 1e-5, "cosine forcing error={coarse}");
        assert!(fine < 0.1 * coarse, "coarse={coarse}, fine={fine}");
    }

    #[test]
    fn error_estimate_is_third_order_for_time_forcing() {
        // y'=cos(t) has J=0, so every stage-level f_t scale error shows up
        // directly in the estimate. The true one-step error is O(h^3).
        let problem = OdeBuilder::<M>::new()
            .rtol(1.0)
            .atol([1.0])
            .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        let t0 = 1.0f64;
        *solver.state_mut().t = t0;
        let mut estimates = Vec::new();
        for h in [0.05, 0.025] {
            let (error_norm, y1, _, _, _) = solver.attempt(h, false).unwrap();
            let estimate = error_norm.sqrt();
            let actual = (y1[0] - ((t0 + h).sin() - t0.sin())).abs();
            assert!(
                estimate < 1.5 * actual && actual < 1.5 * estimate,
                "h={h}, estimate={estimate}, actual={actual}"
            );
            estimates.push(estimate);
        }
        let ratio = estimates[0] / estimates[1];
        assert!(6.0 < ratio && ratio < 10.0, "estimate ratio={ratio}");
    }

    #[test]
    fn prothero_robinson_paper_oracle() {
        // Steinebach (2023): g=sin(t), lambda=-1000,
        // y'=lambda*(y-g)+g', y(0)=g(0), so y(1)=sin(1).
        let error_at_tolerance = |rtol, atol| {
            let problem = OdeBuilder::<M>::new()
                .rtol(rtol)
                .atol([atol])
                .rhs_implicit(
                    |x, _, t, f| f[0] = -1000.0 * (x[0] - t.sin()) + t.cos(),
                    |_, _, _, v, jv| jv[0] = -1000.0 * v[0],
                )
                .init(|_, _, y| y[0] = 0.0, 1)
                .build()
                .unwrap();
            let mut solver = problem.rosenbrock23::<LS>().unwrap();
            advance_to(&mut solver, 1.0);
            (solver.state().y[0] - 1.0f64.sin()).abs()
        };
        let coarse = error_at_tolerance(1e-5, 1e-7);
        let fine = error_at_tolerance(1e-7, 1e-9);
        assert!(fine < 0.1 * coarse, "coarse={coarse}, fine={fine}");
        // The previous 1e-8 bound depended on the defective estimate forcing
        // far more steps than these tolerances require. Check the requested
        // accuracy and convergence trend without rewarding that over-solving.
        assert!(fine < 2e-7, "Prothero-Robinson error={fine}");
    }

    #[test]
    fn time_derivative_probe_must_advance_within_step() {
        // One ULP of time has no representable interior point. Failing with a
        // typed error is preferable to dividing by zero or crossing a stop.
        let problem = OdeBuilder::<M>::new()
            .rhs_implicit(|_, _, t, f| f[0] = t, |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        let t_even = f64::from_bits(1e12f64.to_bits() & !1);
        for t in [t_even, f64::from_bits(t_even.to_bits() + 1)] {
            *solver.state_mut().t = t;
            let h = f64::from_bits(t.to_bits() + 1) - t;
            assert!(matches!(
                solver.attempt(h, false),
                Err(DiffsolError::OdeSolverError(OdeSolverError::Other(_)))
            ));
        }
    }

    #[test]
    fn second_order_convergence() {
        let coarse = fixed_step_error(0.25);
        let fine = fixed_step_error(0.125);
        assert!(coarse > 3.0 * fine, "coarse={coarse}, fine={fine}");
        assert!(coarse < 2e-2, "coarse={coarse}");
    }

    #[test]
    fn rejected_attempt_keeps_initial_state_intact() {
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-8)
            .atol([1e-10])
            .rhs_implicit(
                |x, _, _, f| f[0] = -100.0 * x[0],
                |_, _, _, v, jv| jv[0] = -100.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        *solver.state_mut().h = 0.1;
        solver.step().unwrap();
        assert!(solver.get_statistics().number_of_error_test_failures > 0);
        assert!(
            solver
                .get_statistics()
                .number_of_linear_solver_setups_from_error_test_fail
                > 0
        );
        let t = solver.state().t;
        assert!(t > 0.0 && t < 0.1);
        assert!((solver.state().y[0] - (-100.0 * t).exp()).abs() < 1e-5);
        advance_to(&mut solver, 0.1);
        assert!((solver.state().y[0] - (-10.0f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn unsupported_equation_modes_are_rejected() {
        let mass_problem = OdeBuilder::<M>::new()
            .rhs_implicit(|x, _, _, f| f[0] = -x[0], |_, _, _, v, jv| jv[0] = -v[0])
            .init(|_, _, y| y[0] = 1.0, 1)
            .mass(|v, _, _, beta, y| y[0] = v[0] + beta * y[0])
            .build()
            .unwrap();
        assert!(matches!(
            mass_problem.rosenbrock23::<LS>(),
            Err(DiffsolError::OdeSolverError(
                OdeSolverError::MassMatrixNotSupported
            ))
        ));

        let output_problem = OdeBuilder::<M>::new()
            .rhs_implicit(|x, _, _, f| f[0] = -x[0], |_, _, _, v, jv| jv[0] = -v[0])
            .init(|_, _, y| y[0] = 1.0, 1)
            .out_implicit(
                |x, _, _, out| out[0] = x[0],
                |_, _, _, v, jv| jv[0] = v[0],
                1,
            )
            .integrate_out(true)
            .build()
            .unwrap();
        assert!(matches!(
            output_problem.rosenbrock23::<LS>(),
            Err(DiffsolError::OdeSolverError(
                OdeSolverError::IntegratedOutputNotSupported
            ))
        ));

        let sens_problem = OdeBuilder::<M>::new()
            .p([1.0])
            .rhs_sens_implicit(
                |x, p, _, f| f[0] = -p[0] * x[0],
                |_, p, _, v, jv| jv[0] = -p[0] * v[0],
                |x, _, _, v, fp| fp[0] = -x[0] * v[0],
            )
            .init_sens(|_, _, y| y[0] = 1.0, |_, _, _, dy| dy[0] = 0.0, 1)
            .build()
            .unwrap();
        let state = sens_problem.rk_state_sens(&Tableau::tsit45()).unwrap();
        assert!(matches!(
            Rosenbrock23::<_, LS>::new(&sens_problem, state, LS::default()),
            Err(DiffsolError::OdeSolverError(
                OdeSolverError::SensitivityNotSupported
            ))
        ));
    }

    #[test]
    fn nonautonomous_linear_oracle() {
        // y' = t - 2y, y(0)=1; exact y(t)=t/2-1/4+5/4 exp(-2t).
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .h0(0.1)
            .rhs_implicit(
                |x, _, t, f| f[0] = t - 2.0 * x[0],
                |_, _, _, v, jv| jv[0] = -2.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        advance_to(&mut solver, 1.0);
        let exact = 0.25 + 1.25 * (-2.0f64).exp();
        assert!((solver.state().y[0] - exact).abs() < 2e-5);
        assert!(solver.get_statistics().number_of_linear_solver_setups > 0);
    }

    #[test]
    fn interpolation_and_root_event() {
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-8)
            .atol([1e-10])
            .h0(0.8)
            .rhs_implicit(|_, _, _, f| f[0] = 1.0, |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .root(|x, _, _, r| r[0] = x[0] - 0.5, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        let mut found = None;
        for _ in 0..100 {
            if let OdeSolverStopReason::RootFound(t, index) = solver.step().unwrap() {
                found = Some((t, index));
                break;
            }
        }
        let (root, index) = found.expect("root must be detected");
        assert_eq!(index, 0);
        assert!((root - 0.5).abs() < 1e-6);
        assert!((solver.interpolate(root).unwrap()[0] - 0.5).abs() < 1e-6);
        assert!((solver.interpolate_dy(root).unwrap()[0] - 1.0).abs() < 1e-6);
        solver.state_mut_back(root).unwrap();
        assert!((solver.state().y[0] - 0.5).abs() < 1e-6);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn faer_linear_backend() {
        let problem = OdeBuilder::<crate::FaerMat<f64>>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .rhs_implicit(
                |x, _, _, f| f[0] = -10.0 * x[0],
                |_, _, _, v, jv| jv[0] = -10.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<crate::FaerLU<f64>>().unwrap();
        advance_to(&mut solver, 1.0);
        assert!((solver.state().y[0] - (-10.0f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn stiff_stage_uses_the_jacobian_factorization() {
        // Independent scalar stage calculation for y'=-1000y at h=0.1.
        // An identity factorization produces a very different endpoint.
        let problem = OdeBuilder::<M>::new()
            .rhs_implicit(
                |x, _, _, f| f[0] = -1000.0 * x[0],
                |_, _, _, v, jv| jv[0] = -1000.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        let (_, endpoint, _, _, _) = solver.attempt(0.1, false).unwrap();
        let gamma = 1.0 / (2.0 + 2.0_f64.sqrt());
        let a = 1.0 + gamma * 0.1 * 1000.0;
        let k1 = -1000.0 / a;
        let f_mid = -1000.0 * (1.0 + 0.05 * k1);
        let k2 = k1 + (f_mid - k1) / a;
        let expected = 1.0 + 0.1 * k2;
        assert!((endpoint[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn nonlinear_jacobian_is_evaluated_at_the_current_state() {
        // y'=-y² at y0=2 has J=-4. Evaluating the Jacobian at 2*y0
        // would incorrectly give J=-8 and change this stage endpoint.
        let problem = OdeBuilder::<M>::new()
            .rhs_implicit(
                |x, _, _, f| f[0] = -x[0] * x[0],
                |x, _, _, v, jv| jv[0] = -2.0 * x[0] * v[0],
            )
            .init(|_, _, y| y[0] = 2.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        let (_, endpoint, _, _, _) = solver.attempt(0.1, false).unwrap();
        let gamma = 1.0 / (2.0 + 2.0_f64.sqrt());
        let a = 1.0 + gamma * 0.1 * 4.0;
        let k1 = -4.0 / a;
        let mid = 2.0 + 0.05 * k1;
        let f_mid = -mid * mid;
        let k2 = k1 + (f_mid - k1) / a;
        let expected = 2.0 + 0.1 * k2;
        assert!((endpoint[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn nonlinear_jacobian_refreshes_after_an_accepted_step() {
        use std::{cell::RefCell, rc::Rc};

        let jacobian_states = Rc::new(RefCell::new(Vec::<(f64, f64)>::new()));
        let observed = jacobian_states.clone();
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .h0(0.05)
            .rhs_implicit(
                |x, _, _, f| f[0] = -x[0] * x[0],
                move |x, _, t, v, jv| {
                    observed.borrow_mut().push((t, x[0]));
                    jv[0] = -2.0 * x[0] * v[0];
                },
            )
            .init(|_, _, y| y[0] = 2.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        solver.step().unwrap();
        let first_time = solver.state().t;
        let first_value = solver.state().y[0];
        assert!(first_time > 0.0 && first_value < 2.0);
        solver.step().unwrap();
        let second_time = solver.state().t;
        let second_value = solver.state().y[0];
        assert!(second_time > first_time);
        assert!(
            jacobian_states
                .borrow()
                .iter()
                .any(|(t, y)| { *t == first_time && (*y - first_value).abs() < 1e-12 }),
            "the nonlinear Jacobian was not evaluated at the first accepted state"
        );
        let exact = 2.0 / (1.0 + 2.0 * second_time);
        assert!((second_value - exact).abs() < 1e-4);
    }

    #[test]
    fn stiff_decay_oracle() {
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-6)
            .atol([1e-9])
            .h0(0.1)
            .rhs_implicit(
                |x, _, _, f| f[0] = -1000.0 * x[0],
                |_, _, _, v, jv| jv[0] = -1000.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        advance_to(&mut solver, 0.1);
        assert!(solver.state().y[0].abs() < 1e-4);
    }

    #[test]
    fn ordinary_stop_evaluates_exact_endpoint() {
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        advance_to(&mut solver, 1.0);
        assert_eq!(solver.state().t, 1.0);
        assert!((solver.state().dy[0] - 1.0f64.cos()).abs() < 1e-14);
    }

    #[test]
    fn stop_time_does_not_sample_next_forcing_interval() {
        // A 100-unit forcing jump occurs exactly at t=1. The first interval
        // must follow y'=1 and end at y=1 regardless of the jump.
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .h0(0.4)
            .rhs_implicit(
                |_, _, t, f| f[0] = if t < 1.0 { 1.0 } else { 101.0 },
                |_, _, _, _, jv| jv[0] = 0.0,
            )
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rosenbrock23::<LS>().unwrap();
        solver.set_discontinuity_stop_time(1.0).unwrap();
        *solver.state_mut().h = 0.4;
        loop {
            if let OdeSolverStopReason::TstopReached = solver.step().unwrap() {
                break;
            }
        }
        assert!((solver.state().dy[0] - 1.0).abs() < 1e-10);
        assert!((solver.state().y[0] - 1.0).abs() < 1e-5);
        assert!(solver.get_statistics().number_of_error_test_failures < 5);
        // Apply a bolus at the same boundary, then integrate under the new
        // forcing interval. The solver must re-evaluate the mutated state.
        solver.state_mut().y[0] += 2.0;
        advance_to(&mut solver, 1.1);
        assert!((solver.state().y[0] - 13.1).abs() < 1e-4);
        assert!((solver.state().dy[0] - 101.0).abs() < 1e-10);
    }
}
