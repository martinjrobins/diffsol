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
/// Integrated outputs use quadrature along the required state continuous extension;
/// Rodas5P outputs have global order four.
/// After mutating the state, keep `dy` consistent: Hermite interpolation uses it
/// as the left-endpoint derivative for tableaus without a continuous extension.
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
    scratch: Eqn::V,
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
            scratch: self.scratch.clone(),
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
            scratch: ft.clone(),
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
        // Evaluate the RHS Jacobian at y, rather than the SDIRK stage phi + c*y.
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
                self.rk.integrate_rosenbrock_outputs(h, &mut self.scratch);
            }
            // Use diffsol's shared RK error scaling at the starting state, not the endpoint.
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
    fn interpolate_sens_inplace(&self, t: Eqn::T, out: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_sens_inplace(t, out)
    }
    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.state_mut_back(t, self.rk.problem().integrate_out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ode_equations::test_models::{
            exponential_decay::exponential_decay_problem,
            exponential_decay_with_algebraic::{
                exponential_decay_with_algebraic_adjoint_problem,
                exponential_decay_with_algebraic_problem,
            },
            robertson_ode::robertson_ode,
        },
        ode_solver::tests::{
            test_checkpointing, test_config, test_interpolate, test_interpolate_dy,
            test_ode_solver, test_problem, test_state_mut,
        },
        FaerLU, FaerMat, NalgebraLU, NalgebraMat, OdeBuilder, TableauMat, TableauVec,
    };
    type Mat = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;
    fn advance<'a, E: OdeEquationsImplicit<T = f64> + 'a, S: OdeSolverMethod<'a, E>>(
        s: &mut S,
        t: f64,
    ) {
        s.set_stop_time(t).unwrap();
        while s.step().unwrap() != OdeSolverStopReason::TstopReached {}
    }
    macro_rules! contract {
        ($mat:ty, $ls:ty) => {{
            test_state_mut(test_problem::<$mat>(false).rodas5p::<$ls>().unwrap());
            test_interpolate(test_problem::<$mat>(false).rodas5p::<$ls>().unwrap());
            test_interpolate(test_problem::<$mat>(true).rodas5p::<$ls>().unwrap());
            test_interpolate_dy(test_problem::<$mat>(false).rodas5p::<$ls>().unwrap());
            test_config(robertson_ode::<$mat>(false, 1).0.rodas5p::<$ls>().unwrap());
            let (p, sol) = exponential_decay_problem::<$mat>(false);
            test_checkpointing(
                sol,
                p.rodas5p::<$ls>().unwrap(),
                p.rodas5p::<$ls>().unwrap(),
            );
            for (p, sol) in [
                exponential_decay_problem::<$mat>(false),
                exponential_decay_problem::<$mat>(true),
            ] {
                let mut s = p.rodas5p::<$ls>().unwrap();
                test_ode_solver(&mut s, sol, None, false, false);
                let stats = s.get_statistics();
                assert_eq!(
                    stats.number_of_linear_solver_setups,
                    stats.number_of_steps + stats.number_of_error_test_failures
                );
                assert_eq!(stats.number_of_nonlinear_solver_iterations, 0);
            }
            let (p, sol) = robertson_ode::<$mat>(false, 1);
            test_ode_solver(&mut p.rodas5p::<$ls>().unwrap(), sol, None, false, false);
        }};
    }
    #[test]
    fn nalgebra_shared_contract() {
        contract!(Mat, LS);
    }
    #[test]
    fn faer_shared_contract() {
        contract!(FaerMat<f64>, FaerLU<f64>);
    }
    #[test]
    fn constant_mass_index_one_dae() {
        macro_rules! check {
            ($mat:ty, $ls:ty) => {{
                let (p, sol) = exponential_decay_with_algebraic_problem::<$mat>(false);
                test_ode_solver(&mut p.rodas5p::<$ls>().unwrap(), sol, None, false, false);
                let (p, sol) = exponential_decay_with_algebraic_adjoint_problem::<$mat>(true);
                let mut s = p.rodas5p::<$ls>().unwrap();
                // The shared state harness applies out(y); this fixture instead expects integral(out).
                for point in sol.solution_points {
                    while s.state().t < point.t { s.step().unwrap(); }
                    s.interpolate_out(point.t).unwrap().assert_eq_st(&point.state, 2e-5);
                }
            }};
        }
        check!(Mat, LS);
        check!(FaerMat<f64>, FaerLU<f64>);
    }
    fn sine_problem(
        lambda: f64,
        integrate: bool,
    ) -> OdeSolverProblem<
        impl OdeEquationsImplicit<
            M = Mat,
            V = crate::NalgebraVec<f64>,
            T = f64,
            C = crate::NalgebraContext,
        >,
    > {
        OdeBuilder::<Mat>::new()
            .rtol(10.0)
            .atol([10.0])
            .rhs_implicit(
                move |x, _, t, f| f[0] = lambda * (x[0] - t.sin()) + t.cos(),
                move |_, _, _, v, jv| jv[0] = lambda * v[0],
            )
            .init(|_, _, y| y[0] = 0.0, 1)
            .integrate_out(integrate)
            .out_implicit(
                |x, _, t, g| g[0] = x[0] * x[0] + t.powi(4),
                |_, _, _, _, _| unreachable!("quadrature does not use output Jacobians"),
                1,
            )
            .build()
            .unwrap()
    }
    fn fixed_error(h: f64, lambda: f64) -> f64 {
        let p = sine_problem(lambda, false);
        let mut s = p.rodas5p::<LS>().unwrap();
        *s.state_mut().h = h;
        s.config_mut().maximum_timestep_growth = 1.0;
        s.config_mut().minimum_timestep_growth = 1.0;
        advance(&mut s, 1.0);
        (s.state().y[0] - 1.0f64.sin()).abs()
    }
    #[test]
    fn fifth_order_and_stiff_order_reduction() {
        let coarse = fixed_error(0.2, -2.0);
        let fine = fixed_error(0.1, -2.0);
        assert!(coarse > 20.0 * fine, "{coarse} {fine}");
        assert!(fixed_error(0.2, -1000.0) > fixed_error(0.1, -1000.0));
    }
    #[test]
    fn continuous_extension_midpoint_convergence() {
        let error = |h| {
            let p = sine_problem(0.0, false);
            let mut s = p.rodas5p::<LS>().unwrap();
            *s.state_mut().h = h;
            s.step().unwrap();
            (s.interpolate(h / 2.0).unwrap()[0] - (h / 2.0).sin()).abs()
        };
        assert!(error(0.4) > 12.0 * error(0.2));
    }
    #[test]
    fn embedded_difference_has_local_order_five() {
        let error = |h| {
            let p = sine_problem(-2.0, false);
            let mut s = p.rodas5p::<LS>().unwrap();
            *s.state_mut().h = h;
            s.step().unwrap();
            s.rk.error_norm(h, None::<&mut NoAug<_>>, |_| Ok(()))
                .unwrap()
                .sqrt()
        };
        let ratio = error(0.2) / error(0.1);
        assert!((20.0..45.0).contains(&ratio), "{ratio}");
    }
    #[test]
    fn nonlinear_jacobian_refreshes_at_the_accepted_state() {
        use std::{cell::RefCell, rc::Rc};
        let points = Rc::new(RefCell::new(Vec::new()));
        let observed = points.clone();
        let p = OdeBuilder::<Mat>::new()
            .rtol(1e-7)
            .atol([1e-9])
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
        let mut s = p.rodas5p::<LS>().unwrap();
        s.step().unwrap();
        let (t, y) = (s.state().t, s.state().y[0]);
        s.step().unwrap();
        assert!(points
            .borrow()
            .iter()
            .any(|&(at, x)| at == t && (x - y).abs() < 1e-12));
    }
    #[test]
    fn rejection_preserves_the_initial_solution() {
        let p = OdeBuilder::<Mat>::new()
            .rtol(1e-8)
            .atol([1e-10])
            .rhs_implicit(
                |x, _, _, f| f[0] = -100.0 * x[0],
                |_, _, _, v, jv| jv[0] = -100.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut s = p.rodas5p::<LS>().unwrap();
        *s.state_mut().h = 0.1;
        s.step().unwrap();
        assert!(s.get_statistics().number_of_error_test_failures > 0);
        assert!(s.state().t < 0.1);
        assert!((s.state().y[0] - (-100.0 * s.state().t).exp()).abs() < 2e-5);
        advance(&mut s, 0.1);
        assert!((s.state().y[0] - (-10.0f64).exp()).abs() < 2e-5);
    }
    fn euler(gamma: f64, diagonal: f64, coupling: f64) -> Tableau<f64> {
        Tableau::new_rosenbrock(
            TableauMat::from_slice(1, 1, &[diagonal]),
            TableauVec::from_slice(&[1.0]),
            TableauVec::from_slice(&[0.0]),
            TableauVec::from_slice(&[0.0]),
            1,
            None,
            TableauMat::from_slice(1, 1, &[coupling]),
            gamma,
            TableauVec::from_slice(&[0.0]),
            1,
        )
    }
    #[test]
    fn custom_tableau_and_single_stage_hermite_interpolation() {
        let (p, _) = exponential_decay_problem::<Mat>(false);
        let mut state = p.rodas5p_state::<LS>().unwrap();
        state.h = 0.1;
        let mut s = p
            .rosenbrock_solver::<LS, Mat>(state, euler(1.0, 0.0, 0.0))
            .unwrap();
        let y0 = s.state().y[0];
        let dy0 = s.state().dy[0];
        s.step().unwrap();
        assert!((s.state().y[0] - y0 / 1.01).abs() < 1e-12);
        assert!((s.interpolate_dy(0.0).unwrap()[0] - dy0).abs() < 1e-12);
        assert!((s.interpolate_dy(0.1).unwrap()[0] - s.state().dy[0]).abs() < 1e-12);
    }
    #[test]
    fn invalid_tableaus_are_typed_errors() {
        let (p, _) = exponential_decay_problem::<Mat>(false);
        for tableau in [
            euler(0.0, 0.0, 0.0),
            euler(-1.0, 0.0, 0.0),
            euler(f64::NAN, 0.0, 0.0),
            euler(1.0, 1.0, 0.0),
            euler(1.0, 0.0, 1.0),
            Tableau::esdirk34(),
        ] {
            let error = p
                .rosenbrock_solver::<LS, Mat>(p.rodas5p_state::<LS>().unwrap(), tableau)
                .err()
                .unwrap();
            assert!(matches!(
                error,
                DiffsolError::OdeSolverError(OdeSolverError::InvalidTableau(_))
            ));
        }
        let (p, _) = exponential_decay_with_algebraic_problem::<Mat>(false);
        assert!(p
            .rosenbrock_solver::<LS, Mat>(p.rodas5p_state::<LS>().unwrap(), euler(1.0, 0.0, 0.0))
            .is_err());
    }
    #[test]
    fn constant_integrated_output_is_consistent_and_dense() {
        macro_rules! check {
            ($mat:ty, $ls:ty) => {{
                let p = test_problem::<$mat>(true);
                let mut s = p.rodas5p::<$ls>().unwrap();
                *s.state_mut().h = 0.1;
                s.step().unwrap();
                assert!((s.state().g[0] - 0.1 * s.state().y[0]).abs() < 1e-12);
                assert!(
                    (s.interpolate_out(0.05).unwrap()[0] - 0.05 * s.state().y[0]).abs() < 1e-12
                );
            }};
        }
        check!(Mat, LS);
        check!(FaerMat<f64>, FaerLU<f64>);
    }
    #[test]
    fn nonlinear_integrated_output_converges_and_restarts() {
        let error = |h| {
            let p = sine_problem(-2.0, true);
            let mut s = p.rodas5p::<LS>().unwrap();
            *s.state_mut().h = h;
            s.config_mut().minimum_timestep_growth = 1.0;
            s.config_mut().maximum_timestep_growth = 1.0;
            advance(&mut s, 1.0);

            let state = s.checkpoint();
            let mut restarted = p.rodas5p_solver::<LS>(state).unwrap();
            *restarted.config_mut() = s.config().clone();
            advance(&mut s, 2.0);
            advance(&mut restarted, 2.0);
            assert!((s.state().g[0] - restarted.state().g[0]).abs() < 1e-12);
            (s.interpolate_out(s.state().t).unwrap()[0] - (1.0 - (4.0f64).sin() / 4.0 + 32.0 / 5.0))
                .abs()
        };
        let coarse = error(0.2);
        let fine = error(0.1);
        assert!(coarse > 20.0 * fine, "{coarse} {fine}");
    }
    #[test]
    fn integrated_output_has_its_own_error_control() {
        let p = OdeBuilder::<Mat>::new()
            .rtol(1e-6)
            .atol([1e-8])
            .integrate_out(true)
            .out_rtol(1e-8)
            .out_atol([1e-10])
            .rhs_implicit(|_, _, _, f| f[0] = 0.0, |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 1.0, 1)
            .out_implicit(
                |_, _, t, g| g[0] = t.exp(),
                |_, _, _, _, _| unreachable!("quadrature does not use output Jacobians"),
                1,
            )
            .build()
            .unwrap();
        let mut s = p.rodas5p::<LS>().unwrap();
        *s.state_mut().h = 1.0;
        advance(&mut s, 1.0);
        assert!(s.get_statistics().number_of_error_test_failures > 0);
        assert!((s.state().g[0] - (1.0f64.exp() - 1.0)).abs() < 1e-8);
    }

    #[test]
    fn paper_prothero_robinson_problem_two() {
        const LAMBDA: f64 = 1e5;
        let g = |t: f64| 10.0 - (10.0 + t) * (-t).exp();
        let problem = OdeBuilder::<Mat>::new()
            .rtol(1e-8)
            .atol([1e-10])
            .rhs_implicit(
                |x, _, t, f| {
                    let g = 10.0 - (10.0 + t) * (-t).exp();
                    let dg = (9.0 + t) * (-t).exp();
                    f[0] = -LAMBDA * (x[0] - g) + dg;
                },
                |_, _, _, v, jv| jv[0] = -LAMBDA * v[0],
            )
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        advance(&mut solver, 2.0);
        let error = (solver.state().y[0] - g(2.0)).abs();
        assert!(error < 1e-7, "paper problem 2 error={error}");
    }
    #[test]
    fn paper_table_six_fixed_step_errors_and_order() {
        let published_errors = [1.26e-9, 1.47e-10, 1.78e-11, 2.17e-12];
        let mut errors = Vec::new();
        for (h, published) in [0.25, 0.125, 0.0625, 0.03125]
            .into_iter()
            .zip(published_errors)
        {
            let problem = OdeBuilder::<Mat>::new()
                .rtol(10.0)
                .atol([10.0])
                .rhs_implicit(
                    |x, _, t, f| {
                        let g = 10.0 - (10.0 + t) * (-t).exp();
                        let dg = (9.0 + t) * (-t).exp();
                        f[0] = -1e5 * (x[0] - g) + dg;
                    },
                    |_, _, _, v, jv| jv[0] = -1e5 * v[0],
                )
                .init(|_, _, y| y[0] = 0.0, 1)
                .build()
                .unwrap();
            let mut solver = problem.rodas5p::<LS>().unwrap();
            *solver.state_mut().h = h;
            solver.config_mut().minimum_timestep_growth = 1.0;
            solver.config_mut().maximum_timestep_growth = 1.0;
            advance(&mut solver, 2.0);
            let exact = 10.0 - 12.0 * (-2.0_f64).exp();
            let error = (solver.state().y[0] - exact).abs();
            assert!(
                (error - published).abs() < 0.2 * published,
                "h={h}, error={error}, published={published}"
            );
            errors.push(error);
        }
        for pair in errors.windows(2) {
            let order = (pair[0] / pair[1]).log2();
            assert!((order - 3.0).abs() < 0.2, "observed order={order}");
        }
    }
    #[test]
    fn paper_table_eight_index_two_order_reduction() {
        let published_errors = [9.00e-5, 2.33e-5, 5.94e-6];
        let mut errors = Vec::new();
        for (h, published) in [0.03125, 0.015625, 0.0078125]
            .into_iter()
            .zip(published_errors)
        {
            let problem = OdeBuilder::<Mat>::new()
                .t0(1.0)
                .rtol(10.0)
                .atol([10.0, 10.0])
                .rhs_implicit(
                    |x, _, t, f| {
                        f[0] = x[1];
                        f[1] = x[0] * x[0] - 1.0 / (t * t);
                    },
                    |x, _, _, v, jv| {
                        jv[0] = v[1];
                        jv[1] = 2.0 * x[0] * v[0];
                    },
                )
                .mass(|v, _, _, beta, y| {
                    y[0] = v[0] + beta * y[0];
                    y[1] *= beta;
                })
                .init(
                    |_, _, y| {
                        y[0] = -1.0;
                        y[1] = 1.0;
                    },
                    2,
                )
                .build()
                .unwrap();
            let mut solver = problem
                .rodas5p_solver::<LS>(RkState::new(&problem, 5).unwrap())
                .unwrap();
            *solver.state_mut().h = h;
            solver.config_mut().minimum_timestep_growth = 1.0;
            solver.config_mut().maximum_timestep_growth = 1.0;
            advance(&mut solver, 2.0);
            let y = solver.state().y;
            let error = (y[0] + 0.5).abs().max((y[1] - 0.25).abs());
            assert!(
                (error - published).abs() < 0.2 * published,
                "h={h}, error={error}, published={published}"
            );
            errors.push(error);
        }
        for pair in errors.windows(2) {
            let order = (pair[0] / pair[1]).log2();
            assert!((order - 2.0).abs() < 0.2, "observed order={order}");
        }
    }
    #[test]
    fn paper_polynomial_dense_output_ode_component() {
        for degree in 1_i32..=4 {
            let problem = OdeBuilder::<Mat>::new()
                .rtol(10.0)
                .atol([10.0])
                .rhs_implicit(
                    move |_, _, t, f| f[0] = f64::from(degree) * t.powi(degree - 1),
                    |_, _, _, _, jv| jv[0] = 0.0,
                )
                .init(|_, _, y| y[0] = 0.0, 1)
                .build()
                .unwrap();
            let mut solver = problem.rodas5p::<LS>().unwrap();
            *solver.state_mut().h = 2.0;
            solver.step().unwrap();
            for t in [0.25_f64, 0.5, 1.0, 1.5, 1.75] {
                let error = (solver.interpolate(t).unwrap()[0] - t.powi(degree)).abs();
                assert!(error < 1e-9, "degree={degree}, t={t}, error={error}");
            }
        }
    }
    #[test]
    fn paper_index_one_dae_problem_one() {
        let problem = OdeBuilder::<Mat>::new()
            .t0(2.0)
            .rtol(1e-8)
            .atol([1e-10, 1e-10])
            .rhs_implicit(
                |x, _, t, f| {
                    f[0] = x[1] / x[0];
                    f[1] = x[0] / x[1] - t;
                },
                |x, _, _, v, jv| {
                    jv[0] = v[1] / x[0] - x[1] * v[0] / x[0].powi(2);
                    jv[1] = v[0] / x[1] - x[0] * v[1] / x[1].powi(2);
                },
            )
            .mass(|v, _, _, beta, y| {
                y[0] = v[0] + beta * y[0];
                y[1] *= beta;
            })
            .init(
                |_, _, y| {
                    y[0] = 2.0_f64.ln();
                    y[1] = 2.0_f64.ln() / 2.0;
                },
                2,
            )
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        advance(&mut solver, 4.0);
        let y = solver.state().y;
        assert!((y[0] - 4.0_f64.ln()).abs() < 1e-7);
        assert!((y[1] - 4.0_f64.ln() / 4.0).abs() < 1e-7);
        assert!((y[0] / y[1] - 4.0).abs() < 1e-7);
    }
    #[test]
    fn paper_table_five_fixed_step_dae_errors_and_order() {
        let published_errors = [2.93e-8, 8.56e-10, 2.59e-11];
        let mut errors = Vec::new();
        for (h, published) in [0.125, 0.0625, 0.03125].into_iter().zip(published_errors) {
            let problem = OdeBuilder::<Mat>::new()
                .t0(2.0)
                .rtol(10.0)
                .atol([10.0, 10.0])
                .rhs_implicit(
                    |x, _, t, f| {
                        f[0] = x[1] / x[0];
                        f[1] = x[0] / x[1] - t;
                    },
                    |x, _, _, v, jv| {
                        jv[0] = v[1] / x[0] - x[1] * v[0] / x[0].powi(2);
                        jv[1] = v[0] / x[1] - x[0] * v[1] / x[1].powi(2);
                    },
                )
                .mass(|v, _, _, beta, y| {
                    y[0] = v[0] + beta * y[0];
                    y[1] *= beta;
                })
                .init(
                    |_, _, y| {
                        y[0] = 2.0_f64.ln();
                        y[1] = 2.0_f64.ln() / 2.0;
                    },
                    2,
                )
                .build()
                .unwrap();
            let mut solver = problem.rodas5p::<LS>().unwrap();
            *solver.state_mut().h = h;
            solver.config_mut().minimum_timestep_growth = 1.0;
            solver.config_mut().maximum_timestep_growth = 1.0;
            advance(&mut solver, 4.0);
            let y = solver.state().y;
            let error = (y[0] - 4.0_f64.ln())
                .abs()
                .max((y[1] - 4.0_f64.ln() / 4.0).abs());
            assert!(
                (error - published).abs() < 0.2 * published,
                "h={h}, error={error}, published={published}"
            );
            errors.push(error);
        }
        for pair in errors.windows(2) {
            let order = (pair[0] / pair[1]).log2();
            assert!((order - 5.0).abs() < 0.25, "observed order={order}");
        }
    }
    #[test]
    fn paper_polynomial_dense_output_dae_problem_six() {
        for degree in 1_i32..=5 {
            let problem = OdeBuilder::<Mat>::new()
                .rtol(10.0)
                .atol([10.0, 10.0])
                .rhs_implicit(
                    move |x, _, t, f| {
                        f[0] = f64::from(degree) * t.powi(degree - 1);
                        f[1] = x[0] - x[1];
                    },
                    |_, _, _, v, jv| {
                        jv[0] = 0.0;
                        jv[1] = v[0] - v[1];
                    },
                )
                .mass(|v, _, _, beta, y| {
                    y[0] = v[0] + beta * y[0];
                    y[1] *= beta;
                })
                .init(
                    |_, _, y| {
                        y[0] = 0.0;
                        y[1] = 0.0;
                    },
                    2,
                )
                .build()
                .unwrap();
            let mut solver = problem.rodas5p::<LS>().unwrap();
            *solver.state_mut().h = 2.0;
            solver.step().unwrap();
            assert!((solver.state().y[0] - 2.0_f64.powi(degree)).abs() < 1e-9);
            assert!((solver.state().y[1] - 2.0_f64.powi(degree)).abs() < 1e-9);
            let exact_derivative = f64::from(degree) * 2.0_f64.powi(degree - 1);
            if degree <= 4 {
                assert!((solver.state().dy[0] - exact_derivative).abs() < 1e-9);
                assert!((solver.state().dy[1] - exact_derivative).abs() < 1e-9);
            }
            for t in [0.25_f64, 0.5, 1.0, 1.5, 1.75] {
                let y = solver.interpolate(t).unwrap();
                let exact = t.powi(degree);
                for component in 0..2 {
                    let error = (y[component] - exact).abs();
                    if degree <= 4 {
                        assert!(
                            error < 1e-10,
                            "degree={degree}, t={t}, component={component}, error={error}"
                        );
                    } else if t == 1.0 {
                        // Table 9: fifth-order endpoint accuracy but only
                        // fourth-order continuous interpolation.
                        assert!(
                            (error - 0.312).abs() < 0.02,
                            "degree={degree}, component={component}, error={error}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nonautonomous_constant_mass_dae() {
        // y0 = sin(t), y1 = y0; f_t acts on the differential equation.
        let p = OdeBuilder::<Mat>::new()
            .rtol(1e-8)
            .atol([1e-10, 1e-10])
            .rhs_implicit(
                |x, _, t, f| {
                    f[0] = -x[0] + t.sin() + t.cos();
                    f[1] = x[0] - x[1];
                },
                |_, _, _, v, jv| {
                    jv[0] = -v[0];
                    jv[1] = v[0] - v[1];
                },
            )
            .mass(|v, _, _, beta, y| {
                y[0] = v[0] + beta * y[0];
                y[1] *= beta;
            })
            .init(
                |_, _, y| {
                    y[0] = 0.0;
                    y[1] = 0.0;
                },
                2,
            )
            .build()
            .unwrap();
        let mut s = p.rodas5p::<LS>().unwrap();
        advance(&mut s, 1.0);
        for i in 0..2 {
            assert!((s.state().y[i] - 1.0f64.sin()).abs() < 1e-7);
        }
    }
    #[test]
    fn integrated_output_requires_continuous_extension() {
        let p = sine_problem(0.0, true);
        let err = p
            .rosenbrock_solver::<LS, Mat>(p.rodas5p_state::<LS>().unwrap(), euler(1.0, 0.0, 0.0))
            .err()
            .unwrap();
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::InvalidTableau(_))
        ));
    }
    fn stability_step(z: f64) -> f64 {
        let p = OdeBuilder::<Mat>::new()
            .rtol(1e30)
            .atol([1e30])
            .rhs_implicit(
                move |x, _, _, f| f[0] = z * x[0],
                move |_, _, _, v, jv| jv[0] = z * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut s = p.rodas5p::<LS>().unwrap();
        *s.state_mut().h = 1.0;
        s.step().unwrap();
        s.state().y[0]
    }
    #[test]
    fn rodas5p_transformed_stiff_accuracy_and_l_stability() {
        let t = Tableau::<f64>::rodas5p();
        // In transformed form the final correction u8 completes the last stage state:
        // b = A[7,:] + e8, not b = A[7,:] (the latter has a zero diagonal).
        for i in 0..8 {
            assert!((t.b()[i] - t.a(7, i) - if i == 7 { 1.0 } else { 0.0 }).abs() < 1e-15);
        }
        assert!(stability_step(-1e8).abs() < 1e-6);
        assert!(stability_step(-1e14).abs() < 1e-10);
    }
}
