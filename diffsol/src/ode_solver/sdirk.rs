use crate::error::DiffsolError;
use crate::matrix::MatrixRef;
use crate::ode_solver::runge_kutta::Rk;
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
    OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, StateRef,
    StateRefMut,
};
use num_traits::{FromPrimitive, One};

use super::bdf::BdfStatistics;
use super::config::SdirkConfig;
use super::jacobian_update::SolverState;
use super::method::AugmentedOdeSolverMethod;

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
    nonlinear_solver: NewtonNonlinearSolver<Eqn::M, LS>,
    convergence: Convergence<'a, Eqn::V>,
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
        let mut nonlinear_solver = NewtonNonlinearSolver::new(LS::default());
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
        Self {
            rk: self.rk.clone(),
            convergence: self.convergence.clone(),
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
        self.rk.tableau().a().get_index(1, 1)
    }

    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<M>,
        linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        Rk::<Eqn, M>::check_sdirk_rk(&tableau)?;
        let rk = Rk::new(problem, state, tableau)?;
        let mut ret = Self::_new(rk, problem, linear_solver, true, SdirkConfig::default())?;
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
        let mut jacobian_update = JacobianUpdate::default();
        jacobian_update.update_jacobian(state.h);
        jacobian_update.update_rhs_jacobian();

        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);

        // set max iterations for nonlinear solver
        let mut convergence = Convergence::new(problem.rtol, &problem.atol);
        convergence.set_max_iter(config.maximum_newton_iterations);

        let gamma = rk.tableau().a().get_index(1, 1);
        let op = if integrate_main_eqn {
            let callable = SdirkCallable::new(&problem.eqn, gamma);
            callable.set_h(state.h);
            Some(callable)
        } else {
            None
        };

        Ok(Self {
            rk,
            convergence,
            nonlinear_solver,
            op,
            s_op: None,
            jacobian_update,
            config,
        })
    }

    pub fn new_augmented(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<M>,
        linear_solver: LS,
        augmented_eqn: AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        Rk::<Eqn, M>::check_sdirk_rk(&tableau)?;
        let rk = Rk::new_augmented(problem, state, tableau, &augmented_eqn)?;
        let mut ret = Self::_new(rk, problem, linear_solver, true, SdirkConfig::default())?;

        ret.s_op = if augmented_eqn.integrate_main_eqn() {
            ret.nonlinear_solver.set_problem(ret.op.as_ref().unwrap());
            let callable = SdirkCallable::new_no_jacobian(augmented_eqn, ret.gamma());
            callable.set_h(ret.rk.state().h);
            Some(callable)
        } else {
            ret.op = None;
            let state = ret.rk.state();
            let callable = SdirkCallable::new(augmented_eqn, ret.gamma());
            callable.set_h(state.h);
            ret.nonlinear_solver.set_problem(&callable);
            Some(callable)
        };
        ret.jacobian_updates(ret.rk.state().h, SolverState::Checkpoint);
        Ok(ret)
    }

    fn jacobian_updates(&mut self, h: Eqn::T, state: SolverState) {
        if self.jacobian_update.check_rhs_jacobian_update(h, &state) {
            if let Some(op) = self.op.as_mut() {
                op.set_jacobian_is_stale();
                self.nonlinear_solver.reset_jacobian(
                    op,
                    &self.rk.old_state().dy,
                    self.rk.state().t,
                );
            } else if let Some(s_op) = self.s_op.as_mut() {
                s_op.set_jacobian_is_stale();
                self.nonlinear_solver.reset_jacobian(
                    s_op,
                    &self.rk.old_state().ds[0],
                    self.rk.state().t,
                );
            }
            self.jacobian_update.update_rhs_jacobian();
            self.jacobian_update.update_jacobian(h);
        } else if self.jacobian_update.check_jacobian_update(h, &state) {
            // shouldn't matter what we put in for x cause rhs_jacobian is already updated
            if let Some(op) = self.op.as_ref() {
                self.nonlinear_solver.reset_jacobian(
                    op,
                    &self.rk.old_state().dy,
                    self.rk.state().t,
                );
            } else if let Some(s_op) = self.s_op.as_ref() {
                self.nonlinear_solver.reset_jacobian(
                    s_op,
                    &self.rk.old_state().ds[0],
                    self.rk.state().t,
                );
            }
            self.jacobian_update.update_jacobian(h);
        }
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

    pub fn get_statistics(&self) -> &BdfStatistics {
        self.rk.get_statistics()
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
    type State = RkState<Eqn::V>;
    type Config = SdirkConfig<Eqn::T>;

    fn config(&self) -> &SdirkConfig<Eqn::T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut SdirkConfig<Eqn::T> {
        &mut self.config
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
            let x = &self.rk.state().s[0];
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

    fn set_state(&mut self, state: Self::State) {
        let h = state.h;
        self.rk.set_state(state);

        self.update_op_step_size(h);

        // reinitialise jacobian updates as if a checkpoint was taken
        self.nonlinear_solver.clear_jacobian();
    }

    fn into_state(self) -> RkState<Eqn::V> {
        self.rk.into_state()
    }

    fn checkpoint(&mut self) -> Self::State {
        self.nonlinear_solver.clear_jacobian();
        self.rk.state().clone()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut h = self.rk.start_step()?;

        // setup the operators for the step
        self.update_op_step_size(h);

        // loop until step is accepted
        let mut nattempts = 0;
        let mut updated_jacobian = false;
        let start = if self.rk.skip_first_stage() { 1 } else { 0 };
        let factor = 'step: loop {
            // start a step attempt
            self.rk
                .start_step_attempt(h, self.s_op.as_mut().map(|s_op| s_op.eqn_mut()));
            for i in start..self.rk.tableau().s() {
                if self
                    .rk
                    .do_stage_sdirk(
                        i,
                        h,
                        self.op.as_ref(),
                        self.s_op.as_mut(),
                        &mut self.nonlinear_solver,
                        &mut self.convergence,
                    )
                    .is_err()
                {
                    if !updated_jacobian {
                        // newton iteration did not converge, so update jacobian and try again
                        updated_jacobian = true;
                        self.jacobian_updates(h, SolverState::FirstConvergenceFail);
                    } else {
                        // newton iteration did not converge and jacobian has been updated, so we reduce step size and try again
                        h *= Eqn::T::from_f64(0.3).unwrap();
                        self.update_op_step_size(h);
                        self.jacobian_updates(h, SolverState::SecondConvergenceFail);
                    }
                    self.rk.solve_fail(h, self.config.minimum_timestep)?;
                    // try again....
                    continue 'step;
                }
            }
            let error_norm = self
                .rk
                .error_norm(h, self.s_op.as_mut().map(|s_op| s_op.eqn_mut()));

            let maxiter = self.convergence.max_iter() as f64;
            let niter = self.convergence.niter() as f64;
            let safety_factor = (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter);
            let factor = self.rk.factor(
                error_norm,
                safety_factor,
                self.config.minimum_timestep_shrink,
                self.config.maximum_timestep_growth,
            );
            if error_norm < Eqn::T::one() {
                break factor;
            }
            h *= factor;
            self.update_op_step_size(h);
            self.jacobian_updates(h, SolverState::ErrorTestFail);
            nattempts += 1;
            self.rk.error_test_fail(
                h,
                nattempts,
                self.config.maximum_error_test_failures,
                self.config.minimum_timestep,
            )?;
        };

        // accept the step
        let new_h = h * factor;
        self.update_op_step_size(h);
        self.jacobian_updates(h, SolverState::StepSuccess);
        self.jacobian_update.step();
        self.rk.step_accepted(h, new_h, true)
    }

    fn set_stop_time(&mut self, tstop: <Eqn as Op>::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(tstop)
    }

    fn interpolate_sens_inplace(
        &self,
        t: <Eqn as Op>::T,
        sens: &mut [Eqn::V],
    ) -> Result<(), DiffsolError> {
        self.rk.interpolate_sens_inplace(t, sens)
    }

    fn interpolate_inplace(&self, t: <Eqn>::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_inplace(t, y)
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
}

#[cfg(test)]
mod test {
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
            heat2d::head2d_problem,
            robertson::{robertson, robertson_sens},
            robertson_ode::robertson_ode,
        },
        ode_solver::tests::{
            setup_test_adjoint, setup_test_adjoint_sum_squares, test_adjoint,
            test_adjoint_sum_squares, test_checkpointing, test_config, test_interpolate,
            test_ode_solver, test_problem, test_state_mut, test_state_mut_on_problem,
        },
        Context, DenseMatrix, FaerSparseLU, FaerSparseMat, MatrixCommon, NalgebraLU, NalgebraVec,
        OdeEquations, OdeSolverMethod, Op, Vector, VectorView,
    };

    use num_traits::abs;

    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn sdirk_state_mut() {
        test_state_mut(test_problem::<M>(false).tr_bdf2::<LS>().unwrap());
    }
    #[test]
    fn sdirk_config() {
        test_config(robertson_ode::<M>(false, 1).0.esdirk34::<LS>().unwrap());
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
    fn test_tr_bdf2_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 4
        number_of_steps: 29
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 116
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 118
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
        number_of_linear_solver_setups: 7
        number_of_steps: 55
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 660
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 222
        number_of_jac_muls: 446
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
        number_of_linear_solver_setups: 3
        number_of_steps: 13
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 109
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 111
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
        number_of_linear_solver_setups: 13
        number_of_steps: 7
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 109
        number_of_nonlinear_solver_fails: 3
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 113
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
        number_of_linear_solver_setups: 5
        number_of_steps: 21
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 442
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 140
        number_of_jac_muls: 308
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn sdirk_test_esdirk34_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer, None)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 543
        number_of_jac_muls: 10
        number_of_matrix_evals: 5
        number_of_jac_adj_muls: 348
        "###);
    }

    #[test]
    fn sdirk_test_nalgebra_exponential_decay_algebraic_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, soln) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer, Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 434
        number_of_jac_muls: 9
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 953
        "###);
    }

    #[test]
    fn sdirk_test_esdirk34_exponential_decay_algebraic_adjoint() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer, None)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 496
        number_of_jac_muls: 21
        number_of_matrix_evals: 7
        number_of_jac_adj_muls: 199
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 95
        number_of_steps: 233
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 2024
        number_of_nonlinear_solver_fails: 30
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 2027
        number_of_jac_muls: 36
        number_of_matrix_evals: 12
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_sens() {
        let (problem, soln) = robertson_sens::<M>();
        let mut s = problem.tr_bdf2_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 100
        number_of_steps: 214
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 4894
        number_of_nonlinear_solver_fails: 63
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1652
        number_of_jac_muls: 3317
        number_of_matrix_evals: 22
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 89
        number_of_steps: 145
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 1920
        number_of_nonlinear_solver_fails: 44
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1923
        number_of_jac_muls: 42
        number_of_matrix_evals: 14
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson_sens() {
        let (problem, soln) = robertson_sens::<M>();
        let mut s = problem.esdirk34_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 123
        number_of_steps: 140
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 5152
        number_of_nonlinear_solver_fails: 110
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1753
        number_of_jac_muls: 3498
        number_of_matrix_evals: 30
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_ode() {
        let (problem, soln) = robertson_ode::<M>(false, 1);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 117
        number_of_steps: 307
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 2722
        number_of_nonlinear_solver_fails: 40
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 2724
        number_of_jac_muls: 39
        number_of_matrix_evals: 13
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
    fn test_tstop_tr_bdf2() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[test]
    fn test_root_finder_tr_bdf2() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
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
            let (ys, _ts) = s.solve(10.0).unwrap();
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
}
