use super::method::AugmentedOdeSolverMethod;
use super::runge_kutta::Rk;
use crate::error::DiffsolError;
use crate::ode_solver::bdf::BdfStatistics;
use crate::vector::VectorRef;
use crate::NoAug;
use crate::OdeSolverStopReason;
use crate::RkState;
use crate::Tableau;
use crate::{
    AugmentedOdeEquations, DefaultDenseMatrix, DenseMatrix, ExplicitRkConfig, OdeEquations,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, StateRef, StateRefMut,
};
use num_traits::One;

impl<'a, Eqn, M, AugEqn> AugmentedOdeSolverMethod<'a, Eqn, AugEqn>
    for ExplicitRk<'a, Eqn, M, AugEqn>
where
    Eqn: OdeEquations,
    AugEqn: AugmentedOdeEquations<Eqn>,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugEqn>) {
        (self.rk.into_state(), self.augmented_eqn)
    }
    fn augmented_eqn(&self) -> Option<&AugEqn> {
        self.augmented_eqn.as_ref()
    }
}

/// An explicit Runge-Kutta method.
///
/// The particular method is defined by the [Tableau] used to create the solver.
/// If the `beta` matrix of the [Tableau] is present this is used for interpolation, otherwise hermite interpolation is used.
///
/// Restrictions:
/// - The upper triangular and diagonal parts of the `a` matrix must be zero (i.e. explicit).
/// - The last row of the `a` matrix must be the same as the `b` vector, and the last element of the `c` vector must be 1 (i.e. a stiffly accurate method)
pub struct ExplicitRk<
    'a,
    Eqn,
    M = <<Eqn as Op>::V as DefaultDenseMatrix>::M,
    AugmentedEqn = NoAug<Eqn>,
> where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    rk: Rk<'a, Eqn, M>,
    augmented_eqn: Option<AugmentedEqn>,
    config: ExplicitRkConfig<Eqn::T>,
}

impl<Eqn, M, AugmentedEqn> Clone for ExplicitRk<'_, Eqn, M, AugmentedEqn>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    fn clone(&self) -> Self {
        Self {
            rk: self.rk.clone(),
            augmented_eqn: self.augmented_eqn.clone(),
            config: self.config.clone(),
        }
    }
}

impl<'a, Eqn, M, AugmentedEqn> ExplicitRk<'a, Eqn, M, AugmentedEqn>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<M>,
    ) -> Result<Self, DiffsolError> {
        Rk::<Eqn, M>::check_explicit_rk(problem, &tableau)?;
        Ok(Self {
            rk: Rk::new(problem, state, tableau)?,
            augmented_eqn: None,
            config: ExplicitRkConfig::default(),
        })
    }

    pub fn new_augmented(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<M>,
        augmented_eqn: AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        Rk::<Eqn, M>::check_explicit_rk(problem, &tableau)?;
        Ok(Self {
            rk: Rk::new_augmented(problem, state, tableau, &augmented_eqn)?,
            augmented_eqn: Some(augmented_eqn),
            config: ExplicitRkConfig::default(),
        })
    }

    pub fn get_statistics(&self) -> &BdfStatistics {
        self.rk.get_statistics()
    }
}

impl<'a, Eqn, M, AugmentedEqn> OdeSolverMethod<'a, Eqn> for ExplicitRk<'a, Eqn, M, AugmentedEqn>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    type State = RkState<Eqn::V>;
    type Config = ExplicitRkConfig<Eqn::T>;

    fn config(&self) -> &ExplicitRkConfig<Eqn::T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ExplicitRkConfig<Eqn::T> {
        &mut self.config
    }

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.rk.problem()
    }

    fn jacobian(&self) -> Option<std::cell::Ref<'_, <Eqn>::M>> {
        None
    }

    fn mass(&self) -> Option<std::cell::Ref<'_, <Eqn>::M>> {
        None
    }

    fn order(&self) -> usize {
        self.rk.order()
    }

    fn set_state(&mut self, state: Self::State) {
        self.rk.set_state(state);
    }

    fn into_state(self) -> RkState<Eqn::V> {
        self.rk.into_state()
    }

    fn checkpoint(&mut self) -> Self::State {
        self.rk.checkpoint()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut h = self.rk.start_step()?;

        // loop until step is accepted
        let mut nattempts = 0;
        let factor = loop {
            // start a step attempt
            self.rk.start_step_attempt(h, self.augmented_eqn.as_mut());
            for i in 1..self.rk.tableau().s() {
                self.rk.do_stage(i, h, self.augmented_eqn.as_mut());
            }
            let error_norm = self.rk.error_norm(h, self.augmented_eqn.as_mut());
            let factor = self.rk.factor(
                error_norm,
                1.0,
                self.config.minimum_timestep_shrink,
                self.config.maximum_timestep_growth,
            );
            if error_norm < Eqn::T::one() {
                break factor;
            }
            h *= factor;
            nattempts += 1;
            self.rk.error_test_fail(
                h,
                nattempts,
                self.config.maximum_error_test_failures,
                self.config.minimum_timestep,
            )?;
        };
        self.rk.step_accepted(h, h * factor, false)
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
                exponential_decay_problem_sens, exponential_decay_problem_with_root,
                negative_exponential_decay_problem,
            },
            robertson_ode::robertson_ode,
        },
        ode_solver::tests::{
            setup_test_adjoint, setup_test_adjoint_sum_squares, test_adjoint,
            test_adjoint_sum_squares, test_checkpointing, test_config, test_interpolate,
            test_ode_solver, test_problem, test_state_mut, test_state_mut_on_problem,
        },
        Context, DenseMatrix, MatrixCommon, NalgebraLU, NalgebraVec, OdeEquations, OdeSolverMethod,
        Op, Vector, VectorView,
    };

    use num_traits::abs;

    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn explicit_rk_state_mut() {
        test_state_mut(test_problem::<M>(false).tsit45().unwrap());
    }
    #[test]
    fn explicit_rk_config() {
        test_config(robertson_ode::<M>(false, 1).0.tsit45().unwrap());
    }
    #[test]
    fn explicit_rk_test_interpolate() {
        test_interpolate(test_problem::<M>(false).tsit45().unwrap());
    }

    #[test]
    fn explicit_rk_test_interpolate_out() {
        test_interpolate(test_problem::<M>(true).tsit45().unwrap());
    }

    #[test]
    fn explicit_rk_test_interpolate_sens() {
        test_interpolate(test_problem::<M>(false).tsit45_sens().unwrap());
    }

    #[test]
    fn explicit_rk_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let s1 = problem.tsit45().unwrap();
        let s2 = problem.tsit45().unwrap();
        test_checkpointing(soln, s1, s2);
    }

    #[test]
    fn explicit_rk_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = p.tsit45().unwrap();
        test_state_mut_on_problem(s, soln);
    }

    #[test]
    fn explicit_rk_test_nalgebra_negative_exponential_decay() {
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tsit45_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 0
        number_of_steps: 5
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 32
        number_of_jac_muls: 0
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tsit45_nalgebra_f32_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<NalgebraMat<f32>>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_tsit45_nalgebra_heat1d_diffsl() {
        use crate::ode_equations::test_models::heat1d::heat1d_diffsl_problem;

        let (problem, soln) = heat1d_diffsl_problem::<M, diffsl::LlvmModule, 10>();
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 0
        number_of_steps: 93
        number_of_error_test_failures: 9
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 0
        number_of_jac_muls: 0
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 0
        "###);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_tsit45_cuda_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<crate::CudaMat<f64>>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 0
        number_of_steps: 5
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 32
        number_of_jac_muls: 0
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tsit45_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.tsit45_sens().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 0
        number_of_steps: 8
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 50
        number_of_jac_muls: 98
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn explicit_rk_test_tsit45_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.tsit45().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem.tsit45_solver_adjoint(checkpointer, None).unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 434
        number_of_jac_muls: 8
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 123
        "###);
    }

    #[test]
    fn explicit_rk_test_nalgebra_exponential_decay_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(false);
        let mut s = problem.tsit45().unwrap();
        let (checkpointer, soln) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .tsit45_solver_adjoint(checkpointer, Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 747
        number_of_jac_muls: 0
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 1707
        "###);
    }

    #[test]
    fn test_tstop_tsit45() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[test]
    fn test_root_finder_tsit45() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let mut s = problem.tsit45().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }

    #[test]
    fn test_param_sweep_tsit45() {
        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        let mut ps = Vec::new();
        for y0 in (1..10).map(f64::from) {
            ps.push(problem.context().vector_from_vec(vec![0.1, y0]));
        }

        let mut old_soln: Option<NalgebraVec<f64>> = None;
        for p in ps {
            problem.eqn_mut().set_params(&p);
            let mut s = problem.tsit45().unwrap();
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
    fn test_ball_bounce_tsit45() {
        type M = crate::NalgebraMat<f64>;
        let (x, v, t) = crate::ode_solver::tests::test_ball_bounce(
            crate::ode_solver::tests::test_ball_bounce_problem::<M>()
                .tsit45()
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
