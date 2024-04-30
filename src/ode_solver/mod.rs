pub mod bdf;
pub mod builder;
pub mod equations;
pub mod method;
pub mod problem;
pub mod sdirk;
pub mod tableau;
pub mod test_models;

#[cfg(feature = "diffsl")]
pub mod diffsl;

#[cfg(feature = "sundials")]
pub mod sundials;

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use self::problem::OdeSolverSolution;

    use super::test_models::{
        exponential_decay::exponential_decay_problem,
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem,
        robertson::robertson, robertson_ode::robertson_ode,
    };
    use super::*;
    use crate::linear_solver::nalgebra::lu::LU;
    use crate::matrix::Matrix;
    use crate::nonlinear_solver::newton::NewtonNonlinearSolver;
    use crate::op::filter::FilterCallable;
    use crate::op::unit::UnitCallable;
    use crate::op::{NonLinearOp, Op};
    use crate::scalar::scale;
    use crate::{NonLinearSolver, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState};
    use crate::{Sdirk, Tableau, Vector};
    use num_traits::One;
    use num_traits::Zero;
    use tests::bdf::Bdf;
    use tests::test_models::dydt_y2::dydt_y2_problem;
    use tests::test_models::gaussian_decay::gaussian_decay_problem;

    fn test_ode_solver<M, Eqn>(
        method: &mut impl OdeSolverMethod<Eqn>,
        mut root_solver: impl NonLinearSolver<FilterCallable<Eqn::Rhs>>,
        problem: &OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
    ) where
        M: Matrix,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let state = OdeSolverState::new_consistent(problem, &mut root_solver).unwrap();
        method.set_problem(state, problem);
        for point in solution.solution_points.iter() {
            while method.state().unwrap().t < point.t {
                method.step().unwrap();
            }

            let soln = method.interpolate(point.t).unwrap();

            if let Some(override_tol) = override_tol {
                soln.assert_eq_st(&point.state, override_tol);
            } else {
                let scale = {
                    let problem = method.problem().unwrap();
                    point.state.abs() * scale(problem.rtol) + problem.atol.as_ref()
                };
                let mut error = soln.clone() - &point.state;
                error.component_div_assign(&scale);
                let error_norm = error.norm() / M::T::from((point.state.len() as f64).sqrt());
                assert!(
                    error_norm < M::T::from(10.0),
                    "error_norm: {} at t = {}",
                    error_norm,
                    point.t
                );
            }
        }
    }

    type Mcpu = nalgebra::DMatrix<f64>;

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay() {
        let tableau = Tableau::<Mcpu>::tr_bdf2();
        let mut s = Sdirk::new(tableau, LU::default());
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 4
        number_of_steps: 4
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.1919383103666485
        final_step_size: 0.32116185769995603
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 18
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay() {
        let tableau = Tableau::<Mcpu>::esdirk34();
        let mut s = Sdirk::new(tableau, LU::default());
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 2
        number_of_steps: 2
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.28998214001102113
        final_step_size: 0.8287576160727735
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 14
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 16
        number_of_steps: 19
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 54
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.011892071150027213
        final_step_size: 0.23215911532645564
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 56
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[cfg(feature = "sundials")]
    #[test]
    fn test_sundials_exponential_decay() {
        let mut s = crate::SundialsIda::default();
        let rs = NewtonNonlinearSolver::new(crate::SundialsLinearSolver::new_dense());
        let (problem, soln) = exponential_decay_problem::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 16
        number_of_steps: 24
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 39
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.001
        final_step_size: 0.256
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 39
        number_of_jac_muls: 32
        number_of_matrix_evals: 16
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 17
        number_of_steps: 21
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 58
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.004450050658086208
        final_step_size: 0.20995860176773154
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 62
        number_of_jac_muls: 4
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson() {
        let tableau = Tableau::<Mcpu>::tr_bdf2();
        let mut s = Sdirk::new(tableau, LU::default());
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 435
        number_of_steps: 416
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 12
        initial_step_size: 0.0011378590984747281
        final_step_size: 61530652710.46018
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_rhs_evals: 3189
        number_of_jac_mul_evals: 40
        number_of_mass_evals: 3192
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 13
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson() {
        let tableau = Tableau::<Mcpu>::esdirk34();
        let mut s = Sdirk::new(tableau, LU::default());
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 380
        number_of_steps: 349
        number_of_error_test_failures: 11
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 20
        initial_step_size: 0.00619535739618413
        final_step_size: 58646462255.08058
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_rhs_evals: 3513
        number_of_jac_mul_evals: 55
        number_of_mass_evals: 3516
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 106
        number_of_steps: 345
        number_of_error_test_failures: 5
        number_of_nonlinear_solver_iterations: 985
        number_of_nonlinear_solver_fails: 22
        initial_step_size: 0.0000045643545698038086
        final_step_size: 5435491162.573224
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 988
        number_of_jac_muls: 55
        number_of_matrix_evals: 18
        "###);
    }

    #[cfg(feature = "sundials")]
    #[test]
    fn test_sundials_robertson() {
        let mut s = crate::SundialsIda::default();
        let rs = NewtonNonlinearSolver::new(crate::SundialsLinearSolver::new_dense());
        let (problem, soln) = robertson::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 59
        number_of_steps: 355
        number_of_error_test_failures: 15
        number_of_nonlinear_solver_iterations: 506
        number_of_nonlinear_solver_fails: 5
        initial_step_size: 0.001
        final_step_size: 11535117835.253025
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 507
        number_of_jac_muls: 178
        number_of_matrix_evals: 59
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(true);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 106
        number_of_steps: 345
        number_of_error_test_failures: 5
        number_of_nonlinear_solver_iterations: 985
        number_of_nonlinear_solver_fails: 22
        initial_step_size: 0.0000045643545698038086
        final_step_size: 5435491162.573224
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_rhs_evals: 988
        number_of_jac_mul_evals: 58
        number_of_mass_evals: 990
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_ode() {
        let tableau = Tableau::<Mcpu>::tr_bdf2();
        let mut s = Sdirk::new(tableau, LU::default());
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson_ode::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 242
        number_of_steps: 230
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 12
        initial_step_size: 0.0010137172178872197
        final_step_size: 30691200423.745472
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_rhs_evals: 2354
        number_of_jac_mul_evals: 39
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 13
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson_ode::<Mcpu>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 106
        number_of_steps: 345
        number_of_error_test_failures: 5
        number_of_nonlinear_solver_iterations: 981
        number_of_nonlinear_solver_fails: 22
        initial_step_size: 0.0000038381494276795106
        final_step_size: 5636682847.540523
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 983
        number_of_jac_muls: 54
        number_of_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = dydt_y2_problem::<Mcpu>(false, 10);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 65
        number_of_steps: 205
        number_of_error_test_failures: 10
        number_of_nonlinear_solver_iterations: 593
        number_of_nonlinear_solver_fails: 7
        initial_step_size: 0.0000019982428436469115
        final_step_size: 1.0781694150073
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 595
        number_of_jac_muls: 60
        number_of_matrix_evals: 6
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = dydt_y2_problem::<Mcpu>(true, 10);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 59
        number_of_steps: 196
        number_of_error_test_failures: 11
        number_of_nonlinear_solver_iterations: 425
        number_of_nonlinear_solver_fails: 2
        initial_step_size: 0.0000019982428436469115
        final_step_size: 1.0012225310200868
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 427
        number_of_jac_muls: 12
        number_of_matrix_evals: 2
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = gaussian_decay_problem::<Mcpu>(false, 10);
        test_ode_solver(&mut s, rs, &problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 14
        number_of_steps: 58
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 159
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.0025148668593658707
        final_step_size: 0.19566316816600493
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 161
        number_of_jac_muls: 10
        number_of_matrix_evals: 1
        "###);
    }

    pub struct TestEqnRhs<M> {
        _m: std::marker::PhantomData<M>,
    }

    impl<M: Matrix> Op for TestEqnRhs<M> {
        type T = M::T;
        type V = M::V;
        type M = M;

        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            0
        }
        fn nstates(&self) -> usize {
            1   
        }
    }

    impl<M: Matrix> NonLinearOp for TestEqnRhs<M> {
        fn call_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::V) {
            y[0] = M::T::zero();
        }

        fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
            y[0] = M::T::zero();
        }
    }

    pub struct TestEqn<M: Matrix> {
        rhs: Rc<TestEqnRhs<M>>,
        mass: Rc<UnitCallable<M>>,
    }

    impl<M: Matrix> TestEqn<M> {
        pub fn new() -> Self {
            Self {
                rhs: Rc::new(TestEqnRhs {
                    _m: std::marker::PhantomData,
                }),
                mass: Rc::new(UnitCallable::new(1)),
            }
        }
    }
    
    impl<M: Matrix> OdeEquations for TestEqn<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type Rhs = TestEqnRhs<M>;
        type Mass = UnitCallable<M>;

        fn set_params(&mut self, _p: Self::V) {}

        fn rhs(&self) -> &Rc<Self::Rhs> {
            &self.rhs
        }

        fn mass(&self) -> &Rc<Self::Mass> {
            &self.mass
        }

        fn init(&self, _t: Self::T) -> Self::V {
            M::V::from_element(1, M::T::zero())
        }
    }


    pub fn test_interpolate<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        let problem = OdeSolverProblem::new(
            TestEqn::new(),
            M::T::from(1e-6),
            M::V::from_element(1, M::T::from(1e-6)),
            M::T::zero(),
            M::T::one(),
        );
        let state = OdeSolverState::new(&problem);
        s.set_problem(state.clone(), &problem);
        let t0 = M::T::zero();
        let t1 = M::T::one();
        s.interpolate(t0)
            .unwrap()
            .assert_eq_st(&state.y, M::T::from(1e-9));
        assert!(s.interpolate(t1).is_err());
        s.step().unwrap();
        assert!(s.interpolate(s.state().unwrap().t).is_ok());
        assert!(s.interpolate(s.state().unwrap().t + t1).is_err());
    }

    pub fn test_no_set_problem<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        assert!(s.state().is_none());
        assert!(s.problem().is_none());
        assert!(s.take_state().is_none());
        assert!(s.step().is_err());
        assert!(s.interpolate(M::T::one()).is_err());
    }

    pub fn test_take_state<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        let problem = OdeSolverProblem::new(
            TestEqn::new(),
            M::T::from(1e-6),
            M::V::from_element(1, M::T::from(1e-6)),
            M::T::zero(),
            M::T::one(),
        );
        let state = OdeSolverState::new(&problem);
        s.set_problem(state.clone(), &problem);
        let state2 = s.take_state().unwrap();
        state2.y.assert_eq_st(&state.y, M::T::from(1e-9));
        assert!(s.take_state().is_none());
        assert!(s.state().is_none());
        assert!(s.step().is_err());
        assert!(s.interpolate(M::T::one()).is_err());
    }
}
