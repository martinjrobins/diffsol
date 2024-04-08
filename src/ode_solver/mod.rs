pub mod bdf;
pub mod builder;
pub mod equations;
pub mod method;
pub mod problem;
pub mod sdirk;
pub mod test_models;

#[cfg(feature = "diffsl")]
pub mod diffsl;

#[cfg(feature = "sundials")]
pub mod sundials;

#[cfg(test)]
mod tests {
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
    use crate::op::ode_rhs::OdeRhs;
    use crate::op::Op;
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
        mut root_solver: impl NonLinearSolver<FilterCallable<OdeRhs<Eqn>>>,
        problem: OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
    ) where
        M: Matrix + 'static,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let state = OdeSolverState::new_consistent(&problem, &mut root_solver).unwrap();
        method.set_problem(state, &problem);
        for point in solution.solution_points.iter() {
            while method.state().unwrap().t < point.t {
                method.step().unwrap();
            }

            let soln = method.interpolate(point.t).unwrap();

            if let Some(override_tol) = override_tol {
                soln.assert_eq(&point.state, override_tol);
            } else {
                let tol = {
                    let problem = method.problem().unwrap();
                    soln.abs() * scale(problem.rtol) + problem.atol.as_ref()
                };
                soln.assert_eq(&point.state, M::T::from(10.0) * tol[0]);
            }
        }
    }

    type Mcpu = nalgebra::DMatrix<f64>;

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay() {
        let mut s = Sdirk::new(
            Tableau::<Mcpu>::tr_bdf2(),
            LU::default(),
        );
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 32
        number_of_jac_mul_evals: 2
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_sdirk4_nalgebra_exponential_decay() {
        let mut s = Sdirk::new(
            Tableau::<Mcpu>::sdirk4(),
            LU::default(),
        );
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 32
        number_of_jac_mul_evals: 2
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
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
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 56
        number_of_jac_mul_evals: 2
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[cfg(feature = "sundials")]
    #[test]
    fn test_sundials_exponential_decay() {
        let mut s = crate::SundialsIda::default();
        let rs = NewtonNonlinearSolver::new(crate::SundialsLinearSolver::new_dense());
        let (problem, soln) = exponential_decay_problem::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
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
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 39
        number_of_jac_mul_evals: 32
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 16
        "###);
    }

    #[test]
    fn test_sdirk4_nalgebra_exponential_decay_algebraic() {
        let mut s = Sdirk::new(
            Tableau::<Mcpu>::sdirk4(),
            LU::default(),
        );
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 34
        number_of_jac_mul_evals: 4
        number_of_mass_evals: 36
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 18
        number_of_steps: 21
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 58
        number_of_nonlinear_solver_fails: 2
        initial_step_size: 0.004450050658086208
        final_step_size: 0.20974041151932246
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 62
        number_of_jac_mul_evals: 7
        number_of_mass_evals: 64
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 2
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson() {
        let mut s = Sdirk::new(
            Tableau::<Mcpu>::tr_bdf2(),
            LU::default(),
        );
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 32
        number_of_jac_mul_evals: 2
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_sdirk4_nalgebra_robertson() {
        let mut s = Sdirk::new(
            Tableau::<Mcpu>::sdirk4(),
            LU::default(),
        );
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 34
        number_of_jac_mul_evals: 4
        number_of_mass_evals: 36
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 129
        number_of_steps: 374
        number_of_error_test_failures: 17
        number_of_nonlinear_solver_iterations: 1046
        number_of_nonlinear_solver_fails: 25
        initial_step_size: 0.0000045643545698038086
        final_step_size: 7622676567.923919
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1049
        number_of_jac_mul_evals: 55
        number_of_mass_evals: 1052
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 18
        "###);
    }

    #[cfg(feature = "sundials")]
    #[test]
    fn test_sundials_robertson() {
        let mut s = crate::SundialsIda::default();
        let rs = NewtonNonlinearSolver::new(crate::SundialsLinearSolver::new_dense());
        let (problem, soln) = robertson::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
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
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 507
        number_of_jac_mul_evals: 178
        number_of_mass_evals: 686
        number_of_mass_matrix_evals: 60
        number_of_jacobian_matrix_evals: 59
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson::<Mcpu>(true);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 129
        number_of_steps: 374
        number_of_error_test_failures: 17
        number_of_nonlinear_solver_iterations: 1046
        number_of_nonlinear_solver_fails: 25
        initial_step_size: 0.0000045643545698038086
        final_step_size: 7622676567.923919
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1049
        number_of_jac_mul_evals: 58
        number_of_mass_evals: 1051
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_sdirk4_nalgebra_robertson_ode() {
        let mut s = Sdirk::new(
            Tableau::<Mcpu>::sdirk4(),
            LU::default(),
        );
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson_ode::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 34
        number_of_jac_mul_evals: 4
        number_of_mass_evals: 36
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = robertson_ode::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 105
        number_of_steps: 346
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 941
        number_of_nonlinear_solver_fails: 18
        initial_step_size: 0.0000038381494276795106
        final_step_size: 7310380599.023874
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 943
        number_of_jac_mul_evals: 51
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 17
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = dydt_y2_problem::<Mcpu>(false, 10);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 109
        number_of_steps: 469
        number_of_error_test_failures: 16
        number_of_nonlinear_solver_iterations: 1440
        number_of_nonlinear_solver_fails: 4
        initial_step_size: 0.000000024472764934039197
        final_step_size: 0.6825461945378934
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1442
        number_of_jac_mul_evals: 50
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 5
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = dydt_y2_problem::<Mcpu>(true, 10);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 109
        number_of_steps: 469
        number_of_error_test_failures: 16
        number_of_nonlinear_solver_iterations: 1440
        number_of_nonlinear_solver_fails: 4
        initial_step_size: 0.000000024472764934039197
        final_step_size: 0.6825461945378934
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1442
        number_of_jac_mul_evals: 15
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 5
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(LU::default());
        let (problem, soln) = gaussian_decay_problem::<Mcpu>(false, 10);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 15
        number_of_steps: 56
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 144
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.0025148668593658707
        final_step_size: 0.19513473994542221
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 146
        number_of_jac_mul_evals: 10
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    pub struct TestEqn<M> {
        _m: std::marker::PhantomData<M>,
    }
    impl<M: Matrix> Op for TestEqn<M> {
        type M = M;
        type T = M::T;
        type V = M::V;

        fn nout(&self) -> usize {
            1
        }

        fn nstates(&self) -> usize {
            1
        }

        fn nparams(&self) -> usize {
            1
        }
    }
    impl<M: Matrix> OdeEquations for TestEqn<M> {
        fn set_params(&mut self, _p: Self::V) {}

        fn rhs_inplace(&self, _t: Self::T, _y: &Self::V, rhs_y: &mut Self::V) {
            rhs_y[0] = M::T::zero();
        }

        fn rhs_jac_inplace(&self, _t: Self::T, _x: &Self::V, _v: &Self::V, y: &mut Self::V) {
            y[0] = M::T::zero();
        }

        fn init(&self, _t: Self::T) -> Self::V {
            M::V::from_element(1, M::T::zero())
        }
    }

    pub fn test_interpolate<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        let problem = OdeSolverProblem::new(
            TestEqn {
                _m: std::marker::PhantomData,
            },
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
            .assert_eq(&state.y, M::T::from(1e-9));
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
            TestEqn {
                _m: std::marker::PhantomData,
            },
            M::T::from(1e-6),
            M::V::from_element(1, M::T::from(1e-6)),
            M::T::zero(),
            M::T::one(),
        );
        let state = OdeSolverState::new(&problem);
        s.set_problem(state.clone(), &problem);
        let state2 = s.take_state().unwrap();
        state2.y.assert_eq(&state.y, M::T::from(1e-9));
        assert!(s.take_state().is_none());
        assert!(s.state().is_none());
        assert!(s.step().is_err());
        assert!(s.interpolate(M::T::one()).is_err());
    }
}
