fn main() {
    divan::main();
}

mod exponential_decay {
    use diffsol::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use diffsol::{Bdf, OdeSolverMethod};

    #[divan::bench]
    fn bdf() {
        let mut s = Bdf::default();
        let (problem, _soln) = exponential_decay_problem::<nalgebra::DMatrix<f64>>(false);
        let _y = s.solve(&problem, 1.0);
    }

    #[cfg(feature = "sundials")]
    #[divan::bench]
    fn sundials() {
        let mut s = diffsol::SundialsIda::default();
        let (problem, _soln) = exponential_decay_problem::<diffsol::SundialsMatrix>(false);
        let _y = s.solve(&problem, 1.0);
    }
}

mod robertson_ode {
    use diffsol::{ode_solver::test_models::robertson_ode::robertson_ode, Bdf, OdeSolverMethod};

    #[divan::bench]
    fn bdf() {
        let mut s = Bdf::default();
        let (problem, _soln) = robertson_ode::<nalgebra::DMatrix<f64>>(false);
        let _y = s.solve(&problem, 4.0000e+10);
    }

    #[cfg(feature = "sundials")]
    #[divan::bench]
    fn sundials() {
        let mut s = diffsol::SundialsIda::default();
        let (problem, _soln) = robertson_ode::<diffsol::SundialsMatrix>(false);
        let _y = s.solve(&problem, 4.0000e+10);
    }
}

mod robertson {
    use diffsol::{
        ode_solver::test_models::robertson::robertson, Bdf, NalgebraLU, NewtonNonlinearSolver,
        OdeSolverMethod,
    };

    #[divan::bench]
    fn bdf() {
        let mut s = Bdf::default();
        let (problem, _soln) = robertson::<nalgebra::DMatrix<f64>>(false);
        let mut root = NewtonNonlinearSolver::new(NalgebraLU::default());
        let _y = s.make_consistent_and_solve(&problem, 4.0000e+10, &mut root);
    }

    #[cfg(feature = "sundials")]
    #[divan::bench]
    fn sundials() {
        use diffsol::SundialsLinearSolver;

        let mut s = diffsol::SundialsIda::default();
        let (problem, _soln) = robertson::<diffsol::SundialsMatrix>(false);
        let mut root = NewtonNonlinearSolver::new(SundialsLinearSolver::new_dense());
        let _y = s.make_consistent_and_solve(&problem, 4.0000e+10, &mut root);
    }
}
