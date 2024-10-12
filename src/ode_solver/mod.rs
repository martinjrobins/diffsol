pub mod adjoint_equations;
pub mod bdf;
pub mod bdf_state;
pub mod builder;
pub mod checkpointing;
pub mod equations;
pub mod jacobian_update;
pub mod method;
pub mod problem;
pub mod sdirk;
pub mod sdirk_state;
pub mod sens_equations;
pub mod state;
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
    use method::{AdjointOdeSolverMethod, SensitivitiesOdeSolverMethod};
    use nalgebra::ComplexField;

    use super::*;
    use crate::matrix::Matrix;
    use crate::op::unit::UnitCallable;
    use crate::op::{NonLinearOp, Op};
    use crate::{ConstantOp, DefaultDenseMatrix, DefaultSolver, Vector};
    use crate::{
        OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason,
    };
    use num_traits::One;
    use num_traits::Zero;

    pub fn test_ode_solver<M, Eqn>(
        method: &mut impl SensitivitiesOdeSolverMethod<Eqn>,
        problem: &OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
        use_tstop: bool,
        solve_for_sensitivities: bool,
    ) -> Eqn::V
    where
        M: Matrix,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
        Eqn::M: DefaultSolver,
    {
        if solve_for_sensitivities {
            let sensitivity_error_control = solution.sens_solution_points.is_some();
            let state = OdeSolverState::new_with_sensitivities(problem, method).unwrap();
            method
                .set_problem_with_sensitivities(state, problem, sensitivity_error_control)
                .unwrap();
        } else {
            let state = OdeSolverState::new(problem, method).unwrap();
            method.set_problem(state, problem).unwrap();
        }
        let have_root = problem.eqn.as_ref().root().is_some();
        for (i, point) in solution.solution_points.iter().enumerate() {
            let (soln, sens_soln) = if use_tstop {
                match method.set_stop_time(point.t) {
                    Ok(_) => loop {
                        match method.step() {
                            Ok(OdeSolverStopReason::RootFound(_)) => {
                                assert!(have_root);
                                return method.state().unwrap().y.clone();
                            }
                            Ok(OdeSolverStopReason::TstopReached) => {
                                break (
                                    method.state().unwrap().y.clone(),
                                    method.state().unwrap().s.to_vec(),
                                );
                            }
                            _ => (),
                        }
                    },
                    Err(_) => (
                        method.state().unwrap().y.clone(),
                        method.state().unwrap().s.to_vec(),
                    ),
                }
            } else {
                while method.state().unwrap().t.abs() < point.t.abs() {
                    if let OdeSolverStopReason::RootFound(t) = method.step().unwrap() {
                        assert!(have_root);
                        return method.interpolate(t).unwrap();
                    }
                }
                let soln = method.interpolate(point.t).unwrap();
                let sens_soln = method.interpolate_sens(point.t).unwrap();
                (soln, sens_soln)
            };
            let soln = if let Some(out) = problem.eqn.out() {
                out.call(&soln, point.t)
            } else {
                soln
            };
            assert_eq!(
                soln.len(),
                point.state.len(),
                "soln.len() != point.state.len()"
            );
            if let Some(override_tol) = override_tol {
                soln.assert_eq_st(&point.state, override_tol);
            } else {
                let (rtol, atol) = if problem.eqn.out().is_some() {
                    // problem rtol and atol is on the state, so just use solution tolerance here
                    (solution.rtol, &solution.atol)
                } else {
                    (problem.rtol, problem.atol.as_ref())
                };
                let error = soln.clone() - &point.state;
                let error_norm = error.squared_norm(&point.state, atol, rtol).sqrt();
                assert!(
                    error_norm < M::T::from(15.0),
                    "error_norm: {} at t = {}. soln: {:?}, expected: {:?}",
                    error_norm,
                    point.t,
                    soln,
                    point.state
                );
                if solve_for_sensitivities {
                    if let Some(sens_soln_points) = solution.sens_solution_points.as_ref() {
                        for (j, sens_points) in sens_soln_points.iter().enumerate() {
                            let sens_point = &sens_points[i];
                            let sens_soln = &sens_soln[j];
                            let error = sens_soln.clone() - &sens_point.state;
                            let error_norm =
                                error.squared_norm(&sens_point.state, atol, rtol).sqrt();
                            assert!(
                                error_norm < M::T::from(29.0),
                                "error_norm: {} at t = {}",
                                error_norm,
                                point.t
                            );
                        }
                    }
                }
            }
        }
        method.state().unwrap().y.clone()
    }

    pub fn test_ode_solver_no_sens<M, Eqn>(
        method: &mut impl OdeSolverMethod<Eqn>,
        problem: &OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
        use_tstop: bool,
    ) -> Eqn::V
    where
        M: Matrix,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
        Eqn::M: DefaultSolver,
    {
        let state = OdeSolverState::new(problem, method).unwrap();
        method.set_problem(state, problem).unwrap();
        let have_root = problem.eqn.as_ref().root().is_some();
        for point in solution.solution_points.iter() {
            let soln = if use_tstop {
                match method.set_stop_time(point.t) {
                    Ok(_) => loop {
                        match method.step() {
                            Ok(OdeSolverStopReason::RootFound(_)) => {
                                assert!(have_root);
                                return method.state().unwrap().y.clone();
                            }
                            Ok(OdeSolverStopReason::TstopReached) => {
                                break method.state().unwrap().y.clone();
                            }
                            _ => (),
                        }
                    },
                    Err(_) => method.state().unwrap().y.clone(),
                }
            } else {
                while method.state().unwrap().t.abs() < point.t.abs() {
                    if let OdeSolverStopReason::RootFound(t) = method.step().unwrap() {
                        assert!(have_root);
                        return method.interpolate(t).unwrap();
                    }
                }
                method.interpolate(point.t).unwrap()
            };
            let soln = if let Some(out) = problem.eqn.out() {
                out.call(&soln, point.t)
            } else {
                soln
            };
            assert_eq!(
                soln.len(),
                point.state.len(),
                "soln.len() != point.state.len()"
            );
            if let Some(override_tol) = override_tol {
                soln.assert_eq_st(&point.state, override_tol);
            } else {
                let (rtol, atol) = if problem.eqn.out().is_some() {
                    // problem rtol and atol is on the state, so just use solution tolerance here
                    (solution.rtol, &solution.atol)
                } else {
                    (problem.rtol, problem.atol.as_ref())
                };
                let error = soln.clone() - &point.state;
                let error_norm = error.squared_norm(&point.state, atol, rtol).sqrt();
                assert!(
                    error_norm < M::T::from(15.0),
                    "error_norm: {} at t = {}. soln: {:?}, expected: {:?}",
                    error_norm,
                    point.t,
                    soln,
                    point.state
                );
            }
        }
        method.state().unwrap().y.clone()
    }

    pub fn test_ode_solver_adjoint<M, Eqn, Method>(
        method: &mut Method,
        problem: &OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
    ) -> Method::AdjointSolver
    where
        M: Matrix,
        Method: AdjointOdeSolverMethod<Eqn>,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
        Eqn::M: DefaultSolver,
    {
        let state = OdeSolverState::new(problem, method).unwrap();
        method.set_problem(state, problem).unwrap();
        let t0 = solution.solution_points.first().unwrap().t;
        let t1 = solution.solution_points.last().unwrap().t;
        method.set_stop_time(t1).unwrap();
        let mut nsteps = 0;
        let (rtol, atol) = (solution.rtol, &solution.atol);
        let mut checkpoints = vec![method.checkpoint().unwrap()];
        for point in solution.solution_points.iter() {
            while method.state().unwrap().t.abs() < point.t.abs() {
                method.step().unwrap();
                nsteps += 1;
                if nsteps > 50 {
                    checkpoints.push(method.checkpoint().unwrap());
                    nsteps = 0;
                }
            }
            let soln = method.interpolate_out(point.t).unwrap();
            // problem rtol and atol is on the state, so just use solution tolerance here
            let error = soln.clone() - &point.state;
            let error_norm = error.squared_norm(&point.state, atol, rtol).sqrt();
            assert!(
                error_norm < M::T::from(15.0),
                "error_norm: {} at t = {}. soln: {:?}, expected: {:?}",
                error_norm,
                point.t,
                soln,
                point.state
            );
        }
        checkpoints.push(method.checkpoint().unwrap());
        let mut adjoint_solver = method.new_adjoint_solver(checkpoints, true).unwrap();
        let y_expect = M::V::from_element(problem.eqn.rhs().nstates(), M::T::zero());
        adjoint_solver.state().unwrap().y.assert_eq_st(&y_expect, M::T::from(1e-9));
        for i in 0..problem.eqn.out().unwrap().nout() {
            adjoint_solver.state().unwrap().s[i].assert_eq_st(&y_expect, M::T::from(1e-9));
        }
        let g_expect = M::V::from_element(problem.eqn.rhs().nparams(), M::T::zero());
        for i in 0..problem.eqn.out().unwrap().nout() {
            adjoint_solver.state().unwrap().sg[i].assert_eq_st(&g_expect, M::T::from(1e-9));
        }
        
        adjoint_solver.set_stop_time(t0).unwrap();
        while adjoint_solver.state().unwrap().t.abs() > t0 {
            adjoint_solver.step().unwrap();
        }
        let mut state = adjoint_solver.take_state().unwrap();
        let state_mut = state.as_mut();
        adjoint_solver.problem().unwrap().eqn.correct_sg_for_init(t0, state_mut.s, state_mut.sg);
        
        let points = solution
            .sens_solution_points
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| &x[0])
            .collect::<Vec<_>>();
        for (soln, point) in state_mut.sg.iter().zip(points.iter()) {
            let error = soln.clone() - &point.state;
            let error_norm = error.squared_norm(&point.state, atol, rtol).sqrt();
            assert!(
                error_norm < M::T::from(20.0),
                "error_norm: {} at t = {}. soln: {:?}, expected: {:?}",
                error_norm,
                point.t,
                soln,
                point.state
            );
        }
        adjoint_solver
    }

    pub struct TestEqnInit<M> {
        _m: std::marker::PhantomData<M>,
    }

    impl<M: Matrix> Op for TestEqnInit<M> {
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

    impl<M: Matrix> ConstantOp for TestEqnInit<M> {
        fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
            y[0] = M::T::one();
        }
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
        init: Rc<TestEqnInit<M>>,
    }

    impl<M: Matrix> TestEqn<M> {
        pub fn new() -> Self {
            Self {
                rhs: Rc::new(TestEqnRhs {
                    _m: std::marker::PhantomData,
                }),
                init: Rc::new(TestEqnInit {
                    _m: std::marker::PhantomData,
                }),
            }
        }
    }

    impl<M: Matrix> OdeEquations for TestEqn<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type Rhs = TestEqnRhs<M>;
        type Mass = UnitCallable<M>;
        type Root = UnitCallable<M>;
        type Init = TestEqnInit<M>;
        type Out = UnitCallable<M>;

        fn set_params(&mut self, _p: Self::V) {}

        fn rhs(&self) -> &Rc<Self::Rhs> {
            &self.rhs
        }

        fn mass(&self) -> Option<&Rc<Self::Mass>> {
            None
        }

        fn root(&self) -> Option<&Rc<Self::Root>> {
            None
        }

        fn init(&self) -> &Rc<Self::Init> {
            &self.init
        }

        fn out(&self) -> Option<&Rc<Self::Out>> {
            None
        }
    }

    pub fn test_interpolate<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        let problem = OdeSolverProblem::new(
            TestEqn::new(),
            M::T::from(1e-6),
            M::V::from_element(1, M::T::from(1e-6)),
            M::T::zero(),
            M::T::one(),
            false,
        )
        .unwrap();
        let state = Method::State::new_without_initialise(&problem).unwrap();
        s.set_problem(state.clone(), &problem).unwrap();
        let t0 = M::T::zero();
        let t1 = M::T::one();
        s.interpolate(t0)
            .unwrap()
            .assert_eq_st(state.as_ref().y, M::T::from(1e-9));
        assert!(s.interpolate(t1).is_err());
        s.step().unwrap();
        assert!(s.interpolate(s.state().unwrap().t).is_ok());
        assert!(s.interpolate(s.state().unwrap().t + t1).is_err());
    }

    pub fn test_no_set_problem<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        assert!(s.state().is_none());
        assert!(s.problem().is_none());
        assert!(s.state().is_none());
        assert!(s.step().is_err());
        assert!(s.interpolate(M::T::one()).is_err());
    }

    pub fn test_state_mut<M: Matrix, Method: OdeSolverMethod<TestEqn<M>>>(mut s: Method) {
        let problem = OdeSolverProblem::new(
            TestEqn::new(),
            M::T::from(1e-6),
            M::V::from_element(1, M::T::from(1e-6)),
            M::T::zero(),
            M::T::one(),
            false,
        )
        .unwrap();
        let state = Method::State::new_without_initialise(&problem).unwrap();
        s.set_problem(state.clone(), &problem).unwrap();
        let state2 = s.state().unwrap();
        state2.y.assert_eq_st(state.as_ref().y, M::T::from(1e-9));
        s.state_mut().unwrap().y[0] = M::T::from(std::f64::consts::PI);
        assert_eq!(
            s.state_mut().unwrap().y[0],
            M::T::from(std::f64::consts::PI)
        );
    }

    pub fn test_checkpointing<M, Method, Problem>(
        mut solver1: Method,
        mut solver2: Method,
        problem: OdeSolverProblem<Problem>,
        soln: OdeSolverSolution<M::V>,
    ) where
        M: Matrix + DefaultSolver,
        Method: OdeSolverMethod<Problem>,
        Problem: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let state = OdeSolverState::new(&problem, &solver1).unwrap();
        solver1.set_problem(state, &problem).unwrap();
        let half_i = soln.solution_points.len() / 2;
        let half_t = soln.solution_points[half_i].t;
        while solver1.state().unwrap().t <= half_t {
            solver1.step().unwrap();
        }
        let checkpoint = solver1.checkpoint().unwrap();
        solver2.set_problem(checkpoint, &problem).unwrap();

        // carry on solving with both solvers, they should produce about the same results (probably might diverge a bit, but should always match the solution)
        for point in soln.solution_points.iter().skip(half_i + 1) {
            while solver2.state().unwrap().t < point.t {
                solver1.step().unwrap();
                solver2.step().unwrap();
                let time_error = (solver1.state().unwrap().t - solver2.state().unwrap().t)
                    .abs()
                    / (solver1.state().unwrap().t.abs() * problem.rtol + problem.atol[0]);
                assert!(
                    time_error < M::T::from(20.0),
                    "time_error: {} at t = {}",
                    time_error,
                    solver1.state().unwrap().t
                );
                solver1.state().unwrap().y.assert_eq_norm(
                    solver2.state().unwrap().y,
                    &problem.atol,
                    problem.rtol,
                    M::T::from(20.0),
                );
            }
            let soln = solver1.interpolate(point.t).unwrap();
            soln.assert_eq_norm(&point.state, &problem.atol, problem.rtol, M::T::from(15.0));
            let soln = solver2.interpolate(point.t).unwrap();
            soln.assert_eq_norm(&point.state, &problem.atol, problem.rtol, M::T::from(15.0));
        }
    }

    pub fn test_state_mut_on_problem<Eqn, Method>(
        mut s: Method,
        problem: OdeSolverProblem<Eqn>,
        soln: OdeSolverSolution<Eqn::V>,
    ) where
        Eqn: OdeEquations,
        Method: OdeSolverMethod<Eqn>,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
    {
        // solve for a little bit
        s.solve(&problem, Eqn::T::from(1.0)).unwrap();

        // reinit using state_mut
        let state = Method::State::new_without_initialise(&problem).unwrap();
        s.state_mut().unwrap().y.copy_from(state.as_ref().y);
        *s.state_mut().unwrap().t = state.as_ref().t;

        // solve and check against solution
        for point in soln.solution_points.iter() {
            while s.state().unwrap().t < point.t {
                s.step().unwrap();
            }
            let soln = s.interpolate(point.t).unwrap();
            let error = soln.clone() - &point.state;
            let error_norm = error
                .squared_norm(&error, &problem.atol, problem.rtol)
                .sqrt();
            assert!(
                error_norm < Eqn::T::from(17.0),
                "error_norm: {} at t = {}",
                error_norm,
                point.t
            );
        }
    }
}
