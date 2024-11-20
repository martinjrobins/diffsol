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

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use self::problem::OdeSolverSolution;
    use checkpointing::HermiteInterpolator;
    use method::{AdjointOdeSolverMethod};
    use nalgebra::ComplexField;

    use super::*;
    use crate::matrix::Matrix;
    use crate::op::unit::UnitCallable;
    use crate::{
        op::OpStatistics, CraneliftModule, DenseMatrix, DiffSl, MatrixCommon, NonLinearOpJacobian,
        OdeBuilder, OdeEquations, OdeEquationsAdjoint, OdeEquationsImplicit, OdeEquationsRef,
        OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason,
        VectorView,
    };
    use crate::{ConstantOp, DefaultDenseMatrix, DefaultSolver, NonLinearOp, Op, Vector};
    use num_traits::One;
    use num_traits::Zero;

    pub fn test_ode_solver<'a, M, Eqn, Method>(
        method: &mut Method,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
        use_tstop: bool,
        solve_for_sensitivities: bool,
    ) -> Eqn::V
    where
        M: Matrix,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V> + 'a,
        Eqn::M: DefaultSolver,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        let have_root = method.problem().eqn.as_ref().root().is_some();
        for (i, point) in solution.solution_points.iter().enumerate() {
            let (soln, sens_soln) = if use_tstop {
                match method.set_stop_time(point.t) {
                    Ok(_) => loop {
                        match method.step() {
                            Ok(OdeSolverStopReason::RootFound(_)) => {
                                assert!(have_root);
                                return method.state().y.clone();
                            }
                            Ok(OdeSolverStopReason::TstopReached) => {
                                break (
                                    method.state().y.clone(),
                                    method.state().s.to_vec(),
                                );
                            }
                            _ => (),
                        }
                    },
                    Err(_) => (
                        method.state().y.clone(),
                        method.state().s.to_vec(),
                    ),
                }
            } else {
                while method.state().t.abs() < point.t.abs() {
                    if let OdeSolverStopReason::RootFound(t) = method.step().unwrap() {
                        assert!(have_root);
                        return method.interpolate(t).unwrap();
                    }
                }
                let soln = method.interpolate(point.t).unwrap();
                let sens_soln = method.interpolate_sens(point.t).unwrap();
                (soln, sens_soln)
            };
            let soln = if let Some(out) = method.problem().eqn.out() {
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
                let (rtol, atol) = if method.problem().eqn.out().is_some() {
                    // problem rtol and atol is on the state, so just use solution tolerance here
                    (solution.rtol, &solution.atol)
                } else {
                    (method.problem().rtol, method.problem().atol.as_ref())
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
        method.state().y.clone()
    }

    pub fn test_ode_solver_adjoint<'a, M, Eqn, Method>(
        method: Method,
        solution: OdeSolverSolution<M::V>,
    ) -> Method::AdjointSolver
    where
        M: Matrix,
        Method: AdjointOdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsAdjoint<M = M, T = M::T, V = M::V>,
        Eqn::M: DefaultSolver,
    {
        let t0 = solution.solution_points.first().unwrap().t;
        let t1 = solution.solution_points.last().unwrap().t;
        method.set_stop_time(t1).unwrap();
        let mut nsteps = 0;
        let (rtol, atol) = (solution.rtol, &solution.atol);
        let mut checkpoints = vec![method.checkpoint()];
        let mut ts = Vec::new();
        let mut ys = Vec::new();
        let mut ydots = Vec::new();
        for point in solution.solution_points.iter() {
            while method.state().t.abs() < point.t.abs() {
                ts.push(method.state().t);
                ys.push(method.state().y.clone());
                ydots.push(method.state().dy.clone());
                method.step().unwrap();
                nsteps += 1;
                if nsteps > 50 && method.state().t.abs() < t1.abs() {
                    checkpoints.push(method.checkpoint());
                    nsteps = 0;
                    ts.clear();
                    ys.clear();
                    ydots.clear();
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
        ts.push(method.state().t);
        ys.push(method.state().y.clone());
        ydots.push(method.state().dy.clone());
        checkpoints.push(method.checkpoint());
        let last_segment = HermiteInterpolator::new(ys, ydots, ts);
        let mut adjoint_solver = method
            .into_adjoint_solver(checkpoints, last_segment)
            .unwrap();
        let y_expect = M::V::from_element(method.problem().eqn.rhs().nstates(), M::T::zero());
        adjoint_solver
            .state()
            .y
            .assert_eq_st(&y_expect, M::T::from(1e-9));
        let g_expect = M::V::from_element(method.problem().eqn.rhs().nparams(), M::T::zero());
        for i in 0..method.problem().eqn.out().unwrap().nout() {
            adjoint_solver.state().sg[i].assert_eq_st(&g_expect, M::T::from(1e-9));
        }

        adjoint_solver.set_stop_time(t0).unwrap();
        while adjoint_solver.state().t.abs() > t0 {
            adjoint_solver.step().unwrap();
        }
        let adjoint_problem = adjoint_solver.problem().clone();
        let mut state = adjoint_solver.into_state();
        let state_mut = state.as_mut();
        adjoint_problem
            .eqn
            .correct_sg_for_init(t0, state_mut.s, state_mut.sg);

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
                error_norm < M::T::from(15.0),
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
        fn call_inplace(&self, _t: Self::T, mut y: &mut Self::V) {
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
        fn call_inplace(&self, _x: &Self::V, _t: Self::T, mut y: &mut Self::V) {
            y[0] = M::T::zero();
        }
    }

    impl<M: Matrix> NonLinearOpJacobian for TestEqnRhs<M> {
        fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, mut y: &mut Self::V) {
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

    impl<M: Matrix> Op for TestEqn<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        fn set_params(&mut self, _p: Rc<Self::V>) {}
        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            0
        }
        fn nstates(&self) -> usize {
            1
        }
        fn statistics(&self) -> crate::op::OpStatistics {
            OpStatistics::default()
        }
    }

    impl<'a, M: Matrix> OdeEquationsRef<'a> for TestEqn<M> {
        type Rhs = &'a TestEqnRhs<M>;
        type Mass = &'a UnitCallable<M>;
        type Root = &'a UnitCallable<M>;
        type Init = &'a TestEqnInit<M>;
        type Out = &'a UnitCallable<M>;
    }

    impl<M: Matrix> OdeEquations for TestEqn<M> {
        fn rhs(&self) -> &TestEqnRhs<M> {
            &self.rhs
        }

        fn mass(&self) -> Option<&UnitCallable<M>> {
            None
        }

        fn root(&self) -> Option<&UnitCallable<M>> {
            None
        }

        fn init(&self) -> &TestEqnInit<M> {
            &self.init
        }

        fn out(&self) -> Option<&UnitCallable<M>> {
            None
        }
    }

    pub fn test_interpolate<'a, M: Matrix, Method: OdeSolverMethod<'a, TestEqn<M>>, F: FnOnce(&'a OdeSolverProblem<TestEqn<M>>, Method::State) -> Method>(f: F) {
        let problem = OdeSolverProblem::new(
            Rc::new(TestEqn::new()),
            M::T::from(1e-6),
            Rc::new(M::V::from_element(1, M::T::from(1e-6))),
            None,
            None,
            None,
            None,
            None,
            None,
            M::T::zero(),
            M::T::one(),
            false,
        )
        .unwrap();
        let state = Method::State::new_without_initialise(&problem).unwrap();
        let s = f(&problem, state.clone());
        let t0 = M::T::zero();
        let t1 = M::T::one();
        s.interpolate(t0)
            .unwrap()
            .assert_eq_st(state.as_ref().y, M::T::from(1e-9));
        assert!(s.interpolate(t1).is_err());
        s.step().unwrap();
        assert!(s.interpolate(s.state().t).is_ok());
        assert!(s.interpolate(s.state().t + t1).is_err());
    }

    pub fn test_state_mut<'a, M: Matrix, Method: OdeSolverMethod<'a, TestEqn<M>>, F: FnOnce(&'a OdeSolverProblem<TestEqn<M>>, Method::State) -> Method>(f: F) {
        let problem = OdeSolverProblem::new(
            Rc::new(TestEqn::new()),
            M::T::from(1e-6),
            Rc::new(M::V::from_element(1, M::T::from(1e-6))),
            None,
            None,
            None,
            None,
            None,
            None,
            M::T::zero(),
            M::T::one(),
            false,
        )
        .unwrap();
        let state = Method::State::new_without_initialise(&problem).unwrap();
        let s = f(&problem, state);
        let state2 = s.state();
        state2.y.assert_eq_st(state.as_ref().y, M::T::from(1e-9));
        s.state_mut().y[0] = M::T::from(std::f64::consts::PI);
        assert_eq!(
            s.state_mut().y[0],
            M::T::from(std::f64::consts::PI)
        );
    }

    #[cfg(feature = "diffsl")]
    pub fn test_ball_bounce<'a, M, Method, F>(f: F) -> (Vec<f64>, Vec<f64>, Vec<f64>)
    where
        M: Matrix<T = f64>,
        M: DefaultSolver<T = f64>,
        M::V: DefaultDenseMatrix<T = f64>,
        Method: OdeSolverMethod<'a, DiffSl<M, CraneliftModule>>,
        F: FnOnce(&OdeSolverProblem<DiffSl<M, CraneliftModule>>) -> Method,  
    {
        let eqn = DiffSl::compile(
            "
            g { 9.81 } h { 10.0 }
            u_i {
                x = h,
                v = 0,
            }
            F_i {
                v,
                -g,
            }
            stop {
                x,
            }
        ",
        )
        .unwrap();

        let e = 0.8;
        let problem = OdeBuilder::new().build_from_eqn(eqn).unwrap();
        let solver = f(&problem);

        let final_time = 2.5;

        // solve and apply the remaining doses
        solver.set_stop_time(final_time).unwrap();
        loop {
            match solver.step() {
                Ok(OdeSolverStopReason::InternalTimestep) => (),
                Ok(OdeSolverStopReason::RootFound(t)) => {
                    // get the state when the event occurred
                    let mut y = solver.interpolate(t).unwrap();

                    // update the velocity of the ball
                    y[1] *= -e;

                    // make sure the ball is above the ground
                    y[0] = y[0].max(f64::EPSILON);

                    // set the state to the updated state
                    solver.state_mut().y.copy_from(&y);
                    solver.state_mut().dy[0] = y[1];
                    *solver.state_mut().t = t;

                    break;
                }
                Ok(OdeSolverStopReason::TstopReached) => break,
                Err(_) => panic!("unexpected solver error"),
            }
        }
        // do three more steps after the 1st bound and many sure they are correct
        let mut x = vec![];
        let mut v = vec![];
        let mut t = vec![];
        for _ in 0..3 {
            let ret = solver.step();
            x.push(solver.state().y[0]);
            v.push(solver.state().y[1]);
            t.push(solver.state().t);
            match ret {
                Ok(OdeSolverStopReason::InternalTimestep) => (),
                Ok(OdeSolverStopReason::RootFound(_)) => {
                    panic!("should be an internal timestep but found a root")
                }
                Ok(OdeSolverStopReason::TstopReached) => break,
                _ => panic!("should be an internal timestep"),
            }
        }
        (x, v, t)
    }

    pub fn test_checkpointing<'a, M, Method, Eqn>(
        soln: OdeSolverSolution<M::V>,
        solver1: Method,
        solver2: Method,
    ) where
        M: Matrix + DefaultSolver,
        Method: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicit<M = M, T = M::T, V = M::V> + 'a,
    {
        let half_i = soln.solution_points.len() / 2;
        let half_t = soln.solution_points[half_i].t;
        while solver1.state().t <= half_t {
            solver1.step().unwrap();
        }
        let checkpoint = solver1.checkpoint();
        solver2.set_state(checkpoint);

        // carry on solving with both solvers, they should produce about the same results (probably might diverge a bit, but should always match the solution)
        for point in soln.solution_points.iter().skip(half_i + 1) {
            while solver2.state().t < point.t {
                solver1.step().unwrap();
                solver2.step().unwrap();
                let time_error = (solver1.state().t - solver2.state().t).abs()
                    / (solver1.state().t.abs() * solver1.problem().rtol + solver1.problem().atol[0]);
                assert!(
                    time_error < M::T::from(20.0),
                    "time_error: {} at t = {}",
                    time_error,
                    solver1.state().t
                );
                solver1.state().y.assert_eq_norm(
                    solver2.state().y,
                    &solver1.problem().atol,
                    solver1.problem().rtol,
                    M::T::from(20.0),
                );
            }
            let soln = solver1.interpolate(point.t).unwrap();
            soln.assert_eq_norm(&point.state, &solver1.problem().atol, solver1.problem().rtol, M::T::from(15.0));
            let soln = solver2.interpolate(point.t).unwrap();
            soln.assert_eq_norm(&point.state, &solver1.problem().atol, solver1.problem().rtol, M::T::from(15.0));
        }
    }

    pub fn test_param_sweep<Method, Eqn, F>(
        f: F,
        mut problem: OdeSolverProblem<Eqn>,
        ps: Vec<Eqn::V>,
    ) where
        for <'a> Method: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicit,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        F: Fn(&OdeSolverProblem<Eqn>) -> Method,  

    {
        let mut old_soln = None;
        for p in ps {
            problem.set_params(p).unwrap();
            let s = f(&problem);
            let (ys, _ts) = s.solve(Eqn::T::from(10.0)).unwrap();
            // check that the new solution is different from the old one
            if let Some(old_soln) = &mut old_soln {
                let new_soln = ys.column(ys.ncols() - 1).into_owned();
                let diff = (new_soln - &*old_soln)
                    .squared_norm(old_soln, &problem.atol, problem.rtol)
                    .sqrt();
                assert!(diff > Eqn::T::from(1.0e-6), "diff: {}", diff);
            }
            old_soln = Some(ys.column(ys.ncols() - 1).into_owned());
        }
    }

    pub fn test_state_mut_on_problem<'a, Eqn, Method>(
        mut s: Method,
        soln: OdeSolverSolution<Eqn::V>,
    ) where
        Eqn: OdeEquationsImplicit + 'a,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        // save state and solve for a little bit
        let state = s.checkpoint();
        s.solve(Eqn::T::from(1.0)).unwrap();

        // reinit using state_mut
        let state = Method::State::new_without_initialise(s.problem()).unwrap();
        s.state_mut().y.copy_from(state.as_ref().y);
        *s.state_mut().t = state.as_ref().t;

        // solve and check against solution
        for point in soln.solution_points.iter() {
            while s.state().t < point.t {
                s.step().unwrap();
            }
            let soln = s.interpolate(point.t).unwrap();
            let error = soln.clone() - &point.state;
            let error_norm = error
                .squared_norm(&error, &s.problem().atol, s.problem().rtol)
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
