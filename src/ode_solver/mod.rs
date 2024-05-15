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

    use super::*;
    use crate::matrix::Matrix;
    use crate::op::filter::FilterCallable;
    use crate::op::unit::UnitCallable;
    use crate::op::{NonLinearOp, Op};
    use crate::scalar::scale;
    use crate::Vector;
    use crate::{
        NonLinearSolver, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
        OdeSolverStopReason,
    };
    use num_traits::One;
    use num_traits::Zero;

    pub fn test_ode_solver<M, Eqn>(
        method: &mut impl OdeSolverMethod<Eqn>,
        mut root_solver: impl NonLinearSolver<FilterCallable<Eqn::Rhs>>,
        problem: &OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
        use_tstop: bool,
    ) -> Eqn::V
    where
        M: Matrix,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let mut state = OdeSolverState::new(problem);
        state.set_consistent(problem, &mut root_solver).unwrap();
        state.set_step_size(problem, method.order());
        method.set_problem(state, problem);
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
                                break method.state().unwrap().y.clone()
                            }
                            _ => (),
                        }
                    },
                    Err(_) => method.state().unwrap().y.clone(),
                }
            } else {
                while method.state().unwrap().t < point.t {
                    if let OdeSolverStopReason::RootFound(t) = method.step().unwrap() {
                        assert!(have_root);
                        return method.interpolate(t).unwrap();
                    }
                }
                method.interpolate(point.t).unwrap()
            };

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
                    error_norm < M::T::from(15.0),
                    "error_norm: {} at t = {}",
                    error_norm,
                    point.t
                );
            }
        }
        method.state().unwrap().y.clone()
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
        type Root = UnitCallable<M>;

        fn set_params(&mut self, _p: Self::V) {}

        fn rhs(&self) -> &Rc<Self::Rhs> {
            &self.rhs
        }

        fn mass(&self) -> &Rc<Self::Mass> {
            &self.mass
        }

        fn root(&self) -> Option<&Rc<Self::Root>> {
            None
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
        );
        let state = OdeSolverState::new(&problem);
        s.set_problem(state.clone(), &problem);
        let state2 = s.state().unwrap();
        state2.y.assert_eq_st(&state.y, M::T::from(1e-9));
        s.state_mut().unwrap().y[0] = M::T::from(std::f64::consts::PI);
        assert_eq!(s.state().unwrap().y[0], M::T::from(std::f64::consts::PI));
    }

    pub fn test_state_mut_on_problem<Eqn: OdeEquations, Method: OdeSolverMethod<Eqn>>(
        mut s: Method,
        problem: OdeSolverProblem<Eqn>,
        soln: OdeSolverSolution<Eqn::V>,
    ) {
        // solve for a little bit
        s.solve(&problem, Eqn::T::from(1.0)).unwrap();

        // reinit using state_mut
        let state = OdeSolverState::new(&problem);
        s.state_mut().unwrap().y.copy_from(&state.y);
        s.state_mut().unwrap().t = state.t;

        // solve and check against solution
        for point in soln.solution_points.iter() {
            while s.state().unwrap().t < point.t {
                s.step().unwrap();
            }
            let soln = s.interpolate(point.t).unwrap();

            let scale = {
                let problem = s.problem().unwrap();
                point.state.abs() * scale(problem.rtol) + problem.atol.as_ref()
            };
            let mut error = soln.clone() - &point.state;
            error.component_div_assign(&scale);
            let error_norm = error.norm() / Eqn::T::from((point.state.len() as f64).sqrt());
            assert!(
                error_norm < Eqn::T::from(15.0),
                "error_norm: {} at t = {}",
                error_norm,
                point.t
            );
        }
    }
}
