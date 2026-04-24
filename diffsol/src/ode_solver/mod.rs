pub mod adjoint;
pub mod bdf;
pub mod bdf_state;
pub mod builder;
pub mod checkpointing;
pub mod config;
pub mod explicit_rk;
pub mod jacobian_update;
pub mod method;
pub mod problem;
pub mod runge_kutta;
pub mod sde;
pub mod sdirk;
pub mod sdirk_state;
pub mod sensitivities;
pub mod solution;
pub mod state;
pub mod tableau;

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use self::problem::OdeSolverSolution;

    use super::*;
    use crate::error::{DiffsolError, OdeSolverError};
    use crate::matrix::Matrix;
    use crate::ode_solver::sensitivities::SensitivitiesOdeSolverMethod;
    use crate::ode_solver::solution::Solution;
    use crate::op::unit::UnitCallable;
    use crate::op::ParameterisedOp;
    use crate::Scalar;
    use crate::{
        op::OpStatistics, AdjointEquations, AdjointOdeSolverMethod, Context, DenseMatrix,
        MatrixCommon, MatrixRef, NonLinearOp, NonLinearOpJacobian, OdeEquations,
        OdeEquationsImplicit, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens,
        OdeEquationsRef, OdeSolverConfig, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
        OdeSolverStopReason, Scale, VectorRef, VectorView, VectorViewMut,
    };
    use crate::{
        ConstantOp, ConstantOpSens, DefaultDenseMatrix, DefaultSolver, LinearSolver,
        NonLinearOpSens, Op, Vector,
    };
    use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

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
        Method: OdeSolverMethod<'a, Eqn>,
    {
        let have_root = method.problem().eqn.root().is_some();
        for (i, point) in solution.solution_points.iter().enumerate() {
            let (soln, sens_soln) = if use_tstop {
                match method.set_stop_time(point.t) {
                    Ok(_) => loop {
                        match method.step() {
                            Ok(OdeSolverStopReason::RootFound(_, _)) => {
                                assert!(have_root);
                                return method.state().y.clone();
                            }
                            Ok(OdeSolverStopReason::TstopReached) => {
                                break (method.state().y.clone(), method.state().s.to_vec());
                            }
                            _ => (),
                        }
                    },
                    Err(_) => (method.state().y.clone(), method.state().s.to_vec()),
                }
            } else {
                while method.state().t.abs() < point.t.abs() {
                    if let OdeSolverStopReason::RootFound(t, _) = method.step().unwrap() {
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
                    (method.problem().rtol, &method.problem().atol)
                };
                let error = soln.clone() - &point.state;
                let error_norm = error.squared_norm(&point.state, atol, rtol).sqrt();
                assert!(
                    error_norm < M::T::from_f64(20.0).unwrap(),
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
                                error_norm < M::T::from_f64(29.0).unwrap(),
                                "error_norm: {error_norm} at t = {}, sens index: {j}. soln: {sens_soln:?}, expected: {:?}",
                                point.t,
                                sens_point.state
                            );
                        }
                    }
                }
            }
        }
        method.state().y.clone()
    }

    pub fn setup_test_adjoint<'a, LS, Eqn>(
        problem: &'a mut OdeSolverProblem<Eqn>,
        soln: OdeSolverSolution<Eqn::V>,
    ) -> <Eqn::V as DefaultDenseMatrix>::M
    where
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        LS: LinearSolver<Eqn::M>,
        Eqn::V: DefaultDenseMatrix,
        for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
        for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        let nparams = problem.eqn.nparams();
        let nout = problem.eqn.nout();
        let ctx = problem.eqn.context();
        let mut dgdp = <Eqn::V as DefaultDenseMatrix>::M::zeros(nparams, nout, ctx.clone());
        let final_time = soln.solution_points.last().unwrap().t;
        let mut p_0 = Eqn::V::zeros(nparams, ctx.clone());
        problem.eqn.get_params(&mut p_0);
        let h_base = Eqn::T::from_f64(1e-10).unwrap();
        let mut h = Eqn::V::from_element(nparams, h_base, ctx.clone());
        h.axpy(h_base, &p_0, Eqn::T::one());
        let p_base = p_0.clone();
        for i in 0..nparams {
            p_0.set_index(i, p_base.get_index(i) + h.get_index(i));
            problem.eqn.set_params(&p_0);
            let g_pos = {
                let mut s = problem.bdf::<LS>().unwrap();
                s.set_stop_time(final_time).unwrap();
                while s.step().unwrap() != OdeSolverStopReason::TstopReached {}
                s.state().g.clone()
            };

            p_0.set_index(i, p_base.get_index(i) - h.get_index(i));
            problem.eqn.set_params(&p_0);
            let g_neg = {
                let mut s = problem.bdf::<LS>().unwrap();
                s.set_stop_time(final_time).unwrap();
                while s.step().unwrap() != OdeSolverStopReason::TstopReached {}
                s.state().g.clone()
            };
            p_0.set_index(i, p_base.get_index(i));

            let delta = (g_pos - g_neg) / Scale(Eqn::T::from_f64(2.).unwrap() * h.get_index(i));
            for j in 0..nout {
                dgdp.set_index(i, j, delta.get_index(j));
            }
        }
        problem.eqn.set_params(&p_base);
        dgdp
    }

    /// sum_i^n (soln_i - data_i)^2
    /// sum_i^n (soln_i - data_i)^4
    pub(crate) fn sum_squares<DM>(soln: &DM, data: &DM) -> DM::V
    where
        DM: DenseMatrix,
    {
        let mut ret = DM::V::zeros(2, soln.context().clone());
        for j in 0..soln.ncols() {
            let soln_j = soln.column(j);
            let data_j = data.column(j);
            let delta = soln_j - data_j;
            let norm2 = delta.norm(2);
            ret.set_index(0, ret.get_index(0) + norm2 * norm2);
            let norm4 = delta.norm(4);
            let norm4_sq = norm4 * norm4;
            ret.set_index(1, ret.get_index(1) + norm4_sq * norm4_sq);
        }
        ret
    }

    /// sum_i^n 2 * (soln_i - data_i)
    /// sum_i^n 4 * (soln_i - data_i)^3
    pub(crate) fn dsum_squaresdp<DM>(soln: &DM, data: &DM) -> Vec<DM>
    where
        DM: DenseMatrix,
    {
        let delta = soln.clone() - data;
        let mut delta3 = delta.clone();
        for j in 0..delta3.ncols() {
            let delta_col = delta.column(j).into_owned();

            let mut delta3_col = delta_col.clone();
            delta3_col.component_mul_assign(&delta_col);
            delta3_col.component_mul_assign(&delta_col);

            delta3.column_mut(j).copy_from(&delta3_col);
        }
        let ret = vec![
            delta * Scale(DM::T::from_f64(2.).unwrap()),
            delta3 * Scale(DM::T::from_f64(4.).unwrap()),
        ];
        ret
    }

    pub fn setup_test_adjoint_sum_squares<'a, LS, Eqn>(
        problem: &'a mut OdeSolverProblem<Eqn>,
        times: &[Eqn::T],
    ) -> (
        <Eqn::V as DefaultDenseMatrix>::M,
        <Eqn::V as DefaultDenseMatrix>::M,
    )
    where
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        LS: LinearSolver<Eqn::M>,
        Eqn::V: DefaultDenseMatrix,
        for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
        for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        let nparams = problem.eqn.nparams();
        let nout = 2;
        let ctx = problem.eqn.context();
        let mut dgdp = <Eqn::V as DefaultDenseMatrix>::M::zeros(nparams, nout, ctx.clone());

        let mut p_0 = ctx.vector_zeros(nparams);
        problem.eqn.get_params(&mut p_0);
        let h_base = Eqn::T::from_f64(1e-10).unwrap();
        let mut h = Eqn::V::from_element(nparams, h_base, ctx.clone());
        h.axpy(h_base, &p_0, Eqn::T::one());
        let mut p_data = p_0.clone();
        p_data.axpy(Eqn::T::from_f64(0.1).unwrap(), &p_0, Eqn::T::one());
        let p_base = p_0.clone();

        problem.eqn.set_params(&p_data);
        let data = {
            let mut s = problem.bdf::<LS>().unwrap();
            s.solve_dense(times).unwrap().0
        };

        for i in 0..nparams {
            p_0.set_index(i, p_base.get_index(i) + h.get_index(i));
            problem.eqn.set_params(&p_0);
            let g_pos = {
                let mut s = problem.bdf::<LS>().unwrap();
                let v = s.solve_dense(times).unwrap().0;
                sum_squares(&v, &data)
            };

            p_0.set_index(i, p_base.get_index(i) - h.get_index(i));
            problem.eqn.set_params(&p_0);
            let g_neg = {
                let mut s = problem.bdf::<LS>().unwrap();
                let v = s.solve_dense(times).unwrap().0;
                sum_squares(&v, &data)
            };

            p_0.set_index(i, p_base.get_index(i));

            let delta = (g_pos - g_neg) / Scale(Eqn::T::from_f64(2.).unwrap() * h.get_index(i));
            for j in 0..nout {
                dgdp.set_index(i, j, delta.get_index(j));
            }
        }
        problem.eqn.set_params(&p_base);
        (dgdp, data)
    }

    pub fn single_reset_root_discrete_times<T: Scalar>(t_stop: T) -> Vec<T> {
        let t_root = t_stop / T::from_f64(2.0).unwrap();
        [0.25, 0.75, 1.25, 1.75]
            .into_iter()
            .map(|factor| t_root * T::from_f64(factor).unwrap())
            .collect()
    }

    fn solve_dense_with_single_reset_root<'a, Eqn, Method, BuildForward>(
        build_forward: BuildForward,
        times: &[Eqn::T],
    ) -> <Eqn::V as DefaultDenseMatrix>::M
    where
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        Eqn::V: DefaultDenseMatrix,
        Method: OdeSolverMethod<'a, Eqn>,
        BuildForward: Fn(Option<Method::State>) -> Result<Method, DiffsolError>,
    {
        let mut soln = Solution::<Eqn::V>::new_dense(times.to_vec()).unwrap();
        let first_forward_solver = build_forward(None).unwrap().solve_soln(&mut soln).unwrap();
        match soln.stop_reason {
            Some(OdeSolverStopReason::RootFound(_, 0)) => {}
            Some(OdeSolverStopReason::RootFound(_, idx)) => {
                panic!("expected first solve_soln() segment to stop on root 0, got root {idx}")
            }
            Some(OdeSolverStopReason::TstopReached) => {
                panic!("expected first solve_soln() segment to stop on the interior root")
            }
            Some(OdeSolverStopReason::InternalTimestep) | None => {
                panic!("first solve_soln() segment did not finish with a terminal stop reason")
            }
        }

        let mut state_after_reset = first_forward_solver.state_clone();
        {
            let problem = first_forward_solver.problem();
            let rhs = problem.eqn.rhs();
            let has_mass = problem.eqn.mass().is_some();
            let reset_fn = problem.eqn.reset().unwrap();
            state_after_reset
                .as_mut()
                .state_mut_op(&rhs, has_mass, &reset_fn)
                .unwrap();
        }

        build_forward(Some(state_after_reset))
            .unwrap()
            .solve_soln(&mut soln)
            .unwrap();
        assert!(
            soln.is_complete(),
            "expected stitched solve_soln() output to cover all requested observation times",
        );
        soln.ys
    }

    pub fn setup_test_adjoint_sum_squares_with_single_reset_root<'a, LS, Eqn>(
        problem: &'a mut OdeSolverProblem<Eqn>,
        times: &[Eqn::T],
    ) -> (
        <Eqn::V as DefaultDenseMatrix>::M,
        <Eqn::V as DefaultDenseMatrix>::M,
    )
    where
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        LS: LinearSolver<Eqn::M>,
        Eqn::V: DefaultDenseMatrix,
        for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
        for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        let nparams = problem.eqn.nparams();
        let nout = 2;
        let ctx = problem.eqn.context();
        let mut dgdp = <Eqn::V as DefaultDenseMatrix>::M::zeros(nparams, nout, ctx.clone());

        let mut p_0 = ctx.vector_zeros(nparams);
        problem.eqn.get_params(&mut p_0);
        let h_base = Eqn::T::from_f64(1e-10).unwrap();
        let mut h = Eqn::V::from_element(nparams, h_base, ctx.clone());
        h.axpy(h_base, &p_0, Eqn::T::one());
        let mut p_data = p_0.clone();
        p_data.axpy(Eqn::T::from_f64(0.1).unwrap(), &p_0, Eqn::T::one());
        let p_base = p_0.clone();

        problem.eqn.set_params(&p_data);
        let data = solve_dense_with_single_reset_root::<Eqn, _, _>(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            times,
        );

        for i in 0..nparams {
            p_0.set_index(i, p_base.get_index(i) + h.get_index(i));
            problem.eqn.set_params(&p_0);
            let g_pos = {
                let v = solve_dense_with_single_reset_root::<Eqn, _, _>(
                    |state| match state {
                        Some(state) => problem.bdf_solver(state),
                        None => problem.bdf::<LS>(),
                    },
                    times,
                );
                sum_squares(&v, &data)
            };

            p_0.set_index(i, p_base.get_index(i) - h.get_index(i));
            problem.eqn.set_params(&p_0);
            let g_neg = {
                let v = solve_dense_with_single_reset_root::<Eqn, _, _>(
                    |state| match state {
                        Some(state) => problem.bdf_solver(state),
                        None => problem.bdf::<LS>(),
                    },
                    times,
                );
                sum_squares(&v, &data)
            };

            p_0.set_index(i, p_base.get_index(i));

            let delta = (g_pos - g_neg) / Scale(Eqn::T::from_f64(2.).unwrap() * h.get_index(i));
            for j in 0..nout {
                dgdp.set_index(i, j, delta.get_index(j));
            }
        }
        problem.eqn.set_params(&p_base);
        (dgdp, data)
    }

    pub fn test_adjoint_sum_squares<'a, Eqn, SolverF, SolverB>(
        backwards_solver: SolverB,
        dgdp_check: <Eqn::V as DefaultDenseMatrix>::M,
        forwards_soln: <Eqn::V as DefaultDenseMatrix>::M,
        data: <Eqn::V as DefaultDenseMatrix>::M,
        times: &[Eqn::T],
    ) where
        SolverF: OdeSolverMethod<'a, Eqn>,
        SolverB: AdjointOdeSolverMethod<'a, Eqn, SolverF>,
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        Eqn::V: DefaultDenseMatrix,
        Eqn::M: DefaultSolver,
    {
        let nparams = dgdp_check.nrows();
        let dgdu = dsum_squaresdp(&forwards_soln, &data);

        let atol = Eqn::V::from_element(
            nparams,
            Eqn::T::from_f64(1e-6).unwrap(),
            data.context().clone(),
        );
        let rtol = Eqn::T::from_f64(1e-6).unwrap();
        let state = backwards_solver
            .solve_adjoint_backwards_pass(None, times, dgdu.iter().collect::<Vec<_>>().as_slice())
            .unwrap();
        let gs_adj = state.into_common().sg;
        #[allow(clippy::needless_range_loop)]
        for j in 0..dgdp_check.ncols() {
            gs_adj[j].assert_eq_norm(
                &dgdp_check.column(j).into_owned(),
                &atol,
                rtol,
                Eqn::T::from_f64(260.).unwrap(),
            );
        }
    }

    pub fn test_adjoint<'a, Eqn, SolverF, SolverB>(
        backwards_solver: SolverB,
        dgdp_check: <Eqn::V as DefaultDenseMatrix>::M,
    ) where
        SolverF: OdeSolverMethod<'a, Eqn>,
        SolverB: AdjointOdeSolverMethod<'a, Eqn, SolverF>,
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        Eqn::V: DefaultDenseMatrix,
        Eqn::M: DefaultSolver,
    {
        let nout = backwards_solver.problem().eqn.nout();
        let atol = Eqn::V::from_element(
            nout,
            Eqn::T::from_f64(1e-6).unwrap(),
            dgdp_check.context().clone(),
        );
        let rtol = Eqn::T::from_f64(1e-6).unwrap();
        let state = backwards_solver
            .solve_adjoint_backwards_pass(None, &[], &[])
            .unwrap();
        let gs_adj = state.into_common().sg;
        #[allow(clippy::needless_range_loop)]
        for j in 0..dgdp_check.ncols() {
            gs_adj[j].assert_eq_norm(
                &dgdp_check.column(j).into_owned(),
                &atol,
                rtol,
                Eqn::T::from_f64(40.).unwrap(),
            );
        }
    }

    pub struct TestEqnInit<M: Matrix> {
        ctx: M::C,
    }

    impl<M: Matrix> Op for TestEqnInit<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;

        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            1
        }
        fn nstates(&self) -> usize {
            1
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }

    impl<M: Matrix> ConstantOp for TestEqnInit<M> {
        fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
            y.fill(M::T::one());
        }
    }

    impl<M: Matrix> ConstantOpSens for TestEqnInit<M> {
        fn sens_mul_inplace(&self, _t: Self::T, _v: &Self::V, sens: &mut Self::V) {
            sens.fill(M::T::zero());
        }
    }

    pub struct TestEqnRhs<M: Matrix> {
        ctx: M::C,
    }

    impl<M: Matrix> Op for TestEqnRhs<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;

        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            1
        }
        fn nstates(&self) -> usize {
            1
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }

    impl<M: Matrix> NonLinearOp for TestEqnRhs<M> {
        fn call_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::V) {
            y.fill(M::T::zero());
        }
    }

    impl<M: Matrix> NonLinearOpJacobian for TestEqnRhs<M> {
        fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
            y.fill(M::T::zero());
        }
    }

    impl<M: Matrix> NonLinearOpSens for TestEqnRhs<M> {
        fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, sens: &mut Self::V) {
            sens.fill(M::T::zero());
        }
    }

    pub struct TestEqnOut<M: Matrix> {
        ctx: M::C,
    }

    impl<M: Matrix> Op for TestEqnOut<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;

        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            1
        }
        fn nstates(&self) -> usize {
            1
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }

    impl<M: Matrix> NonLinearOp for TestEqnOut<M> {
        fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
            y.copy_from(x);
        }
    }

    impl<M: Matrix> NonLinearOpJacobian for TestEqnOut<M> {
        fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(v);
        }
    }

    impl<M: Matrix> NonLinearOpSens for TestEqnOut<M> {
        fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, sens: &mut Self::V) {
            sens.fill(M::T::zero());
        }
    }

    pub struct TestEqn<M: Matrix> {
        rhs: Rc<TestEqnRhs<M>>,
        init: Rc<TestEqnInit<M>>,
        out: Rc<TestEqnOut<M>>,
        ctx: M::C,
    }

    impl<M: Matrix> TestEqn<M> {
        pub fn new() -> Self {
            let ctx = M::C::default();
            Self {
                rhs: Rc::new(TestEqnRhs { ctx: ctx.clone() }),
                init: Rc::new(TestEqnInit { ctx: ctx.clone() }),
                out: Rc::new(TestEqnOut { ctx: ctx.clone() }),
                ctx,
            }
        }
    }

    impl<M: Matrix> Op for TestEqn<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;
        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            1
        }
        fn nstates(&self) -> usize {
            1
        }
        fn statistics(&self) -> crate::op::OpStatistics {
            OpStatistics::default()
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }

    impl<'a, M: Matrix> OdeEquationsRef<'a> for TestEqn<M> {
        type Rhs = &'a TestEqnRhs<M>;
        type Mass = ParameterisedOp<'a, UnitCallable<M>>;
        type Root = ParameterisedOp<'a, UnitCallable<M>>;
        type Init = &'a TestEqnInit<M>;
        type Out = &'a TestEqnOut<M>;
        type Reset = ParameterisedOp<'a, UnitCallable<M>>;
    }

    impl<M: Matrix> OdeEquations for TestEqn<M> {
        fn rhs(&self) -> &TestEqnRhs<M> {
            &self.rhs
        }

        fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
            None
        }

        fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
            None
        }

        fn init(&self) -> &TestEqnInit<M> {
            &self.init
        }

        fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
            Some(&self.out)
        }
        fn set_params(&mut self, _p: &Self::V) {
            unimplemented!()
        }
        fn get_params(&self, _p: &mut Self::V) {
            unimplemented!()
        }
    }

    pub fn test_problem<M: Matrix>(integrate_out: bool) -> OdeSolverProblem<TestEqn<M>> {
        let eqn = TestEqn::<M>::new();
        let atol = eqn
            .context()
            .vector_from_element(1, M::T::from_f64(1e-6).unwrap());
        OdeSolverProblem::new(
            eqn,
            M::T::from_f64(1e-6).unwrap(),
            atol,
            None,
            None,
            None,
            None,
            None,
            None,
            M::T::zero(),
            M::T::one(),
            integrate_out,
            Default::default(),
            Default::default(),
        )
        .unwrap()
    }

    pub fn test_interpolate<'a, M: Matrix, Method: OdeSolverMethod<'a, TestEqn<M>>>(mut s: Method) {
        let state = s.checkpoint();
        let integrating_sens = !s.state().s.is_empty();
        let integrating_out = s.problem().integrate_out;
        let t0 = state.as_ref().t;
        let t1 = t0 + M::T::from_f64(1e6).unwrap();
        s.interpolate(t0)
            .unwrap()
            .assert_eq_st(state.as_ref().y, M::T::from_f64(1e-9).unwrap());
        assert!(s.interpolate(t1).is_err());
        assert!(s.interpolate_out(t1).is_err());
        if integrating_sens {
            assert!(s.interpolate_sens(t1).is_err());
        } else {
            assert!(s.interpolate_sens(t0).is_ok());
        }
        s.step().unwrap();
        let tmid = t0 + (s.state().t - t0) / M::T::from_f64(2.0).unwrap();
        assert!(s.interpolate(s.state().t).is_ok());
        assert!(s.interpolate(tmid).is_ok());
        if integrating_out {
            assert!(s.interpolate_out(s.state().t).is_ok());
        } else {
            assert!(s.interpolate_out(s.state().t).is_err());
        }
        assert!(s.interpolate_sens(s.state().t).is_ok());
        assert!(s.interpolate(s.state().t + t1).is_err());
        assert!(s.interpolate_out(s.state().t + t1).is_err());
        if integrating_sens {
            assert!(s.interpolate_sens(s.state().t + t1).is_err());
        } else {
            assert!(s.interpolate_sens(s.state().t + t1).is_ok());
        }

        let mut y_wrong_length = M::V::zeros(2, s.problem().context().clone());
        assert!(s
            .interpolate_inplace(s.state().t, &mut y_wrong_length)
            .is_err());
        let mut g_wrong_length = M::V::zeros(2, s.problem().context().clone());
        assert!(s
            .interpolate_out_inplace(s.state().t, &mut g_wrong_length)
            .is_err());
        let mut s_wrong_length = vec![
            M::V::zeros(1, s.problem().context().clone()),
            M::V::zeros(1, s.problem().context().clone()),
        ];
        assert!(s
            .interpolate_sens_inplace(s.state().t, &mut s_wrong_length)
            .is_err());
        let mut s_wrong_vec_length = if integrating_sens {
            vec![M::V::zeros(2, s.problem().context().clone())]
        } else {
            vec![]
        };
        if integrating_sens {
            assert!(s
                .interpolate_sens_inplace(s.state().t, &mut s_wrong_vec_length)
                .is_err());
        } else {
            assert!(s
                .interpolate_sens_inplace(s.state().t, &mut s_wrong_vec_length)
                .is_ok());
        }

        s.state_mut().y.fill(M::T::from_f64(3.0).unwrap());
        assert!(s.interpolate(s.state().t).is_ok());
        if integrating_out {
            assert!(s.interpolate_out(s.state().t).is_ok());
        }
        if integrating_sens {
            assert!(s.interpolate_sens(s.state().t).is_ok());
        }
        assert!(s.interpolate(tmid).is_err());
        assert!(s.interpolate_out(tmid).is_err());
        if integrating_sens {
            assert!(s.interpolate_sens(tmid).is_err());
        } else {
            assert!(s.interpolate_sens(tmid).is_ok());
        }
    }

    pub fn test_interpolate_dy<'a, M: Matrix, Method: OdeSolverMethod<'a, TestEqn<M>>>(
        mut s: Method,
    ) {
        // Error before first step: t is in the future
        let t_future = s.state().t + M::T::from_f64(1e6).unwrap();
        assert!(s.interpolate_dy(t_future).is_err());

        let t0 = s.state().t;
        s.step().unwrap();
        let t1 = s.state().t;
        let dt = t1 - t0;
        let tmid = t0 + dt / M::T::from_f64(2.0).unwrap();

        // Wrong vector length should return error
        let mut dy_wrong = M::V::zeros(2, s.problem().context().clone());
        assert!(s.interpolate_dy_inplace(t1, &mut dy_wrong).is_err());

        // t after current time should return error
        assert!(s.interpolate_dy(t1 + M::T::from_f64(1e6).unwrap()).is_err());

        // interpolate_dy should be consistent with finite-difference of interpolate (step 1)
        let eps = dt.abs() * M::T::from_f64(1e-5).unwrap();
        let y_plus = s.interpolate(tmid + eps).unwrap();
        let y_minus = s.interpolate(tmid - eps).unwrap();
        let fd_dy = (y_plus - y_minus) * Scale(M::T::one() / (M::T::from_f64(2.0).unwrap() * eps));
        let dy = s.interpolate_dy(tmid).unwrap();
        dy.assert_eq_norm(
            &fd_dy,
            &s.problem().atol,
            s.problem().rtol,
            M::T::from_f64(1e3).unwrap(),
        );

        // take a second step and check consistency again
        let t1 = s.state().t;
        s.step().unwrap();
        let t2 = s.state().t;
        let dt2 = t2 - t1;
        let tmid2 = t1 + dt2 / M::T::from_f64(2.0).unwrap();
        let eps2 = dt2.abs() * M::T::from_f64(1e-5).unwrap();
        let y_plus = s.interpolate(tmid2 + eps2).unwrap();
        let y_minus = s.interpolate(tmid2 - eps2).unwrap();
        let fd_dy2 =
            (y_plus - y_minus) * Scale(M::T::one() / (M::T::from_f64(2.0).unwrap() * eps2));
        let dy2 = s.interpolate_dy(tmid2).unwrap();
        dy2.assert_eq_norm(
            &fd_dy2,
            &s.problem().atol,
            s.problem().rtol,
            M::T::from_f64(1e3).unwrap(),
        );
    }

    pub fn test_config<'a, Eqn: OdeEquations + 'a, Method: OdeSolverMethod<'a, Eqn>>(
        mut s: Method,
    ) {
        *s.config_mut().as_base_mut().minimum_timestep = Eqn::T::from_f64(1.0e8).unwrap();
        assert_eq!(
            *s.config().as_base_ref().minimum_timestep,
            Eqn::T::from_f64(1.0e8).unwrap()
        );
        // force a step size reduction
        *s.state_mut().h = Eqn::T::from_f64(0.1).unwrap();

        let mut failed = false;
        for _ in 0..10 {
            if let Err(DiffsolError::OdeSolverError(OdeSolverError::StepSizeTooSmall { time: _ })) =
                s.step()
            {
                failed = true;
                break;
            }
        }
        assert!(failed);
    }

    pub fn test_state_mut<'a, M: Matrix, Method: OdeSolverMethod<'a, TestEqn<M>>>(mut s: Method) {
        let state = s.checkpoint();
        let state2 = s.state();
        state2
            .y
            .assert_eq_st(state.as_ref().y, M::T::from_f64(1e-9).unwrap());
        s.state_mut()
            .y
            .set_index(0, M::T::from_f64(std::f64::consts::PI).unwrap());
        assert_eq!(
            s.state_mut().y.get_index(0),
            M::T::from_f64(std::f64::consts::PI).unwrap()
        );
    }

    #[cfg(feature = "diffsl-cranelift")]
    pub fn test_ball_bounce_problem<M: crate::MatrixHost<T = f64>>(
    ) -> OdeSolverProblem<crate::DiffSl<M, crate::CraneliftJitModule>> {
        crate::OdeBuilder::<M>::new()
            .build_from_diffsl(
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
            .unwrap()
    }

    #[cfg(feature = "diffsl-cranelift")]
    pub fn test_ball_bounce<'a, M, Method>(mut solver: Method) -> (Vec<f64>, Vec<f64>, Vec<f64>)
    where
        M: crate::MatrixHost<T = f64>,
        M: DefaultSolver<T = f64>,
        M::V: DefaultDenseMatrix<T = f64>,
        Method: OdeSolverMethod<'a, crate::DiffSl<M, crate::CraneliftJitModule>>,
    {
        let e = 0.8;

        let final_time = 2.5;

        // solve and apply the remaining doses
        solver.set_stop_time(final_time).unwrap();
        loop {
            match solver.step() {
                Ok(OdeSolverStopReason::InternalTimestep) => (),
                Ok(OdeSolverStopReason::RootFound(t, _)) => {
                    // get the state when the event occurred
                    let mut y = solver.interpolate(t).unwrap();

                    // update the velocity of the ball
                    y.set_index(1, y.get_index(1) * -e);

                    // make sure the ball is above the ground
                    y.set_index(0, y.get_index(0).max(f64::EPSILON));

                    // set the state to the updated state
                    solver.state_mut().y.copy_from(&y);
                    solver.state_mut().dy.set_index(0, y.get_index(1));
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
            x.push(solver.state().y.get_index(0));
            v.push(solver.state().y.get_index(1));
            t.push(solver.state().t);
            match ret {
                Ok(OdeSolverStopReason::InternalTimestep) => (),
                Ok(OdeSolverStopReason::RootFound(_, _)) => {
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
        mut solver1: Method,
        mut solver2: Method,
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
        let checkpoint_t = checkpoint.as_ref().t;
        solver2.set_state(checkpoint);

        // carry on solving with both solvers, they should produce about the same results (probably might diverge a bit, but should always match the solution)
        for point in soln.solution_points.iter().skip(half_i + 1) {
            // point should be past checkpoint
            if point.t < checkpoint_t {
                continue;
            }
            while solver2.state().t < point.t {
                solver1.step().unwrap();
                solver2.step().unwrap();
                let time_error = (solver1.state().t - solver2.state().t).abs()
                    / (solver1.state().t.abs() * solver1.problem().rtol
                        + solver1.problem().atol.get_index(0));
                assert!(
                    time_error < M::T::from_f64(20.0).unwrap(),
                    "time_error: {} at t = {}",
                    time_error,
                    solver1.state().t
                );
                solver1.state().y.assert_eq_norm(
                    solver2.state().y,
                    &solver1.problem().atol,
                    solver1.problem().rtol,
                    M::T::from_f64(20.0).unwrap(),
                );
            }
            let soln = solver1.interpolate(point.t).unwrap();
            soln.assert_eq_norm(
                &point.state,
                &solver1.problem().atol,
                solver1.problem().rtol,
                M::T::from_f64(15.0).unwrap(),
            );
            let soln = solver2.interpolate(point.t).unwrap();
            soln.assert_eq_norm(
                &point.state,
                &solver1.problem().atol,
                solver1.problem().rtol,
                M::T::from_f64(15.0).unwrap(),
            );
        }
    }

    pub fn test_state_mut_on_problem<'a, Eqn, Method>(
        mut s: Method,
        soln: OdeSolverSolution<Eqn::V>,
    ) where
        Eqn: OdeEquationsImplicit + 'a,
        Method: OdeSolverMethod<'a, Eqn>,
        Eqn::V: DefaultDenseMatrix,
    {
        // save state and solve for a little bit
        let state = s.checkpoint();
        s.solve(Eqn::T::one()).unwrap();

        // reinit using state_mut
        s.state_mut().y.copy_from(state.as_ref().y);
        s.state_mut().dy.copy_from(state.as_ref().dy);
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
                error_norm < Eqn::T::from_f64(19.0).unwrap(),
                "error_norm: {} at t = {}",
                error_norm,
                point.t
            );
        }
    }

    /// Test that `step()` returns `RootFound(t, index)` with the correct root index.
    ///
    /// The problem must have a root function with **two** outputs and **no** Reset:
    ///   - Root 0 fires first (at `t ≈ 5.108`, `y[0] ≈ 0.6` for the exponential-decay test model)
    ///   - Root 1 fires second
    ///
    /// The test asserts that the first `RootFound` reports index 0 and the time
    /// matches `t_root_0_expected` within `tol`.
    pub fn test_root_found_index<'a, Eqn, Method>(
        mut solver: Method,
        soln: &OdeSolverSolution<Eqn::V>,
        expected_root_index: usize,
        tol: Eqn::T,
    ) where
        Eqn: OdeEquations + 'a,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        let t_root_expected = soln.solution_points[0].t;
        solver
            .set_stop_time(Eqn::T::from_f64(100.0).unwrap())
            .unwrap();
        loop {
            match solver.step().unwrap() {
                // RED: `RootFound` currently has one field; adding `index` makes this fail.
                OdeSolverStopReason::RootFound(t, index) => {
                    assert_eq!(
                        index, expected_root_index,
                        "expected root index {expected_root_index} but got {index}",
                    );
                    assert!(
                        (t - t_root_expected).abs() < tol,
                        "expected t ≈ {t_root_expected:?}, got {t:?}",
                    );
                    break;
                }
                OdeSolverStopReason::TstopReached => {
                    panic!("reached tstop without finding a root")
                }
                OdeSolverStopReason::InternalTimestep => {}
            }
        }
    }

    /// Test that `solve()` automatically applies resets at roots and continues
    /// integrating until `final_time`.
    pub fn test_solve_with_reset<'a, Eqn, Method>(
        mut solver: Method,
        soln: &OdeSolverSolution<Eqn::V>,
    ) where
        Eqn: OdeEquations + 'a,
        Eqn::V: DefaultDenseMatrix,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        let final_time = Eqn::T::from_f64(100.0).unwrap();
        let (ys, ts, stop_reason) = solver.solve(final_time).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        let t_last = *ts.last().unwrap();
        let time_tol = soln.rtol * final_time.abs() + soln.atol.get_index(0);
        assert!(
            (t_last - final_time).abs() < Eqn::T::from_f64(30.0).unwrap() * time_tol,
            "expected solve() to reach final_time ≈ {:?}, got {:?}",
            final_time,
            t_last,
        );
        assert!(
            (solver.state().t - final_time).abs() < Eqn::T::from_f64(30.0).unwrap() * time_tol,
            "expected solver state at final_time ≈ {:?}, got {:?}",
            final_time,
            solver.state().t,
        );

        let expected = &soln.solution_points[0];
        let root_time_tol = soln.rtol * expected.t.abs() + soln.atol.get_index(0);
        let root_col = ts
            .iter()
            .position(|&t| (t - expected.t).abs() < Eqn::T::from_f64(30.0).unwrap() * root_time_tol)
            .expect("expected solve() output to include the second-root/reset time");
        let root_expected = Eqn::V::from_element(
            expected.state.len(),
            Eqn::T::from_f64(0.4).unwrap(),
            expected.state.context().clone(),
        );
        let root_state = ys.column(root_col).into_owned();
        let root_error = root_state - &root_expected;
        let root_error_norm = root_error
            .squared_norm(&root_expected, &soln.atol, soln.rtol)
            .sqrt();
        let error_threshold = Eqn::T::from_f64(20.0).unwrap();
        assert!(
            root_error_norm < error_threshold,
            "expected reset state y=0.4 at second-root time; WRMS error norm {root_error_norm:?} ≥ {error_threshold:?}",
        );

        let reset_value = Eqn::T::from_f64(0.4).unwrap();
        let reset_tol = Eqn::T::from_f64(30.0).unwrap()
            * (soln.rtol * reset_value.abs() + soln.atol.get_index(0));
        let last_reset_col = (0..ts.len())
            .rev()
            .find(|&i| (ys.get_index(0, i) - reset_value).abs() < reset_tol)
            .expect("expected solve() output to include at least one reset state");
        let final_time_f64 = final_time.to_f64().unwrap();
        let last_reset_time_f64 = ts[last_reset_col].to_f64().unwrap();
        let expected_final_value =
            Eqn::T::from_f64(0.4 * (-0.1 * (final_time_f64 - last_reset_time_f64)).exp()).unwrap();
        let expected_final = Eqn::V::from_element(
            expected.state.len(),
            expected_final_value,
            expected.state.context().clone(),
        );
        let final_state = ys.column(ts.len() - 1).into_owned();
        let final_error = final_state - &expected_final;
        let final_error_norm = final_error
            .squared_norm(&expected_final, &soln.atol, soln.rtol)
            .sqrt();
        assert!(
            final_error_norm < error_threshold,
            "final state mismatch after automatic reset continuation: WRMS error norm {final_error_norm:?} ≥ {error_threshold:?}",
        );
    }

    /// Test that `solve_dense()` automatically applies resets at roots and
    /// continues filling the requested evaluation times.
    pub fn test_solve_dense_with_reset<'a, Eqn, Method>(
        mut solver: Method,
        soln: &OdeSolverSolution<Eqn::V>,
    ) where
        Eqn: OdeEquations + 'a,
        Eqn::V: DefaultDenseMatrix,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        let t_stop = soln.solution_points[0].t;
        let final_time = t_stop * Eqn::T::from_f64(2.0).unwrap();
        let mut probe_solver = solver.clone();
        let (probe_ys, probe_ts, probe_stop_reason) = probe_solver.solve(final_time).unwrap();
        assert_eq!(probe_stop_reason, OdeSolverStopReason::TstopReached);

        let reset_time_tol =
            Eqn::T::from_f64(30.0).unwrap() * (soln.rtol * t_stop.abs() + soln.atol.get_index(0));
        let post_event_dt = Eqn::T::from_f64(1e-6).unwrap();
        let reset_value = Eqn::T::from_f64(0.4).unwrap();
        let reset_value_tol = Eqn::T::from_f64(30.0).unwrap()
            * (soln.rtol * reset_value.abs() + soln.atol.get_index(0));
        let reset_col = (0..probe_ts.len())
            .find(|&i| {
                (probe_ts[i] - t_stop).abs() < reset_time_tol
                    && (probe_ys.get_index(0, i) - reset_value).abs() < reset_value_tol
            })
            .expect("expected solve() probe output to contain the second-root reset state");
        let t_event = probe_ts[reset_col];
        let t_eval = vec![Eqn::T::zero(), t_event, t_event + post_event_dt, final_time];

        let (ret, stop_reason) = solver.solve_dense(&t_eval).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        assert!(
            ret.ncols() == t_eval.len(),
            "expected solve_dense() to fill all requested evaluation times"
        );
        let time_tol = soln.rtol * final_time.abs() + soln.atol.get_index(0);
        assert!(
            (solver.state().t - final_time).abs() < Eqn::T::from_f64(30.0).unwrap() * time_tol,
            "expected solver state at final_time ≈ {:?}, got {:?}",
            final_time,
            solver.state().t,
        );

        let error_threshold = Eqn::T::from_f64(20.0).unwrap();
        let pre_reset_state = ret.column(1).into_owned();
        let pre_reset_error = pre_reset_state - &soln.solution_points[0].state;
        let pre_reset_error_norm = pre_reset_error
            .squared_norm(&soln.solution_points[0].state, &soln.atol, soln.rtol)
            .sqrt();
        assert!(
            pre_reset_error_norm < error_threshold,
            "expected pre-reset state at event time; WRMS norm {pre_reset_error_norm:?} >= {error_threshold:?}",
        );

        let expected_post_reset_value =
            reset_value * (-Eqn::T::from_f64(0.1).unwrap() * post_event_dt).exp();
        let expected_post_reset = Eqn::V::from_element(
            soln.solution_points[0].state.len(),
            expected_post_reset_value,
            soln.solution_points[0].state.context().clone(),
        );
        let post_reset_state = ret.column(2).into_owned();
        let post_reset_error = post_reset_state - &expected_post_reset;
        let post_reset_error_norm = post_reset_error
            .squared_norm(&expected_post_reset, &soln.atol, soln.rtol)
            .sqrt();
        assert!(
            post_reset_error_norm < error_threshold,
            "expected reset state just after event time; WRMS norm {post_reset_error_norm:?} >= {error_threshold:?}",
        );
    }

    /// Test that `solve_dense_sensitivities()` applies root-aware resets and
    /// continues filling the requested evaluation times.
    pub fn test_solve_dense_sensitivities_with_reset<'a, Eqn, Method>(
        mut solver: Method,
        soln: &OdeSolverSolution<Eqn::V>,
    ) where
        Eqn: OdeEquationsImplicitSens + 'a,
        Eqn::V: DefaultDenseMatrix,
        Method: SensitivitiesOdeSolverMethod<'a, Eqn>,
    {
        let t_stop = soln.solution_points[0].t;
        let t_event = Eqn::T::from_f64(10.0 * (5.0_f64 / 3.0_f64).ln()).unwrap();

        let post_event_dt = Eqn::T::from_f64(1e-6).unwrap();
        let t_eval = vec![Eqn::T::zero(), t_event, t_event + post_event_dt, t_stop];
        let (ret, ret_sens, stop_reason) = solver.solve_dense_sensitivities(&t_eval).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        assert_eq!(ret.ncols(), t_eval.len());
        for ret_sens_j in &ret_sens {
            assert_eq!(ret_sens_j.ncols(), t_eval.len());
        }

        let error_threshold = Eqn::T::from_f64(100.0).unwrap();
        let ctx = soln.solution_points[0].state.context().clone();
        let nstates = soln.solution_points[0].state.len();

        let post_reset_y = Eqn::T::from_f64(2.6).unwrap()
            * (-Eqn::T::from_f64(0.1).unwrap() * post_event_dt).exp();
        let post_reset_t = t_event + post_event_dt;
        let expected_post_reset = Eqn::V::from_element(nstates, post_reset_y, ctx.clone());
        let expected_post_reset_sk =
            Eqn::V::from_element(nstates, -post_reset_y * post_reset_t, ctx.clone());
        let expected_post_reset_sy0 = Eqn::V::from_element(nstates, post_reset_y, ctx);

        let col = 2;
        let ey = ret.column(col).into_owned() - &expected_post_reset;
        let esk = ret_sens[0].column(col).into_owned() - &expected_post_reset_sk;
        let esy0 = ret_sens[1].column(col).into_owned() - &expected_post_reset_sy0;
        let norm = (ey.squared_norm(&expected_post_reset, &soln.atol, soln.rtol)
            + esk.squared_norm(&expected_post_reset_sk, &soln.atol, soln.rtol)
            + esy0.squared_norm(&expected_post_reset_sy0, &soln.atol, soln.rtol))
        .sqrt();
        assert!(
            norm < error_threshold,
            "dense sensitivity mismatch just after reset; combined WRMS {norm:?} >= {error_threshold:?}",
        );
    }

    pub fn test_solve_adjoint_with_single_reset_root<
        'a,
        Eqn,
        MethodF,
        MethodB,
        BuildForward,
        BuildAdjointState,
        BuildAdjointFromState,
    >(
        build_forward: BuildForward,
        soln: &OdeSolverSolution<Eqn::V>,
        build_adjoint_state: BuildAdjointState,
        build_adjoint_from_state: BuildAdjointFromState,
    ) where
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        MethodF: OdeSolverMethod<'a, Eqn>,
        MethodB: AdjointOdeSolverMethod<'a, Eqn, MethodF, State = MethodF::State>,
        BuildForward: Fn(Option<MethodF::State>) -> Result<MethodF, DiffsolError>,
        BuildAdjointState:
            Fn(&mut AdjointEquations<'a, Eqn, MethodF>) -> Result<MethodF::State, DiffsolError>,
        BuildAdjointFromState:
            Fn(MethodF::State, AdjointEquations<'a, Eqn, MethodF>) -> Result<MethodB, DiffsolError>,
    {
        let expected_out = &soln.solution_points[0];
        let forward_stop_time = expected_out.t + Eqn::T::from_f64(1.0).unwrap();

        let mut first_forward_solver = build_forward(None).unwrap();
        let (pre_reset_checkpointer, _, _, _) = first_forward_solver
            .solve_with_checkpointing(forward_stop_time, None)
            .unwrap();
        let fwd_state_minus = first_forward_solver.into_state();
        let mut state_after_reset = fwd_state_minus.clone();
        let problem = pre_reset_checkpointer.problem();
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let reset_fn = problem.eqn.reset().unwrap();
        state_after_reset
            .as_mut()
            .state_mut_op(&rhs, has_mass, &reset_fn)
            .unwrap();
        let fwd_state_plus = state_after_reset.clone();

        let mut second_forward_solver = build_forward(Some(state_after_reset)).unwrap();
        let (post_reset_checkpointer, _, _, post_reset_stop_reason) = second_forward_solver
            .solve_with_checkpointing(forward_stop_time, None)
            .unwrap();
        let final_forward_state = second_forward_solver.into_state();
        let t_second_root = final_forward_state.as_ref().t;

        let out_error = final_forward_state.as_ref().g.clone() - &expected_out.state;
        let out_norm = out_error
            .squared_norm(&expected_out.state, &soln.atol, soln.rtol)
            .sqrt();
        assert!(
            out_norm < Eqn::T::from_f64(50.0).unwrap(),
            "forward integrated output mismatch at second root: actual {:?}, expected {:?}, WRMS {out_norm:?}",
            final_forward_state.as_ref().g,
            expected_out.state,
        );
        let time_tol = soln.rtol * expected_out.t.abs() + soln.atol.get_index(0);
        assert!(
            (t_second_root - expected_out.t).abs() < Eqn::T::from_f64(30.0).unwrap() * time_tol,
            "expected second root time ≈ {:?}, got {:?}",
            expected_out.t,
            t_second_root,
        );

        let mut post_reset_adjoint_eqn =
            problem.adjoint_equations(post_reset_checkpointer.clone(), None);
        let mut post_reset_adjoint_state =
            build_adjoint_state(&mut post_reset_adjoint_eqn).unwrap();
        let post_reset_root_idx = match post_reset_stop_reason {
            OdeSolverStopReason::RootFound(_, idx) => idx,
            OdeSolverStopReason::TstopReached => {
                panic!("expected second forward segment to stop on a root, got TstopReached")
            }
            OdeSolverStopReason::InternalTimestep => {
                panic!("expected second forward segment to stop on a root, got InternalTimestep")
            }
        };
        post_reset_adjoint_state
            .as_mut()
            .state_mut_adjoint_terminal_root(
                &mut post_reset_adjoint_eqn,
                post_reset_root_idx,
                &final_forward_state,
            )
            .unwrap();
        let post_reset_adjoint =
            build_adjoint_from_state(post_reset_adjoint_state, post_reset_adjoint_eqn).unwrap();
        let mut adjoint_state = post_reset_adjoint
            .solve_adjoint_backwards_pass(Some(fwd_state_minus.as_ref().t), &[], &[])
            .unwrap();
        let t0 = pre_reset_checkpointer.problem().t0;
        let ctx = pre_reset_checkpointer.problem().context().clone();
        let reset_problem = pre_reset_checkpointer.problem();
        let mut pre_reset_adjoint_eqn = problem.adjoint_equations(pre_reset_checkpointer, None);
        {
            let reset_fn = reset_problem.eqn.reset().unwrap();
            let root_fn = reset_problem.eqn.root().unwrap();
            adjoint_state
                .as_mut()
                .state_mut_op_with_adjoint_and_reset(
                    &mut pre_reset_adjoint_eqn,
                    &reset_fn,
                    &root_fn,
                    0,
                    &fwd_state_minus,
                    &fwd_state_plus,
                )
                .unwrap();
        }
        let pre_reset_adjoint =
            build_adjoint_from_state(adjoint_state, pre_reset_adjoint_eqn).unwrap();
        let adjoint_state = pre_reset_adjoint
            .solve_adjoint_backwards_pass(None, &[], &[])
            .unwrap();

        let sens_points = soln.sens_solution_points.as_ref().unwrap();
        let expected_grad = Eqn::V::from_vec(
            sens_points
                .iter()
                .map(|pts| pts[0].state.get_index(0))
                .collect(),
            ctx.clone(),
        );
        let atol = Eqn::V::from_element(expected_grad.len(), Eqn::T::from_f64(1e-6).unwrap(), ctx);
        let t0_tol = Eqn::T::from_f64(10.0).unwrap() * Eqn::T::EPSILON;
        assert!(
            (adjoint_state.as_ref().t - t0).abs() <= t0_tol,
            "expected adjoint final time {:?}, got {:?}",
            t0,
            adjoint_state.as_ref().t,
        );
        adjoint_state.as_ref().sg[0].assert_eq_norm(
            &expected_grad,
            &atol,
            Eqn::T::from_f64(1e-6).unwrap(),
            Eqn::T::from_f64(60.0).unwrap(),
        );
    }

    pub fn test_solve_adjoint_sum_squares_with_single_reset_root<
        'a,
        Eqn,
        MethodF,
        MethodB,
        BuildForward,
        BuildAdjointState,
        BuildAdjointFromState,
    >(
        build_forward: BuildForward,
        soln: &OdeSolverSolution<Eqn::V>,
        build_adjoint_state: BuildAdjointState,
        build_adjoint_from_state: BuildAdjointFromState,
        dgdp_check: <Eqn::V as DefaultDenseMatrix>::M,
        data: <Eqn::V as DefaultDenseMatrix>::M,
        times: &[Eqn::T],
    ) where
        Eqn: OdeEquationsImplicitAdjoint + 'a,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        MethodF: OdeSolverMethod<'a, Eqn>,
        MethodB: AdjointOdeSolverMethod<'a, Eqn, MethodF, State = MethodF::State>,
        BuildForward: Fn(Option<MethodF::State>) -> Result<MethodF, DiffsolError>,
        BuildAdjointState:
            Fn(&mut AdjointEquations<'a, Eqn, MethodF>) -> Result<MethodF::State, DiffsolError>,
        BuildAdjointFromState:
            Fn(MethodF::State, AdjointEquations<'a, Eqn, MethodF>) -> Result<MethodB, DiffsolError>,
    {
        let expected_out = &soln.solution_points[0];
        let forward_stop_time = expected_out.t + Eqn::T::from_f64(1.0).unwrap();
        let forwards_soln =
            solve_dense_with_single_reset_root::<Eqn, MethodF, _>(&build_forward, times);
        assert_eq!(
            forwards_soln.ncols(),
            times.len(),
            "expected stitched forward samples to cover every requested observation time",
        );
        let dgdu = dsum_squaresdp(&forwards_soln, &data);
        let dgdu_refs = dgdu.iter().collect::<Vec<_>>();

        let mut first_forward_solver = build_forward(None).unwrap();
        let (pre_reset_checkpointer, _, _, pre_reset_stop_reason) = first_forward_solver
            .solve_with_checkpointing(forward_stop_time, None)
            .unwrap();
        let fwd_state_minus = first_forward_solver.into_state();
        match pre_reset_stop_reason {
            OdeSolverStopReason::RootFound(_, 0) => {}
            OdeSolverStopReason::RootFound(_, idx) => {
                panic!("expected first checkpointed segment to stop on root 0, got root {idx}")
            }
            OdeSolverStopReason::TstopReached => {
                panic!("expected first checkpointed segment to stop on the interior root")
            }
            OdeSolverStopReason::InternalTimestep => {
                panic!("first checkpointed segment ended without a terminal stop reason")
            }
        }

        let mut state_after_reset = fwd_state_minus.clone();
        let problem = pre_reset_checkpointer.problem();
        let rhs = problem.eqn.rhs();
        let has_mass = problem.eqn.mass().is_some();
        let reset_fn = problem.eqn.reset().unwrap();
        state_after_reset
            .as_mut()
            .state_mut_op(&rhs, has_mass, &reset_fn)
            .unwrap();
        let fwd_state_plus = state_after_reset.clone();

        let mut second_forward_solver = build_forward(Some(state_after_reset)).unwrap();
        let (post_reset_checkpointer, _, _, post_reset_stop_reason) = second_forward_solver
            .solve_with_checkpointing(forward_stop_time, None)
            .unwrap();
        let final_forward_state = second_forward_solver.into_state();
        let t_second_root = final_forward_state.as_ref().t;

        let time_tol = soln.rtol * expected_out.t.abs() + soln.atol.get_index(0);
        assert!(
            (t_second_root - expected_out.t).abs() < Eqn::T::from_f64(30.0).unwrap() * time_tol,
            "expected second root time ≈ {:?}, got {:?}",
            expected_out.t,
            t_second_root,
        );

        let mut post_reset_adjoint_eqn =
            problem.adjoint_equations(post_reset_checkpointer.clone(), Some(dgdu.len()));
        let mut post_reset_adjoint_state =
            build_adjoint_state(&mut post_reset_adjoint_eqn).unwrap();
        let post_reset_root_idx = match post_reset_stop_reason {
            OdeSolverStopReason::RootFound(_, idx) => idx,
            OdeSolverStopReason::TstopReached => {
                panic!("expected second forward segment to stop on a root, got TstopReached")
            }
            OdeSolverStopReason::InternalTimestep => {
                panic!("expected second forward segment to stop on a root, got InternalTimestep")
            }
        };
        post_reset_adjoint_state
            .as_mut()
            .state_mut_adjoint_terminal_root(
                &mut post_reset_adjoint_eqn,
                post_reset_root_idx,
                &final_forward_state,
            )
            .unwrap();
        let post_reset_adjoint =
            build_adjoint_from_state(post_reset_adjoint_state, post_reset_adjoint_eqn).unwrap();
        let mut adjoint_state = post_reset_adjoint
            .solve_adjoint_backwards_pass(
                Some(fwd_state_minus.as_ref().t),
                times,
                dgdu_refs.as_slice(),
            )
            .unwrap();

        let t0 = pre_reset_checkpointer.problem().t0;
        let ctx = pre_reset_checkpointer.problem().context().clone();
        let reset_problem = pre_reset_checkpointer.problem();
        let mut pre_reset_adjoint_eqn =
            problem.adjoint_equations(pre_reset_checkpointer, Some(dgdu.len()));
        {
            let reset_fn = reset_problem.eqn.reset().unwrap();
            let root_fn = reset_problem.eqn.root().unwrap();
            adjoint_state
                .as_mut()
                .state_mut_op_with_adjoint_and_reset(
                    &mut pre_reset_adjoint_eqn,
                    &reset_fn,
                    &root_fn,
                    0,
                    &fwd_state_minus,
                    &fwd_state_plus,
                )
                .unwrap();
        }
        let pre_reset_adjoint =
            build_adjoint_from_state(adjoint_state, pre_reset_adjoint_eqn).unwrap();
        let adjoint_state = pre_reset_adjoint
            .solve_adjoint_backwards_pass(None, times, dgdu_refs.as_slice())
            .unwrap();

        let nparams = dgdp_check.nrows();
        let atol = Eqn::V::from_element(nparams, Eqn::T::from_f64(1e-6).unwrap(), ctx);
        let t0_tol = Eqn::T::from_f64(10.0).unwrap() * Eqn::T::EPSILON;
        assert!(
            (adjoint_state.as_ref().t - t0).abs() <= t0_tol,
            "expected adjoint final time {:?}, got {:?}",
            t0,
            adjoint_state.as_ref().t,
        );
        #[allow(clippy::needless_range_loop)]
        for j in 0..dgdp_check.ncols() {
            adjoint_state.as_ref().sg[j].assert_eq_norm(
                &dgdp_check.column(j).into_owned(),
                &atol,
                Eqn::T::from_f64(1e-6).unwrap(),
                Eqn::T::from_f64(260.0).unwrap(),
            );
        }
    }
}
