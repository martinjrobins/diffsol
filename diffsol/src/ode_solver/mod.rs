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
pub mod state;
pub mod tableau;

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use self::problem::OdeSolverSolution;
    use nalgebra::ComplexField;

    use super::*;
    use crate::error::{DiffsolError, OdeSolverError};
    use crate::matrix::Matrix;
    use crate::op::unit::UnitCallable;
    use crate::op::ParameterisedOp;
    use crate::{
        op::OpStatistics, AdjointOdeSolverMethod, Context, DenseMatrix, MatrixCommon, MatrixRef,
        NonLinearOpJacobian, OdeEquations, OdeEquationsImplicit, OdeEquationsImplicitAdjoint,
        OdeEquationsRef, OdeSolverConfig, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
        OdeSolverStopReason, Scale, VectorRef, VectorView, VectorViewMut,
    };
    use crate::{
        ConstantOp, ConstantOpSens, DefaultDenseMatrix, DefaultSolver, LinearSolver, NonLinearOp,
        NonLinearOpSens, Op, Vector,
    };
    use num_traits::{FromPrimitive, One, Zero};

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
                            Ok(OdeSolverStopReason::RootFound(_)) => {
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
                    (method.problem().rtol, &method.problem().atol)
                };
                let error = soln.clone() - &point.state;
                let error_norm = error.squared_norm(&point.state, atol, rtol).sqrt();
                assert!(
                    error_norm < M::T::from_f64(15.0).unwrap(),
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
            let mut s = problem.bdf::<LS>().unwrap();
            s.set_stop_time(final_time).unwrap();
            while s.step().unwrap() != OdeSolverStopReason::TstopReached {}
            let g_pos = s.state().g.clone();

            p_0.set_index(i, p_base.get_index(i) - h.get_index(i));
            problem.eqn.set_params(&p_0);
            let mut s = problem.bdf::<LS>().unwrap();
            s.set_stop_time(final_time).unwrap();
            while s.step().unwrap() != OdeSolverStopReason::TstopReached {}
            let g_neg = s.state().g.clone();
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
            ret.set_index(0, ret.get_index(0) + delta.norm(2).powi(2));
            ret.set_index(1, ret.get_index(1) + delta.norm(4).powi(4));
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
        let mut s = problem.bdf::<LS>().unwrap();
        let data = s.solve_dense(times).unwrap();

        for i in 0..nparams {
            p_0.set_index(i, p_base.get_index(i) + h.get_index(i));
            problem.eqn.set_params(&p_0);
            let mut s = problem.bdf::<LS>().unwrap();
            let v = s.solve_dense(times).unwrap();
            let g_pos = sum_squares(&v, &data);

            p_0.set_index(i, p_base.get_index(i) - h.get_index(i));
            problem.eqn.set_params(&p_0);
            let mut s = problem.bdf::<LS>().unwrap();
            let v = s.solve_dense(times).unwrap();
            let g_neg = sum_squares(&v, &data);

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
            .solve_adjoint_backwards_pass(times, dgdu.iter().collect::<Vec<_>>().as_slice())
            .unwrap();
        let gs_adj = state.into_common().sg;
        #[allow(clippy::needless_range_loop)]
        for j in 0..dgdp_check.ncols() {
            gs_adj[j].assert_eq_norm(
                &dgdp_check.column(j).into_owned(),
                &atol,
                rtol,
                Eqn::T::from_f64(66.).unwrap(),
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
            .solve_adjoint_backwards_pass(&[], &[])
            .unwrap();
        let gs_adj = state.into_common().sg;
        #[allow(clippy::needless_range_loop)]
        for j in 0..dgdp_check.ncols() {
            gs_adj[j].assert_eq_norm(
                &dgdp_check.column(j).into_owned(),
                &atol,
                rtol,
                Eqn::T::from_f64(33.).unwrap(),
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

    pub fn test_config<'a, Eqn: OdeEquations + 'a, Method: OdeSolverMethod<'a, Eqn>>(
        mut s: Method,
    ) {
        *s.config_mut().as_base_mut().minimum_timestep = Eqn::T::from_f64(1.0e8).unwrap();
        assert_eq!(
            *s.config().as_base_ref().minimum_timestep,
            Eqn::T::from_f64(1.0e8).unwrap()
        );
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
                Ok(OdeSolverStopReason::RootFound(t)) => {
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
}
