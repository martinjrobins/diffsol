use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error, AdjointEquations, AugmentedOdeEquations, AugmentedOdeSolverMethod,
    DefaultDenseMatrix, DefaultSolver, DenseMatrix, LinearSolver, Matrix, MatrixCommon, MatrixOp,
    NonLinearOpAdjoint, NonLinearOpSensAdjoint, OdeEquations, OdeEquationsImplicitAdjoint,
    OdeSolverMethod, OdeSolverState, OdeSolverStopReason, Op, Vector, VectorIndex,
};

use num_traits::{One, Zero};
use std::ops::{AddAssign, SubAssign};

pub trait AdjointOdeSolverMethod<'a, Eqn, Solver>:
    AugmentedOdeSolverMethod<'a, Eqn, AdjointEquations<'a, Eqn, Solver>>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    Solver: OdeSolverMethod<'a, Eqn>,
{
    /// Backwards pass for adjoint sensitivity analysis
    ///
    /// The overall goal is to compute the gradient of an output function `G` with respect to the model parameters `p`
    ///
    /// If `dgdu_eval` is empty, then `G` is the integral of the model output function `u` over time
    ///
    /// $$
    /// G = \int_{t_0}^{t_{\text{final}}} u(y(t)) dt
    /// $$
    ///
    /// where `y(t)` is the solution of the model at time `t`
    ///
    /// If `dgdu_eval` is non empty, then the output function `G` made from the sum of a sequence of `n` functions `g_i`
    /// operating on the model output function u at timepoints `t_i`
    ///
    /// $$
    /// G = \int_{t_0}^{t_{\text{final}}} \sum_{i=0}^{n-1} g_i(u(y(t_i)))) \delta(t - t_i) dt
    /// $$
    ///
    /// For example, if `G` is the standard sum of squared errors, then `g_i = (u(y(t_i)) - d_i)^2`,
    /// where `d_i` is the measured value of the output at time `t_i`
    ///
    /// The user passes in the gradient of `g_i` with respect to `u_i` for each timepoint `i` in `dgdu_eval`.
    /// For example, if `g_i = (u(y(t_i)) - d_i)^2`, then `dgdu_i = 2(u(y(t_i)) - d_i)`, where `u(y(t_i))`
    /// can be obtained from the forward pass.
    ///
    /// The input `dgdu_eval` is a vector so users can supply multiple sets of `g_i` functions, and each
    /// element of the vector is a dense matrix of size `n_o x n`, where `n_o`` is the number of outputs in the model
    /// and `n` is the number of timepoints. The i-th column of `dgdu_eval` is the gradient of `g_i` with respect to `u_i`.
    /// The input `t_eval` is a vector of length `n`, where the i-th element is the timepoint `t_i`.
    fn solve_adjoint_backwards_pass(
        mut self,
        t_eval: &[Eqn::T],
        dgdu_eval: &[&<Eqn::V as DefaultDenseMatrix>::M],
    ) -> Result<Self::State, DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Eqn::M: DefaultSolver,
    {
        // check that aug_eqn exists
        if self.augmented_eqn().is_none() {
            return Err(ode_solver_error!(Other, "No augmented equations"));
        }

        // t_eval should be in increasing order
        if t_eval.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ode_solver_error!(
                Other,
                "t_eval should be in increasing order"
            ));
        }

        let have_neqn = self.augmented_eqn().unwrap().max_index();
        if dgdu_eval.is_empty() {
            // if dgdus is empty, then the number of outputs in the model is used
            let expected_neqn = self.problem().eqn.out().map(|o| o.nout()).unwrap_or(0);
            if self.augmented_eqn().unwrap().max_index() != expected_neqn {
                return Err(ode_solver_error!(
                    Other,
                    format!("Number of augmented equations does not match number of model outputs: {} != {}", have_neqn, expected_neqn)
                ));
            }
        } else {
            // if dgdu_eval is not empty, check that aug_eqn has dgdu_eval.len() outputs
            let expected_neqn = dgdu_eval.len();
            if have_neqn != expected_neqn {
                return Err(ode_solver_error!(
                    Other,
                    format!("Number of outputs in augmented equations does not match number of outputs in dgdu_eval: {} != {}", have_neqn, expected_neqn)
                ));
            }
        }
        // check that nrows of each dgdu_eval is the same as the number of outputs in the model
        let nout = self.problem().eqn.nout();
        if dgdu_eval.iter().any(|dgdu| dgdu.nrows() != nout) {
            return Err(ode_solver_error!(
                Other,
                "Number of outputs does not match number of rows in gradient"
            ));
        }

        let mut integrate_delta_g = if have_neqn > 0 && !dgdu_eval.is_empty() {
            let integrate_delta_g =
                IntegrateDeltaG::<_, <Eqn::M as DefaultSolver>::LS>::new(&self)?;
            Some(integrate_delta_g)
        } else {
            None
        };

        // solve the adjoint problem stopping at each t_eval
        for (i, t) in t_eval.iter().enumerate().rev() {
            // integrate to t if not already there
            match self.set_stop_time(*t) {
                Ok(_) => while self.step()? != OdeSolverStopReason::TstopReached {},
                Err(DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime)) => {}
                e => e?,
            }

            if let Some(integrate_delta_g) = integrate_delta_g.as_mut() {
                let dudg_i = dgdu_eval.iter().map(|dgdu| dgdu.column(i));
                integrate_delta_g.integrate_delta_g(&mut self, dudg_i)?;
            }
        }

        // keep integrating until t0
        let t0 = self.problem().t0;
        match self.set_stop_time(t0) {
            Ok(_) => while self.step()? != OdeSolverStopReason::TstopReached {},
            Err(DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime)) => {}
            e => e?,
        }

        // correct the adjoint solution for the initial conditions
        let (mut state, aug_eqn) = self.into_state_and_eqn();
        let aug_eqn = aug_eqn.unwrap();
        let state_mut = state.as_mut();
        aug_eqn.correct_sg_for_init(t0, state_mut.s, state_mut.sg);

        // return the solution
        Ok(state)
    }
}

impl<'a, Eqn, S, Solver> AdjointOdeSolverMethod<'a, Eqn, S> for Solver
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    S: OdeSolverMethod<'a, Eqn>,
    Solver: AugmentedOdeSolverMethod<'a, Eqn, AdjointEquations<'a, Eqn, S>>,
{
}

struct BlockInfoSol<M: Matrix, LS: LinearSolver<M>> {
    pub block: MatrixOp<M>,
    pub src_indices: <M::V as Vector>::Index,
    pub solver: LS,
}

struct BlockInfo<M: Matrix> {
    pub block: MatrixOp<M>,
    pub src_indices: <M::V as Vector>::Index,
}

struct PartitionInfo<I> {
    pub algebraic_indices: I,
    pub differential_indices: I,
}

struct IntegrateDeltaG<M: Matrix, LS: LinearSolver<M>> {
    pub rhs_jac_aa: Option<BlockInfoSol<M, LS>>,
    pub rhs_jac_ad: Option<BlockInfo<M>>,
    pub mass_dd: Option<BlockInfoSol<M, LS>>,
    pub partition: Option<PartitionInfo<<M::V as Vector>::Index>>,
    pub tmp_algebraic: M::V,
    pub tmp_differential: M::V,
    pub tmp_differential2: M::V,
    pub tmp_nparams: M::V,
    pub tmp_nstates: M::V,
    pub tmp_nstates2: M::V,
    pub tmp_nout: M::V,
}

impl<M, LS> IntegrateDeltaG<M, LS>
where
    M: Matrix,
    LS: LinearSolver<M>,
{
    fn new<'a, Eqn, Solver>(solver: &Solver) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<M = M, V = M::V, T = M::T, C = M::C> + 'a,
        Solver: OdeSolverMethod<'a, Eqn>,
    {
        let eqn = &solver.problem().eqn;
        let ctx = solver.problem().eqn.context();
        let (partition, mass_dd, rhs_jac_aa, rhs_jac_ad) = if let Some(_mass) = eqn.mass() {
            let mass_matrix = solver.mass().unwrap();
            let (algebraic_indices, differential_indices) =
                mass_matrix.partition_indices_by_zero_diagonal();

            // setup mass solver
            let [(dd, dd_idx), _, _, _] = mass_matrix.split(&algebraic_indices);
            let mut mass_dd = BlockInfoSol {
                block: MatrixOp::new(dd),
                src_indices: dd_idx,
                solver: LS::default(),
            };
            mass_dd.solver.set_problem(&mass_dd.block);

            // setup jacobian solver if there are algebraic indices
            let (rhs_jac_aa, rhs_jac_ad) = if algebraic_indices.len() > 0 {
                let jacobian = solver
                    .jacobian()
                    .ok_or(DiffsolError::from(OdeSolverError::JacobianNotAvailable))?;
                let [_, (ad, ad_idx), _, (aa, aa_idx)] = jacobian.split(&algebraic_indices);
                let mut rhs_jac_aa = BlockInfoSol {
                    block: MatrixOp::new(aa),
                    src_indices: aa_idx,
                    solver: LS::default(),
                };
                rhs_jac_aa.solver.set_problem(&rhs_jac_aa.block);
                let rhs_jac_ad = BlockInfo {
                    block: MatrixOp::new(ad),
                    src_indices: ad_idx,
                };
                (Some(rhs_jac_aa), Some(rhs_jac_ad))
            } else {
                (None, None)
            };
            let partition = PartitionInfo {
                algebraic_indices,
                differential_indices,
            };
            (Some(partition), Some(mass_dd), rhs_jac_aa, rhs_jac_ad)
        } else {
            (None, None, None, None)
        };
        let nparams = eqn.rhs().nparams();
        let nstates = eqn.rhs().nstates();
        let nout = eqn.out().map(|o| o.nout()).unwrap_or(nstates);
        let tmp_nstates = M::V::zeros(nstates, ctx.clone());
        let tmp_nstates2 = M::V::zeros(nstates, ctx.clone());
        let tmp_nparams = M::V::zeros(nparams, ctx.clone());
        let tmp_nout = M::V::zeros(nout, ctx.clone());
        let nalgebraic = partition
            .as_ref()
            .map(|p| p.algebraic_indices.len())
            .unwrap_or(0);
        let ndifferential = nstates - nalgebraic;
        let tmp_algebraic = M::V::zeros(nalgebraic, ctx.clone());
        let tmp_differential = M::V::zeros(ndifferential, ctx.clone());
        let tmp_differential2 = M::V::zeros(ndifferential, ctx.clone());
        Ok(Self {
            rhs_jac_aa,
            rhs_jac_ad,
            mass_dd,
            tmp_nparams,
            tmp_algebraic,
            partition,
            tmp_nstates,
            tmp_nout,
            tmp_differential,
            tmp_differential2,
            tmp_nstates2,
        })
    }
    fn integrate_delta_g<'a, 'b, Eqn, S1, Solver>(
        &mut self,
        solver: &mut Solver,
        dgdus: impl Iterator<Item = <M::V as Vector>::View<'b>>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T> + 'a,
        Solver: AdjointOdeSolverMethod<'a, Eqn, S1>,
        S1: OdeSolverMethod<'a, Eqn>,
    {
        // interpolate forward state y
        let t = solver.state().t;
        solver
            .augmented_eqn()
            .unwrap()
            .interpolate_forward_state(t, &mut self.tmp_nstates)?;

        // if there are algebraic indices, setup the solver for (f*_y^a)^{-1} and M_dd*^-1
        if let Some(rhs_jac_aa) = self.rhs_jac_aa.as_mut() {
            let jacobian = solver.jacobian().unwrap();
            let rhs_jac_ad = self.rhs_jac_ad.as_mut().unwrap();
            rhs_jac_ad
                .block
                .m_mut()
                .gather(&jacobian, &rhs_jac_ad.src_indices);
            rhs_jac_aa
                .block
                .m_mut()
                .gather(&jacobian, &rhs_jac_aa.src_indices);
            rhs_jac_aa.solver.set_linearisation(
                &rhs_jac_aa.block,
                &self.tmp_algebraic,
                Eqn::T::zero(),
            );
        };

        // if there is a mass matrix, setup the solver for M_dd*^-1
        if let Some(mass_dd) = self.mass_dd.as_mut() {
            let mass = solver.mass().unwrap();
            mass_dd.block.m_mut().gather(&mass, &mass_dd.src_indices);
            mass_dd.solver.set_linearisation(
                &mass_dd.block,
                &self.tmp_differential,
                Eqn::T::zero(),
            );
        }

        // tmp_nout = all
        // tmp_nstates = all
        // tmp_nstates = out
        // tmp_differential = algebraic or mass
        // tmp_algebraic = algebraic
        // tmp_differential2 = algebraic
        // tmp_nparams = out

        let out = solver.augmented_eqn().unwrap().eqn().out();
        let sol_mdd_opt = self.mass_dd.as_ref().map(|m| &m.solver);
        let sol_jaa_opt = self.rhs_jac_aa.as_ref().map(|m| &m.solver);
        let state_mut = solver.state_mut();
        for ((s_i, sg_i), dgdu) in state_mut
            .s
            .iter_mut()
            .zip(state_mut.sg.iter_mut())
            .zip(dgdus)
        {
            // start from dgdu^T, where u is the output of the model
            self.tmp_nout.copy_from_view(&dgdu);

            // has out, mass and algebraic indices (requires tmp_nstates2, tmp_differential2, tmp_differential, tmp_algebraic
            if let (Some(out), Some(sol_mdd), Some(sol_jaa)) =
                (out.as_ref(), sol_mdd_opt, sol_jaa_opt)
            {
                let p = self.partition.as_ref().unwrap();
                // calculate -dgdy^T = -u_y^T * dgdu^T (if u = y, then dgdy = dgdu)
                out.jac_transpose_mul_inplace(
                    &self.tmp_nstates,
                    t,
                    &self.tmp_nout,
                    &mut self.tmp_nstates2,
                );

                // calculate -M_dd^-1 * dgdy_d^T (if M = I, then dgdy = dgdu)
                self.tmp_differential
                    .gather(&self.tmp_nstates2, &p.differential_indices);
                self.tmp_algebraic
                    .gather(&self.tmp_nstates2, &p.algebraic_indices);
                sol_mdd.solve_in_place(&mut self.tmp_differential)?;

                // add -f*_y^d (f*_y^a)^{-1} g*_y^a + M_dd^-1 g*_y^d to differential part of s
                self.tmp_differential2.gather(s_i, &p.differential_indices);
                self.tmp_differential2.sub_assign(&self.tmp_differential);

                sol_jaa.solve_in_place(&mut self.tmp_algebraic)?;
                let rhs_jac_ad = self.rhs_jac_ad.as_ref().unwrap().block.m();
                rhs_jac_ad.gemv(
                    M::T::one(),
                    &self.tmp_algebraic,
                    M::T::one(),
                    &mut self.tmp_differential2,
                );
                self.tmp_differential2.scatter(&p.differential_indices, s_i);
            // just has out and mass, no algebraic indices (requires tmp_nstates2)
            } else if let (Some(out), Some(sol_mdd)) = (out.as_ref(), sol_mdd_opt) {
                // calculate -dgdy^T = -u_y^T * dgdu^T (if u = y, then dgdy = dgdu)
                out.jac_transpose_mul_inplace(
                    &self.tmp_nstates,
                    t,
                    &self.tmp_nout,
                    &mut self.tmp_nstates2,
                );
                // calculate -M_dd^-1 * dgdy^T (if M = I, then dgdy = dgdu)
                sol_mdd.solve_in_place(&mut self.tmp_nstates2)?;
                s_i.sub_assign(&self.tmp_nstates2);
            // just has out, no mass (requires tmp_nstates2)
            } else if let Some(out) = out.as_ref() {
                // calculate -dgdy^T = -u_y^T * dgdu^T (if u = y, then dgdy = dgdu)
                out.jac_transpose_mul_inplace(
                    &self.tmp_nstates,
                    t,
                    &self.tmp_nout,
                    &mut self.tmp_nstates2,
                );
                s_i.sub_assign(&self.tmp_nstates2);
            // no out, has mass and algebraic indices (requires tmp_differential, tmp_algebraic, tmp_differential2)
            } else if let (Some(sol_mdd), Some(sol_jaa)) = (sol_mdd_opt, sol_jaa_opt) {
                let p = self.partition.as_ref().unwrap();
                self.tmp_differential
                    .gather(&self.tmp_nout, &p.differential_indices);
                self.tmp_algebraic
                    .gather(&self.tmp_nout, &p.algebraic_indices);

                // calculate M_dd^-1 * dgdy^T (if M = I, then dgdy = dgdu)
                sol_mdd.solve_in_place(&mut self.tmp_differential)?;

                // add -f*_y^d (f*_y^a)^{-1} g*_y^a + M_dd^-1 g*_y^d to differential part of s
                self.tmp_differential2.gather(s_i, &p.differential_indices);
                self.tmp_differential2.add_assign(&self.tmp_differential);
                sol_jaa.solve_in_place(&mut self.tmp_algebraic)?;
                let rhs_jac_ad = self.rhs_jac_aa.as_ref().unwrap().block.m();
                rhs_jac_ad.gemv(
                    M::T::one(),
                    &self.tmp_algebraic,
                    M::T::one(),
                    &mut self.tmp_differential2,
                );
                self.tmp_differential2.scatter(&p.differential_indices, s_i);
            // no out, has mass, no algebraic indices
            } else if let Some(sol_mdd) = sol_mdd_opt {
                // calculate M_dd^-1 * dgdy^T (if M = I, then dgdy = dgdu)
                sol_mdd.solve_in_place(&mut self.tmp_nout)?;
                s_i.add_assign(&self.tmp_nout);
            // no out, no mass
            } else {
                // add dgdy^T to s
                s_i.add_assign(&self.tmp_nout);
            }

            // add -g_p^T(x, t) to sg if there is an output function (requires tmp_nparams)
            // g_p = g_u^T * u_y^T * y_p^T
            // g_p^T =  u_p^T * g_u^T (u is the output of the model, so u_p^T is the sens_tranpose)
            if let Some(out) = out.as_ref() {
                out.sens_transpose_mul_inplace(
                    &self.tmp_nstates,
                    t,
                    &self.tmp_nout,
                    &mut self.tmp_nparams,
                );
                sg_i.sub_assign(&self.tmp_nparams);
            }
        }
        Ok(())
    }
}
