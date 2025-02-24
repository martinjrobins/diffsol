use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error, AdjointEquations, AugmentedOdeEquations, AugmentedOdeSolverMethod,
    DefaultDenseMatrix, DefaultSolver, DenseMatrix, LinearOp, LinearSolver, Matrix, MatrixCommon,
    MatrixOp, NonLinearOpAdjoint, NonLinearOpSensAdjoint, OdeEquations, OdeEquationsAdjoint,
    OdeSolverMethod, OdeSolverState, OdeSolverStopReason, Op, Vector, VectorIndex,
};

use num_traits::{One, Zero};
use std::ops::AddAssign;

pub trait AdjointOdeSolverMethod<'a, Eqn, Solver>:
    AugmentedOdeSolverMethod<'a, Eqn, AdjointEquations<'a, Eqn, Solver>>
where
    Eqn: OdeEquationsAdjoint + 'a,
    Solver: OdeSolverMethod<'a, Eqn>,
{
    /// Backwards pass for adjoint sensitivity analysis
    ///
    /// The overall goal is to compute the gradient of an output function G with respect to the model parameters p
    ///
    /// If `dgdu_eval` is empty, then G is the integral of the model output function u over time
    ///
    /// $$
    /// G = \int_{t_0}^{t_{\text{final}}} u(y(t)) dt
    /// $$
    ///
    /// where `y(t)` is the solution of the model at time `t`
    ///
    /// If `dgdu_eval` is non empty, then the output function G made from the sum of a sequence of n functions g_i
    ///
    /// $$
    /// G = \int_{t_0}^{t_{\text{final}}} \sum_{i=0}^{n-1} g_i(u(y(t_i)))) \delta(t - t_i) dt
    /// $$
    ///
    /// where $g(t)$ is the output of the model at time $t_i$
    ///
    /// For example, if G is the standard sum of squared errors, then $g_i = (u(y(t_i)) - d_i)^2$,
    /// where $d_i$ is the measured value of the output at time $t_i$
    ///
    /// The user passes in the gradient of g_i with respect to u_i for each timepoint i in `dgdu_eval`.
    /// For example, if g_i = (u(y(t_i)) - d_i)^2, then dgdu_i = 2(u(y(t_i)) - d_i), where $u(y(t_i))$
    /// can be obtained from the forward pass.
    ///
    /// The input `dgdu_eval` is a vector so users can supply multiple sets of `g_i` functions, and each
    /// element of the vector is a dense matrix of size n_o x n, where n_o is the number of outputs in the model
    /// and n is the number of timepoints. The i-th column of `dgdu_eval` is the gradient of g_i with respect to u_i.
    /// The input `t_eval` is a vector of length n, where the i-th element is the timepoint t_i.
    fn solve_adjoint_backwards_pass(
        mut self,
        t_eval: &[Eqn::T],
        dgdu_eval: &[&mut <Eqn::V as DefaultDenseMatrix>::M],
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
        let nout = self.problem().eqn.out().map(|o| o.nout()).unwrap_or(0);
        if dgdu_eval.iter().any(|dgdu| dgdu.nrows() != nout) {
            return Err(ode_solver_error!(
                Other,
                "Number of outputs does not match number of rows in gradient"
            ));
        }

        let mut integrate_delta_g = if have_neqn > 0 {
            Some(IntegrateDeltaG::<_, <Eqn::M as DefaultSolver>::LS>::new(
                &self,
            )?)
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
        match self.set_stop_time(self.problem().t0) {
            Ok(_) => while self.step()? != OdeSolverStopReason::TstopReached {},
            Err(DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime)) => {}
            e => e?,
        }

        // correct the adjoint solution for the initial conditions
        let (mut state, aug_eqn) = self.into_state_and_eqn();
        let aug_eqn = aug_eqn.unwrap();
        let state_mut = state.as_mut();
        aug_eqn.correct_sg_for_init(t_eval[0], state_mut.s, state_mut.sg);

        // return the solution
        Ok(state)
    }
}

impl<'a, Eqn, S, Solver> AdjointOdeSolverMethod<'a, Eqn, S> for Solver
where
    Eqn: OdeEquationsAdjoint + 'a,
    S: OdeSolverMethod<'a, Eqn>,
    Solver: AugmentedOdeSolverMethod<'a, Eqn, AdjointEquations<'a, Eqn, S>>,
{
}

struct IntegrateDeltaG<M: Matrix, LS: LinearSolver<M>> {
    pub rhs_jac_aa: Option<MatrixOp<M>>,
    pub solver: Option<LS>,
    pub algebraic_indices: <M::V as Vector>::Index,
    pub differential_indices: <M::V as Vector>::Index,
    pub tmp_algebraic: M::V,
    pub tmp_differential: M::V,
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
        Eqn: OdeEquations<M = M, V = M::V, T = M::T> + 'a,
        Solver: OdeSolverMethod<'a, Eqn>,
    {
        let eqn = &solver.problem().eqn;
        let (algebraic_indices, differential_indices) = if let Some(mass) = eqn.mass() {
            mass.matrix(M::T::zero())
                .diagonal()
                .partition_indices(|x| x == M::T::zero())
        } else {
            (
                <M::V as Vector>::Index::zeros(0),
                <M::V as Vector>::Index::zeros(0),
            )
        };
        let nparams = eqn.rhs().nparams();
        let nstates = eqn.rhs().nstates();
        let nout = eqn.out().map(|o| o.nout()).unwrap_or(nstates);
        let tmp_nstates = M::V::zeros(nstates);
        let tmp_nstates2 = M::V::zeros(nstates);
        let tmp_nparams = M::V::zeros(nparams);
        let tmp_nout = M::V::zeros(nout);
        let tmp_algebraic = M::V::zeros(algebraic_indices.len());
        let tmp_differential = M::V::zeros(nstates - algebraic_indices.len());
        if algebraic_indices.len() > 0 {
            let jacobian = solver
                .jacobian()
                .ok_or(DiffsolError::from(OdeSolverError::JacobianNotAvailable))?;
            let (_, _, _, rhs_jac_aa) = jacobian.split_at_indices(&algebraic_indices);
            let rhs_jac_aa_op = MatrixOp::new(rhs_jac_aa);
            let mut solver = LS::default();
            solver.set_problem(&rhs_jac_aa_op);
            Ok(Self {
                rhs_jac_aa: Some(rhs_jac_aa_op),
                solver: Some(solver),
                tmp_nparams,
                tmp_algebraic,
                algebraic_indices,
                tmp_nstates,
                tmp_nout,
                tmp_differential,
                differential_indices,
                tmp_nstates2,
            })
        } else {
            Ok(Self {
                rhs_jac_aa: None,
                solver: None,
                tmp_algebraic,
                tmp_nparams,
                algebraic_indices,
                tmp_nstates,
                tmp_nout,
                tmp_differential,
                differential_indices,
                tmp_nstates2,
            })
        }
    }
    fn integrate_delta_g<'a, 'b, Eqn, S1, Solver>(
        &mut self,
        solver: &mut Solver,
        dgdus: impl Iterator<Item = <M::V as Vector>::View<'b>>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsAdjoint<M = M, V = M::V, T = M::T> + 'a,
        Solver: AdjointOdeSolverMethod<'a, Eqn, S1>,
        S1: OdeSolverMethod<'a, Eqn>,
    {
        // interpolate forward state y
        let t = solver.state().t;
        solver
            .augmented_eqn()
            .unwrap()
            .interpolate_forward_state(t, &mut self.tmp_nstates)?;

        // if there are algebraic indices, setup the solver for (f*_y^a)^{-1}
        let rhs_jac_ad_op = if self.algebraic_indices.len() > 0 {
            let jacobian = solver.jacobian().unwrap();
            let (_, _, jac_ad, jac_aa) = jacobian.split_at_indices(&self.algebraic_indices);

            // init solver for (f*_u^a)^{-1}
            let rhs_jac_aa = self.rhs_jac_aa.as_mut().unwrap();
            rhs_jac_aa.m_mut().copy_from(&jac_aa);

            // linearisation does not depend on x or t
            self.solver.as_mut().unwrap().set_linearisation(
                rhs_jac_aa,
                &self.tmp_algebraic,
                Eqn::T::zero(),
            );
            Some(jac_ad)
        } else {
            None
        };

        let out = solver.augmented_eqn().unwrap().eqn().out();
        let state_mut = solver.state_mut();
        for ((s_i, sg_i), dgdu) in state_mut
            .s
            .iter_mut()
            .zip(state_mut.sg.iter_mut())
            .zip(dgdus)
        {
            self.tmp_nout.copy_from_view(&dgdu);
            let neg_dgdy = if let Some(out) = out.as_ref() {
                out.jac_transpose_mul_inplace(
                    &self.tmp_nstates,
                    t,
                    &self.tmp_nout,
                    &mut self.tmp_nstates2,
                );
                &self.tmp_nstates2
            } else {
                &self.tmp_nout
            };
            if self.algebraic_indices.len() > 0 {
                // add f*_y^d (f*_y^a)^{-1} g*_y^a - g*_y^d to differential part of s
                // note g_y^T = u_y^T * g_u^T, where u_y^T is the out adjoint
                self.tmp_differential
                    .gather(neg_dgdy, &self.differential_indices);
                self.tmp_algebraic.gather(neg_dgdy, &self.algebraic_indices);
                s_i.add_assign(&self.tmp_differential);
                self.solver
                    .as_ref()
                    .unwrap()
                    .solve_in_place(&mut self.tmp_algebraic)?;
                rhs_jac_ad_op.as_ref().unwrap().gemv(
                    -M::T::one(),
                    &self.tmp_algebraic,
                    M::T::one(),
                    s_i,
                );
            } else {
                // add -g*_y^T(x, t) to s
                s_i.add_assign(neg_dgdy);
            }
            // add -g_p^T(x, t) to sg
            // g_p = g_u^T * u_y^T * y_p^T
            // g_p^T =  u_p^T * g_u^T (u is the output of the model, so u_p^T is the sens_tranpose)
            if let Some(out) = out.as_ref() {
                out.sens_transpose_mul_inplace(
                    &self.tmp_nstates,
                    t,
                    &self.tmp_nout,
                    &mut self.tmp_nparams,
                );
                sg_i.add_assign(&self.tmp_nparams);
            }
        }
        Ok(())
    }
}
