use crate::{
    error::DiffsolError, error::OdeSolverError, ode_solver_error, AugmentedOdeSolverMethod,
    Context, DefaultDenseMatrix, DenseMatrix, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens,
    OdeEquationsImplicitSens, OdeSolverStopReason, Op, SensEquations, Vector, VectorViewMut,
};
use num_traits::{One, Zero};
use std::ops::AddAssign;

pub trait SensitivitiesOdeSolverMethod<'a, Eqn>:
    AugmentedOdeSolverMethod<'a, Eqn, SensEquations<'a, Eqn>>
where
    Eqn: OdeEquationsImplicitSens + 'a,
{
    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a tuple `(y, sens)`, where `y` is a dense matrix of solution values at timepoints given by `t_eval`,
    /// and `sens` is a Vec of dense matrices, the ith element of the Vec are the the sensitivities with respect to the ith parameter.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    #[allow(clippy::type_complexity)]
    fn solve_dense_sensitivities(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<
        (
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<<Eqn::V as DefaultDenseMatrix>::M>,
        ),
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicitSens,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        if self.problem().integrate_out {
            return Err(ode_solver_error!(
                Other,
                "Cannot integrate out when solving for sensitivities"
            ));
        }
        let (mut tmp_nout, mut tmp_nparms, nrows) = if let Some(out) = self.problem().eqn.out() {
            (
                Some(Eqn::V::zeros(out.nout(), self.problem().context().clone())),
                Some(Eqn::V::zeros(
                    out.nparams(),
                    self.problem().context().clone(),
                )),
                out.nout(),
            )
        } else {
            (None, None, self.problem().eqn.rhs().nout())
        };

        let nstates = self.problem().eqn.rhs().nstates();
        let nparams = self.problem().eqn.rhs().nparams();
        let ctx = self.problem().context().clone();

        let mut y = Eqn::V::zeros(nstates, ctx.clone());
        let mut s = vec![Eqn::V::zeros(nstates, ctx.clone()); nparams];

        let mut ret = self
            .problem()
            .context()
            .dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());
        let mut ret_sens = vec![
            self.problem()
                .context()
                .dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());
            nparams
        ];

        // check t_eval is increasing and all values are >= the current time
        let t0 = self.state().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        let t_final = *t_eval.last().unwrap();
        self.set_stop_time(t_final)?;

        let mut col = 0usize;
        let mut t_i = 0usize;

        'outer: while t_i < t_eval.len() {
            let t_target = t_eval[t_i];

            while self.state().t < t_target {
                match self.step()? {
                    OdeSolverStopReason::InternalTimestep => {}
                    OdeSolverStopReason::TstopReached => break,
                    OdeSolverStopReason::RootFound(t_root, root_idx) => {
                        // ----- write t_eval points strictly before t_root -----
                        while t_i < t_eval.len() && t_eval[t_i] < t_root {
                            self.interpolate_inplace(t_eval[t_i], &mut y)?;
                            self.interpolate_sens_inplace(t_eval[t_i], &mut s)?;
                            if let Some(out) = self.problem().eqn.out() {
                                let tmp_nout = tmp_nout.as_mut().unwrap();
                                let tmp_nparams = tmp_nparms.as_mut().unwrap();
                                out.call_inplace(&y, t_eval[t_i], tmp_nout);
                                ret.column_mut(col).copy_from(&*tmp_nout);
                                for (j, s_j) in s.iter_mut().enumerate() {
                                    let mut col_v = ret_sens[j].column_mut(col);
                                    tmp_nparams.set_index(j, Eqn::T::one());
                                    out.jac_mul_inplace(&y, t_eval[t_i], s_j, tmp_nout);
                                    col_v.copy_from(&*tmp_nout);
                                    out.sens_mul_inplace(&y, t_eval[t_i], tmp_nparams, tmp_nout);
                                    col_v.add_assign(&*tmp_nout);
                                    tmp_nparams.set_index(j, Eqn::T::zero());
                                }
                            } else {
                                ret.column_mut(col).copy_from(&y);
                                for (j, s_j) in s.iter().enumerate() {
                                    ret_sens[j].column_mut(col).copy_from(s_j);
                                }
                            }
                            col += 1;
                            t_i += 1;
                        }

                        // Pin state (y, dy, t, s) to t_root via interpolation.
                        self.state_mut_back(t_root)?;
                        // Populate local buffers from the pinned state for possible output writing.
                        y.copy_from(self.state().y);
                        for (j, s_j) in s.iter_mut().enumerate() {
                            s_j.copy_from(&self.state().s[j]);
                        }

                        // ----- Handle reset at root index 0 -----
                        if root_idx == 0 {
                            if let Some(reset_fn) = self.problem().eqn.reset() {
                                // Apply reset: updates state.y, state.dy, and
                                // s_new[j] = J_R(y_before, t_root) · s_old[j].
                                self.state_mut_op_with_sens(&reset_fn)?;
                                self.set_stop_time(t_final)?;
                                continue 'outer;
                            }
                        }

                        // ----- Non-reset root: write root state and return early -----
                        // y and s are already interpolated at t_root
                        if let Some(out) = self.problem().eqn.out() {
                            let tmp_nout = tmp_nout.as_mut().unwrap();
                            let tmp_nparams = tmp_nparms.as_mut().unwrap();
                            out.call_inplace(&y, t_root, tmp_nout);
                            ret.column_mut(col).copy_from(&*tmp_nout);
                            for (j, s_j) in s.iter_mut().enumerate() {
                                let mut col_v = ret_sens[j].column_mut(col);
                                tmp_nparams.set_index(j, Eqn::T::one());
                                out.jac_mul_inplace(&y, t_root, s_j, tmp_nout);
                                col_v.copy_from(&*tmp_nout);
                                out.sens_mul_inplace(&y, t_root, tmp_nparams, tmp_nout);
                                col_v.add_assign(&*tmp_nout);
                                tmp_nparams.set_index(j, Eqn::T::zero());
                            }
                        } else {
                            ret.column_mut(col).copy_from(&y);
                            for (j, s_j) in s.iter().enumerate() {
                                ret_sens[j].column_mut(col).copy_from(s_j);
                            }
                        }
                        col += 1;
                        ret.resize_cols(col);
                        for rs in ret_sens.iter_mut() {
                            rs.resize_cols(col);
                        }
                        return Ok((ret, ret_sens));
                    }
                }
            }

            // Write t_eval[t_i] to output using interpolation
            self.interpolate_inplace(t_target, &mut y)?;
            self.interpolate_sens_inplace(t_target, &mut s)?;
            if let Some(out) = self.problem().eqn.out() {
                let tmp_nout = tmp_nout.as_mut().unwrap();
                let tmp_nparams = tmp_nparms.as_mut().unwrap();
                out.call_inplace(&y, t_target, tmp_nout);
                ret.column_mut(col).copy_from(&*tmp_nout);
                for (j, s_j) in s.iter_mut().enumerate() {
                    let mut col_v = ret_sens[j].column_mut(col);
                    tmp_nparams.set_index(j, Eqn::T::one());
                    out.jac_mul_inplace(&y, t_target, s_j, tmp_nout);
                    col_v.copy_from(&*tmp_nout);
                    out.sens_mul_inplace(&y, t_target, tmp_nparams, tmp_nout);
                    col_v.add_assign(&*tmp_nout);
                    tmp_nparams.set_index(j, Eqn::T::zero());
                }
            } else {
                ret.column_mut(col).copy_from(&y);
                for (j, s_j) in s.iter().enumerate() {
                    ret_sens[j].column_mut(col).copy_from(s_j);
                }
            }
            col += 1;
            t_i += 1;
        }

        ret.resize_cols(col);
        for rs in ret_sens.iter_mut() {
            rs.resize_cols(col);
        }
        Ok((ret, ret_sens))
    }
}

impl<'a, M, Eqn> SensitivitiesOdeSolverMethod<'a, Eqn> for M
where
    M: AugmentedOdeSolverMethod<'a, Eqn, SensEquations<'a, Eqn>>,
    Eqn: OdeEquationsImplicitSens + 'a,
{
}
