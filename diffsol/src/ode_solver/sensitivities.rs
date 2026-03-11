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
        let eqn_out = self.problem().eqn.out();

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

        // Write one column of (y, s) output at time t into ret/ret_sens[col].
        // ret, ret_sens, and col are passed explicitly so they remain directly accessible
        // outside the closure (avoiding borrow conflicts with resize/return).
        let mut write_col = |y: &Eqn::V,
                             s: &[Eqn::V],
                             t: Eqn::T,
                             col: usize,
                             ret: &mut <Eqn::V as DefaultDenseMatrix>::M,
                             ret_sens: &mut [<Eqn::V as DefaultDenseMatrix>::M]|
         -> Result<(), DiffsolError> {
            if let Some(out) = eqn_out.as_ref() {
                let tmp_nout = tmp_nout.as_mut().unwrap();
                let tmp_nparams = tmp_nparms.as_mut().unwrap();
                out.call_inplace(y, t, tmp_nout);
                ret.column_mut(col).copy_from(&*tmp_nout);
                for (j, s_j) in s.iter().enumerate() {
                    let mut col_v = ret_sens[j].column_mut(col);
                    tmp_nparams.set_index(j, Eqn::T::one());
                    out.jac_mul_inplace(y, t, s_j, tmp_nout);
                    col_v.copy_from(&*tmp_nout);
                    out.sens_mul_inplace(y, t, tmp_nparams, tmp_nout);
                    col_v.add_assign(&*tmp_nout);
                    tmp_nparams.set_index(j, Eqn::T::zero());
                }
            } else {
                ret.column_mut(col).copy_from(y);
                for (j, s_j) in s.iter().enumerate() {
                    ret_sens[j].column_mut(col).copy_from(s_j);
                }
            }
            Ok(())
        };

        'outer: while t_i < t_eval.len() {
            let t_target = t_eval[t_i];

            while self.state().t < t_target {
                match self.step()? {
                    OdeSolverStopReason::InternalTimestep => {}
                    OdeSolverStopReason::TstopReached => break,
                    OdeSolverStopReason::RootFound(t_root, root_idx) => {
                        // Write any t_eval points strictly before t_root.
                        while t_i < t_eval.len() && t_eval[t_i] < t_root {
                            self.interpolate_inplace(t_eval[t_i], &mut y)?;
                            self.interpolate_sens_inplace(t_eval[t_i], &mut s)?;
                            write_col(&y, &s, t_eval[t_i], col, &mut ret, &mut ret_sens)?;
                            col += 1;
                            t_i += 1;
                        }

                        self.state_mut_back(t_root)?;

                        if root_idx == 0 {
                            if let Some(reset_fn) = self.problem().eqn.reset() {
                                self.state_mut_op_with_sens(&reset_fn)?;
                                continue 'outer;
                            }
                        }

                        // Non-reset root: write root state and return early.
                        y.copy_from(self.state().y);
                        for (j, s_j) in s.iter_mut().enumerate() {
                            s_j.copy_from(&self.state().s[j]);
                        }
                        write_col(&y, &s, t_root, col, &mut ret, &mut ret_sens)?;
                        col += 1;
                        ret.resize_cols(col);
                        for rs in ret_sens.iter_mut() {
                            rs.resize_cols(col);
                        }
                        return Ok((ret, ret_sens));
                    }
                }
            }

            self.interpolate_inplace(t_target, &mut y)?;
            self.interpolate_sens_inplace(t_target, &mut s)?;
            write_col(&y, &s, t_target, col, &mut ret, &mut ret_sens)?;
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
