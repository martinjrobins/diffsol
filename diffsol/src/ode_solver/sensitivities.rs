use crate::{
    error::DiffsolError, error::OdeSolverError, ode_solver_error, AugmentedOdeSolverMethod,
    Context, DefaultDenseMatrix, DefaultSolver, DenseMatrix, NonLinearOp, NonLinearOpJacobian,
    NonLinearOpSens, OdeEquationsImplicitSens, OdeSolverStopReason, Op, SensEquations, Vector,
    VectorViewMut,
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
        Eqn::M: DefaultSolver,
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

        let mut y = Eqn::V::zeros(
            self.problem().eqn.rhs().nstates(),
            self.problem().context().clone(),
        );

        let mut s = vec![
            Eqn::V::zeros(
                self.problem().eqn.rhs().nstates(),
                self.problem().context().clone(),
            );
            self.problem().eqn.rhs().nparams()
        ];

        let mut ret = self
            .problem()
            .context()
            .dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());
        let mut ret_sens = vec![
            self.problem()
                .context()
                .dense_mat_zeros::<Eqn::V>(nrows, t_eval.len());
            self.problem().eqn.rhs().nparams()
        ];

        // check t_eval is increasing and all values are greater than or equal to the current time
        let t0 = self.state().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().take(t_eval.len() - 1).enumerate() {
            while self.state().t < *t {
                step_reason = self.step()?;
            }
            self.interpolate_inplace(*t, &mut y)?;
            self.interpolate_sens_inplace(*t, &mut s)?;
            if let Some(out) = self.problem().eqn.out() {
                let tmp_nout = tmp_nout.as_mut().unwrap();
                let tmp_nparams = tmp_nparms.as_mut().unwrap();
                out.call_inplace(&y, *t, tmp_nout);
                ret.column_mut(i).copy_from(tmp_nout);
                for (j, s_j) in s.iter_mut().enumerate() {
                    // compute J * s_j + dF/dp * e_j where e_j is the jth basis vector
                    let mut ret_sens = ret_sens[j].column_mut(i);
                    tmp_nparams.set_index(j, Eqn::T::one());
                    out.jac_mul_inplace(&y, *t, s_j, tmp_nout);
                    ret_sens.copy_from(tmp_nout);
                    out.sens_mul_inplace(&y, *t, tmp_nparams, tmp_nout);
                    ret_sens.add_assign(&*tmp_nout);
                    tmp_nparams.set_index(j, Eqn::T::zero());
                }
            } else {
                ret.column_mut(i).copy_from(&y);
                for (j, s_j) in s.iter().enumerate() {
                    ret_sens[j].column_mut(i).copy_from(s_j);
                }
            }
        }

        // do final step
        while step_reason != OdeSolverStopReason::TstopReached {
            step_reason = self.step()?;
        }
        let y = self.state().y;
        let s = self.state().s;
        let i = t_eval.len() - 1;
        let t = t_eval.last().unwrap();
        if let Some(out) = self.problem().eqn.out() {
            let tmp_nout = tmp_nout.as_mut().unwrap();
            let tmp_nparams = tmp_nparms.as_mut().unwrap();
            out.call_inplace(y, *t, tmp_nout);
            ret.column_mut(i).copy_from(tmp_nout);
            for (j, s_j) in s.iter().enumerate() {
                // compute J * s_j + dF/dp * e_j where e_j is the jth basis vector
                let mut ret_sens = ret_sens[j].column_mut(i);
                tmp_nparams.set_index(j, Eqn::T::one());
                out.jac_mul_inplace(y, *t, s_j, tmp_nout);
                ret_sens.copy_from(tmp_nout);
                out.sens_mul_inplace(y, *t, tmp_nparams, tmp_nout);
                ret_sens.add_assign(&*tmp_nout);
                tmp_nparams.set_index(j, Eqn::T::zero());
            }
        } else {
            ret.column_mut(i).copy_from(y);
            for (j, s_j) in s.iter().enumerate() {
                ret_sens[j].column_mut(i).copy_from(s_j);
            }
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
