use crate::{
    error::DiffsolError, Context, DefaultDenseMatrix, DenseMatrix, OdeEquations, OdeSolverProblem,
    OdeSolverState, StateRef, StateRefMut, Vector,
};

use super::state::{zeros_mat_with_directions, zeros_with_directions, StateCommon};
use num_traits::Zero;

/// What a Runge-Kutta step needs, beyond the end-of-step state, to replay itself for
/// [crate::StepVjp].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RkStepRecord<T> {
    /// Step size of the accepted attempt. Unlike the state's `h`, the step size the next step
    /// starts with, a later `set_stop_time` does not change it.
    pub(crate) h: T,
}

/// State container for the Runge-Kutta solvers. For the common state API use `as_ref` and `as_mut` methods.
///
/// After a step, `diff` holds the stage increments `h k_i` of that step, from which the step can
/// be interpolated and replayed.
#[derive(Clone)]
pub struct RkState<V, M = <V as DefaultDenseMatrix>::M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V>,
{
    pub(crate) diff: M,
    pub(crate) y: V,
    pub(crate) dy: V,
    pub(crate) g: V,
    pub(crate) dg: V,
    pub(crate) s: V,
    pub(crate) ds: V,
    pub(crate) sg: V,
    pub(crate) dsg: V,
    pub(crate) t: V::T,
    pub(crate) h: V::T,
    /// The last step taken to reach this state, if any.
    pub(crate) last_step: Option<RkStepRecord<V::T>>,
}

impl<V, M> RkState<V, M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V, C = V::C>,
{
    pub(crate) fn new_empty(ctx: V::C) -> Self {
        let default_v = V::zeros(0, ctx.clone());
        Self {
            diff: M::zeros(0, 0, ctx),
            y: default_v.clone(),
            dy: default_v.clone(),
            g: default_v.clone(),
            dg: default_v.clone(),
            s: default_v.clone(),
            ds: default_v.clone(),
            sg: default_v.clone(),
            dsg: default_v.clone(),
            t: V::T::zero(),
            h: V::T::zero(),
            last_step: None,
        }
    }

    /// Make `diff` hold `nstages` stage increments of the state, forgetting the last step if it
    /// did not.
    pub(crate) fn set_nstages(&mut self, nstages: usize) {
        let nstates = self.y.len();
        if self.diff.nrows() != nstates
            || self.diff.ncols() != nstages
            || self.diff.context().nbatch() != self.y.context().nbatch()
        {
            self.diff = M::zeros(nstates, nstages, self.y.context().clone());
            self.last_step = None;
        }
    }
}

impl<V, M> OdeSolverState<V> for RkState<V, M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V, C = V::C>,
{
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        _ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        Ok(())
    }

    fn set_augmented_problem<Eqn: OdeEquations, AugmentedEqn: crate::AugmentedOdeEquations<Eqn>>(
        &mut self,
        _ode_problem: &OdeSolverProblem<Eqn>,
        _augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError> {
        Ok(())
    }

    fn new_from_common(state: StateCommon<V>) -> Self {
        let diff = M::zeros(0, 0, state.y.context().clone());
        Self {
            diff,
            y: state.y,
            dy: state.dy,
            g: state.g,
            dg: state.dg,
            s: state.s,
            ds: state.ds,
            sg: state.sg,
            dsg: state.dsg,
            t: state.t,
            h: state.h,
            last_step: None,
        }
    }

    fn into_common(self) -> StateCommon<V> {
        StateCommon {
            y: self.y,
            dy: self.dy,
            g: self.g,
            dg: self.dg,
            s: self.s,
            ds: self.ds,
            sg: self.sg,
            dsg: self.dsg,
            t: self.t,
            h: self.h,
        }
    }

    fn as_mut(&mut self) -> StateRefMut<'_, V> {
        // a modified state is no longer the end of the recorded step
        self.last_step = None;
        StateRefMut {
            y: &mut self.y,
            dy: &mut self.dy,
            g: &mut self.g,
            dg: &mut self.dg,
            s: &mut self.s,
            ds: &mut self.ds,
            sg: &mut self.sg,
            dsg: &mut self.dsg,
            t: &mut self.t,
            h: &mut self.h,
        }
    }

    fn as_ref(&self) -> StateRef<'_, V> {
        StateRef {
            y: &self.y,
            dy: &self.dy,
            g: &self.g,
            dg: &self.dg,
            s: &self.s,
            ds: &self.ds,
            sg: &self.sg,
            dsg: &self.dsg,
            t: self.t,
            h: self.h,
        }
    }

    /// Builds the new state directly, rather than cloning this one only to replace every vector
    /// and matrix, and keeps the step data.
    fn zeros_like_with_directions(&self, ndirections: usize) -> Result<Self, DiffsolError> {
        let v = |v: &V| zeros_with_directions(v, ndirections);
        Ok(Self {
            diff: zeros_mat_with_directions(&self.diff, ndirections)?,
            y: v(&self.y)?,
            dy: v(&self.dy)?,
            g: v(&self.g)?,
            dg: v(&self.dg)?,
            s: v(&self.s)?,
            ds: v(&self.ds)?,
            sg: v(&self.sg)?,
            dsg: v(&self.dsg)?,
            t: self.t,
            h: self.h,
            last_step: self.last_step,
        })
    }
}
