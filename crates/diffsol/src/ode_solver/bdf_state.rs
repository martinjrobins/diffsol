use crate::{
    error::DiffsolError, ode_solver_error, scalar::IndexType, AugmentedOdeEquations, Context,
    DefaultDenseMatrix, DenseMatrix, OdeEquations, OdeSolverProblem, OdeSolverState, Op, StateRef,
    StateRefMut, Vector, VectorViewMut,
};
use num_traits::Zero;

use super::state::StateCommon;

/// What a [crate::Bdf] step needs to replay itself for [crate::StepJvp] and [crate::StepVjp]:
/// the scalars that, with the end-of-step state, fix the step as a linear map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BdfStepRecord<T> {
    /// Step size of the history the step started from: `h_init` after a first-order
    /// initialisation, otherwise the step size the previous step ended with.
    pub(crate) h_ref: T,
    /// Step size of the accepted attempt.
    pub(crate) h: T,
    /// Order of the accepted attempt.
    pub(crate) order: usize,
    /// Whether the step started from a first-order initialisation of `diff` from `(y, dy)`.
    pub(crate) from_init: bool,
}

/// State container for the BDF integrator. For the common state API use `as_ref` and `as_mut` methods.
///
/// The history `diff` is the integrator's real state; after a step `y` and `dy` are readouts of
/// it: `y = diff[:, 0]`, and `dy` is `diff[:, 1] / h` of the accepted attempt, taken before any
/// step-size rescale at the end of the step.
#[derive(Clone)]
pub struct BdfState<V, M = <V as DefaultDenseMatrix>::M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V>,
{
    pub(crate) order: usize,
    pub(crate) diff: M,
    pub(crate) sdiff: M,
    pub(crate) gdiff: M,
    pub(crate) sgdiff: M,
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
    pub(crate) diff_initialised: bool,
    pub(crate) sdiff_initialised: bool,
    pub(crate) gdiff_initialised: bool,
    pub(crate) sgdiff_initialised: bool,
    /// The last step taken to reach this state, if any.
    pub(crate) last_step: Option<BdfStepRecord<V::T>>,
    /// The step size `diff` was last initialised to first order with, until a step runs from it.
    pub(crate) h_init: Option<V::T>,
    /// The step size when [crate::OdeSolverMethod::step] last returned. Unlike `h`, a later
    /// `set_stop_time` rescale does not change it.
    pub(crate) h_end: V::T,
}

/// Highest BDF order.
pub(crate) const MAX_ORDER: IndexType = 5;

impl<V, M> BdfState<V, M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V, C = V::C>,
{
    pub(crate) fn new_empty(ctx: V::C) -> Self {
        let default_v = V::zeros(0, ctx.clone());
        let default_m = M::zeros(0, 0, ctx.clone());
        Self {
            order: 1,
            diff: default_m.clone(),
            sdiff: default_m.clone(),
            gdiff: default_m.clone(),
            sgdiff: default_m.clone(),
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
            diff_initialised: false,
            sdiff_initialised: false,
            gdiff_initialised: false,
            sgdiff_initialised: false,
            last_step: None,
            h_init: None,
            h_end: V::T::zero(),
        }
    }

    pub fn initialise_diff_to_first_order(&mut self) {
        self.order = 1usize;
        self.h_init = Some(self.h);
        self.diff.column_mut(0).copy_from(&self.y);
        self.diff.column_mut(1).axpy(self.h, &self.dy, M::T::zero());
        self.diff_initialised = true;
    }

    pub fn initialise_sdiff_to_first_order(&mut self) {
        self.sdiff.column_mut(0).copy_from(&self.s);
        self.sdiff
            .column_mut(1)
            .axpy(self.h, &self.ds, M::T::zero());
        self.sdiff_initialised = true;
    }

    pub fn initialise_gdiff_to_first_order(&mut self) {
        self.gdiff.column_mut(0).copy_from(&self.g);
        self.gdiff
            .column_mut(1)
            .axpy(self.h, &self.dg, M::T::zero());
        self.gdiff_initialised = true;
    }

    pub fn initialise_sgdiff_to_first_order(&mut self) {
        self.sgdiff.column_mut(0).copy_from(&self.sg);
        self.sgdiff
            .column_mut(1)
            .axpy(self.h, &self.dsg, M::T::zero());
        self.sgdiff_initialised = true;
    }
}

impl<V, M> OdeSolverState<V> for BdfState<V, M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V, C = V::C>,
{
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        let nstates = ode_problem.eqn.rhs().nstates();
        if self.diff.nrows() != nstates {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        let expected_gdiff_len = if ode_problem.integrate_out {
            ode_problem.eqn.nout()
        } else {
            0
        };
        if self.gdiff.nrows() != expected_gdiff_len {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.diff_initialised {
            self.initialise_diff_to_first_order();
        }
        if !self.gdiff_initialised {
            self.initialise_gdiff_to_first_order();
        }
        Ok(())
    }

    fn set_augmented_problem<Eqn: OdeEquations, AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError> {
        let nlanes = augmented_eqn.aug_context().nbatch();
        let nstates = ode_problem.eqn.rhs().nstates();
        if self.sdiff.nrows() != nstates || self.sdiff.context().nbatch() != nlanes {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        let sgdiff_size = augmented_eqn.out().map(|out| out.nout()).unwrap_or(0);
        if self.sgdiff.nrows() != sgdiff_size || self.sgdiff.context().nbatch() != nlanes {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.sdiff_initialised {
            self.initialise_sdiff_to_first_order();
        }
        if !self.sgdiff_initialised {
            self.initialise_sgdiff_to_first_order();
        }
        Ok(())
    }

    fn new_from_common(state: super::state::StateCommon<V>) -> Self {
        let StateCommon {
            y,
            dy,
            g,
            dg,
            s,
            ds,
            sg,
            dsg,
            t,
            h,
        } = state;
        let nstates = y.len();
        let ctx = y.context();
        let diff = M::zeros(nstates, MAX_ORDER + 3, ctx.clone());
        // the augmented differences carry the augmented batch count of the state they track
        let sdiff = M::zeros(s.len(), MAX_ORDER + 3, s.context().clone());
        let gdiff = M::zeros(g.len(), MAX_ORDER + 3, ctx.clone());
        let sgdiff = M::zeros(sg.len(), MAX_ORDER + 3, sg.context().clone());
        Self {
            order: 1,
            diff,
            sdiff,
            gdiff,
            sgdiff,
            y,
            dy,
            g,
            dg,
            s,
            ds,
            sg,
            dsg,
            t,
            h,
            diff_initialised: false,
            sdiff_initialised: false,
            gdiff_initialised: false,
            sgdiff_initialised: false,
            last_step: None,
            h_init: None,
            h_end: h,
        }
    }

    fn zeros_like(&self) -> Self {
        let zeros = |m: &M| M::zeros(m.nrows(), m.ncols(), m.context().clone());
        let mut z = self.clone();
        z.diff = zeros(&self.diff);
        z.sdiff = zeros(&self.sdiff);
        z.gdiff = zeros(&self.gdiff);
        z.sgdiff = zeros(&self.sgdiff);
        for v in [
            &mut z.y, &mut z.dy, &mut z.g, &mut z.dg, &mut z.s, &mut z.ds, &mut z.sg, &mut z.dsg,
        ] {
            v.fill(V::T::zero());
        }
        z
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

    fn as_mut(&mut self) -> StateRefMut<'_, V> {
        self.diff_initialised = false;
        self.sdiff_initialised = false;
        self.gdiff_initialised = false;
        self.sgdiff_initialised = false;
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
}
