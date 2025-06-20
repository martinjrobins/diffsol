use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    scalar::IndexType,
    scale, AugmentedOdeEquations, DefaultDenseMatrix, DenseMatrix, OdeEquations, OdeSolverProblem,
    OdeSolverState, Op, StateRef, StateRefMut, Vector, VectorViewMut,
};
use std::ops::MulAssign;

use super::state::StateCommon;

#[derive(Clone)]
pub struct BdfState<V, M = <V as DefaultDenseMatrix>::M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V>,
{
    pub(crate) order: usize,
    pub(crate) diff: M,
    pub(crate) sdiff: Vec<M>,
    pub(crate) gdiff: M,
    pub(crate) sgdiff: Vec<M>,
    pub(crate) y: V,
    pub(crate) dy: V,
    pub(crate) g: V,
    pub(crate) dg: V,
    pub(crate) s: Vec<V>,
    pub(crate) ds: Vec<V>,
    pub(crate) sg: Vec<V>,
    pub(crate) dsg: Vec<V>,
    pub(crate) t: V::T,
    pub(crate) h: V::T,
    pub(crate) diff_initialised: bool,
    pub(crate) sdiff_initialised: bool,
    pub(crate) gdiff_initialised: bool,
    pub(crate) sgdiff_initialised: bool,
}

impl<V, M> BdfState<V, M>
where
    V: Vector + DefaultDenseMatrix,
    M: DenseMatrix<T = V::T, V = V>,
{
    pub(crate) const MAX_ORDER: IndexType = 5;

    pub fn initialise_diff_to_first_order(&mut self) {
        self.order = 1usize;
        self.diff.column_mut(0).copy_from(&self.y);
        self.diff.column_mut(1).copy_from(&self.dy);
        self.diff.column_mut(1).mul_assign(scale(self.h));
        self.diff_initialised = true;
    }

    pub fn initialise_sdiff_to_first_order(&mut self) {
        let naug = self.sdiff.len();
        for i in 0..naug {
            let sdiff = &mut self.sdiff[i];
            let s = &self.s[i];
            let ds = &self.ds[i];
            sdiff.column_mut(0).copy_from(s);
            sdiff.column_mut(1).copy_from(ds);
            sdiff.column_mut(1).mul_assign(scale(self.h));
        }
        self.sdiff_initialised = true;
    }

    pub fn initialise_gdiff_to_first_order(&mut self) {
        self.gdiff.column_mut(0).copy_from(&self.g);
        self.gdiff.column_mut(1).copy_from(&self.dg);
        self.gdiff.column_mut(1).mul_assign(scale(self.h));
        self.gdiff_initialised = true;
    }

    pub fn initialise_sgdiff_to_first_order(&mut self) {
        let naug = self.sgdiff.len();
        for i in 0..naug {
            let sgdiff = &mut self.sgdiff[i];
            let sg = &self.sg[i];
            let dsg = &self.dsg[i];
            sgdiff.column_mut(0).copy_from(sg);
            sgdiff.column_mut(1).copy_from(dsg);
            sgdiff.column_mut(1).mul_assign(scale(self.h));
        }
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
        let expected_gdiff_len = if let Some(out) = ode_problem.eqn.out() {
            if ode_problem.integrate_out {
                out.nout()
            } else {
                0
            }
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
        let naug = augmented_eqn.max_index();
        let nstates = ode_problem.eqn.rhs().nstates();
        if self.sdiff.len() != naug || self.sdiff[0].nrows() != nstates {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        let (sgdiff_len, sgdiff_size) = if let Some(_out) = augmented_eqn.out() {
            if let Some(out) = augmented_eqn.out() {
                (naug, out.nout())
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };
        if self.sgdiff.len() != sgdiff_len
            || (sgdiff_len > 0 && self.sgdiff[0].nrows() != sgdiff_size)
        {
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
        let diff = M::zeros(nstates, Self::MAX_ORDER + 3, ctx.clone());
        let sdiff = vec![M::zeros(nstates, Self::MAX_ORDER + 3, ctx.clone()); s.len()];
        let gdiff = M::zeros(g.len(), Self::MAX_ORDER + 3, ctx.clone());
        let sgdiff = if !sg.is_empty() {
            vec![M::zeros(sg[0].len(), Self::MAX_ORDER + 3, ctx.clone()); sg.len()]
        } else {
            Vec::new()
        };
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
