use crate::{
    error::DiffsolError, OdeEquations, OdeSolverProblem, OdeSolverState, StateRef, StateRefMut,
    Vector,
};

use super::state::StateCommon;

#[derive(Clone)]
pub struct RkState<V: Vector> {
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
}

impl<V> RkState<V> where V: Vector {}

impl<V> OdeSolverState<V> for RkState<V>
where
    V: Vector,
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
        Self {
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
}
