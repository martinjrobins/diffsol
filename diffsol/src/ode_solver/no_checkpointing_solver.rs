use std::{cell::Ref, marker::PhantomData};

use crate::{
    error::DiffsolError, OdeEquations, OdeSolverConfig, OdeSolverConfigMut, OdeSolverConfigRef,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason, StateRef, StateRefMut,
};

pub struct NoCheckpointingSolverConfig<T>(PhantomData<T>);

impl<T> Clone for NoCheckpointingSolverConfig<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for NoCheckpointingSolverConfig<T> {}

impl<T> Default for NoCheckpointingSolverConfig<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T> OdeSolverConfig<T> for NoCheckpointingSolverConfig<T> {
    fn as_base_ref(&self) -> OdeSolverConfigRef<'_, T> {
        panic!("checkpoint replay solver config is unavailable")
    }

    fn as_base_mut(&mut self) -> OdeSolverConfigMut<'_, T> {
        panic!("checkpoint replay solver config is unavailable")
    }
}

pub struct NoCheckpointingSolver<Eqn, State>(PhantomData<(Eqn, State)>)
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>;

impl<Eqn, State> Clone for NoCheckpointingSolver<Eqn, State>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
{
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<Eqn, State> Default for NoCheckpointingSolver<Eqn, State>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'a, Eqn, State> OdeSolverMethod<'a, Eqn> for NoCheckpointingSolver<Eqn, State>
where
    Eqn: OdeEquations + 'a,
    State: OdeSolverState<Eqn::V>,
{
    type State = State;
    type Config = NoCheckpointingSolverConfig<Eqn::T>;

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn checkpoint(&mut self) -> Self::State {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state_clone(&self) -> Self::State {
        panic!("checkpoint replay solver is unavailable")
    }

    fn set_state(&mut self, _state: Self::State) {
        panic!("checkpoint replay solver is unavailable")
    }

    fn into_state(self) -> Self::State {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state(&self) -> StateRef<'_, Eqn::V> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn config(&self) -> &Self::Config {
        panic!("checkpoint replay solver config is unavailable")
    }

    fn config_mut(&mut self) -> &mut Self::Config {
        panic!("checkpoint replay solver config is unavailable")
    }

    fn jacobian(&self) -> Option<Ref<'_, Eqn::M>> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn mass(&self) -> Option<Ref<'_, Eqn::M>> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn set_stop_time(&mut self, _tstop: Eqn::T) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_inplace(&self, _t: Eqn::T, _y: &mut Eqn::V) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_dy_inplace(&self, _t: Eqn::T, _dy: &mut Eqn::V) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_out_inplace(&self, _t: Eqn::T, _g: &mut Eqn::V) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_sens_inplace(
        &self,
        _t: Eqn::T,
        _sens: &mut [Eqn::V],
    ) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state_mut_back(&mut self, _t: Eqn::T) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn order(&self) -> usize {
        panic!("checkpoint replay solver is unavailable")
    }
}
