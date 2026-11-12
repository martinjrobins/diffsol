use std::{
    any::Any,
    sync::{Arc, Mutex, MutexGuard},
};

use diffsol::{DefaultSolver, DenseVjpTape, DiffsolError, OdeEquationsImplicit};

use crate::{
    error::DiffsolRtError, linear_solver_type::LinearSolverType,
    ode_solver_tag::OdeSolverMethodTag, ode_solver_type::OdeSolverType, scalar_type::Scalar,
};

pub(crate) trait VjpCheckpoint: Any + Send {
    fn as_any(&self) -> &dyn Any;
    fn method(&self) -> OdeSolverType;
    fn linear_solver(&self) -> LinearSolverType;
    /// The number of times of the forward solve.
    fn nt(&self) -> usize;
}

impl dyn VjpCheckpoint + '_ {
    pub(crate) fn data<M, Eqn, Tag>(&self) -> Result<&VjpCheckpointData<M, Eqn, Tag>, DiffsolError>
    where
        M: diffsol::Matrix<T: Scalar> + DefaultSolver + 'static,
        Eqn: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C> + 'static,
        Tag: OdeSolverMethodTag<M, Eqn> + 'static,
    {
        self.as_any()
            .downcast_ref::<VjpCheckpointData<M, Eqn, Tag>>()
            .ok_or_else(|| {
                DiffsolError::Other(
                    "VJP checkpoint is incompatible with this ODE solver".to_string(),
                )
            })
    }
}

/// The data of a forward pass of [crate::OdeWrapper::solve_vjp_fwd]: the tape of the solve,
/// the parameters of all the lanes, and the solver that made the tape, which must also do the
/// backward pass.
pub(crate) struct VjpCheckpointData<M, Eqn, Tag>
where
    M: diffsol::Matrix<T: Scalar> + DefaultSolver,
    Eqn: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C>,
    Tag: OdeSolverMethodTag<M, Eqn>,
{
    pub(crate) tape: DenseVjpTape<M::V, Tag::State>,
    pub(crate) params: M::V,
    method: OdeSolverType,
    linear_solver: LinearSolverType,
    _eqn: std::marker::PhantomData<fn() -> Eqn>,
}

impl<M, Eqn, Tag> VjpCheckpointData<M, Eqn, Tag>
where
    M: diffsol::Matrix<T: Scalar> + DefaultSolver,
    Eqn: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C>,
    Tag: OdeSolverMethodTag<M, Eqn>,
{
    pub(crate) fn new(
        tape: DenseVjpTape<M::V, Tag::State>,
        params: M::V,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
    ) -> Self {
        Self {
            tape,
            params,
            method,
            linear_solver,
            _eqn: std::marker::PhantomData,
        }
    }
}

impl<M, Eqn, Tag> VjpCheckpoint for VjpCheckpointData<M, Eqn, Tag>
where
    M: diffsol::Matrix<T: Scalar> + DefaultSolver + 'static,
    M::V: Send,
    Tag::State: Send,
    Eqn: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C> + 'static,
    Tag: OdeSolverMethodTag<M, Eqn> + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn method(&self) -> OdeSolverType {
        self.method
    }

    fn linear_solver(&self) -> LinearSolverType {
        self.linear_solver
    }

    fn nt(&self) -> usize {
        self.tape.t_eval().len()
    }
}

/// The checkpoint of a forward pass of [crate::OdeWrapper::solve_vjp_fwd], for one or more
/// backward passes with [crate::OdeWrapper::solve_vjp_bkwd]. It can be shared across
/// threads.
#[derive(Clone)]
pub struct VjpCheckpointWrapper(Arc<Mutex<Box<dyn VjpCheckpoint>>>);

impl VjpCheckpointWrapper {
    pub(crate) fn new(checkpoint: Box<dyn VjpCheckpoint>) -> Self {
        Self(Arc::new(Mutex::new(checkpoint)))
    }

    pub(crate) fn guard(&self) -> Result<MutexGuard<'_, Box<dyn VjpCheckpoint>>, DiffsolRtError> {
        self.0
            .lock()
            .map_err(|_| DiffsolError::Other("VJP checkpoint mutex poisoned".to_string()).into())
    }
}
