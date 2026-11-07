use diffsol::matrix::MatrixRef;
use diffsol::{
    AdjointEquations, AdjointOdeSolverMethod, Bdf, BdfState, CheckpointingPath, DefaultDenseMatrix,
    DefaultSolver, DiffsolError, ExplicitRk, LinearSolver, Matrix, OdeEquations,
    OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, RkState, Sdirk, SensEquations, SensitivitiesOdeSolverMethod, VectorRef,
};

use crate::scalar_type::Scalar;

pub(crate) trait OdeSolverMethodTag<M, Eqn>
where
    M: Matrix<T: Scalar> + DefaultSolver,
    Eqn: OdeEquations<M = M, T = M::T, V = M::V, C = M::C>,
{
    type State: OdeSolverState<M::V>;

    type OdeSolverMethod<'a, LS>: OdeSolverMethod<'a, Eqn, State = Self::State>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>: SensitivitiesOdeSolverMethod<'a, Eqn>
        + OdeSolverMethod<'a, Eqn, State = Self::State>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>;

    type AdjointOdeSolverMethod<'a, LS, S>: AdjointOdeSolverMethod<'a, Eqn, S>
        + OdeSolverMethod<'a, Eqn, State = Self::State>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: Self::State,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    fn uninitialised_solver<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        let state = Self::State::new_without_initialise(problem)?;
        Self::solver_with_state::<LS>(problem, state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: Self::State,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        checkpointing: CheckpointingPath<Eqn, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: Self::State,
        adjoint_eqn: AdjointEquations<'a, Eqn, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;
}

pub(crate) struct BdfTag;
pub(crate) struct Esdirk34Tag;
pub(crate) struct TrBdf2Tag;
pub(crate) struct Tsit45Tag;

impl<M, Eqn> OdeSolverMethodTag<M, Eqn> for BdfTag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = BdfState<M::V>;

    type OdeSolverMethod<'a, LS>
        = Bdf<'a, Eqn, LS>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = Bdf<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M, SensEquations<'a, Eqn>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = Bdf<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M, AdjointEquations<'a, Eqn, S>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf::<LS>()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: BdfState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf_solver::<LS>(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf_sens::<LS>()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: BdfState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf_solver_sens::<LS>(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        checkpointing: CheckpointingPath<Eqn, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.bdf_solver_adjoint::<LS, _>(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: BdfState<M::V>,
        adjoint_eqn: AdjointEquations<'a, Eqn, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
    }
}

impl<M, Eqn> OdeSolverMethodTag<M, Eqn> for Esdirk34Tag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = RkState<M::V>;

    type OdeSolverMethod<'a, LS>
        = Sdirk<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = Sdirk<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M, SensEquations<'a, Eqn>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = Sdirk<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M, AdjointEquations<'a, Eqn, S>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34::<LS>()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34_solver::<LS>(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34_sens::<LS>()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34_solver_sens::<LS>(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        checkpointing: CheckpointingPath<Eqn, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.esdirk34_solver_adjoint::<LS, _>(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
        adjoint_eqn: AdjointEquations<'a, Eqn, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.esdirk34_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
    }
}

impl<M, Eqn> OdeSolverMethodTag<M, Eqn> for TrBdf2Tag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = RkState<M::V>;

    type OdeSolverMethod<'a, LS>
        = Sdirk<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = Sdirk<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M, SensEquations<'a, Eqn>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = Sdirk<'a, Eqn, LS, <M::V as DefaultDenseMatrix>::M, AdjointEquations<'a, Eqn, S>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2::<LS>()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2_solver::<LS>(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2_sens::<LS>()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2_solver_sens::<LS>(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        checkpointing: CheckpointingPath<Eqn, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tr_bdf2_solver_adjoint::<LS, _>(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
        adjoint_eqn: AdjointEquations<'a, Eqn, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tr_bdf2_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
    }
}

impl<M, Eqn> OdeSolverMethodTag<M, Eqn> for Tsit45Tag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    Eqn: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = RkState<M::V>;

    type OdeSolverMethod<'a, LS>
        = ExplicitRk<'a, Eqn, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = ExplicitRk<'a, Eqn, <M::V as DefaultDenseMatrix>::M, SensEquations<'a, Eqn>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = ExplicitRk<'a, Eqn, <M::V as DefaultDenseMatrix>::M, AdjointEquations<'a, Eqn, S>>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45_solver(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45_sens()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45_solver_sens(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        checkpointing: CheckpointingPath<Eqn, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tsit45_solver_adjoint(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<M::V>,
        adjoint_eqn: AdjointEquations<'a, Eqn, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        Eqn: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tsit45_solver_adjoint_from_state(state, adjoint_eqn)
    }
}
