use diffsol::matrix::MatrixRef;
use diffsol::{
    AdjointEquations, AdjointOdeSolverMethod, Bdf, BdfState, CheckpointingPath, CodegenModule,
    DefaultDenseMatrix, DefaultSolver, DiffSl, DiffsolError, ExplicitRk, LinearSolver, Matrix,
    NewtonNonlinearSolver, NoLineSearch, OdeEquations,
    OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, RkState, Sdirk, SensEquations, SensitivitiesOdeSolverMethod, VectorRef,
};

use crate::scalar_type::Scalar;

pub(crate) trait OdeSolverMethodTag<M, CG>
where
    M: Matrix<T: Scalar> + DefaultSolver,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquations,
{
    type State: OdeSolverState<M::V>;

    type OdeSolverMethod<'a, LS>: OdeSolverMethod<'a, DiffSl<M, CG>, State = Self::State>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>: SensitivitiesOdeSolverMethod<'a, DiffSl<M, CG>>
        + OdeSolverMethod<'a, DiffSl<M, CG>, State = Self::State>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>;

    type AdjointOdeSolverMethod<'a, LS, S>: AdjointOdeSolverMethod<'a, DiffSl<M, CG>, S>
        + OdeSolverMethod<'a, DiffSl<M, CG>, State = Self::State>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: Self::State,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    fn uninitialised_solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquations<M = M, T = M::T, V = M::V, C = M::C>,
    {
        let state = Self::State::new_without_initialise(problem)?;
        Self::solver_with_state::<LS>(problem, state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: Self::State,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        checkpointing: CheckpointingPath<DiffSl<M, CG>, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: Self::State,
        adjoint_eqn: AdjointEquations<'a, DiffSl<M, CG>, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;
}

pub(crate) struct BdfTag;
pub(crate) struct Esdirk34Tag;
pub(crate) struct TrBdf2Tag;
pub(crate) struct Tsit45Tag;

impl<M, CG> OdeSolverMethodTag<M, CG> for BdfTag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = BdfState<M::V>;

    type OdeSolverMethod<'a, LS>
        = Bdf<'a, DiffSl<M, CG>, NewtonNonlinearSolver<M, LS, NoLineSearch>>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = Bdf<
        'a,
        DiffSl<M, CG>,
        NewtonNonlinearSolver<M, LS, NoLineSearch>,
        <M::V as DefaultDenseMatrix>::M,
        SensEquations<'a, DiffSl<M, CG>>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = Bdf<
        'a,
        DiffSl<M, CG>,
        NewtonNonlinearSolver<M, LS, NoLineSearch>,
        <M::V as DefaultDenseMatrix>::M,
        AdjointEquations<'a, DiffSl<M, CG>, S>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf::<LS>()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: BdfState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf_solver::<LS>(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf_sens::<LS>()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: BdfState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.bdf_solver_sens::<LS>(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        checkpointing: CheckpointingPath<DiffSl<M, CG>, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.bdf_solver_adjoint::<LS, _>(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: BdfState<M::V>,
        adjoint_eqn: AdjointEquations<'a, DiffSl<M, CG>, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
    }
}

impl<M, CG> OdeSolverMethodTag<M, CG> for Esdirk34Tag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = RkState<M::V>;

    type OdeSolverMethod<'a, LS>
        = Sdirk<'a, DiffSl<M, CG>, LS, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = Sdirk<
        'a,
        DiffSl<M, CG>,
        LS,
        <M::V as DefaultDenseMatrix>::M,
        SensEquations<'a, DiffSl<M, CG>>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = Sdirk<
        'a,
        DiffSl<M, CG>,
        LS,
        <M::V as DefaultDenseMatrix>::M,
        AdjointEquations<'a, DiffSl<M, CG>, S>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34::<LS>()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34_solver::<LS>(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34_sens::<LS>()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.esdirk34_solver_sens::<LS>(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        checkpointing: CheckpointingPath<DiffSl<M, CG>, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.esdirk34_solver_adjoint::<LS, _>(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
        adjoint_eqn: AdjointEquations<'a, DiffSl<M, CG>, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.esdirk34_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
    }
}

impl<M, CG> OdeSolverMethodTag<M, CG> for TrBdf2Tag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = RkState<M::V>;

    type OdeSolverMethod<'a, LS>
        = Sdirk<'a, DiffSl<M, CG>, LS, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = Sdirk<
        'a,
        DiffSl<M, CG>,
        LS,
        <M::V as DefaultDenseMatrix>::M,
        SensEquations<'a, DiffSl<M, CG>>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = Sdirk<
        'a,
        DiffSl<M, CG>,
        LS,
        <M::V as DefaultDenseMatrix>::M,
        AdjointEquations<'a, DiffSl<M, CG>, S>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2::<LS>()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2_solver::<LS>(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2_sens::<LS>()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tr_bdf2_solver_sens::<LS>(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        checkpointing: CheckpointingPath<DiffSl<M, CG>, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tr_bdf2_solver_adjoint::<LS, _>(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
        adjoint_eqn: AdjointEquations<'a, DiffSl<M, CG>, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tr_bdf2_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn)
    }
}

impl<M, CG> OdeSolverMethodTag<M, CG> for Tsit45Tag
where
    M: Matrix<T: Scalar> + DefaultSolver,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type State = RkState<M::V>;

    type OdeSolverMethod<'a, LS>
        = ExplicitRk<'a, DiffSl<M, CG>, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type SensOdeSolverMethod<'a, LS>
        = ExplicitRk<
        'a,
        DiffSl<M, CG>,
        <M::V as DefaultDenseMatrix>::M,
        SensEquations<'a, DiffSl<M, CG>>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    type AdjointOdeSolverMethod<'a, LS, S>
        = ExplicitRk<
        'a,
        DiffSl<M, CG>,
        <M::V as DefaultDenseMatrix>::M,
        AdjointEquations<'a, DiffSl<M, CG>, S>,
    >
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45()
    }

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45_solver(state)
    }

    fn solver_sens<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45_sens()
    }

    fn solver_sens_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
    ) -> Result<Self::SensOdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
    {
        problem.tsit45_solver_sens(state)
    }

    fn solver_adjoint<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        checkpointing: CheckpointingPath<DiffSl<M, CG>, S::State>,
        solver: Option<S>,
        nout_override: Option<usize>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tsit45_solver_adjoint(checkpointing, solver, nout_override)
    }

    fn solver_adjoint_from_state<'a, LS, S>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: RkState<M::V>,
        adjoint_eqn: AdjointEquations<'a, DiffSl<M, CG>, S>,
    ) -> Result<Self::AdjointOdeSolverMethod<'a, LS, S>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>,
        S: OdeSolverMethod<'a, DiffSl<M, CG>>,
        DiffSl<M, CG>: OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    {
        problem.tsit45_solver_adjoint_from_state(state, adjoint_eqn)
    }
}
