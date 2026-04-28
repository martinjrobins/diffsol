use diffsol::matrix::MatrixRef;
use diffsol::{
    Bdf, BdfState, CodegenModule, DefaultDenseMatrix, DiffSl, DiffsolError, ExplicitRk,
    LinearSolver, Matrix, NewtonNonlinearSolver, NoLineSearch, OdeEquations, OdeEquationsImplicit,
    OdeSolverMethod, OdeSolverProblem, RkState, Sdirk, VectorRef,
};

use crate::scalar_type::Scalar;

pub(crate) trait OdeSolverMethodTag<M, CG>
where
    M: Matrix<T: Scalar>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquations,
{
    type OdeSolverMethod<'a, LS>: OdeSolverMethod<'a, DiffSl<M, CG>>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    fn solver<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

    fn solver_with_state<'a, LS>(
        problem: &'a OdeSolverProblem<DiffSl<M, CG>>,
        state: <Self::OdeSolverMethod<'a, LS> as OdeSolverMethod<'a, DiffSl<M, CG>>>::State,
    ) -> Result<Self::OdeSolverMethod<'a, LS>, DiffsolError>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;
}

pub(crate) struct BdfTag;
pub(crate) struct Esdirk34Tag;
pub(crate) struct TrBdf2Tag;
pub(crate) struct Tsit45Tag;

impl<M, CG> OdeSolverMethodTag<M, CG> for BdfTag
where
    M: Matrix<T: Scalar>,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type OdeSolverMethod<'a, LS>
        = Bdf<'a, DiffSl<M, CG>, NewtonNonlinearSolver<M, LS, NoLineSearch>>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

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
}

impl<M, CG> OdeSolverMethodTag<M, CG> for Esdirk34Tag
where
    M: Matrix<T: Scalar>,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type OdeSolverMethod<'a, LS>
        = Sdirk<'a, DiffSl<M, CG>, LS, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

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
}

impl<M, CG> OdeSolverMethodTag<M, CG> for TrBdf2Tag
where
    M: Matrix<T: Scalar>,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquationsImplicit<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type OdeSolverMethod<'a, LS>
        = Sdirk<'a, DiffSl<M, CG>, LS, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

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
}

impl<M, CG> OdeSolverMethodTag<M, CG> for Tsit45Tag
where
    M: Matrix<T: Scalar>,
    M::V: DefaultDenseMatrix<T = M::T, C = M::C>,
    CG: CodegenModule,
    DiffSl<M, CG>: OdeEquations<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    type OdeSolverMethod<'a, LS>
        = ExplicitRk<'a, DiffSl<M, CG>, <M::V as DefaultDenseMatrix>::M>
    where
        M: 'a,
        CG: 'a,
        LS: LinearSolver<M>;

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
}
