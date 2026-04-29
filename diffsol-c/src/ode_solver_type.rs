// Solver method Python enum. This is used to select the overarching solver
// stragegy like bdf or esdirk34 in diffsol.

use diffsol::error::{DiffsolError, OdeSolverError};
use diffsol::ode_equations::OdeEquationsImplicitSens;
use diffsol::{
    matrix::MatrixRef, DefaultDenseMatrix, DenseMatrix, DiffSl, LinearSolver, Matrix,
    OdeSolverProblem, OdeSolverState, VectorHost, VectorRef, VectorView,
};
use diffsol::{
    ode_solver_error, AdjointOdeSolverMethod, CheckpointingPath, CodegenModule, DefaultSolver,
    OdeEquations, OdeSolverMethod, OdeSolverStopReason, SensitivitiesOdeSolverMethod, Solution,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::adjoint_checkpoint::{AdjointCheckpoint, AdjointCheckpointData};
use crate::ode_solver_tag::{BdfTag, Esdirk34Tag, OdeSolverMethodTag, TrBdf2Tag, Tsit45Tag};
use crate::scalar_type::Scalar;
use crate::utils::is_sens_available;
use crate::{
    linear_solver_type::LinearSolverType,
    valid_linear_solver::{KluValidator, LuValidator},
};

/// Enumerates the possible ODE solver methods for diffsol. See the solver descriptions in the diffsol documentation (https://github.com/martinjrobins/diffsol) for more details.
///
/// :attr bdf: Backward Differentiation Formula (BDF) method for stiff ODEs and singular mass matrices
/// :attr esdirk34: Explicit Singly Diagonally Implicit Runge-Kutta (ESDIRK) method for moderately stiff ODEs and singular mass matrices.
/// :attr tr_bdf2: Trapezoidal Backward Differentiation Formula of order 2 (TR-BDF2) method for moderately stiff ODEs and singular mass matrices.
/// :attr tsit45: Tsitouras 4/5th order Explicit Runge-Kutta (TSIT45) method for non-stiff ODEs. This is an explicit method, it cannot handle singular mass matrices and does not require a linear solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OdeSolverType {
    Bdf,
    Esdirk34,
    TrBdf2,
    Tsit45,
}

fn solve_with_tag<M, CG, LS, Tag>(
    problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
    mut soln: Solution<M::V>,
) -> Result<Solution<M::V>, DiffsolError>
where
    M: Matrix<T: Scalar>,
    CG: CodegenModule,
    M::V: VectorHost + DefaultDenseMatrix,
    LS: LinearSolver<M>,
    Tag: OdeSolverMethodTag<M, CG>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    let mut solver = Tag::solver::<LS>(problem)?;
    while !soln.is_complete() {
        solver = solver.solve_soln(&mut soln)?;
        let root_idx = match soln.stop_reason {
            Some(OdeSolverStopReason::RootFound(_, root_idx)) if !soln.is_complete() => root_idx,
            _ => continue,
        };
        if problem.eqn.reset().is_none() {
            soln.truncate(problem, solver.state())?;
            return Ok(soln);
        }
        let mut state = solver.into_state();
        problem.eqn.set_model_index(root_idx);
        state.as_mut().apply_reset(&problem.eqn)?;
        solver = Tag::solver_with_state::<LS>(problem, state)?;
    }
    Ok(soln)
}

fn solve_fwd_sens_with_tag<M, CG, LS, Tag>(
    problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
    t_eval: &[M::T],
) -> Result<Solution<M::V>, DiffsolError>
where
    M: Matrix<T: Scalar>,
    CG: CodegenModule,
    M::V: VectorHost + DefaultDenseMatrix,
    LS: LinearSolver<M>,
    Tag: OdeSolverMethodTag<M, CG>,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    let mut soln = Solution::new_dense(t_eval.to_vec())?;
    let mut solver = Tag::solver_sens::<LS>(problem)?;
    while !soln.is_complete() {
        solver = solver.solve_soln_sensitivities(&mut soln)?;
        let root_idx = match soln.stop_reason {
            Some(OdeSolverStopReason::RootFound(_, root_idx)) if !soln.is_complete() => root_idx,
            _ => continue,
        };
        if problem.eqn.reset().is_none() {
            soln.truncate_sens(problem, solver.state())?;
            return Ok(soln);
        }
        let mut state = solver.into_state();
        problem.eqn.set_model_index(root_idx);
        state
            .as_mut()
            .apply_reset_with_sens(&problem.eqn, root_idx)?;
        solver = Tag::solver_sens_with_state::<LS>(problem, state)?;
    }
    Ok(soln)
}

#[allow(clippy::type_complexity)]
fn solve_with_checkpointing_with_tag<M, CG, LS, Tag>(
    problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
    mut soln: Solution<M::V>,
) -> Result<(Solution<M::V>, CheckpointingPath<DiffSl<M, CG>, Tag::State>), DiffsolError>
where
    M: Matrix<T: Scalar>,
    CG: CodegenModule,
    M::V: VectorHost + DefaultDenseMatrix,
    LS: LinearSolver<M>,
    Tag: OdeSolverMethodTag<M, CG>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    let mut solver = Tag::solver::<LS>(problem)?;
    let mut checkpointing = Vec::new();
    while !soln.is_complete() {
        solver = solver.solve_soln_with_checkpointing(&mut soln, &mut checkpointing, None)?;
        let root_idx = match soln.stop_reason {
            Some(OdeSolverStopReason::RootFound(_, root_idx)) if !soln.is_complete() => root_idx,
            _ => continue,
        };
        if problem.eqn.reset().is_none() {
            soln.truncate(problem, solver.state())?;
            return Ok((soln, checkpointing));
        }
        let mut state = solver.into_state();
        problem.eqn.set_model_index(root_idx);
        state.as_mut().apply_reset(&problem.eqn)?;
        solver = Tag::solver_with_state::<LS>(problem, state)?;
    }
    Ok((soln, checkpointing))
}

fn integral_from_soln<V>(soln: &Solution<V>) -> Result<V, DiffsolError>
where
    V: DefaultDenseMatrix,
{
    if soln.ts.is_empty() {
        return Err(ode_solver_error!(
            Other,
            "Continuous adjoint solve returned no integral samples"
        ));
    }
    Ok(soln.ys.column(soln.ts.len() - 1).into_owned())
}

#[allow(clippy::type_complexity)]
fn solve_adjoint_fwd_with_tag<M, CG, LS, Tag>(
    problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
    t_eval: &[M::T],
    params: &[f64],
    method: OdeSolverType,
    linear_solver: LinearSolverType,
) -> Result<(Solution<M::V>, Box<dyn AdjointCheckpoint>), DiffsolError>
where
    M: Matrix<T: Scalar> + 'static,
    CG: CodegenModule + 'static,
    M::V: VectorHost + DefaultDenseMatrix,
    LS: LinearSolver<M>,
    DiffSl<M, CG>: OdeEquations<M = M, T = M::T, V = M::V, C = M::C>,
    Tag: OdeSolverMethodTag<M, CG> + 'static,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    let soln = Solution::new_dense(t_eval.to_vec())?;
    let (soln, checkpointing) = solve_with_checkpointing_with_tag::<M, CG, LS, Tag>(problem, soln)?;
    Ok((
        soln,
        Box::new(AdjointCheckpointData::<M, CG, Tag>::new(
            checkpointing,
            params.to_vec(),
            method,
            linear_solver,
        )),
    ))
}

fn solve_continuous_adjoint_with_tag<M, CG, LS, Tag>(
    problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
    final_time: M::T,
    method: OdeSolverType,
) -> Result<(M::V, Vec<M::V>), DiffsolError>
where
    M: Matrix<T: Scalar> + DefaultSolver + 'static,
    CG: CodegenModule + 'static,
    M::V: VectorHost + DefaultDenseMatrix,
    LS: LinearSolver<M>,
    Tag: OdeSolverMethodTag<M, CG> + 'static,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>
        + diffsol::OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    let soln = Solution::new(final_time);
    let (soln, checkpointing) = solve_with_checkpointing_with_tag::<M, CG, LS, Tag>(problem, soln)?;
    let integral = integral_from_soln(&soln)?;
    let sg = match method {
        OdeSolverType::Bdf => solve_adjoint_bkwds_with_fwd_bkwd_tag::<M, CG, LS, LS, Tag, BdfTag>(
            problem,
            &soln,
            checkpointing,
            &[],
            None,
        ),
        OdeSolverType::Esdirk34 => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            LS,
            LS,
            Tag,
            Esdirk34Tag,
        >(problem, &soln, checkpointing, &[], None),
        OdeSolverType::TrBdf2 => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            LS,
            LS,
            Tag,
            TrBdf2Tag,
        >(problem, &soln, checkpointing, &[], None),
        OdeSolverType::Tsit45 => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            LS,
            LS,
            Tag,
            Tsit45Tag,
        >(problem, &soln, checkpointing, &[], None),
    }?;
    Ok((integral, sg))
}

fn solve_adjoint_bkwds_with_fwd_tag<M, CG, FwdLS, BwdLS, Tag>(
    problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
    checkpoint: &AdjointCheckpointData<M, CG, Tag>,
    backwards_method: OdeSolverType,
    dgdu_eval: &<M::V as DefaultDenseMatrix>::M,
    t_eval: &[M::T],
) -> Result<Vec<M::V>, DiffsolError>
where
    M: Matrix<T: Scalar> + DefaultSolver + 'static,
    CG: CodegenModule + 'static,
    M::V: VectorHost + DefaultDenseMatrix,
    FwdLS: LinearSolver<M>,
    BwdLS: LinearSolver<M>,
    Tag: OdeSolverMethodTag<M, CG> + 'static,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>
        + diffsol::OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    // TODO: can we avoid cloning here? Adjoint equations require ownership of the checkpointing segments so maybe not
    // unless we can change the adjoint equations to take references to the checkpointing segments instead
    let checkpointing = checkpoint.checkpointing.clone();
    let soln = Solution::new_dense(t_eval.to_vec())?;

    // we will only consider a single output g for now, so nout_override is 1
    let dgdu_eval = [dgdu_eval];
    match backwards_method {
        OdeSolverType::Bdf => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            FwdLS,
            BwdLS,
            Tag,
            BdfTag,
        >(problem, &soln, checkpointing, &dgdu_eval, Some(1)),
        OdeSolverType::Esdirk34 => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            FwdLS,
            BwdLS,
            Tag,
            Esdirk34Tag,
        >(problem, &soln, checkpointing, &dgdu_eval, Some(1)),
        OdeSolverType::TrBdf2 => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            FwdLS,
            BwdLS,
            Tag,
            TrBdf2Tag,
        >(problem, &soln, checkpointing, &dgdu_eval, Some(1)),
        OdeSolverType::Tsit45 => solve_adjoint_bkwds_with_fwd_bkwd_tag::<
            M,
            CG,
            FwdLS,
            BwdLS,
            Tag,
            Tsit45Tag,
        >(problem, &soln, checkpointing, &dgdu_eval, Some(1)),
    }
}

fn solve_adjoint_bkwds_with_fwd_bkwd_tag<'solver, M, CG, FwdLS, BwdLS, FwdTag, BwdTag>(
    problem: &'solver mut OdeSolverProblem<DiffSl<M, CG>>,
    soln: &Solution<M::V>,
    mut checkpointing: CheckpointingPath<DiffSl<M, CG>, FwdTag::State>,
    dgdu_eval: &[&<M::V as DefaultDenseMatrix>::M],
    nout_override: Option<usize>,
) -> Result<Vec<M::V>, DiffsolError>
where
    M: Matrix<T: Scalar> + DefaultSolver + 'solver,
    CG: CodegenModule + 'solver,
    M::V: VectorHost + DefaultDenseMatrix,
    FwdLS: LinearSolver<M>,
    BwdLS: LinearSolver<M>,
    FwdTag: OdeSolverMethodTag<M, CG>,
    BwdTag: OdeSolverMethodTag<M, CG>,
    DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>
        + diffsol::OdeEquationsImplicitAdjoint<M = M, T = M::T, V = M::V, C = M::C>,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    let checkpointing_len = checkpointing.len();
    if checkpointing_len == 0 {
        return Err(ode_solver_error!(
            Other,
            "Adjoint backward pass requires at least one checkpointing segment"
        ));
    }

    let t_eval = if dgdu_eval.is_empty() {
        &[]
    } else {
        soln.ts.as_slice()
    };

    let current_checkpointing = checkpointing
        .pop()
        .ok_or_else(|| ode_solver_error!(Other, "Adjoint backward pass returned no state"))?;
    let model_index = checkpointing
        .last()
        .map(|segment| {
            segment
                .terminal_reset_root_idx()
                .expect("Missing reset root index")
        })
        .unwrap_or(0);
    problem.eqn_mut().set_model_index(model_index);
    let fwd_solver = FwdTag::uninitialised_solver::<FwdLS>(&*problem)?;
    let mut adjoint = BwdTag::solver_adjoint::<BwdLS, _>(
        &*problem,
        vec![current_checkpointing],
        Some(fwd_solver),
        nout_override,
    )?;
    loop {
        let (mut state, adjoint_checkpointing) =
            adjoint.solve_adjoint_backwards_pass(t_eval, dgdu_eval)?;
        let Some(previous_checkpointing) = checkpointing.pop() else {
            return Ok(state.into_common().sg);
        };
        let model_index = checkpointing
            .last()
            .map(|segment| {
                segment
                    .terminal_reset_root_idx()
                    .expect("Missing reset root index")
            })
            .unwrap_or(0);
        let fwd_state_minus = previous_checkpointing.last_checkpoint();
        let fwd_state_plus = adjoint_checkpointing
            .first()
            .ok_or_else(|| {
                ode_solver_error!(Other, "Adjoint backward pass returned no checkpointing")
            })?
            .first_checkpoint();
        state.as_mut().apply_reset_with_adjoint(
            problem.eqn(),
            previous_checkpointing.terminal_reset_root_idx().unwrap(),
            fwd_state_minus.as_ref(),
            fwd_state_plus.as_ref(),
            problem.integrate_out,
        )?;
        problem.eqn_mut().set_model_index(model_index);
        let fwd_solver = FwdTag::uninitialised_solver::<FwdLS>(&*problem)?;
        // TODO: remove clone here
        let adjoint_eqn = problem.adjoint_equations(
            vec![previous_checkpointing],
            Some(fwd_solver),
            nout_override,
        );

        adjoint = BwdTag::solver_adjoint_from_state::<BwdLS, _>(&*problem, state, adjoint_eqn)?;
    }
}

impl OdeSolverType {
    pub(crate) fn solve<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        final_time: M::T,
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                solve_with_tag::<M, CG, LS, BdfTag>(problem, Solution::new(final_time))
            }
            OdeSolverType::Esdirk34 => {
                solve_with_tag::<M, CG, LS, Esdirk34Tag>(problem, Solution::new(final_time))
            }
            OdeSolverType::TrBdf2 => {
                solve_with_tag::<M, CG, LS, TrBdf2Tag>(problem, Solution::new(final_time))
            }
            OdeSolverType::Tsit45 => {
                solve_with_tag::<M, CG, LS, Tsit45Tag>(problem, Solution::new(final_time))
            }
        }
    }

    pub(crate) fn solve_dense<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                solve_with_tag::<M, CG, LS, BdfTag>(problem, Solution::new_dense(t_eval.to_vec())?)
            }
            OdeSolverType::Esdirk34 => solve_with_tag::<M, CG, LS, Esdirk34Tag>(
                problem,
                Solution::new_dense(t_eval.to_vec())?,
            ),
            OdeSolverType::TrBdf2 => solve_with_tag::<M, CG, LS, TrBdf2Tag>(
                problem,
                Solution::new_dense(t_eval.to_vec())?,
            ),
            OdeSolverType::Tsit45 => solve_with_tag::<M, CG, LS, Tsit45Tag>(
                problem,
                Solution::new_dense(t_eval.to_vec())?,
            ),
        }
    }

    fn check_sens_available() -> Result<(), DiffsolError> {
        if !is_sens_available() {
            return Err(DiffsolError::Other(
                "Sensitivity analysis is not supported on Windows, please use a linux or macOS system.".to_string(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_fwd_sens<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => solve_fwd_sens_with_tag::<M, CG, LS, BdfTag>(problem, t_eval),
            OdeSolverType::Esdirk34 => {
                solve_fwd_sens_with_tag::<M, CG, LS, Esdirk34Tag>(problem, t_eval)
            }
            OdeSolverType::TrBdf2 => {
                solve_fwd_sens_with_tag::<M, CG, LS, TrBdf2Tag>(problem, t_eval)
            }
            OdeSolverType::Tsit45 => {
                solve_fwd_sens_with_tag::<M, CG, LS, Tsit45Tag>(problem, t_eval)
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_adjoint_fwd<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
        params: &[f64],
        linear_solver: LinearSolverType,
    ) -> Result<(Solution<M::V>, Box<dyn AdjointCheckpoint>), DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + 'static,
        CG: CodegenModule + 'static,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => solve_adjoint_fwd_with_tag::<M, CG, LS, BdfTag>(
                problem,
                t_eval,
                params,
                *self,
                linear_solver,
            ),
            OdeSolverType::Esdirk34 => solve_adjoint_fwd_with_tag::<M, CG, LS, Esdirk34Tag>(
                problem,
                t_eval,
                params,
                *self,
                linear_solver,
            ),
            OdeSolverType::TrBdf2 => solve_adjoint_fwd_with_tag::<M, CG, LS, TrBdf2Tag>(
                problem,
                t_eval,
                params,
                *self,
                linear_solver,
            ),
            OdeSolverType::Tsit45 => solve_adjoint_fwd_with_tag::<M, CG, LS, Tsit45Tag>(
                problem,
                t_eval,
                params,
                *self,
                linear_solver,
            ),
        }
    }

    pub(crate) fn solve_continuous_adjoint<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        final_time: M::T,
    ) -> Result<(M::V, Vec<M::V>), DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + 'static,
        CG: CodegenModule + 'static,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>
            + diffsol::OdeEquationsImplicitAdjoint,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => {
                solve_continuous_adjoint_with_tag::<M, CG, LS, BdfTag>(problem, final_time, *self)
            }
            OdeSolverType::Esdirk34 => solve_continuous_adjoint_with_tag::<M, CG, LS, Esdirk34Tag>(
                problem, final_time, *self,
            ),
            OdeSolverType::TrBdf2 => solve_continuous_adjoint_with_tag::<M, CG, LS, TrBdf2Tag>(
                problem, final_time, *self,
            ),
            OdeSolverType::Tsit45 => solve_continuous_adjoint_with_tag::<M, CG, LS, Tsit45Tag>(
                problem, final_time, *self,
            ),
        }
    }

    pub(crate) fn solve_adjoint_bkwd<M, CG, BwdLS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        checkpoint: &dyn AdjointCheckpoint,
        dgdu_eval: &<M::V as DefaultDenseMatrix>::M,
        t_eval: &[M::T],
    ) -> Result<Vec<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + LuValidator<M> + KluValidator<M> + 'static,
        CG: CodegenModule + 'static,
        M::V: VectorHost + DefaultDenseMatrix,
        BwdLS: LinearSolver<M>,
        DiffSl<M, CG>: OdeEquationsImplicitSens<M = M, T = M::T, V = M::V, C = M::C>
            + diffsol::OdeEquationsImplicitAdjoint,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match checkpoint.method() {
            OdeSolverType::Bdf => {
                let data = checkpoint.data::<M, CG, BdfTag>()?;
                match data.linear_solver() {
                    LinearSolverType::Default => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as DefaultSolver>::LS,
                            BwdLS,
                            BdfTag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Lu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as LuValidator<M>>::LS,
                            BwdLS,
                            BdfTag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Klu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as KluValidator<M>>::LS,
                            BwdLS,
                            BdfTag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                }
            }
            OdeSolverType::Esdirk34 => {
                let data = checkpoint.data::<M, CG, Esdirk34Tag>()?;
                match data.linear_solver() {
                    LinearSolverType::Default => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as DefaultSolver>::LS,
                            BwdLS,
                            Esdirk34Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Lu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as LuValidator<M>>::LS,
                            BwdLS,
                            Esdirk34Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Klu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as KluValidator<M>>::LS,
                            BwdLS,
                            Esdirk34Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                }
            }
            OdeSolverType::TrBdf2 => {
                let data = checkpoint.data::<M, CG, TrBdf2Tag>()?;
                match data.linear_solver() {
                    LinearSolverType::Default => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as DefaultSolver>::LS,
                            BwdLS,
                            TrBdf2Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Lu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as LuValidator<M>>::LS,
                            BwdLS,
                            TrBdf2Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Klu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as KluValidator<M>>::LS,
                            BwdLS,
                            TrBdf2Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                }
            }
            OdeSolverType::Tsit45 => {
                let data = checkpoint.data::<M, CG, Tsit45Tag>()?;
                match data.linear_solver() {
                    LinearSolverType::Default => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as DefaultSolver>::LS,
                            BwdLS,
                            Tsit45Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Lu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as LuValidator<M>>::LS,
                            BwdLS,
                            Tsit45Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                    LinearSolverType::Klu => {
                        solve_adjoint_bkwds_with_fwd_tag::<
                            M,
                            CG,
                            <M as KluValidator<M>>::LS,
                            BwdLS,
                            Tsit45Tag,
                        >(problem, data, *self, dgdu_eval, t_eval)
                    }
                }
            }
        }
    }
}

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod tests {
    use diffsol::{
        CodegenModuleCompile, CodegenModuleJit, DefaultDenseMatrix, DefaultSolver, DenseMatrix,
        Matrix, MatrixCommon, OdeBuilder, OdeSolverProblem, Op, Vector,
    };

    #[cfg(feature = "diffsl-llvm")]
    use crate::linear_solver_type::LinearSolverType;
    use crate::test_support::{
        assert_close, hybrid_logistic_diffsl_code, hybrid_logistic_state, logistic_diffsl_code,
        logistic_state, LOGISTIC_X0,
    };
    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::{hybrid_logistic_state_dr, logistic_state_dr};
    use crate::valid_linear_solver::LuValidator;

    use super::OdeSolverType;

    type M = diffsol::NalgebraMat<f64>;

    fn build_problem<CG>(code: &str) -> OdeSolverProblem<diffsol::DiffSl<M, CG>>
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        OdeBuilder::<M>::new()
            .p([2.0])
            .rtol(1e-6)
            .atol([1e-6])
            .build_from_diffsl::<CG>(code)
            .unwrap()
    }

    fn assert_dense_solution_matches_expected(
        soln: &diffsol::Solution<diffsol::NalgebraVec<f64>>,
        t_eval: &[f64],
        expected: impl Fn(f64) -> f64,
    ) {
        assert_eq!(soln.ts, t_eval);
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                soln.ys.get_index(0, i),
                expected(t),
                5e-4,
                &format!("solution[{i}]"),
            );
        }
    }

    fn test_all_solver_variants<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve::<M, CG, <M as DefaultSolver>::LS>(&mut problem, 1.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 1.0, 5e-4, "solve final time");
            assert_close(
                soln.ys.get_index(0, soln.ts.len() - 1),
                logistic_state(LOGISTIC_X0, 2.0, 1.0),
                5e-4,
                "solve final value",
            );

            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve_dense::<M, CG, <M as DefaultSolver>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                logistic_state(LOGISTIC_X0, 2.0, t)
            });
        }
    }

    fn test_all_hybrid_solver_variants<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve::<M, CG, <M as DefaultSolver>::LS>(&mut problem, 2.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 2.0, 5e-4, "hybrid final time");
            assert_close(
                soln.ys.get_index(0, soln.ts.len() - 1),
                hybrid_logistic_state(2.0, 2.0),
                5e-4,
                "hybrid final value",
            );

            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_dense::<M, CG, <M as DefaultSolver>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                hybrid_logistic_state(2.0, t)
            });
        }
    }

    fn test_all_solver_variants_with_lu<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, 1.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 1.0, 5e-4, "lu solve final time");

            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve_dense::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                logistic_state(LOGISTIC_X0, 2.0, t)
            });
        }
    }

    fn test_all_hybrid_solver_variants_with_lu<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, 2.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 2.0, 5e-4, "lu hybrid final time");

            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_dense::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                hybrid_logistic_state(2.0, t)
            });
        }
    }

    fn assert_direct_hybrid_restart_path_for_method<CG>(method: OdeSolverType)
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];

        let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
        let soln = method
            .solve::<M, CG, <M as DefaultSolver>::LS>(&mut problem, 2.0)
            .unwrap();
        assert_close(
            *soln.ts.last().unwrap(),
            2.0,
            5e-4,
            "direct hybrid restart final time",
        );
        assert_close(
            soln.ys.get_index(0, soln.ts.len() - 1),
            hybrid_logistic_state(2.0, 2.0),
            5e-4,
            "direct hybrid restart final value",
        );

        let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
        let soln = method
            .solve_dense::<M, CG, <M as DefaultSolver>::LS>(&mut problem, &t_eval)
            .unwrap();
        assert_dense_solution_matches_expected(&soln, &t_eval, |t| hybrid_logistic_state(2.0, t));
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_all_sensitivity_solver_variants() {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let soln = method
                .solve_fwd_sens::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    logistic_state_dr(LOGISTIC_X0, 2.0, t),
                    5e-4,
                    &format!("fwd_sens[{i}]"),
                );
            }

            let mut problem = build_problem::<diffsol::LlvmModule>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_fwd_sens::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    hybrid_logistic_state_dr(2.0, t),
                    5e-4,
                    &format!("hybrid_fwd_sens[{i}]"),
                );
            }
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_lu_sensitivity_and_adjoint_solver_variants() {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let soln = method
                .solve_fwd_sens::<M, diffsol::LlvmModule, <M as LuValidator<M>>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    logistic_state_dr(LOGISTIC_X0, 2.0, t),
                    5e-4,
                    &format!("lu fwd_sens[{i}]"),
                );
            }
        }

        let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
        let adjoint_t_eval = [0.0, 0.25, 0.5, 1.0];
        let (soln, checkpoint) = OdeSolverType::Bdf
            .solve_adjoint_fwd::<M, diffsol::LlvmModule, <M as LuValidator<M>>::LS>(
                &mut problem,
                &adjoint_t_eval,
                &[2.0],
                LinearSolverType::Lu,
            )
            .unwrap();
        let dgdu = <<M as MatrixCommon>::V as DefaultDenseMatrix>::M::zeros(
            problem.eqn.nout(),
            soln.ts.len(),
            problem.context().to_owned(),
        );
        let gradient = OdeSolverType::TrBdf2
            .solve_adjoint_bkwd::<M, diffsol::LlvmModule, <M as LuValidator<M>>::LS>(
                &mut problem,
                checkpoint.as_ref(),
                &dgdu,
                &soln.ts,
            )
            .unwrap();
        assert_eq!(gradient.len(), 1);
        assert!(gradient[0].get_index(0).is_finite());
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_direct_hybrid_sensitivity_restart_paths() {
        let t_eval = [0.5, 1.0, 2.5, 3.0, 4.5];
        for method in [
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_fwd_sens::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.ys.get_index(0, i),
                    hybrid_logistic_state(2.0, t),
                    5e-4,
                    &format!("direct hybrid value[{i}]"),
                );
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    hybrid_logistic_state_dr(2.0, t),
                    5e-4,
                    &format!("direct hybrid fwd sens[{i}]"),
                );
            }
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_adjoint_backwards_methods_and_klu_branch() {
        for backwards_method in [OdeSolverType::Esdirk34, OdeSolverType::TrBdf2] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let t_eval = [0.0, 0.25, 0.5, 1.0];
            let (soln, checkpoint) = OdeSolverType::Bdf
                .solve_adjoint_fwd::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                    &[2.0],
                    LinearSolverType::Default,
                )
                .unwrap();
            let dgdu = <<M as MatrixCommon>::V as DefaultDenseMatrix>::M::zeros(
                problem.eqn.nout(),
                soln.ts.len(),
                problem.context().to_owned(),
            );
            let gradient = backwards_method
                .solve_adjoint_bkwd::<M, diffsol::LlvmModule, <M as crate::valid_linear_solver::KluValidator<M>>::LS>(
                    &mut problem,
                    checkpoint.as_ref(),
                    &dgdu,
                    &soln.ts,
                )
                .unwrap();
            assert_eq!(gradient.len(), 1);
            assert!(gradient[0].get_index(0).is_finite());
        }

        let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let (soln, checkpoint) = OdeSolverType::Tsit45
            .solve_adjoint_fwd::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                &mut problem,
                &t_eval,
                &[2.0],
                LinearSolverType::Default,
            )
            .unwrap();
        let dgdu = <<M as MatrixCommon>::V as DefaultDenseMatrix>::M::zeros(
            problem.eqn.nout(),
            soln.ts.len(),
            problem.context().to_owned(),
        );
        let gradient = OdeSolverType::Bdf
            .solve_adjoint_bkwd::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                &mut problem,
                checkpoint.as_ref(),
                &dgdu,
                &soln.ts,
            )
            .unwrap();
        assert_eq!(gradient.len(), 1);
        assert!(gradient[0].get_index(0).is_finite());
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_all_adjoint_solver_variants() {
        let t_eval = [0.0, 0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let (soln, checkpoint) = method
                .solve_adjoint_fwd::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                    &[2.0],
                    LinearSolverType::Default,
                )
                .unwrap();
            let dgdu = <<M as MatrixCommon>::V as DefaultDenseMatrix>::M::zeros(
                problem.eqn.nout(),
                soln.ts.len(),
                problem.context().to_owned(),
            );
            let gradient = OdeSolverType::Bdf
                .solve_adjoint_bkwd::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    checkpoint.as_ref(),
                    &dgdu,
                    &soln.ts,
                )
                .unwrap();
            assert_eq!(gradient.len(), 1);
            assert!(gradient[0].get_index(0).is_finite());
        }
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn runtime_dispatch_solves_all_variants_for_cranelift() {
        test_all_solver_variants::<diffsol::CraneliftJitModule>();
        test_all_solver_variants_with_lu::<diffsol::CraneliftJitModule>();
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn runtime_dispatch_solves_all_hybrid_variants_for_cranelift() {
        test_all_hybrid_solver_variants::<diffsol::CraneliftJitModule>();
        test_all_hybrid_solver_variants_with_lu::<diffsol::CraneliftJitModule>();
        assert_direct_hybrid_restart_path_for_method::<diffsol::CraneliftJitModule>(
            OdeSolverType::Esdirk34,
        );
        assert_direct_hybrid_restart_path_for_method::<diffsol::CraneliftJitModule>(
            OdeSolverType::TrBdf2,
        );
        assert_direct_hybrid_restart_path_for_method::<diffsol::CraneliftJitModule>(
            OdeSolverType::Tsit45,
        );
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_variants_for_llvm() {
        test_all_solver_variants::<diffsol::LlvmModule>();
        test_all_solver_variants_with_lu::<diffsol::LlvmModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_hybrid_variants_for_llvm() {
        test_all_hybrid_solver_variants::<diffsol::LlvmModule>();
        test_all_hybrid_solver_variants_with_lu::<diffsol::LlvmModule>();
        assert_direct_hybrid_restart_path_for_method::<diffsol::LlvmModule>(
            OdeSolverType::Esdirk34,
        );
        assert_direct_hybrid_restart_path_for_method::<diffsol::LlvmModule>(OdeSolverType::TrBdf2);
        assert_direct_hybrid_restart_path_for_method::<diffsol::LlvmModule>(OdeSolverType::Tsit45);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_forward_sensitivity_variants_for_llvm() {
        test_all_sensitivity_solver_variants();
        test_lu_sensitivity_and_adjoint_solver_variants();
        test_direct_hybrid_sensitivity_restart_paths();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_adjoint_variants_for_llvm() {
        test_all_adjoint_solver_variants();
        test_adjoint_backwards_methods_and_klu_branch();
    }
}
