use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error, AdjointEquations, AugmentedOdeEquations, AugmentedOdeSolverMethod,
    CheckpointingPath, DefaultDenseMatrix, DefaultSolver, DenseMatrix, LinearSolver, Matrix,
    MatrixCommon, MatrixOp, NonLinearOpAdjoint, NonLinearOpSensAdjoint, OdeEquations,
    OdeEquationsImplicitAdjoint, OdeSolverMethod, OdeSolverState, OdeSolverStopReason, Op,
    StateRef, Vector, VectorIndex,
};

use num_traits::{One, Zero};
use std::ops::{AddAssign, SubAssign};

pub trait AdjointOdeSolverMethod<'a, Eqn, Solver>:
    AugmentedOdeSolverMethod<'a, Eqn, AdjointEquations<'a, Eqn, Solver>>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    Solver: OdeSolverMethod<'a, Eqn>,
{
    /// Apply the problem reset correction to the adjoint state at a checkpoint
    /// path boundary.
    fn apply_reset_with_adjoint(
        &mut self,
        root_idx: usize,
        fwd_state_minus: StateRef<'_, Eqn::V>,
        fwd_state_plus: StateRef<'_, Eqn::V>,
    ) -> Result<(), DiffsolError> {
        let integrate_out = self.problem().integrate_out;
        let (mut state, adj_eqn) = self
            .state_and_augmented_eqn_mut()
            .ok_or_else(|| ode_solver_error!(Other, "No augmented equations"))?;
        state.apply_reset_with_adjoint(
            adj_eqn.eqn(),
            root_idx,
            fwd_state_minus,
            fwd_state_plus,
            integrate_out,
        )
    }

    /// Backwards pass for adjoint sensitivity analysis
    ///
    /// The overall goal is to compute the gradient of an output function `G` with respect to the model parameters `p`
    ///
    /// If `dgdu_eval` is empty, then `G` is the integral of the model output function `u` over time
    ///
    /// $$
    /// G = \int_{t_0}^{t_{\text{final}}} u(y(t)) dt
    /// $$
    ///
    /// where `y(t)` is the solution of the model at time `t`
    ///
    /// If `dgdu_eval` is non empty, then the output function `G` made from the sum of a sequence of `n` functions `g_i`
    /// operating on the model output function u at timepoints `t_i`
    ///
    /// $$
    /// G = \int_{t_0}^{t_{\text{final}}} \sum_{i=0}^{n-1} g_i(u(y(t_i)))) \delta(t - t_i) dt
    /// $$
    ///
    /// For example, if `G` is the standard sum of squared errors, then `g_i = (u(y(t_i)) - d_i)^2`,
    /// where `d_i` is the measured value of the output at time `t_i`
    ///
    /// The user passes in the gradient of `g_i` with respect to `u_i` for each timepoint `i` in `dgdu_eval`.
    /// For example, if `g_i = (u(y(t_i)) - d_i)^2`, then `dgdu_i = 2(u(y(t_i)) - d_i)`, where `u(y(t_i))`
    /// can be obtained from the forward pass.
    ///
    /// The input `dgdu_eval` is a vector so users can supply multiple sets of `g_i` functions, and each
    /// element of the vector is a dense matrix of size `n_o x n`, where `n_o`` is the number of outputs in the model
    /// and `n` is the number of timepoints. The i-th column of `dgdu_eval` is the gradient of `g_i` with respect to `u_i`.
    /// The input `t_eval` is a vector of length `n`, where the i-th element is the timepoint `t_i`.
    ///
    #[allow(clippy::type_complexity)]
    fn solve_adjoint_backwards_pass(
        mut self,
        t_eval: &[Eqn::T],
        dgdu_eval: &[&<Eqn::V as DefaultDenseMatrix>::M],
    ) -> Result<(Self::State, CheckpointingPath<Eqn, Solver::State>), DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Eqn::M: DefaultSolver,
    {
        let have_neqn = validate_adjoint_backwards_inputs(&self, t_eval, dgdu_eval)?;

        let mut integrate_delta_g = if have_neqn > 0 && !dgdu_eval.is_empty() {
            let integrate_delta_g =
                IntegrateDeltaG::<_, <Eqn::M as DefaultSolver>::LS>::new(&self)?;
            Some(integrate_delta_g)
        } else {
            None
        };
        let problem_t0 = self.problem().t0;
        let solve_t1 = self.state().t;
        let checkpointing_len = self.augmented_eqn().unwrap().checkpointing_len();
        let (first_checkpoint_t, _) = self.augmented_eqn().unwrap().checkpointing_bounds(0);
        let (_, last_checkpoint_t) = self
            .augmented_eqn()
            .unwrap()
            .checkpointing_bounds(checkpointing_len - 1);
        let path_starts_at_problem_t0 = problem_t0 == first_checkpoint_t;
        if solve_t1 != last_checkpoint_t {
            return Err(ode_solver_error!(
                Other,
                "Adjoint solver current time does not match the last checkpointing segment end time"
            ));
        }

        for segment_index in (0..checkpointing_len).rev() {
            let (segment_first_t, segment_end_t) = self
                .augmented_eqn()
                .unwrap()
                .checkpointing_bounds(segment_index);

            solve_adjoint_backwards_segment(
                &mut self,
                segment_first_t,
                segment_end_t,
                segment_index + 1 < checkpointing_len,
                t_eval,
                dgdu_eval,
                integrate_delta_g.as_mut(),
            )?;

            if segment_index > 0 {
                let checkpointing = self
                    .augmented_eqn_mut()
                    .unwrap()
                    .pop_last_checkpointing()
                    .unwrap();
                let fwd_state_plus = checkpointing.first_checkpoint();
                let fwd_state_minus = self
                    .augmented_eqn()
                    .unwrap()
                    .checkpointing_last_state(segment_index - 1);
                let root_idx = self
                    .augmented_eqn()
                    .unwrap()
                    .checkpointing_terminal_reset_root_idx(segment_index - 1)
                    .ok_or_else(|| {
                        ode_solver_error!(
                            Other,
                            "Missing reset root metadata between checkpointing segments"
                        )
                    })?;
                self.apply_reset_with_adjoint(
                    root_idx,
                    fwd_state_minus.as_ref(),
                    fwd_state_plus.as_ref(),
                )?;
            }
        }

        let (mut state, aug_eqn) = self.into_state_and_eqn();
        let aug_eqn = aug_eqn.unwrap();
        if path_starts_at_problem_t0 {
            let state_mut = state.as_mut();
            aug_eqn.correct_sg_for_init(problem_t0, state_mut.s, state_mut.sg);
        }

        Ok((state, aug_eqn.into_checkpointing()))
    }
}

fn validate_adjoint_backwards_inputs<'a, Eqn, Solver, AdjointSolver>(
    solver: &AdjointSolver,
    t_eval: &[Eqn::T],
    dgdu_eval: &[&<Eqn::V as DefaultDenseMatrix>::M],
) -> Result<usize, DiffsolError>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    Eqn::V: DefaultDenseMatrix,
    Solver: OdeSolverMethod<'a, Eqn>,
    AdjointSolver: AdjointOdeSolverMethod<'a, Eqn, Solver>,
{
    if solver.augmented_eqn().is_none() {
        return Err(ode_solver_error!(Other, "No augmented equations"));
    }

    if t_eval.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ode_solver_error!(
            Other,
            "t_eval should be in increasing order"
        ));
    }

    let have_neqn = solver.augmented_eqn().unwrap().max_index();
    if dgdu_eval.is_empty() {
        let expected_neqn = solver.problem().eqn.out().map(|o| o.nout()).unwrap_or(0);
        if have_neqn != expected_neqn {
            return Err(ode_solver_error!(
                Other,
                format!("Number of augmented equations does not match number of model outputs: {} != {}", have_neqn, expected_neqn)
            ));
        }
    } else {
        let expected_neqn = dgdu_eval.len();
        if have_neqn != expected_neqn {
            return Err(ode_solver_error!(
                Other,
                format!("Number of outputs in augmented equations does not match number of outputs in dgdu_eval: {} != {}", have_neqn, expected_neqn)
            ));
        }
    }

    let nout = solver.problem().eqn.nout();
    if dgdu_eval.iter().any(|dgdu| dgdu.nrows() != nout) {
        return Err(ode_solver_error!(
            Other,
            "Number of outputs does not match number of rows in gradient"
        ));
    }
    if dgdu_eval.iter().any(|dgdu| dgdu.ncols() != t_eval.len()) {
        return Err(ode_solver_error!(
            Other,
            "Number of solution timepoints does not match number of columns in gradient"
        ));
    }

    Ok(have_neqn)
}

fn solve_adjoint_backwards_segment<'a, Eqn, Solver, AdjointSolver>(
    solver: &mut AdjointSolver,
    solve_t0: Eqn::T,
    solve_t1: Eqn::T,
    exclude_t1: bool,
    t_eval: &[Eqn::T],
    dgdu_eval: &[&<Eqn::V as DefaultDenseMatrix>::M],
    mut integrate_delta_g: Option<&mut IntegrateDeltaG<Eqn::M, <Eqn::M as DefaultSolver>::LS>>,
) -> Result<(), DiffsolError>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    Eqn::V: DefaultDenseMatrix,
    Eqn::M: DefaultSolver,
    Solver: OdeSolverMethod<'a, Eqn>,
    AdjointSolver: AdjointOdeSolverMethod<'a, Eqn, Solver>,
{
    for (i, t) in t_eval
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, t)| **t <= solve_t1 && **t >= solve_t0)
        .filter(|(_, t)| !(exclude_t1 && **t == solve_t1))
    {
        match solver.set_stop_time(*t) {
            Ok(_) => while solver.step()? != OdeSolverStopReason::TstopReached {},
            Err(DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime)) => {}
            e => e?,
        }

        if let Some(integrate_delta_g) = integrate_delta_g.as_deref_mut() {
            let dudg_i = dgdu_eval.iter().map(|dgdu| dgdu.column(i));
            integrate_delta_g.integrate_delta_g(solver, dudg_i)?;
        }
    }

    match solver.set_stop_time(solve_t0) {
        Ok(_) => while solver.step()? != OdeSolverStopReason::TstopReached {},
        Err(DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime)) => {}
        e => e?,
    }

    Ok(())
}

struct BlockInfoSol<M: Matrix, LS: LinearSolver<M>> {
    pub block: MatrixOp<M>,
    pub src_indices: <M::V as Vector>::Index,
    pub solver: LS,
}

struct BlockInfo<M: Matrix> {
    pub block: MatrixOp<M>,
    pub src_indices: <M::V as Vector>::Index,
}

struct PartitionInfo<I> {
    pub algebraic_indices: I,
    pub differential_indices: I,
}

/// Holds mutable borrows of the scratch buffers in [`IntegrateDeltaG`],
/// for passing them into the discrete delta helper functions.
struct DeltaGBuf<'a, M: Matrix> {
    tmp_nstates: &'a M::V,
    tmp_nstates2: &'a mut M::V,
    tmp_differential: &'a mut M::V,
    tmp_algebraic: &'a mut M::V,
    tmp_differential2: &'a mut M::V,
    tmp_nparams: &'a mut M::V,
    tmp_nout: &'a mut M::V,
}

/// Discrete delta: output operator `g(y)`, mass matrix `M = diag(M_dd, 0)`,
/// and algebraic indices.
///
/// Let `G_y = ∂g/∂y`, `G_p = ∂g/∂p`, and let `A = -F_y^T` be the adjoint
/// Jacobian, partitioned by differential (`d`) and algebraic (`a`) indices:
///
/// ```text
///     A = [A_dd  A_da]
///         [A_ad  A_aa]
/// ```
///
/// Given the discrete cost gradient `dgdu` at this evaluation time:
///
/// ```text
/// v   = G_y^T · dgdu
/// v_d, v_a = partition of v
///
/// λ_d^+ = λ_d^- + M_dd^{-1} · (v_d - A_da · A_aa^{-1} · v_a)
/// sg^+  = sg^-  + G_p^T · dgdu + F_{p,a}^T · A_aa^{-1} · v_a
/// ```
#[allow(clippy::too_many_arguments)]
fn apply_delta_g_out_mass_alg<M: Matrix, OutOp, RhsOp>(
    buf: &mut DeltaGBuf<'_, M>,
    s_i: &mut M::V,
    sg_i: &mut M::V,
    out: &OutOp,
    fwd_rhs: &RhsOp,
    p: &PartitionInfo<<M::V as Vector>::Index>,
    rhs_jac_ad: &M,
    sol_mdd: &impl LinearSolver<M>,
    sol_jaa: &impl LinearSolver<M>,
    t: M::T,
) -> Result<(), DiffsolError>
where
    OutOp: NonLinearOpAdjoint<V = M::V, T = M::T, M = M>
        + NonLinearOpSensAdjoint<V = M::V, T = M::T, M = M>,
    RhsOp: NonLinearOpSensAdjoint<V = M::V, T = M::T, M = M>,
{
    // g_p contribution: sg -= -g_p^T * dgdu  =>  sg += g_p^T * dgdu
    out.sens_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nout, buf.tmp_nparams);
    sg_i.sub_assign(&*buf.tmp_nparams);

    // -g_y^T * dgdu
    out.jac_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nout, buf.tmp_nstates2);

    // gather v_d and v_a from -g_y^T * dgdu
    buf.tmp_differential
        .gather(buf.tmp_nstates2, &p.differential_indices);
    buf.tmp_algebraic
        .gather(buf.tmp_nstates2, &p.algebraic_indices);

    // accumulate (v_d - A_da * A_aa^{-1} * v_a) in tmp_differential
    sol_jaa.solve_in_place(buf.tmp_algebraic)?;
    rhs_jac_ad.gemv(
        M::T::one(),
        buf.tmp_algebraic,
        -M::T::one(),
        buf.tmp_differential,
    );

    // differential update: λ_d += M_dd^{-1} * (v_d - A_da * A_aa^{-1} * v_a)
    sol_mdd.solve_in_place(buf.tmp_differential)?;
    buf.tmp_differential2.gather(s_i, &p.differential_indices);
    buf.tmp_differential2.add_assign(&*buf.tmp_differential);
    buf.tmp_differential2.scatter(&p.differential_indices, s_i);

    // parameter contribution from the algebraic constraint:
    // sg += -F_{p,a}^T * A_aa^{-1} * (-g_{y,a}^T * dgdu)
    buf.tmp_nstates2.fill(M::T::zero());
    buf.tmp_algebraic
        .scatter(&p.algebraic_indices, buf.tmp_nstates2);
    fwd_rhs.sens_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nstates2, buf.tmp_nparams);
    sg_i.add_assign(&*buf.tmp_nparams);
    Ok(())
}

/// Discrete delta: output operator `g(y)` and mass matrix, no algebraic indices.
///
/// ```text
/// v = G_y^T · dgdu
///
/// λ^+ = λ^- + M^{-1} · v
/// sg^+ = sg^- + G_p^T · dgdu
/// ```
fn apply_delta_g_out_mass<M: Matrix, OutOp>(
    buf: &mut DeltaGBuf<'_, M>,
    s_i: &mut M::V,
    sg_i: &mut M::V,
    out: &OutOp,
    sol_mdd: &impl LinearSolver<M>,
    t: M::T,
) -> Result<(), DiffsolError>
where
    OutOp: NonLinearOpAdjoint<V = M::V, T = M::T, M = M>
        + NonLinearOpSensAdjoint<V = M::V, T = M::T, M = M>,
{
    out.sens_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nout, buf.tmp_nparams);
    sg_i.sub_assign(&*buf.tmp_nparams);

    out.jac_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nout, buf.tmp_nstates2);
    sol_mdd.solve_in_place(buf.tmp_nstates2)?;
    s_i.sub_assign(&*buf.tmp_nstates2);
    Ok(())
}

/// Discrete delta: output operator `g(y)`, no mass matrix.
///
/// ```text
/// v = G_y^T · dgdu
///
/// λ^+ = λ^- + v
/// sg^+ = sg^- + G_p^T · dgdu
/// ```
fn apply_delta_g_out<M: Matrix, OutOp>(
    buf: &mut DeltaGBuf<'_, M>,
    s_i: &mut M::V,
    sg_i: &mut M::V,
    out: &OutOp,
    t: M::T,
) where
    OutOp: NonLinearOpAdjoint<V = M::V, T = M::T, M = M>
        + NonLinearOpSensAdjoint<V = M::V, T = M::T, M = M>,
{
    out.sens_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nout, buf.tmp_nparams);
    sg_i.sub_assign(&*buf.tmp_nparams);

    out.jac_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nout, buf.tmp_nstates2);
    s_i.sub_assign(&*buf.tmp_nstates2);
}

/// Discrete delta: no output operator, mass matrix and algebraic indices.
///
/// Without an output function, `dgdu` is applied directly to the adjoint state.
/// Let `A = -F_y^T` be the adjoint Jacobian, partitioned `A_dd, A_da, A_ad, A_aa`.
///
/// ```text
/// λ_d^+ = λ_d^- + M_dd^{-1} · (dgdu_d - A_da · A_aa^{-1} · dgdu_a)
/// sg^+  = sg^-  + F_{p,a}^T · A_aa^{-1} · dgdu_a
/// ```
fn apply_delta_g_no_out_mass_alg<M: Matrix, RhsOp>(
    buf: &mut DeltaGBuf<'_, M>,
    s_i: &mut M::V,
    sg_i: &mut M::V,
    p: &PartitionInfo<<M::V as Vector>::Index>,
    rhs_jac_ad: &M,
    fwd_rhs: &RhsOp,
    sol_mdd: &impl LinearSolver<M>,
    sol_jaa: &impl LinearSolver<M>,
    t: M::T,
) -> Result<(), DiffsolError>
where
    RhsOp: NonLinearOpSensAdjoint<V = M::V, T = M::T, M = M>,
{
    buf.tmp_differential
        .gather(buf.tmp_nout, &p.differential_indices);
    buf.tmp_algebraic.gather(buf.tmp_nout, &p.algebraic_indices);

    // accumulate (dgdu_d - A_da * A_aa^{-1} * dgdu_a) in tmp_differential
    sol_jaa.solve_in_place(buf.tmp_algebraic)?;
    rhs_jac_ad.gemv(
        -M::T::one(),
        buf.tmp_algebraic,
        M::T::one(),
        buf.tmp_differential,
    );

    // differential update: λ_d += M_dd^{-1} * (dgdu_d - A_da * A_aa^{-1} * dgdu_a)
    sol_mdd.solve_in_place(buf.tmp_differential)?;
    buf.tmp_differential2.gather(s_i, &p.differential_indices);
    buf.tmp_differential2.add_assign(&*buf.tmp_differential);
    buf.tmp_differential2.scatter(&p.differential_indices, s_i);

    // parameter contribution from the algebraic constraint:
    // sg += F_{p,a}^T * A_aa^{-1} * dgdu_a = sg - (-F_{p,a}^T * A_aa^{-1} * dgdu_a)
    buf.tmp_nstates2.fill(M::T::zero());
    buf.tmp_algebraic
        .scatter(&p.algebraic_indices, buf.tmp_nstates2);
    fwd_rhs.sens_transpose_mul_inplace(buf.tmp_nstates, t, buf.tmp_nstates2, buf.tmp_nparams);
    sg_i.sub_assign(&*buf.tmp_nparams);
    Ok(())
}

/// Discrete delta: no output operator, mass matrix, no algebraic indices.
///
/// ```text
/// λ^+ = λ^- + M^{-1} · dgdu
/// ```
fn apply_delta_g_no_out_mass<M: Matrix>(
    buf: &mut DeltaGBuf<'_, M>,
    s_i: &mut M::V,
    sol_mdd: &impl LinearSolver<M>,
) -> Result<(), DiffsolError> {
    sol_mdd.solve_in_place(buf.tmp_nout)?;
    s_i.add_assign(&*buf.tmp_nout);
    Ok(())
}

/// Discrete delta: no output operator, no mass matrix.
///
/// ```text
/// λ^+ = λ^- + dgdu
/// ```
fn apply_delta_g_no_out<M: Matrix>(buf: &DeltaGBuf<'_, M>, s_i: &mut M::V) {
    s_i.add_assign(&*buf.tmp_nout);
}

struct IntegrateDeltaG<M: Matrix, LS: LinearSolver<M>> {
    pub rhs_jac_aa: Option<BlockInfoSol<M, LS>>,
    pub rhs_jac_ad: Option<BlockInfo<M>>,
    pub mass_dd: Option<BlockInfoSol<M, LS>>,
    pub partition: Option<PartitionInfo<<M::V as Vector>::Index>>,
    pub tmp_algebraic: M::V,
    pub tmp_differential: M::V,
    pub tmp_differential2: M::V,
    pub tmp_nparams: M::V,
    pub tmp_nstates: M::V,
    pub tmp_nstates2: M::V,
    pub tmp_nout: M::V,
}

impl<M, LS> IntegrateDeltaG<M, LS>
where
    M: Matrix,
    LS: LinearSolver<M>,
{
    fn new<'a, Eqn, Solver>(solver: &Solver) -> Result<Self, DiffsolError>
    where
        Eqn: OdeEquations<M = M, V = M::V, T = M::T, C = M::C> + 'a,
        Solver: OdeSolverMethod<'a, Eqn>,
    {
        let eqn = &solver.problem().eqn;
        let ctx = solver.problem().eqn.context();
        let (partition, mass_dd, rhs_jac_aa, rhs_jac_ad) = if let Some(_mass) = eqn.mass() {
            let mass_matrix = solver.mass().unwrap();
            let (algebraic_indices, differential_indices) =
                mass_matrix.partition_indices_by_zero_diagonal();

            // setup mass solver
            let [(dd, dd_idx), _, _, _] = mass_matrix.split(&algebraic_indices);
            let mut mass_dd = BlockInfoSol {
                block: MatrixOp::new(dd),
                src_indices: dd_idx,
                solver: LS::default(),
            };
            mass_dd.solver.set_problem(&mass_dd.block);

            // setup jacobian solver if there are algebraic indices
            let (rhs_jac_aa, rhs_jac_ad) = if algebraic_indices.len() > 0 {
                let jacobian = solver
                    .jacobian()
                    .ok_or(DiffsolError::from(OdeSolverError::JacobianNotAvailable))?;
                let [_, (ad, ad_idx), _, (aa, aa_idx)] = jacobian.split(&algebraic_indices);
                let mut rhs_jac_aa = BlockInfoSol {
                    block: MatrixOp::new(aa),
                    src_indices: aa_idx,
                    solver: LS::default(),
                };
                rhs_jac_aa.solver.set_problem(&rhs_jac_aa.block);
                let rhs_jac_ad = BlockInfo {
                    block: MatrixOp::new(ad),
                    src_indices: ad_idx,
                };
                (Some(rhs_jac_aa), Some(rhs_jac_ad))
            } else {
                (None, None)
            };
            let partition = PartitionInfo {
                algebraic_indices,
                differential_indices,
            };
            (Some(partition), Some(mass_dd), rhs_jac_aa, rhs_jac_ad)
        } else {
            (None, None, None, None)
        };
        let nparams = eqn.rhs().nparams();
        let nstates = eqn.rhs().nstates();
        let nout = eqn.out().map(|o| o.nout()).unwrap_or(nstates);
        let tmp_nstates = M::V::zeros(nstates, ctx.clone());
        let tmp_nstates2 = M::V::zeros(nstates, ctx.clone());
        let tmp_nparams = M::V::zeros(nparams, ctx.clone());
        let tmp_nout = M::V::zeros(nout, ctx.clone());
        let nalgebraic = partition
            .as_ref()
            .map(|p| p.algebraic_indices.len())
            .unwrap_or(0);
        let ndifferential = nstates - nalgebraic;
        let tmp_algebraic = M::V::zeros(nalgebraic, ctx.clone());
        let tmp_differential = M::V::zeros(ndifferential, ctx.clone());
        let tmp_differential2 = M::V::zeros(ndifferential, ctx.clone());
        Ok(Self {
            rhs_jac_aa,
            rhs_jac_ad,
            mass_dd,
            tmp_nparams,
            tmp_algebraic,
            partition,
            tmp_nstates,
            tmp_nout,
            tmp_differential,
            tmp_differential2,
            tmp_nstates2,
        })
    }
    fn integrate_delta_g<'a, 'b, Eqn, S1, Solver>(
        &mut self,
        solver: &mut Solver,
        dgdus: impl Iterator<Item = <M::V as Vector>::View<'b>>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T> + 'a,
        Solver: AdjointOdeSolverMethod<'a, Eqn, S1>,
        S1: OdeSolverMethod<'a, Eqn>,
    {
        // interpolate forward state y
        let t = solver.state().t;
        solver
            .augmented_eqn()
            .unwrap()
            .interpolate_forward_state(t, &mut self.tmp_nstates)?;

        // if there are algebraic indices, setup the solver for (f*_y^a)^{-1} and M_dd*^-1
        if let Some(rhs_jac_aa) = self.rhs_jac_aa.as_mut() {
            let jacobian = solver.jacobian().unwrap();
            let rhs_jac_ad = self.rhs_jac_ad.as_mut().unwrap();
            rhs_jac_ad
                .block
                .m_mut()
                .gather(&jacobian, &rhs_jac_ad.src_indices);
            rhs_jac_aa
                .block
                .m_mut()
                .gather(&jacobian, &rhs_jac_aa.src_indices);
            rhs_jac_aa.solver.set_linearisation(
                &rhs_jac_aa.block,
                &self.tmp_algebraic,
                Eqn::T::zero(),
            );
        };

        // if there is a mass matrix, setup the solver for M_dd*^-1
        if let Some(mass_dd) = self.mass_dd.as_mut() {
            let mass = solver.mass().unwrap();
            mass_dd.block.m_mut().gather(&mass, &mass_dd.src_indices);
            mass_dd.solver.set_linearisation(
                &mass_dd.block,
                &self.tmp_differential,
                Eqn::T::zero(),
            );
        }

        // tmp_nout = all
        // tmp_nstates = all
        // tmp_nstates = out
        // tmp_differential = algebraic or mass
        // tmp_algebraic = algebraic
        // tmp_differential2 = algebraic
        // tmp_nparams = out

        let out = solver.augmented_eqn().unwrap().eqn().out();
        let fwd_rhs = solver.augmented_eqn().unwrap().eqn().rhs();
        let state_mut = solver.state_mut();
        for ((s_i, sg_i), dgdu) in state_mut
            .s
            .iter_mut()
            .zip(state_mut.sg.iter_mut())
            .zip(dgdus)
        {
            self.tmp_nout.copy_from_view(&dgdu);

            let mut buf = DeltaGBuf {
                tmp_nstates: &self.tmp_nstates,
                tmp_nstates2: &mut self.tmp_nstates2,
                tmp_differential: &mut self.tmp_differential,
                tmp_algebraic: &mut self.tmp_algebraic,
                tmp_differential2: &mut self.tmp_differential2,
                tmp_nparams: &mut self.tmp_nparams,
                tmp_nout: &mut self.tmp_nout,
            };
            let sol_mdd_opt = self.mass_dd.as_ref().map(|m| &m.solver);
            let sol_jaa_opt = self.rhs_jac_aa.as_ref().map(|m| &m.solver);

            if let (Some(out), Some(sol_mdd), Some(sol_jaa)) =
                (out.as_ref(), sol_mdd_opt, sol_jaa_opt)
            {
                let p = self.partition.as_ref().unwrap();
                let rhs_jac_ad = self.rhs_jac_ad.as_ref().unwrap().block.m();
                apply_delta_g_out_mass_alg(
                    &mut buf, s_i, sg_i, out, &fwd_rhs, p, rhs_jac_ad, sol_mdd, sol_jaa, t,
                )?;
            } else if let (Some(out), Some(sol_mdd)) = (out.as_ref(), sol_mdd_opt) {
                apply_delta_g_out_mass(&mut buf, s_i, sg_i, out, sol_mdd, t)?;
            } else if let Some(out) = out.as_ref() {
                apply_delta_g_out(&mut buf, s_i, sg_i, out, t);
            } else if let (Some(sol_mdd), Some(sol_jaa)) = (sol_mdd_opt, sol_jaa_opt) {
                let p = self.partition.as_ref().unwrap();
                let rhs_jac_ad = self.rhs_jac_ad.as_ref().unwrap().block.m();
                apply_delta_g_no_out_mass_alg(
                    &mut buf, s_i, sg_i, p, rhs_jac_ad, &fwd_rhs, sol_mdd, sol_jaa, t,
                )?;
            } else if let Some(sol_mdd) = sol_mdd_opt {
                apply_delta_g_no_out_mass(&mut buf, s_i, sol_mdd)?;
            } else {
                apply_delta_g_no_out(&buf, s_i);
            }
        }

        Ok(())
    }
}
