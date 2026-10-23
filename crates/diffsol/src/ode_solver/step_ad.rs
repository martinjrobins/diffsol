//! Reverse-mode derivatives of single accepted solver steps.
//!
//! These differentiate what the solver actually computed ("discretise-then-optimise"), in
//! contrast to the continuous adjoint equations of [crate::AdjointOdeSolverMethod]. Error
//! control is not differentiated: the step size, the order and the accept/reject decisions of
//! each step are treated as fixed, and only the accepted attempt contributes.
//!
//! Cotangents have the same type as the solver state, `Self::State`. What they hold, and which
//! of their fields are read, depends on the solver; see its implementation. Allocate them with
//! [crate::OdeSolverState::zeros_like].
//!
//! A cotangent can carry several directions at once in its batch lanes: allocate it with
//! [crate::OdeSolverState::zeros_like_with_directions], which gives it `n` lanes for each lane of
//! the state, with lane `b * n + k` holding direction `k` of batch `b`. The parameter cotangent
//! `p_bar` then has one lane per cotangent lane. Every direction shares the step's linear
//! solves, which iterate until the worst of them has converged.
//!
//! A reverse-mode pass saves a [crate::OdeSolverMethod::state_clone] after every step, then
//! pulls a cotangent back through the steps that ended at them, in reverse:
//! ```ignore
//! for end in checkpoints.iter().rev() {
//!     solver.step_vjp(end, &mut cotangent, &mut p_bar)?;
//! }
//! ```
//! [StepVjp::solve_dense_vjp] does all of this for a loss summed over a set of times: the
//! forward solve, the checkpoints, the pull back and the derivative of the initial state.
//!
//! For the forward mode, use the forward sensitivities of
//! [crate::SensitivitiesOdeSolverMethod]. They differentiate the same steps the same way: for
//! [crate::Bdf] and [crate::Sdirk] both are solved to the same `derivative_solve_tolerance` (see
//! [crate::OdeSolverOptions]), so the two agree as closely as that tolerance allows. The
//! transposed solves of [StepVjp::step_vjp] are held to it more strictly: they converge only once
//! an update is below it, not on an error estimated from their convergence rate. For
//! [crate::ExplicitRk] neither needs a solve, so they agree to rounding. Leave the
//! sensitivities out of error control (the default, see [crate::OdeBuilder::sens_rtol]) for the
//! steps to be those of a run without them, as long as no sensitivity solve fails.

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error, ConstantOpSensAdjoint, Context, DefaultSolver, LinearOp, LinearSolver,
    Matrix, MatrixOp, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSensAdjoint,
    OdeEquationsImplicitAdjoint, OdeSolverMethod, OdeSolverState, OdeSolverStopReason, Op, Vector,
    VectorIndex,
};
use diffsol_la::LinearSolver as LaLinearSolver;
use num_traits::Zero;
use std::ops::{AddAssign, SubAssign};

/// Reverse-mode derivative (vector-Jacobian product) of a checkpointed step.
pub trait StepVjp<'a, Eqn: OdeEquationsImplicitAdjoint + 'a>: OdeSolverMethod<'a, Eqn> {
    /// Pull `cotangent`, a cotangent state at `end`, back through the step that ended there.
    /// `end` is a state the solver had straight after a step, as saved by
    /// [OdeSolverMethod::state_clone]; the solver's own state is not used or changed. On return
    /// `cotangent` is the cotangent state at the start of that step, and the parameter cotangent
    /// has been added to `p_bar` (of length `nparams`).
    ///
    /// Fails with `NoStepToDifferentiate` if `end` is not the end of a step (it is the initial
    /// state, or was modified after the step), with `StepDifferentiationSizeMismatch` if
    /// `cotangent` does not have the sizes of `end` (including the number of stages, for a
    /// Runge-Kutta solver) or `p_bar` does not have `nparams` elements, and with
    /// `StepDifferentiationBatchMismatch` if the lanes of `cotangent` and `p_bar` do not fit
    /// those of `end` (see the [module docs](self)). On any other error, `cotangent` and `p_bar`
    /// are left partly transformed and must not be reused.
    ///
    /// Its linear solves use the solver's own Newton matrix as a preconditioner, kept up to date
    /// as a step would: on the solver's Jacobian update schedule, checked at the start of each
    /// replayed step (a step checks it at its end) and advanced by it, and, if a solve fails, with the Jacobian updates a step makes after a failed Newton solve,
    /// as long as they change the matrix, without the smaller step size a replayed step cannot
    /// take. As h is fixed, a Runge-Kutta stage whose solve fails again evaluates the Jacobian at
    /// that stage's value instead, making the preconditioner exact for that stage, and retries
    /// only that stage.
    ///
    /// The mass matrix is taken to be independent of the parameters, as for the forward
    /// sensitivities: for a model whose mass matrix depends on them, `p_bar` misses that term.
    fn step_vjp(
        &mut self,
        end: &Self::State,
        cotangent: &mut Self::State,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError>;

    /// Solve forwards to the times `t_eval` and return the loss `G` and its gradient with
    /// respect to the parameters, `dG/dp`, by pulling the loss back through every step with
    /// [Self::step_vjp].
    ///
    /// The loss is a sum of functions `g_i` of the model output `u` (the state `y` if the model
    /// has no output function) at the times `t_eval`:
    ///
    /// $$
    /// G = \sum_{i=0}^{n-1} g_i(u(y(t_i)))
    /// $$
    ///
    /// Both the loss and its gradient are given as closures, called once for each time during
    /// the forward solve as `(i, t_i, u_i, out)`:
    /// - `g` adds `g_i(u_i)` to `out`, the running loss, of length 1 and starting at zero;
    /// - `dgdu` writes the gradient of `g_i` with respect to `u_i` into `out`, of length `nout`
    ///   and zeroed before the call.
    ///
    /// Several losses can share the backward linear solves: with `ndirections` of them, `out`
    /// has `ndirections` lanes for each lane of the state, with lane `b * ndirections + k` for
    /// loss `k` of batch `b` (see the [module docs](self)); `u_i` has the lanes of the state. The
    /// returned loss and gradient (of length `nparams`) have the lanes of `out`.
    ///
    /// The solve starts from the solver's current state, at or before `t_eval[0]`, and stops at
    /// each time in `t_eval` instead of interpolating to it, as there is no derivative of the
    /// interpolation. The state after every step is kept for the backward pass. If the solve
    /// starts at the problem's initial time, the dependence of the initial state (and of its
    /// time derivative, through the consistent initialisation of the solver's constructor) on
    /// the parameters is included; otherwise the derivative with respect to the starting state
    /// is not added.
    ///
    /// As for [Self::step_vjp], this is the derivative of the steps the solver took, with their
    /// step sizes and orders fixed, and a mass matrix that depends on the parameters is not
    /// differentiated. Roots are not supported: a step that finds one is an error.
    fn solve_dense_vjp<G, Dg>(
        &mut self,
        t_eval: &[Eqn::T],
        ndirections: usize,
        mut g: G,
        mut dgdu: Dg,
    ) -> Result<(Eqn::V, Eqn::V), DiffsolError>
    where
        G: FnMut(usize, Eqn::T, &Eqn::V, &mut Eqn::V),
        Dg: FnMut(usize, Eqn::T, &Eqn::V, &mut Eqn::V),
        Eqn::M: DefaultSolver,
        Self: Sized,
    {
        if ndirections == 0 {
            return Err(ode_solver_error!(Other, "ndirections must be at least 1"));
        }
        if t_eval.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ode_solver_error!(
                Other,
                "t_eval should be in increasing order"
            ));
        }
        let start = self.state_clone();
        let start_t = start.as_ref().t;
        if t_eval.first().is_some_and(|t| *t < start_t) {
            return Err(ode_solver_error!(
                Other,
                "t_eval should not start before the solver's current time"
            ));
        }

        let problem = self.problem();
        let eqn = &problem.eqn;
        let (nstates, nparams) = (eqn.rhs().nstates(), eqn.rhs().nparams());
        let nout = eqn.out().map(|o| o.nout()).unwrap_or(nstates);
        let state_ctx = start.as_ref().y.context().clone();
        let lanes = state_ctx.nbatch() * ndirections;
        let ctx = state_ctx.clone_with_nbatch(lanes)?;

        // forward: a checkpoint after every step, and the loss and its gradient at each time
        let mut checkpoints = Vec::new();
        // (number of checkpoints when t_i was reached, dg_i/du)
        let mut dgdus = Vec::with_capacity(t_eval.len());
        let mut loss = Eqn::V::zeros(1, ctx.clone());
        let mut u = Eqn::V::zeros(nout, state_ctx.clone());
        for (i, &t) in t_eval.iter().enumerate() {
            match self.set_stop_time(t) {
                Ok(_) => loop {
                    let reason = self.step()?;
                    checkpoints.push(self.state_clone());
                    match reason {
                        OdeSolverStopReason::TstopReached => break,
                        OdeSolverStopReason::InternalTimestep => {}
                        OdeSolverStopReason::RootFound(..) => {
                            return Err(ode_solver_error!(
                                Other,
                                "solve_dense_vjp does not support roots"
                            ))
                        }
                    }
                },
                Err(DiffsolError::OdeSolverError(OdeSolverError::StopTimeAtCurrentTime)) => {}
                e => e?,
            }
            let y = self.state().y;
            match eqn.out() {
                Some(out) => out.call_inplace(y, t, &mut u),
                None => u.copy_from(y),
            }
            g(i, t, &u, &mut loss);
            let mut dgdu_i = Eqn::V::zeros(nout, ctx.clone());
            dgdu(i, t, &u, &mut dgdu_i);
            dgdus.push((checkpoints.len(), dgdu_i));
        }

        // backward: add each dg_i/dy to the cotangent where t_i was reached, and pull it back
        let mut cot = checkpoints
            .last()
            .unwrap_or(&start)
            .zeros_like_with_directions(ndirections)?;
        let mut p_bar = Eqn::V::zeros(nparams, ctx.clone());
        let mut tmp_nstates = Eqn::V::zeros(nstates, ctx.clone());
        let mut tmp_nparams = Eqn::V::zeros(nparams, ctx.clone());
        let mut add_dgdy = |state: &Self::State, dgdu_i: &Eqn::V, cot: &mut Self::State| {
            let (y, t) = (state.as_ref().y, state.as_ref().t);
            let cot_y = cot.as_mut().y;
            match eqn.out() {
                // the out operators give -g_y^T v and -g_p^T v
                Some(out) => {
                    out.jac_transpose_mul_inplace(y, t, dgdu_i, &mut tmp_nstates);
                    cot_y.sub_assign(&tmp_nstates);
                    out.sens_transpose_mul_inplace(y, t, dgdu_i, &mut tmp_nparams);
                    p_bar.sub_assign(&tmp_nparams);
                }
                None => cot_y.add_assign(dgdu_i),
            }
        };
        let mut dgdus = dgdus.into_iter().rev().peekable();
        let mut p_bar_steps = Eqn::V::zeros(nparams, ctx.clone());
        for (k, checkpoint) in checkpoints.iter().enumerate().rev() {
            while let Some((_, dgdu_i)) = dgdus.next_if(|(n, _)| *n == k + 1) {
                add_dgdy(checkpoint, &dgdu_i, &mut cot);
            }
            self.step_vjp(checkpoint, &mut cot, &mut p_bar_steps)?;
        }
        for (_, dgdu_i) in dgdus {
            add_dgdy(&start, &dgdu_i, &mut cot);
        }
        p_bar.add_assign(&p_bar_steps);

        if start_t == problem.t0 {
            initial_state_vjp(problem, &start, &mut cot, &mut p_bar)?;
        }
        Ok((loss, p_bar))
    }
}

/// Add to `p_bar` the derivative of the problem's initial state `start`, whose cotangent is
/// `cot`, with respect to the parameters, as the solvers' constructors compute it: `y` from
/// the init operator and `dy = f(y)`, or, with algebraic states, the consistent `y` and `dy`
/// of [crate::StateRefMut::set_consistent].
fn initial_state_vjp<Eqn, S>(
    problem: &crate::OdeSolverProblem<Eqn>,
    start: &S,
    cot: &mut S,
    p_bar: &mut Eqn::V,
) -> Result<(), DiffsolError>
where
    Eqn: OdeEquationsImplicitAdjoint,
    Eqn::M: DefaultSolver,
    S: OdeSolverState<Eqn::V>,
{
    let eqn = &problem.eqn;
    let (y0, t0) = (start.as_ref().y, start.as_ref().t);
    let ctx = cot.as_ref().y.context().clone();
    let rhs = eqn.rhs();
    let mut tmp_nstates = Eqn::V::zeros(rhs.nstates(), ctx.clone());
    let mut tmp_nparams = Eqn::V::zeros(rhs.nparams(), ctx.clone());

    let algebraic = eqn.mass().map(|mass| {
        let mass = mass.matrix(t0);
        let (algebraic, differential) = mass.partition_indices_by_zero_diagonal();
        (mass, algebraic, differential)
    });
    // lambda, the cotangent of the residual f(y) - M dy of the initialisation
    let lambda = match algebraic {
        Some((mass, algebraic, differential)) if algebraic.len() > 0 => {
            // the consistent initialisation solves for w = (dy_d, y_a) from
            //   R(w) = (f_d(y) - M_dd dy_d, f_a(y)) = 0
            // and sets dy_a = 0, so with J_w^T lambda = -w_bar, where
            //   J_w^T = [-M_dd^T, 0; f_dv^T, f_av^T]:
            //   lambda_d = M_dd^{-T} dy_bar_d
            //   lambda_a = -f_av^{-T} (y_bar_a + f_dv^T lambda_d)
            let mut lambda_d = Eqn::V::zeros(differential.len(), ctx.clone());
            lambda_d.gather(cot.as_ref().dy, &differential);
            let [(m_dd, _), _, _, _] = mass.split(&algebraic);
            let m_dd = MatrixOp::new(m_dd);
            let mut solver = <Eqn::M as DefaultSolver>::LS::default();
            solver.set_problem(&m_dd);
            LinearSolver::set_linearisation(&mut solver, &m_dd, &lambda_d, t0);
            LaLinearSolver::solve_transpose_in_place(&solver, &mut lambda_d)?;

            // -f_dv^T lambda_d, in the algebraic rows of -f_y^T (lambda_d, 0)
            let mut lambda = Eqn::V::zeros(rhs.nstates(), ctx.clone());
            lambda_d.scatter(&differential, &mut lambda);
            rhs.jac_transpose_mul_inplace(y0, t0, &lambda, &mut tmp_nstates);
            let mut lambda_a = Eqn::V::zeros(algebraic.len(), ctx.clone());
            lambda_a.gather(&tmp_nstates, &algebraic);
            let mut y_bar_a = Eqn::V::zeros(algebraic.len(), ctx.clone());
            y_bar_a.gather(cot.as_ref().y, &algebraic);
            lambda_a.sub_assign(&y_bar_a);
            let [_, _, _, (f_av, _)] = rhs.jacobian(y0, t0).split(&algebraic);
            let f_av = MatrixOp::new(f_av);
            let mut solver = <Eqn::M as DefaultSolver>::LS::default();
            solver.set_problem(&f_av);
            LinearSolver::set_linearisation(&mut solver, &f_av, &lambda_a, t0);
            LaLinearSolver::solve_transpose_in_place(&solver, &mut lambda_a)?;
            lambda_a.scatter(&algebraic, &mut lambda);
            // y_a is solved for, so its initial value does not matter
            cot.as_mut().y.assign_at_indices(&algebraic, Eqn::T::zero());
            lambda
        }
        // dy = f(y)
        _ => cot.as_ref().dy.clone(),
    };
    // the rhs operators give -f_y^T lambda and -f_p^T lambda
    rhs.jac_transpose_mul_inplace(y0, t0, &lambda, &mut tmp_nstates);
    rhs.sens_transpose_mul_inplace(y0, t0, &lambda, &mut tmp_nparams);
    p_bar.sub_assign(&tmp_nparams);
    let cot_y = cot.as_mut().y;
    cot_y.sub_assign(&tmp_nstates);
    if let Some(algebraic) = eqn
        .mass()
        .map(|m| m.matrix(t0).partition_indices_by_zero_diagonal().0)
    {
        cot_y.assign_at_indices(&algebraic, Eqn::T::zero());
    }
    // the init operator gives -y0_p^T v
    eqn.init()
        .sens_transpose_mul_inplace(t0, cot_y, &mut tmp_nparams);
    p_bar.sub_assign(&tmp_nparams);
    Ok(())
}

/// Check the inputs of a [StepVjp::step_vjp], with the states given as their `(y, dy, diff)`:
/// that `end`'s `diff` has `diff_cols` columns (a state of another tableau or order may not),
/// that `cotangent` has the sizes of `end` and `p_bar` has `nparams` elements, and that their
/// lanes fit (see the [module docs](self)).
pub(crate) fn check_step_vjp_inputs<M: Matrix>(
    end: (&M::V, &M::V, &M),
    diff_cols: usize,
    cotangent: (&M::V, &M::V, &M),
    p_bar: &M::V,
    nparams: usize,
) -> Result<(), DiffsolError> {
    let (end_y, end_dy, end_diff) = end;
    let (y, dy, diff) = cotangent;
    // sizes as (rows, columns of a matrix), formatted only on a mismatch
    let shape = |m: &M| (m.nrows(), Some(m.ncols()));
    let vector = |v: &M::V| (v.len(), None);
    for (what, found, expected) in [
        (
            "diff of the state",
            shape(end_diff),
            (end_y.len(), Some(diff_cols)),
        ),
        ("y", vector(y), vector(end_y)),
        ("dy", vector(dy), vector(end_dy)),
        ("diff", shape(diff), shape(end_diff)),
        ("the parameter cotangent", vector(p_bar), (nparams, None)),
    ] {
        if found != expected {
            let size = |(rows, cols): (usize, Option<usize>)| match cols {
                Some(cols) => format!("{rows} x {cols}"),
                None => rows.to_string(),
            };
            return Err(DiffsolError::from(
                OdeSolverError::StepDifferentiationSizeMismatch {
                    what,
                    found: size(found),
                    expected: size(expected),
                },
            ));
        }
    }
    let state_lanes = end_y.context().nbatch();
    let lanes = y.context().nbatch();
    let mismatch = |what, found, expected: String| {
        Err(DiffsolError::from(
            OdeSolverError::StepDifferentiationBatchMismatch {
                what,
                found,
                expected,
                state_lanes,
            },
        ))
    };
    if !lanes.is_multiple_of(state_lanes) {
        return mismatch("y", lanes, format!("a multiple of {state_lanes}"));
    }
    for (what, found) in [
        ("dy", dy.context().nbatch()),
        ("diff", diff.context().nbatch()),
    ] {
        if found != lanes {
            return mismatch(what, found, lanes.to_string());
        }
    }
    let p_lanes = p_bar.context().nbatch();
    if p_lanes != lanes {
        return mismatch("the parameter cotangent", p_lanes, lanes.to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Tests for the step derivatives. The reference derivatives are finite differences of
    //! single steps and, for whole trajectories, the solver's own forward sensitivities.

    //!
    //! Every solve runs at [RTOL] and [ATOL], and every check compares in the solvers' weighted
    //! norm at the same tolerances, scaled by a factor of its own: the checks compare derivatives
    //! over whole steps or trajectories, whose errors exceed the local errors the solvers
    //! control.

    use crate::{OdeEquations, OdeSolverProblem, Vector};

    const P: [f64; 3] = [1.5, 3.0, 1.0];
    const T_FINAL: f64 = 5.0;

    /// The relative tolerance of every solve and check.
    const RTOL: f64 = 1e-6;
    /// The absolute tolerance of every solve and check.
    const ATOL: f64 = 1e-10;

    /// `problem` with its solves at [RTOL] and [ATOL].
    fn with_tolerances<Eqn>(mut problem: OdeSolverProblem<Eqn>) -> OdeSolverProblem<Eqn>
    where
        Eqn: OdeEquations<T = f64>,
    {
        let ctx = problem.atol.context().clone();
        problem.rtol = RTOL;
        problem.atol = Eqn::V::from_element(problem.atol.len(), ATOL, ctx);
        problem
    }

    /// `value` (with a lane for each batch) is `reference` within `factor` in the solvers'
    /// weighted norm at [RTOL] and [ATOL].
    fn assert_close<V: Vector<T = f64>>(value: &V, reference: &V, factor: f64) {
        let atol = V::from_element(reference.len(), ATOL, reference.context().clone());
        value.assert_eq_norm(reference, &atol, RTOL, factor);
    }

    /// A basis of the directions `(y_dot, dy_dot, p_dot)` a step's inputs are perturbed along:
    /// each of `y`, `dy` and `p` on its own for the ODE. For the DAE, the directions keep the
    /// state consistent with the linearised constraint `y0 + y1 + y2 = p2`, and its time
    /// derivative `dy0 + dy1 + dy2 = 0`, as an inconsistent state makes a step jump, whatever
    /// its size: `y2` and `dy2` follow the other components.
    fn input_directions<V: Vector<T = f64>>(dae: bool) -> Vec<(V, V, V)> {
        let unit = |k: usize| {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            e
        };
        let zero = [0.0; 3];
        let consistent = |mut y: [f64; 3], mut dy: [f64; 3], p: [f64; 3]| {
            if dae {
                y[2] = p[2] - y[0] - y[1];
                dy[2] = -dy[0] - dy[1];
            }
            let v = |x: [f64; 3]| V::from_vec(x.to_vec(), Default::default());
            (v(y), v(dy), v(p))
        };
        let free = if dae { 2 } else { 3 };
        let ys = (0..free).map(|k| consistent(unit(k), zero, zero));
        let dys = (0..free).map(|k| consistent(zero, unit(k), zero));
        let ps = (0..3).map(|k| consistent(zero, zero, unit(k)));
        ys.chain(dys).chain(ps).collect()
    }

    /// `$items` if the flag is `true`, nothing if it is `false`.
    macro_rules! if_flag {
        (true, { $($items:tt)* }) => { $($items)* };
        (false, { $($items:tt)* }) => {};
    }

    /// The tests for one solver and linear solver: on matrix type `$M`, for the solver built by
    /// the problem methods `$new`, `$from_state` (from a state of type `$State`, with a config of
    /// type `$Config`) and, with forward sensitivities, `$new_sens`, each given as the method
    /// name with any generic arguments. `dae` is whether the solver supports the DAE form of the
    /// model, and `refreshes` whether its derivatives can update the Newton matrix (an explicit
    /// solver has none). A single step is solved to the tolerances, so its checks allow the
    /// solvers' norm (a factor of 1); `trajectory_factor` is that of the checks over whole
    /// trajectories, whose errors accumulate over the steps.
    macro_rules! step_ad_tests {
        (
            $mod_name:ident,
            M = $M:ty,
            State = $State:ty,
            Config = $Config:ty,
            new = { $($new:tt)* },
            from_state = { $($from_state:tt)* },
            new_sens = { $($new_sens:tt)* },
            dae = $dae:tt,
            refreshes = $refreshes:tt,
            trajectory_factor = $trajectory_factor:expr $(,)?
        ) => {
            mod $mod_name {
                use super::{assert_close, input_directions, with_tolerances, P, RTOL, T_FINAL};

                /// The forms of the model the solver supports: the ODE, and the DAE if `dae`.
                const FORMS: &[bool] = if $dae { &[false, true] } else { &[false] };
                use crate::error::{DiffsolError, OdeSolverError};
                use crate::ode_equations::test_models::ad_test_model::{
                    ad_test_problem, AdTestEqn,
                };
                use crate::ode_solver::OdeSolverStatistics;
                use crate::{
                    scale, Context, MatrixCommon, OdeEquations, OdeSolverMethod, OdeSolverProblem,
                    OdeSolverState, OdeSolverStopReason, StepVjp, Vector, VectorView,
                };

                type M = $M;
                type V = <M as MatrixCommon>::V;

                /// The factor of the checks over whole trajectories.
                const TRAJECTORY_FACTOR: f64 = $trajectory_factor;
                type Problem = OdeSolverProblem<AdTestEqn<M>>;

                type E = AdTestEqn<M>;

                fn new_solver(
                    p: &Problem,
                ) -> impl OdeSolverMethod<'_, E, State = $State, Config = $Config> + StepVjp<'_, E>
                {
                    p.$($new)*().unwrap()
                }

                fn solver_from_state(
                    p: &Problem,
                    state: $State,
                ) -> impl OdeSolverMethod<'_, E, State = $State> + StepVjp<'_, E> {
                    p.$($from_state)*(state).unwrap()
                }

                /// A solver with forward sensitivities.
                fn new_sens_solver(
                    p: &Problem,
                ) -> impl OdeSolverMethod<'_, E, State = $State, Config = $Config> + StepVjp<'_, E>
                {
                    p.$($new_sens)*().unwrap()
                }

                /// The test model.
                fn problem(dae: bool) -> Problem {
                    with_tolerances(ad_test_problem::<M>(P, dae, RTOL))
                }

                fn problem_at(p: &V, dae: bool) -> Problem {
                    let mut problem = problem(dae);
                    problem.eqn.set_params(p);
                    problem
                }

                fn params(problem: &Problem) -> V {
                    let mut p = V::zeros(3, Default::default());
                    problem.eqn.get_params(&mut p);
                    p
                }

                /// `x + e d`
                fn axpy(x: &V, e: f64, d: &V) -> V {
                    x.clone() + d.clone() * scale(e)
                }

                /// A copy of `state` with `y` and `dy` set, which also rebuilds its history.
                fn with_y_dy<S: OdeSolverState<V>>(state: &S, y: &V, dy: &V) -> S {
                    let mut s = state.clone();
                    let s_mut = s.as_mut();
                    s_mut.y.copy_from(y);
                    s_mut.dy.copy_from(dy);
                    s
                }

                fn y_dy<S: OdeSolverState<V>>(s: &S) -> (V, V) {
                    (s.as_ref().y.clone(), s.as_ref().dy.clone())
                }

                /// Lane `k` of a batched vector.
                fn lane(v: &V, k: usize) -> V {
                    v.get_batch(k).into_owned()
                }

                /// A solver started from `start`, after one step towards `tstop`.
                fn step_from(
                    problem: &Problem,
                    start: $State,
                    tstop: f64,
                ) -> impl OdeSolverMethod<'_, E, State = $State> + StepVjp<'_, E> {
                    let mut solver = solver_from_state(problem, start);
                    solver.set_stop_time(tstop).unwrap();
                    solver.step().unwrap();
                    solver
                }

                /// One step, from the state after `after` steps with its history rebuilt from
                /// `(y, dy)` (as after `state_mut`), and with a stop time at `tstop_fraction` of
                /// the step if given. The step's Jacobian, from its outputs `(y, h dy)` to its
                /// inputs `(y, dy, p)`, by `step_vjp` (a row for each output) is checked against
                /// central finite differences of `step()` (a column for each input direction).
                /// Each entry is a single partial derivative, so a single step solved to the
                /// tolerances matches them in the solvers' norm. The output `h dy` is the step's
                /// increment, at the scale of `y`: `dy` itself carries rounding errors of
                /// `|y| / h` epsilons, which the finite differences cannot resolve when `h` is
                /// small.
                fn rebuilt_step(dae: bool, after: usize, tstop_fraction: Option<f64>) {
                    // the finite differences are the reference, so the Newton solves of their
                    // steps are converged well below the tolerances: their error is divided by
                    // the perturbation
                    let reference = |mut problem: Problem| {
                        problem.ode_options.nonlinear_solver_tolerance = 1e-8;
                        problem
                    };
                    let problem0 = reference(problem(dae));
                    let mut solver = new_solver(&problem0);
                    solver.set_stop_time(T_FINAL).unwrap();
                    for _ in 0..after {
                        solver.step().unwrap();
                    }
                    let state = solver.state_clone();
                    // restart from the state's (y, dy) with its history rebuilt, as after
                    // state_mut, so that the step's inputs are ones the test can perturb
                    let (y0, dy0) = y_dy(&state);
                    let t0 = state.as_ref().t;
                    let mut start = with_y_dy(&state, &y0, &dy0);

                    // finite differences, on steps that must take the same step size. The Newton
                    // error in dy = (history) / h grows as h shrinks, so eps is not too small
                    let p0 = params(&problem0);
                    let directions = input_directions::<V>(dae);
                    let eps = 1e-4;
                    // a perturbed problem and restart, along direction d: parameters
                    // p + e p_dot, y + e y_dot and dy + e dy_dot
                    let problem_e = |(_, _, p_dot): &(V, V, V), e: f64| {
                        reference(problem_at(&axpy(&p0, e, p_dot), dae))
                    };
                    let restart_e = |start: &$State, (y_dot, dy_dot, _): &(V, V, V), e: f64| {
                        with_y_dy(start, &axpy(&y0, e, y_dot), &axpy(&dy0, e, dy_dot))
                    };

                    // at a step size whose first attempt is accepted, for the step and all the
                    // perturbed ones: the size of a retried step depends on the error estimate,
                    // so it would differ between them. Probe steps find one: a perturbed dy is
                    // not f(y), which a method that takes its first stage from dy sees as error
                    // TODO: once https://github.com/martinjrobins/diffsol/issues/394 is fixed
                    // then we can turn off error control and this test would be simpler
                    let h_accepted = directions
                        .iter()
                        .flat_map(|d| [(d, 0.0), (d, eps), (d, -eps)])
                        .map(|(d, e)| {
                            step_from(&problem_e(d, e), restart_e(&start, d, e), T_FINAL)
                                .state()
                                .t
                                - t0
                        })
                        .fold(f64::INFINITY, f64::min);
                    *start.as_mut().h = 0.9 * h_accepted;
                    let tstop = tstop_fraction
                        .map(|f| t0 + f * start.as_ref().h)
                        .unwrap_or(T_FINAL);

                    let mut solver = step_from(&problem0, start.clone(), tstop);
                    let t_end = solver.state().t;
                    if tstop_fraction.is_some() {
                        assert!(
                            t_end < t0 + start.as_ref().h,
                            "the stop time did not shorten the step"
                        );
                    }

                    // the outputs (y, dy) after a step perturbed along d
                    let perturbed = |d: &(V, V, V), e: f64| {
                        let problem = problem_e(d, e);
                        let solver = step_from(&problem, restart_e(&start, d, e), tstop);
                        assert_eq!(solver.state().t, t_end, "the perturbed step changed size");
                        let (y, dy) = y_dy(&solver.state_clone());
                        [y.clone_as_vec(), dy.clone_as_vec()].concat()
                    };
                    // fd[k][j]: the derivative of output k along direction j
                    let fd_columns: Vec<Vec<f64>> = directions
                        .iter()
                        .map(|d| {
                            let (plus, minus) = (perturbed(d, eps), perturbed(d, -eps));
                            plus.iter().zip(&minus).map(|(a, b)| (a - b) * 0.5 / eps).collect()
                        })
                        .collect();
                    let n_out = fd_columns[0].len();
                    // the rows of the outputs (y, h dy)
                    let h = t_end - t0;
                    let row_scale = move |k: usize| if k < 3 { 1.0 } else { h };
                    let fd: Vec<f64> = (0..n_out)
                        .flat_map(|k| fd_columns.iter().map(move |column| row_scale(k) * column[k]))
                        .collect();

                    // vjp[k][j]: the cotangent of output k pulled back, against direction j
                    let end = solver.state_clone();
                    let unit = |k: usize| {
                        let mut e = V::zeros(3, Default::default());
                        e.set_index(k, 1.0);
                        e
                    };
                    let zero = V::zeros(3, Default::default());
                    let vjp: Vec<f64> = (0..n_out)
                        .flat_map(|k| {
                            let (c_y, c_dy) = if k < 3 {
                                (unit(k), zero.clone())
                            } else {
                                (zero.clone(), unit(k - 3))
                            };
                            let mut cot = with_y_dy(&end.zeros_like(), &c_y, &c_dy);
                            let mut p_bar = V::zeros(3, Default::default());
                            solver.step_vjp(&end, &mut cot, &mut p_bar).unwrap();
                            directions
                                .iter()
                                .map(|(y_dot, dy_dot, p_dot)| {
                                    row_scale(k)
                                        * (cot.as_ref().y.dot(y_dot)
                                            + cot.as_ref().dy.dot(dy_dot)
                                            + p_bar.dot(p_dot))
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    assert_close(
                        &V::from_vec(vjp, Default::default()),
                        &V::from_vec(fd, Default::default()),
                        1.0,
                    );
                }

                fn rebuilt_steps(dae: bool) {
                    // the first step, steps partway through after order and step-size changes,
                    // and one whose first attempt is shortened by a stop time. A solver that
                    // takes fewer steps to the stop time has them clamped to the last but one
                    let problem = problem(dae);
                    let mut solver = new_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let last = solve_to_stop(&mut solver).len() - 2;
                    for after in [0, 5, 40, 100] {
                        rebuilt_step(dae, after.min(last), None);
                    }
                    rebuilt_step(dae, 40.min(last), Some(0.4));
                }

                #[test]
                fn rebuilt_steps_ode() {
                    rebuilt_steps(false);
                }

                if_flag!($dae, {
                    #[test]
                    fn rebuilt_steps_dae() {
                        rebuilt_steps(true);
                    }
                });

                /// Step `solver` until it reaches its stop time with `step`, and return the state
                /// after each step.
                fn solve_to_stop<'a, S: OdeSolverMethod<'a, E, State = $State>>(
                    solver: &mut S,
                ) -> Vec<$State> {
                    let mut checkpoints = Vec::new();
                    loop {
                        let reason = solver.step().unwrap();
                        checkpoints.push(solver.state_clone());
                        if let OdeSolverStopReason::TstopReached = reason {
                            break;
                        }
                    }
                    checkpoints
                }

                /// A cotangent pulled back from the end of `checkpoints[..end]` to `start`, a
                /// trajectory of a solver with forward sensitivities, against those
                /// sensitivities: for each parameter `k`, the derivative of `<c, (y, dy)>` at the
                /// end, `<c, (s_k, ds_k)>`, is the cotangent's at the start against
                /// `(s_k, ds_k)` there, plus `p_bar[k]`.
                fn assert_vjp_matches_sensitivities<'a, S>(
                    solver: &mut S,
                    start: &$State,
                    checkpoints: &[$State],
                    end: usize,
                ) where
                    S: StepVjp<'a, E, State = $State>,
                {
                    let (c_y, c_dy): (V, V) = (
                        V::rand(3, 1.3, Default::default()),
                        V::rand(3, 2.1, Default::default()),
                    );
                    let last = &checkpoints[end - 1];
                    let mut cot = with_y_dy(&last.zeros_like(), &c_y, &c_dy);
                    let mut p_bar = V::zeros(3, Default::default());
                    for checkpoint in checkpoints[..end].iter().rev() {
                        solver.step_vjp(checkpoint, &mut cot, &mut p_bar).unwrap();
                    }
                    let (s_end, ds_end) = (last.as_ref().s, last.as_ref().ds);
                    let (s0, ds0) = (start.as_ref().s, start.as_ref().ds);
                    let (forward, backward): (Vec<f64>, Vec<f64>) = (0..3)
                        .map(|k| {
                            let forward = c_y.dot(&lane(s_end, k)) + c_dy.dot(&lane(ds_end, k));
                            let backward = cot.as_ref().y.dot(&lane(s0, k))
                                + cot.as_ref().dy.dot(&lane(ds0, k))
                                + p_bar.get_index(k);
                            (forward, backward)
                        })
                        .unzip();
                    assert_close(
                        &V::from_vec(backward, Default::default()),
                        &V::from_vec(forward, Default::default()),
                        TRAJECTORY_FACTOR,
                    );
                }

                /// The cotangent pulled back through the checkpoints of a solver with forward
                /// sensitivities matches them, for trajectories ending at many steps along the
                /// way, so that the comparison includes steps whose step size changes at their end.
                fn sensitivities(dae: bool) {
                    let problem = problem(dae);
                    let mut solver = new_sens_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let start = solver.state_clone();
                    let checkpoints = solve_to_stop(&mut solver);
                    let n = checkpoints.len();
                    // each pull back runs through every step before its end, so a few ends
                    // spread over the trajectory test all the steps, at a cost linear in n
                    for end in (1..n).step_by(n.div_ceil(10)).chain([n]) {
                        assert_vjp_matches_sensitivities(&mut solver, &start, &checkpoints, end);
                    }
                }

                #[test]
                fn sensitivities_ode() {
                    sensitivities(false);
                }

                if_flag!($dae, {
                    #[test]
                    fn sensitivities_dae() {
                        sensitivities(true);
                    }
                });

                /// A `set_stop_time` between steps rescales the solver's history after the
                /// checkpoint of the step before was taken: the next step's replay starts from the
                /// history as that checkpoint has it. Before the first step, the stop time
                /// shortens the first step, from the initial history.
                #[test]
                fn set_stop_time_between_steps() {
                    for steps_before in [0, 10] {
                        let problem = problem(false);
                        let mut solver = new_sens_solver(&problem);
                        let start = solver.state_clone();
                        let mut checkpoints = Vec::new();
                        for _ in 0..steps_before {
                            solver.step().unwrap();
                            checkpoints.push(solver.state_clone());
                        }
                        let (t, h) = (solver.state().t, solver.state().h);
                        solver.set_stop_time(t + 0.3 * h).unwrap();
                        assert!(
                            solver.state().h != h,
                            "the stop time did not rescale the state"
                        );
                        checkpoints.extend(solve_to_stop(&mut solver));
                        let n = checkpoints.len();
                        assert_vjp_matches_sensitivities(&mut solver, &start, &checkpoints, n);
                    }
                }

                /// A cotangent carrying three directions in its lanes, pulled back through a
                /// trajectory, against each direction on its own.
                fn batched_directions(dae: bool) {
                    let problem = problem(dae);
                    let mut solver = new_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let checkpoints = solve_to_stop(&mut solver);
                    let end = checkpoints.last().unwrap();
                    let mut cot = end.zeros_like_with_directions(3).unwrap();
                    let ctx3 = cot.as_ref().y.context().clone();
                    let (c_y, c_dy) =
                        (V::rand(3, 2.3, ctx3.clone()), V::rand(3, 2.9, ctx3.clone()));
                    cot = with_y_dy(&cot, &c_y, &c_dy);
                    let mut p_bar = V::zeros(3, ctx3.clone());
                    let mut cots: Vec<_> = (0..3)
                        .map(|k| with_y_dy(&end.zeros_like(), &lane(&c_y, k), &lane(&c_dy, k)))
                        .collect();
                    let mut p_bars: Vec<_> =
                        (0..3).map(|_| V::zeros(3, Default::default())).collect();
                    for checkpoint in checkpoints.iter().rev() {
                        solver.step_vjp(checkpoint, &mut cot, &mut p_bar).unwrap();
                        for (c, p) in cots.iter_mut().zip(&mut p_bars) {
                            solver.step_vjp(checkpoint, c, p).unwrap();
                        }
                    }
                    // a batched solve iterates until its worst lane converges, so each lane
                    // agrees with its own run to within the solves' tolerance
                    let assert_lanes = |batched: &V, single: &[V]| {
                        for (k, single) in single.iter().enumerate() {
                            assert_close(&lane(batched, k), single, TRAJECTORY_FACTOR);
                        }
                    };
                    let (y, dy): (Vec<_>, Vec<_>) = cots.iter().map(y_dy).unzip();
                    assert_lanes(cot.as_ref().y, &y);
                    assert_lanes(cot.as_ref().dy, &dy);
                    assert_lanes(&p_bar, &p_bars);
                }

                #[test]
                fn batched_directions_ode() {
                    batched_directions(false);
                }

                if_flag!($dae, {
                    #[test]
                    fn batched_directions_dae() {
                        batched_directions(true);
                    }
                });

                /// A parameter cotangent with fewer lanes than the cotangent, or with the wrong
                /// number of elements, is an error.
                #[test]
                fn input_mismatch() {
                    let problem = problem(false);
                    let mut solver = new_solver(&problem);
                    solver.step().unwrap();
                    let end = solver.state_clone();
                    let ctx = end.as_ref().y.context().clone();
                    let with_lanes = |n: usize| V::zeros(3, ctx.clone_with_nbatch(n).unwrap());
                    let mut cot = end.zeros_like_with_directions(3).unwrap();
                    let r = solver.step_vjp(&end, &mut cot.clone(), &mut with_lanes(1));
                    assert!(
                        matches!(
                            r,
                            Err(DiffsolError::OdeSolverError(
                                OdeSolverError::StepDifferentiationBatchMismatch { .. }
                            ))
                        ),
                        "1-lane p_bar, 3-lane cotangent"
                    );
                    let r = solver.step_vjp(&end, &mut end.zeros_like(), &mut V::zeros(2, ctx));
                    assert!(
                        matches!(
                            r,
                            Err(DiffsolError::OdeSolverError(
                                OdeSolverError::StepDifferentiationSizeMismatch { .. }
                            ))
                        ),
                        "p_bar of 2 elements, 3 parameters"
                    );
                    solver.step_vjp(&end, &mut cot, &mut with_lanes(3)).unwrap();
                }

                /// `with_vjp` is `plain` with `n` Jacobian updates added, where `updates(n)`: those
                /// a failed solve is retried after, those the schedule makes for each step, and
                /// any for a matrix that was not yet set.
                fn assert_only_jacobian_updates_added(
                    with_vjp: &OdeSolverStatistics,
                    plain: &OdeSolverStatistics,
                    updates: impl Fn(usize) -> bool,
                ) {
                    let mut with_vjp = with_vjp.clone();
                    let n = with_vjp.number_of_linear_solver_setups
                        - plain.number_of_linear_solver_setups;
                    assert!(updates(n), "unexpected number of Jacobian updates: {n}");
                    with_vjp.number_of_linear_solver_setups =
                        plain.number_of_linear_solver_setups;
                    with_vjp.number_of_linear_solver_setups_from_first_convergence_fail =
                        plain.number_of_linear_solver_setups_from_first_convergence_fail;
                    with_vjp.number_of_linear_solver_setups_from_second_convergence_fail =
                        plain.number_of_linear_solver_setups_from_second_convergence_fail;
                    with_vjp.number_of_linear_solver_setups_from_checkpoint =
                        plain.number_of_linear_solver_setups_from_checkpoint;
                    with_vjp.number_of_linear_solver_setups_from_step_success =
                        plain.number_of_linear_solver_setups_from_step_success;
                    assert_eq!(format!("{with_vjp:?}"), format!("{plain:?}"));
                }

                /// Pulling a cotangent back through the checkpoints leaves the solver's statistics
                /// alone, apart from the Jacobian updates it makes.
                #[test]
                fn step_vjp_adds_only_jacobian_updates() {
                    let mut problem = problem($dae);
                    problem.ode_options.derivative_solve_max_iter = 1;
                    let mut solver = new_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let checkpoints = solve_to_stop(&mut solver);
                    let plain = solver.get_statistics().clone();
                    let end = checkpoints.last().unwrap();
                    let mut cot = with_y_dy(
                        &end.zeros_like(),
                        &V::rand(3, 1.3, Default::default()),
                        &V::rand(3, 2.1, Default::default()),
                    );
                    let mut p_bar = V::zeros(3, Default::default());
                    for checkpoint in checkpoints.iter().rev() {
                        let _ = solver.step_vjp(checkpoint, &mut cot, &mut p_bar);
                    }
                    assert_only_jacobian_updates_added(solver.get_statistics(), &plain, |n| {
                        if $refreshes {
                            n > 0
                        } else {
                            n == 0
                        }
                    });
                }

                /// A `step` after a `step_vjp` takes the step it would have taken anyway: the
                /// derivative leaves nothing behind that changes the forward solve.
                #[test]
                fn step_after_step_vjp() {
                    for &dae in FORMS {
                        let problem = problem(dae);
                        let mut plain = new_solver(&problem);
                        let mut solver = new_solver(&problem);
                        let mut p_bar = V::zeros(3, Default::default());
                        for _ in 0..20 {
                            plain.step().unwrap();
                            solver.step().unwrap();
                            let end = solver.state_clone();
                            let mut cot = with_y_dy(
                                &end.zeros_like(),
                                &V::rand(3, 1.3, Default::default()),
                                &V::rand(3, 2.1, Default::default()),
                            );
                            solver.step_vjp(&end, &mut cot, &mut p_bar).unwrap();
                        }
                        assert_close(solver.state().y, plain.state().y, TRAJECTORY_FACTOR);
                    }
                }

                fn assert_no_step<'a, S>(solver: &mut S, what: &str)
                where
                    S: StepVjp<'a, AdTestEqn<M>>,
                {
                    let end = solver.state_clone();
                    let mut t = end.zeros_like();
                    let mut p_bar = V::zeros(3, Default::default());
                    assert!(
                        solver.step_vjp(&end, &mut t, &mut p_bar).is_err(),
                        "vjp after {what}"
                    );
                }

                /// `step_vjp` needs a state at the end of a step: not one before any step was
                /// taken, nor one modified or moved back within the step since.
                #[test]
                fn no_step() {
                    let problem = problem(false);
                    let mut solver = new_solver(&problem);
                    assert_no_step(&mut solver, "no step");

                    solver.step().unwrap();
                    solver.state_mut();
                    assert_no_step(&mut solver, "state_mut");

                    solver.step().unwrap();
                    // the next step is at most twice the last one, so this is within it
                    let t = solver.state().t - 0.25 * solver.state().h;
                    solver.state_mut_back(t).unwrap();
                    assert_no_step(&mut solver, "state_mut_back");
                }
            }
        };
    }

    step_ad_tests!(
        bdf_dense,
        M = crate::NalgebraMat<f64>,
        State = crate::BdfState<crate::NalgebraVec<f64>>,
        Config = crate::BdfConfig<f64>,
        new = { bdf::<crate::NalgebraLU<f64>> },
        from_state = { bdf_solver::<crate::NalgebraLU<f64>> },
        new_sens = { bdf_sens::<crate::NalgebraLU<f64>> },
        dae = true,
        refreshes = true,
        trajectory_factor = 50.0,
    );
    step_ad_tests!(
        bdf_sparse,
        M = crate::FaerSparseMat<f64>,
        State = crate::BdfState<crate::FaerVec<f64>>,
        Config = crate::BdfConfig<f64>,
        new = { bdf::<crate::FaerSparseLU<f64>> },
        from_state = { bdf_solver::<crate::FaerSparseLU<f64>> },
        new_sens = { bdf_sens::<crate::FaerSparseLU<f64>> },
        dae = true,
        refreshes = true,
        trajectory_factor = 50.0,
    );
    step_ad_tests!(
        tr_bdf2_dense,
        M = crate::NalgebraMat<f64>,
        State = crate::RkState<crate::NalgebraVec<f64>>,
        Config = crate::SdirkConfig<f64>,
        new = { tr_bdf2::<crate::NalgebraLU<f64>> },
        from_state = { tr_bdf2_solver::<crate::NalgebraLU<f64>> },
        new_sens = { tr_bdf2_sens::<crate::NalgebraLU<f64>> },
        dae = true,
        refreshes = true,
        trajectory_factor = 1.0,
    );
    step_ad_tests!(
        tr_bdf2_sparse,
        M = crate::FaerSparseMat<f64>,
        State = crate::RkState<crate::FaerVec<f64>>,
        Config = crate::SdirkConfig<f64>,
        new = { tr_bdf2::<crate::FaerSparseLU<f64>> },
        from_state = { tr_bdf2_solver::<crate::FaerSparseLU<f64>> },
        new_sens = { tr_bdf2_sens::<crate::FaerSparseLU<f64>> },
        dae = true,
        refreshes = true,
        trajectory_factor = 1.0,
    );
    step_ad_tests!(
        esdirk34_dense,
        M = crate::NalgebraMat<f64>,
        State = crate::RkState<crate::NalgebraVec<f64>>,
        Config = crate::SdirkConfig<f64>,
        new = { esdirk34::<crate::NalgebraLU<f64>> },
        from_state = { esdirk34_solver::<crate::NalgebraLU<f64>> },
        new_sens = { esdirk34_sens::<crate::NalgebraLU<f64>> },
        dae = true,
        refreshes = true,
        trajectory_factor = 10.0,
    );
    step_ad_tests!(
        tsit45_dense,
        M = crate::NalgebraMat<f64>,
        State = crate::RkState<crate::NalgebraVec<f64>>,
        Config = crate::ExplicitRkConfig<f64>,
        new = { tsit45 },
        from_state = { tsit45_solver },
        new_sens = { tsit45_sens },
        dae = false,
        refreshes = false,
        trajectory_factor = 1.0,
    );

    /// [crate::StepVjp::solve_dense_vjp] against forward sensitivities, for the loss `G = 1/2 sum_i |u(t_i)|^2`. They are different
    /// discretisations of the same gradient, so they agree to the solver tolerances.
    ///
    mod gradients {
        use super::{assert_close, with_tolerances, P, RTOL};
        use crate::ode_equations::test_models::ad_test_model::ad_test_problem;
        use crate::ode_equations::test_models::{
            exponential_decay::{
                exponential_decay_problem_adjoint, exponential_decay_problem_sens_with_out,
            },
            robertson::{robertson_adjoint, robertson_sens},
            robertson_ode::robertson_ode_elem_problem,
        };
        use crate::{
            Context, DenseMatrix, MatrixCommon, NalgebraLU, NalgebraMat, NalgebraVec, StepVjp,
            Vector, VectorView,
        };
        use crate::{OdeEquationsImplicitSens, SensitivitiesOdeSolverMethod};

        type M = NalgebraMat<f64>;
        type V = NalgebraVec<f64>;
        type LS = NalgebraLU<f64>;

        /// The loss of a dense solution, `1/2 sum_i |u(t_i)|^2`, in each lane.
        fn dense_loss(soln: &M) -> V {
            let ctx = *soln.column(0).into_owned().context();
            let mut loss = V::zeros(1, ctx);
            for i in 0..soln.ncols() {
                let u = soln.column(i).into_owned();
                loss.for_each_batch_host([&u], |l, [u], _| {
                    l[0] += 0.5 * u.iter().map(|x| x * x).sum::<f64>()
                });
            }
            loss
        }

        /// `g += 1/2 |u|^2`, in every lane
        fn half_squares(_i: usize, _t: f64, u: &V, g: &mut V) {
            g.for_each_batch_host([u], |g, [u], _| {
                g[0] += 0.5 * u.iter().map(|x| x * x).sum::<f64>()
            });
        }

        /// `dg/du = u`, in every lane
        fn identity(_i: usize, _t: f64, u: &V, dgdu: &mut V) {
            dgdu.for_each_batch_host([u], |d, [u], _| d.copy_from_slice(u));
        }

        /// The loss and its gradient by forward sensitivities, from `solver`.
        fn sens_gradient<'a, Eqn, S>(t_eval: &[f64], mut solver: S) -> (V, V)
        where
            Eqn: OdeEquationsImplicitSens<T = f64, V = V, M = M> + 'a,
            S: SensitivitiesOdeSolverMethod<'a, Eqn>,
        {
            let (soln, sens, _) = solver.solve_dense_sensitivities(t_eval).unwrap();
            let ctx = *soln.column(0).into_owned().context();
            // the entries of lane `k` of every column
            let lane_entries = |m: &M, k: usize| {
                (0..m.ncols())
                    .flat_map(|i| lane(&m.column(i).into_owned(), k).clone_as_vec())
                    .collect::<Vec<_>>()
            };
            let lanes = ctx.nbatch();
            let ys: Vec<Vec<f64>> = (0..lanes).map(|k| lane_entries(&soln, k)).collect();
            let loss = dense_loss(&soln);
            let grads: Vec<Vec<f64>> = (0..lanes)
                .map(|k| {
                    sens.iter()
                        .map(|s| {
                            lane_entries(s, k)
                                .iter()
                                .zip(&ys[k])
                                .map(|(s, y)| s * y)
                                .sum()
                        })
                        .collect()
                })
                .collect();
            let mut grad = V::zeros(sens.len(), ctx);
            grad.for_each_batch_host([], |g, [], k| g.copy_from_slice(&grads[k]));
            (loss, grad)
        }

        /// Lane `k` of a batched vector.
        fn lane(v: &V, k: usize) -> V {
            v.get_batch(k).into_owned()
        }

        fn t_evals(problem_times: impl Iterator<Item = f64>, tmax: f64) -> Vec<f64> {
            problem_times.filter(|t| *t > 0.0 && *t <= tmax).collect()
        }

        // Against forward sensitivities. The continuous adjoint does not yet solve the DAE form
        // of Robertson (its jumps at the output times leave the algebraic adjoint states
        // inconsistent), and with tight tolerances it fails on the ODE form before Robertson's
        // long horizon, which these reach. The AD test model's initial state and its consistent
        // initialisation depend on the parameters, which tests the derivative of the initial
        // state.
        macro_rules! sens_gradient_tests {
            ($($name:ident: $problem:expr, $sens_problem:expr, $t_eval:expr,
               $solver:ident $(::<$ls:ty>)?, $sens:ident $(::<$sls:ty>)?, $factor:expr;)+) => {
                $(
                    #[test]
                    fn $name() {
                        let problem = with_tolerances($problem);
                        let sens_problem = with_tolerances($sens_problem);
                        let t_eval: Vec<f64> = $t_eval;
                        let (loss, grad) = problem
                            .$solver$(::<$ls>)?()
                            .unwrap()
                            .solve_dense_vjp(&t_eval, 1, half_squares, identity)
                            .unwrap();
                        let (sens_loss, sens_grad) =
                            sens_gradient(&t_eval, sens_problem.$sens$(::<$sls>)?().unwrap());
                        assert_close(&loss, &sens_loss, $factor);
                        assert_close(&grad, &sens_grad, $factor);
                    }
                )+
            };
        }

        fn robertson_t_eval() -> Vec<f64> {
            let (_, soln) = robertson_adjoint::<M>();
            t_evals(soln.solution_points.iter().map(|p| p.t), 4e8)
        }

        // The two differentiate different steps, as solve_dense_vjp stops at each output time
        // and the sensitivity solve interpolates, so they agree to the solver tolerances.
        sens_gradient_tests! {
            robertson_bdf: robertson_adjoint::<M>().0, robertson_sens::<M>().0,
                robertson_t_eval(), bdf::<LS>, bdf_sens::<LS>, 3.0;
            robertson_esdirk34: robertson_adjoint::<M>().0, robertson_sens::<M>().0,
                robertson_t_eval(), esdirk34::<LS>, esdirk34_sens::<LS>, 10.0;
            robertson_ode_bdf: robertson_ode_elem_problem::<M>(2).0,
                robertson_ode_elem_problem::<M>(2).0, robertson_t_eval(), bdf::<LS>, bdf_sens::<LS>, 200.0;
            robertson_ode_esdirk34: robertson_ode_elem_problem::<M>(2).0,
                robertson_ode_elem_problem::<M>(2).0, robertson_t_eval(), esdirk34::<LS>, esdirk34_sens::<LS>,
                30.0;
            ad_test_dae_bdf: ad_test_problem::<M>(P, true, RTOL), ad_test_problem::<M>(P, true, RTOL),
                vec![1.0, 2.0, 5.0], bdf::<LS>, bdf_sens::<LS>, 1.0;
            ad_test_dae_esdirk34: ad_test_problem::<M>(P, true, RTOL),
                ad_test_problem::<M>(P, true, RTOL), vec![1.0, 2.0, 5.0], esdirk34::<LS>, esdirk34_sens::<LS>,
                2.0;
            ad_test_ode_tr_bdf2: ad_test_problem::<M>(P, false, RTOL),
                ad_test_problem::<M>(P, false, RTOL), vec![0.0, 1.0, 5.0], tr_bdf2::<LS>, tr_bdf2_sens::<LS>,
                10.0;
            exponential_decay_tsit45: exponential_decay_problem_adjoint::<M>(false, false).0,
                exponential_decay_problem_sens_with_out::<M>(false).0,
                (1..10).map(f64::from).collect(), tsit45, tsit45_sens, 1.0;
        }

        /// Two losses carried in the lanes of one solve, `G_0 = 1/2 sum |u|^2` and
        /// `G_1 = sum u_0`, against a solve for each.
        #[test]
        fn two_directions() {
            let problem = with_tolerances(robertson_adjoint::<M>().0);
            let t_eval = robertson_t_eval();
            let first = |_: usize, _: f64, u: &V, g: &mut V| {
                g.for_each_batch_host([u], |g, [u], _| g[0] += u[0])
            };
            let first_grad = |_: usize, _: f64, _u: &V, d: &mut V| {
                d.for_each_batch_host([], |d, [], _| d[0] = 1.0)
            };
            let (loss, grad) = problem
                .bdf::<LS>()
                .unwrap()
                .solve_dense_vjp(
                    &t_eval,
                    2,
                    |_, _, u, g| {
                        g.for_each_batch_host([u], |g, [u], lane| {
                            g[0] += if lane % 2 == 0 {
                                0.5 * u.iter().map(|x| x * x).sum::<f64>()
                            } else {
                                u[0]
                            }
                        })
                    },
                    |_, _, u, d| {
                        d.for_each_batch_host([u], |d, [u], lane| {
                            if lane % 2 == 0 {
                                d.copy_from_slice(u)
                            } else {
                                d[0] = 1.0
                            }
                        })
                    },
                )
                .unwrap();
            assert_eq!(grad.context().nbatch(), 2);
            let singles = [
                problem
                    .bdf::<LS>()
                    .unwrap()
                    .solve_dense_vjp(&t_eval, 1, half_squares, identity)
                    .unwrap(),
                problem
                    .bdf::<LS>()
                    .unwrap()
                    .solve_dense_vjp(&t_eval, 1, first, first_grad)
                    .unwrap(),
            ];
            // a batched solve iterates until its worst lane converges, so each lane agrees with
            // its own solve to within the solves' tolerance
            for (k, (single_loss, single_grad)) in singles.iter().enumerate() {
                assert_close(&lane(&loss, k), single_loss, 1.0);
                assert_close(&lane(&grad, k), single_grad, 1.0);
            }
        }

        /// The derivative of the consistent initial state, against central finite differences,
        /// for a DAE whose consistent `dy` and algebraic `y` depend on the parameters:
        /// `x' = -p0 x + z`, `0 = p1 x + p2 - z`, from `x = p0`.
        #[test]
        fn initial_state_derivative() {
            use crate::{OdeBuilder, OdeSolverState};
            let problem_at = |p: [f64; 3]| {
                let problem = OdeBuilder::<M>::new()
                    .p(p)
                    .rhs_adjoint_implicit(
                        |x: &[f64], p: &[f64], _t: f64, y: &mut [f64]| {
                            y[0] = -p[0] * x[0] + x[1];
                            y[1] = p[1] * x[0] + p[2] - x[1];
                        },
                        |_x: &[f64], p: &[f64], _t: f64, v: &[f64], y: &mut [f64]| {
                            y[0] = -p[0] * v[0] + v[1];
                            y[1] = p[1] * v[0] - v[1];
                        },
                        // -J^T v
                        |_x: &[f64], p: &[f64], _t: f64, v: &[f64], y: &mut [f64]| {
                            y[0] = p[0] * v[0] - p[1] * v[1];
                            y[1] = -v[0] + v[1];
                        },
                        // -f_p^T v
                        |x: &[f64], _p: &[f64], _t: f64, v: &[f64], y: &mut [f64]| {
                            y[0] = x[0] * v[0];
                            y[1] = -x[0] * v[1];
                            y[2] = -v[1];
                        },
                    )
                    .mass_adjoint(
                        |x: &[f64], _p: &[f64], _t: f64, beta: f64, y: &mut [f64]| {
                            y[0] = x[0] + beta * y[0];
                            y[1] *= beta;
                        },
                        |x: &[f64], _p: &[f64], _t: f64, beta: f64, y: &mut [f64]| {
                            y[0] = x[0] + beta * y[0];
                            y[1] *= beta;
                        },
                    )
                    .init_adjoint(
                        |p: &[f64], _t: f64, y: &mut [f64]| {
                            y[0] = p[0];
                            y[1] = 0.0;
                        },
                        // -y0_p^T v
                        |_p: &[f64], _t: f64, v: &[f64], y: &mut [f64]| {
                            y[0] = -v[0];
                            y[1] = 0.0;
                            y[2] = 0.0;
                        },
                        2,
                    )
                    .build()
                    .unwrap();
                with_tolerances(problem)
            };
            let p = [0.5, 2.0, 0.3];
            let (c_y, c_dy) = ([0.7, -1.3], [1.1, 0.4]);
            let objective = |p: [f64; 3]| {
                let state = problem_at(p).bdf_state::<LS>().unwrap();
                let (y, dy) = (
                    state.as_ref().y.clone_as_vec(),
                    state.as_ref().dy.clone_as_vec(),
                );
                (0..2).map(|i| c_y[i] * y[i] + c_dy[i] * dy[i]).sum::<f64>()
            };

            let problem = problem_at(p);
            let start = problem.bdf_state::<LS>().unwrap();
            let mut cot = start.zeros_like();
            cot.as_mut()
                .y
                .copy_from(&V::from_vec(c_y.to_vec(), Default::default()));
            cot.as_mut()
                .dy
                .copy_from(&V::from_vec(c_dy.to_vec(), Default::default()));
            let mut p_bar = V::zeros(3, Default::default());
            super::super::initial_state_vjp(&problem, &start, &mut cot, &mut p_bar).unwrap();

            let eps = 1e-6;
            let fd: Vec<f64> = (0..3)
                .map(|k| {
                    let (mut plus, mut minus) = (p, p);
                    plus[k] += eps;
                    minus[k] -= eps;
                    (objective(plus) - objective(minus)) / (2.0 * eps)
                })
                .collect();
            assert_close(&p_bar, &V::from_vec(fd, Default::default()), 1.0);
        }

        #[test]
        fn rejects_bad_inputs() {
            let (problem, _) = robertson_adjoint::<M>();
            let mut solver = problem.bdf::<LS>().unwrap();
            assert!(solver
                .solve_dense_vjp(&[1.0, 0.5], 1, half_squares, identity)
                .is_err());
            assert!(solver
                .solve_dense_vjp(&[1.0], 0, half_squares, identity)
                .is_err());
        }
    }
}
