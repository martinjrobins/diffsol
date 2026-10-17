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
//!
//! For the forward mode, use the forward sensitivities of
//! [crate::SensitivitiesOdeSolverMethod]. With [crate::Bdf], they differentiate the same steps
//! the same way, solved to the same `derivative_solve_tolerance` (see [crate::BdfConfig]), so the
//! two agree as closely as that tolerance. Leave the sensitivities out of error control (the
//! default, see [crate::OdeBuilder::sens_rtol]) for the steps to be those of a run without them,
//! as long as no sensitivity solve needs a fresh Newton matrix.

use crate::{error::DiffsolError, OdeEquationsImplicitAdjoint, OdeSolverMethod};

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
    /// `cotangent` does not have the sizes of `end` or `p_bar` does not have `nparams` elements,
    /// and with `StepDifferentiationBatchMismatch` if the lanes of `cotangent` and `p_bar` do not
    /// fit those of `end` (see the [module docs](self)). On any other error, `cotangent` and `p_bar` are left partly
    /// transformed and must not be reused.
    ///
    /// Its linear solves use the solver's own Newton matrix as a preconditioner. If that is too
    /// poor for them to converge, it is refreshed, which a later [OdeSolverMethod::step] then
    /// starts from.
    fn step_vjp(
        &mut self,
        end: &Self::State,
        cotangent: &mut Self::State,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError>;
}

#[cfg(test)]
mod tests {
    //! The step derivatives tested through the public API only: cotangents are seeded and read
    //! through the state's `y` and `dy`, so the tests apply to any solver that implements
    //! [super::StepVjp]. They are written once in `step_ad_tests!` and run for each solver and
    //! linear solver. The reference derivatives are finite differences of single steps and, for
    //! whole trajectories, the solver's own forward sensitivities.

    use crate::Vector;

    pub(super) const P: [f64; 3] = [1.5, 3.0, 1.0];
    pub(super) const T_FINAL: f64 = 5.0;
    /// A derivative solve tolerance, a millionth of the error tolerance, tight enough for the
    /// derivatives to agree to the tests' tolerances.
    pub(super) const DERIVATIVE_TOL: f64 = 1e-6;
    /// An iteration limit on the derivative solves low enough that some of them need a fresh
    /// preconditioner, but not so low that they then fail.
    pub(super) const MAX_ITER_FORCING_REFRESHES: usize = 2;

    /// A pseudo-random vector of the model's size.
    pub(super) fn rand<V: Vector<T = f64>>(seed: f64) -> V {
        let mut v = V::zeros(3, Default::default());
        v.for_each_elem([], move |v, [], _, i| *v = ((i as f64 + 1.0) * seed).sin());
        v
    }

    pub(super) fn dot<V: Vector<T = f64>>(a: &V, b: &V) -> f64 {
        let mut d = V::zeros(1, a.context().clone());
        V::reduce_elem(
            &mut d,
            [a, b],
            0.0,
            |[a, b], _, i| a[i] * b[i],
            |x, y| x + y,
        );
        d.get_index(0)
    }

    pub(super) fn assert_close(a: f64, b: f64, rtol: f64, what: &str) {
        let scale = 1.0 + a.abs().max(b.abs());
        assert!((a - b).abs() <= rtol * scale, "{what}: {a} != {b}");
    }

    /// Pseudo-random `(y, dy)` perturbation directions. For the DAE they satisfy the linearised
    /// constraint `y0 + y1 + y2 = p2` with `p_dot`, so that a perturbed state stays consistent:
    /// an inconsistent one makes a step jump, whatever its size.
    pub(super) fn tangent_y_dy<V: Vector<T = f64>>(seed: f64, dae: bool, p_dot: &V) -> (V, V) {
        let (mut y, mut dy): (V, V) = (rand(seed), rand(1.7 * seed));
        if dae {
            y.for_each_batch([p_dot], |y, [p], _| y[2] = p[2] - y[0] - y[1]);
            dy.for_each_batch([], |dy, [], _| dy[2] = -dy[0] - dy[1]);
        }
        (y, dy)
    }

    /// The tests for one solver and linear solver: on matrix type `$M`, with linear solver `$LS`,
    /// for the solver built by the problem methods `$new`, `$from_state` (from a state of type
    /// `$State`, with a config of type `$Config`) and, with forward sensitivities, `$new_sens`.
    macro_rules! step_ad_tests {
        (
            $mod_name:ident,
            M = $M:ty,
            LS = $LS:ty,
            State = $State:ty,
            Config = $Config:ty,
            new = $new:ident,
            from_state = $from_state:ident,
            new_sens = $new_sens:ident $(,)?
        ) => {
            mod $mod_name {
                use super::{
                    assert_close, dot, rand, tangent_y_dy, DERIVATIVE_TOL,
                    MAX_ITER_FORCING_REFRESHES, P, T_FINAL,
                };
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
                type Problem = OdeSolverProblem<AdTestEqn<M>>;

                type E = AdTestEqn<M>;

                /// A solver with its derivative solves converged to `DERIVATIVE_TOL`.
                fn new_solver(
                    p: &Problem,
                ) -> impl OdeSolverMethod<'_, E, State = $State, Config = $Config> + StepVjp<'_, E>
                {
                    let mut solver = p.$new::<$LS>().unwrap();
                    solver.config_mut().derivative_solve_tolerance = DERIVATIVE_TOL;
                    solver
                }

                /// As [new_solver], started from `state`.
                fn solver_from_state(
                    p: &Problem,
                    state: $State,
                ) -> impl OdeSolverMethod<'_, E, State = $State> + StepVjp<'_, E> {
                    let mut solver = p.$from_state::<$LS>(state).unwrap();
                    solver.config_mut().derivative_solve_tolerance = DERIVATIVE_TOL;
                    solver
                }

                /// A solver with forward sensitivities, its derivative solves converged to
                /// `DERIVATIVE_TOL`.
                fn new_sens_solver(
                    p: &Problem,
                ) -> impl OdeSolverMethod<'_, E, State = $State, Config = $Config> + StepVjp<'_, E>
                {
                    let mut solver = p.$new_sens::<$LS>().unwrap();
                    solver.config_mut().derivative_solve_tolerance = DERIVATIVE_TOL;
                    solver
                }

                fn zeros() -> V {
                    V::zeros(3, Default::default())
                }

                /// A tolerance for [Vector::assert_eq_norm], as its `atol` with `tol` as its `rtol`.
                fn tol(tol: f64) -> V {
                    V::from_element(3, tol, Default::default())
                }

                fn problem(dae: bool) -> Problem {
                    ad_test_problem::<M>(P, dae, 1e-8)
                }

                fn problem_at(p: &V, dae: bool) -> Problem {
                    let mut problem = problem(dae);
                    problem.eqn.set_params(p);
                    problem
                }

                fn params(problem: &Problem) -> V {
                    let mut p = zeros();
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

                /// A pseudo-random vector of the model's size with the lanes of `ctx`.
                fn rand_lanes(ctx: &<M as MatrixCommon>::C, seed: f64) -> V {
                    let mut v = V::zeros(3, ctx.clone());
                    v.for_each_elem([], move |v, [], lane, i| {
                        *v = ((3 * lane + i) as f64 + 1.0) * seed;
                        *v = v.sin();
                    });
                    v
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
                /// the step if given. `step_vjp` of the step is checked against central finite
                /// differences of `step()` in `(y, dy, p)`, in a pseudo-random direction.
                fn rebuilt_step(dae: bool, after: usize, tstop_fraction: Option<f64>, seed: f64) {
                    let problem0 = problem(dae);
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
                    // at a step size whose first attempt is accepted: the size of a retried
                    // step depends on the error estimate, so it would differ between the
                    // perturbed steps. A probe step finds one
                    // TODO: once https://github.com/martinjrobins/diffsol/issues/394 is fixed
                    // then we can turn off error control and this test would be simpler
                    let h_accepted = step_from(&problem0, start.clone(), T_FINAL).state().t - t0;
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

                    // finite differences, on steps that must take the same step size. The Newton
                    // error in dy = (history) / h grows as h shrinks, so eps is not too small
                    let p0 = params(&problem0);
                    let p_dot: V = rand(0.31 * seed);
                    let (y_dot, dy_dot) = tangent_y_dy(seed, dae, &p_dot);
                    let eps = 1e-4;
                    let fd = |a: V, b: &V| (a - b) * scale(0.5 / eps);
                    // a perturbed restart: parameters p + e p_dot, y + e y_dot and dy + e dy_dot
                    let perturbed = |e: f64| {
                        let problem = problem_at(&axpy(&p0, e, &p_dot), dae);
                        let restart =
                            with_y_dy(&start, &axpy(&y0, e, &y_dot), &axpy(&dy0, e, &dy_dot));
                        let solver = step_from(&problem, restart, tstop);
                        assert_eq!(solver.state().t, t_end, "the perturbed step changed size");
                        y_dy(&solver.state_clone())
                    };
                    let ((y_p, dy_p), (y_m, dy_m)) = (perturbed(eps), perturbed(-eps));
                    let (y_fd, dy_fd) = (fd(y_p, &y_m), fd(dy_p, &dy_m));

                    // <c_end, fd(t)> = <vjp(c_end), t> + <p_bar, p_dot>
                    let (c_y, c_dy): (V, V) = (rand(2.9 * seed), rand(3.7 * seed));
                    let end = solver.state_clone();
                    let mut cot = with_y_dy(&end.zeros_like(), &c_y, &c_dy);
                    let mut p_bar = zeros();
                    solver.step_vjp(&end, &mut cot, &mut p_bar).unwrap();
                    let forward = dot(&c_y, &y_fd) + dot(&c_dy, &dy_fd);
                    let backward = dot(cot.as_ref().y, &y_dot)
                        + dot(cot.as_ref().dy, &dy_dot)
                        + dot(&p_bar, &p_dot);
                    assert_close(forward, backward, 1e-5, "finite differences");
                }

                fn rebuilt_steps(dae: bool) {
                    // the first step, steps partway through after order and step-size changes,
                    // and one whose first attempt is shortened by a stop time
                    for (i, after) in [0, 5, 40, 100].into_iter().enumerate() {
                        rebuilt_step(dae, after, None, 0.3 + i as f64);
                    }
                    rebuilt_step(dae, 40, Some(0.4), 7.1);
                }

                #[test]
                fn rebuilt_steps_ode() {
                    rebuilt_steps(false);
                }

                #[test]
                fn rebuilt_steps_dae() {
                    rebuilt_steps(true);
                }

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
                    let (c_y, c_dy): (V, V) = (rand(1.3), rand(2.1));
                    let last = &checkpoints[end - 1];
                    let mut cot = with_y_dy(&last.zeros_like(), &c_y, &c_dy);
                    let mut p_bar = zeros();
                    for checkpoint in checkpoints[..end].iter().rev() {
                        solver.step_vjp(checkpoint, &mut cot, &mut p_bar).unwrap();
                    }
                    let (s_end, ds_end) = (last.as_ref().s, last.as_ref().ds);
                    let (s0, ds0) = (start.as_ref().s, start.as_ref().ds);
                    for k in 0..3 {
                        let forward = dot(&c_y, &lane(s_end, k)) + dot(&c_dy, &lane(ds_end, k));
                        let backward = dot(cot.as_ref().y, &lane(s0, k))
                            + dot(cot.as_ref().dy, &lane(ds0, k))
                            + p_bar.get_index(k);
                        assert_close(
                            forward,
                            backward,
                            1e-7,
                            &format!("parameter {k}, trajectory to step {end}"),
                        );
                    }
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
                    for end in (1..n).step_by(3).chain([n]) {
                        assert_vjp_matches_sensitivities(&mut solver, &start, &checkpoints, end);
                    }
                }

                #[test]
                fn sensitivities_ode() {
                    sensitivities(false);
                }

                #[test]
                fn sensitivities_dae() {
                    sensitivities(true);
                }

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
                    let (c_y, c_dy) = (rand_lanes(&ctx3, 2.3), rand_lanes(&ctx3, 2.9));
                    cot = with_y_dy(&cot, &c_y, &c_dy);
                    let mut p_bar = V::zeros(3, ctx3.clone());
                    let mut cots: Vec<_> = (0..3)
                        .map(|k| with_y_dy(&end.zeros_like(), &lane(&c_y, k), &lane(&c_dy, k)))
                        .collect();
                    let mut p_bars: Vec<_> = (0..3).map(|_| zeros()).collect();
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
                            lane(batched, k).assert_eq_norm(single, &tol(1e-6), 1e-6, 1.0);
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

                #[test]
                fn batched_directions_dae() {
                    batched_directions(true);
                }

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

                /// `with_derivatives` is `plain` with `refreshes` derivative refreshes added.
                fn assert_only_derivative_refreshes_added(
                    with_derivatives: &OdeSolverStatistics,
                    plain: &OdeSolverStatistics,
                    refreshes: impl Fn(usize) -> bool,
                ) {
                    let mut with_derivatives = with_derivatives.clone();
                    let n = with_derivatives.number_of_derivative_jacobian_refreshes;
                    assert!(refreshes(n), "unexpected number of refreshes: {n}");
                    with_derivatives.number_of_derivative_jacobian_refreshes = 0;
                    assert_eq!(format!("{with_derivatives:?}"), format!("{plain:?}"));
                }

                /// Pulling a cotangent back through the checkpoints leaves the solver's statistics
                /// alone, apart from the derivatives' own refreshes.
                #[test]
                fn step_vjp_leaves_the_statistics_unchanged() {
                    let problem = problem(true);
                    let mut solver = new_solver(&problem);
                    solver.config_mut().derivative_solve_max_iter = 1;
                    solver.set_stop_time(T_FINAL).unwrap();
                    let checkpoints = solve_to_stop(&mut solver);
                    let plain = solver.get_statistics().clone();
                    let end = checkpoints.last().unwrap();
                    let mut cot = with_y_dy(&end.zeros_like(), &rand(1.3), &rand(2.1));
                    let mut p_bar = zeros();
                    for checkpoint in checkpoints.iter().rev() {
                        let _ = solver.step_vjp(checkpoint, &mut cot, &mut p_bar);
                    }
                    assert_only_derivative_refreshes_added(solver.get_statistics(), &plain, |n| {
                        n > 0
                    });
                }

                /// A sensitivity solve that the Newton matrix is too poor a preconditioner for
                /// (forced with a tiny iteration limit, which a fresh matrix still meets at
                /// `DERIVATIVE_TOL`) refreshes it and retries, rather than rejecting the step, and
                /// still reaches the same sensitivities.
                #[test]
                fn sensitivity_solve_refreshes() {
                    for dae in [false, true] {
                        let problem = problem(dae);
                        let mut plain = new_sens_solver(&problem);
                        plain.set_stop_time(T_FINAL).unwrap();
                        solve_to_stop(&mut plain);
                        let mut solver = new_sens_solver(&problem);
                        solver.config_mut().derivative_solve_max_iter = MAX_ITER_FORCING_REFRESHES;
                        solver.set_stop_time(T_FINAL).unwrap();
                        solve_to_stop(&mut solver);
                        let stats = solver.get_statistics();
                        assert!(stats.number_of_derivative_jacobian_refreshes > 0);
                        assert_eq!(
                            stats.number_of_nonlinear_solver_fails, 0,
                            "a step was rejected"
                        );
                        solver
                            .state()
                            .s
                            .assert_eq_norm(plain.state().s, &tol(1e-6), 1e-6, 1.0);
                    }
                }

                /// With the sensitivities left out of error control, a solver with them takes
                /// exactly the steps of one without, whatever the derivative tolerance, as long
                /// as their solves need no fresh preconditioner.
                #[test]
                fn sensitivities_take_the_same_steps() {
                    for dae in [false, true] {
                        let problem = problem(dae);
                        let mut plain = new_solver(&problem);
                        plain.set_stop_time(T_FINAL).unwrap();
                        let plain_steps = solve_to_stop(&mut plain);
                        let mut solver = new_sens_solver(&problem);
                        solver.set_stop_time(T_FINAL).unwrap();
                        let steps = solve_to_stop(&mut solver);
                        let stats = solver.get_statistics();
                        assert_eq!(stats.number_of_derivative_jacobian_refreshes, 0);
                        assert_eq!(steps.len(), plain_steps.len());
                        for (s, p) in steps.iter().zip(&plain_steps) {
                            assert_eq!(s.as_ref().t, p.as_ref().t);
                        }
                        solver.state().y.assert_eq_st(plain.state().y, 0.0);
                    }
                }

                /// A `step` after a `step_vjp` takes the step it would have taken anyway: the
                /// derivative leaves nothing behind that changes the forward solve.
                #[test]
                fn step_after_step_vjp() {
                    for dae in [false, true] {
                        let problem = problem(dae);
                        let mut plain = new_solver(&problem);
                        let mut solver = new_solver(&problem);
                        let mut p_bar = zeros();
                        for _ in 0..20 {
                            plain.step().unwrap();
                            solver.step().unwrap();
                            let end = solver.state_clone();
                            let mut cot = with_y_dy(&end.zeros_like(), &rand(1.3), &rand(2.1));
                            solver.step_vjp(&end, &mut cot, &mut p_bar).unwrap();
                        }
                        solver
                            .state()
                            .y
                            .assert_eq_norm(plain.state().y, &tol(1e-8), 1e-8, 1.0);
                    }
                }

                /// After a `step_vjp`, the solver's own state is as it was: a `checkpoint`, which
                /// refactorises the Newton matrix (so whatever the derivative's own refreshes left
                /// behind no longer matters), and the steps after it are those of a solver that
                /// took no derivatives.
                #[test]
                fn checkpoint_after_step_vjp() {
                    for dae in [false, true] {
                        let problem = problem(dae);
                        let mut plain = new_solver(&problem);
                        let mut solver = new_solver(&problem);
                        let mut p_bar = zeros();
                        for _ in 0..20 {
                            plain.step().unwrap();
                            solver.step().unwrap();
                            let end = solver.state_clone();
                            let mut cot = with_y_dy(&end.zeros_like(), &rand(1.3), &rand(2.1));
                            solver.step_vjp(&end, &mut cot, &mut p_bar).unwrap();
                            plain.checkpoint();
                            solver.checkpoint();
                        }
                        assert_only_derivative_refreshes_added(
                            solver.get_statistics(),
                            plain.get_statistics(),
                            |_| true,
                        );
                        solver.state().y.assert_eq_st(plain.state().y, 0.0);
                    }
                }

                fn assert_no_step<'a, S>(solver: &mut S, what: &str)
                where
                    S: StepVjp<'a, AdTestEqn<M>>,
                {
                    let end = solver.state_clone();
                    let mut t = end.zeros_like();
                    let mut p_bar = zeros();
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
        LS = crate::NalgebraLU<f64>,
        State = crate::BdfState<crate::NalgebraVec<f64>>,
        Config = crate::BdfConfig<f64>,
        new = bdf,
        from_state = bdf_solver,
        new_sens = bdf_sens,
    );
    step_ad_tests!(
        bdf_sparse,
        M = crate::FaerSparseMat<f64>,
        LS = crate::FaerSparseLU<f64>,
        State = crate::BdfState<crate::FaerVec<f64>>,
        Config = crate::BdfConfig<f64>,
        new = bdf,
        from_state = bdf_solver,
        new_sens = bdf_sens,
    );
}
