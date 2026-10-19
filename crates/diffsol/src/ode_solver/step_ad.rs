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
//! [crate::SensitivitiesOdeSolverMethod]. They differentiate the same steps the same way: for
//! [crate::Bdf] and [crate::Sdirk] both are solved to the same `derivative_solve_tolerance` (see
//! [crate::OdeSolverOptions]), so the two agree as closely as that tolerance, and for
//! [crate::ExplicitRk] neither needs a solve, so they agree to rounding. Leave the
//! sensitivities out of error control (the default, see [crate::OdeBuilder::sens_rtol]) for the
//! steps to be those of a run without them, as long as no sensitivity solve fails.

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
    /// `cotangent` does not have the sizes of `end` or `p_bar` does not have `nparams` elements.
    ///
    /// Its linear solves use the solver's own Newton matrix as a preconditioner, kept up to date
    /// as a step would: on the solver's Jacobian update schedule, advanced by each replayed step,
    /// and, if a solve fails, with the Jacobian updates a step makes after a failed Newton solve,
    /// without the smaller step size a replayed step cannot take.
    ///
    /// The mass matrix is taken to be independent of the parameters, as for the forward
    /// sensitivities: for a model whose mass matrix depends on them, `p_bar` misses that term.
    fn step_vjp(
        &mut self,
        end: &Self::State,
        cotangent: &mut Self::State,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError>;
}

#[cfg(test)]
mod tests {
    //! Tests for the step derivatives. The reference derivatives are finite differences of 
    //! single steps and, for whole trajectories, the solver's own forward sensitivities.

    use crate::Vector;

    const P: [f64; 3] = [1.5, 3.0, 1.0];
    const T_FINAL: f64 = 5.0;
    /// A derivative solve tolerance, a millionth of the error tolerance, tight enough for the
    /// derivatives to agree to the tests' tolerances.
    const DERIVATIVE_TOL: f64 = 1e-6;

    fn assert_close(a: f64, b: f64, rtol: f64, what: &str) {
        let scale = 1.0 + a.abs().max(b.abs());
        assert!((a - b).abs() <= rtol * scale, "{what}: {a} != {b}");
    }

    /// Pseudo-random `(y, dy)` perturbation directions. For the DAE they satisfy the linearised
    /// constraint `y0 + y1 + y2 = p2` with `p_dot`, so that a perturbed state stays consistent:
    /// an inconsistent one makes a step jump, whatever its size.
    fn tangent_y_dy<V: Vector<T = f64>>(seed: f64, dae: bool, p_dot: &V) -> (V, V) {
        let (mut y, mut dy): (V, V) = (
            V::rand(3, seed, Default::default()),
            V::rand(3, 1.7 * seed, Default::default()),
        );
        if dae {
            y.for_each_batch([p_dot], |y, [p], _| y[2] = p[2] - y[0] - y[1]);
            dy.for_each_batch([], |dy, [], _| dy[2] = -dy[0] - dy[1]);
        }
        (y, dy)
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
    /// solver has none).
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
            refreshes = $refreshes:tt $(,)?
        ) => {
            mod $mod_name {
                use super::{assert_close, tangent_y_dy, DERIVATIVE_TOL, P, T_FINAL};

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

                /// A tolerance for [Vector::assert_eq_norm], as its `atol` with `tol` as its `rtol`.
                fn tol(tol: f64) -> V {
                    V::from_element(3, tol, Default::default())
                }

                /// The test model, with its derivative solves converged to `DERIVATIVE_TOL`.
                fn problem(dae: bool) -> Problem {
                    let mut problem = ad_test_problem::<M>(P, dae, 1e-8);
                    problem.ode_options.derivative_solve_tolerance = DERIVATIVE_TOL;
                    problem
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
                /// the step if given. `step_vjp` of the step is checked against central finite
                /// differences of `step()` in `(y, dy, p)`, in a pseudo-random direction.
                fn rebuilt_step(dae: bool, after: usize, tstop_fraction: Option<f64>, seed: f64) {
                    // the steps' Newton solves converged as tightly as the derivatives': a method
                    // that takes its first stage from a perturbed dy has to take small steps
                    // below, and the Newton error in dy = K / h grows as h shrinks
                    let converged = |mut problem: Problem| {
                        problem.ode_options.nonlinear_solver_tolerance = DERIVATIVE_TOL;
                        problem
                    };
                    let problem0 = converged(problem(dae));
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
                    let p_dot: V = V::rand(3, 0.31 * seed, Default::default());
                    let (y_dot, dy_dot) = tangent_y_dy(seed, dae, &p_dot);
                    let eps = 1e-4;
                    let fd = |a: V, b: &V| (a - b) * scale(0.5 / eps);
                    // a perturbed problem and restart: parameters p + e p_dot, y + e y_dot and
                    // dy + e dy_dot
                    let problem_e = |e: f64| converged(problem_at(&axpy(&p0, e, &p_dot), dae));
                    let restart_e = |start: &$State, e: f64| {
                        with_y_dy(start, &axpy(&y0, e, &y_dot), &axpy(&dy0, e, &dy_dot))
                    };

                    // at a step size whose first attempt is accepted, for the step and both
                    // perturbed ones: the size of a retried step depends on the error estimate,
                    // so it would differ between them. Probe steps find one: a perturbed dy is
                    // not f(y), which a method that takes its first stage from dy sees as error
                    // TODO: once https://github.com/martinjrobins/diffsol/issues/394 is fixed
                    // then we can turn off error control and this test would be simpler
                    let h_accepted = [0.0, eps, -eps]
                        .map(|e| {
                            step_from(&problem_e(e), restart_e(&start, e), T_FINAL)
                                .state()
                                .t
                                - t0
                        })
                        .into_iter()
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

                    let perturbed = |e: f64| {
                        let problem = problem_e(e);
                        let solver = step_from(&problem, restart_e(&start, e), tstop);
                        assert_eq!(solver.state().t, t_end, "the perturbed step changed size");
                        y_dy(&solver.state_clone())
                    };
                    let ((y_p, dy_p), (y_m, dy_m)) = (perturbed(eps), perturbed(-eps));
                    let (y_fd, dy_fd) = (fd(y_p, &y_m), fd(dy_p, &dy_m));

                    // <c_end, fd(t)> = <vjp(c_end), t> + <p_bar, p_dot>
                    let (c_y, c_dy): (V, V) = (
                        V::rand(3, 2.9 * seed, Default::default()),
                        V::rand(3, 3.7 * seed, Default::default()),
                    );
                    let end = solver.state_clone();
                    let mut cot = with_y_dy(&end.zeros_like(), &c_y, &c_dy);
                    let mut p_bar = V::zeros(3, Default::default());
                    solver.step_vjp(&end, &mut cot, &mut p_bar).unwrap();
                    let forward = c_y.dot(&y_fd) + c_dy.dot(&dy_fd);
                    let backward = cot.as_ref().y.dot(&y_dot)
                        + cot.as_ref().dy.dot(&dy_dot)
                        + p_bar.dot(&p_dot);
                    assert_close(forward, backward, 1e-5, "finite differences");
                }

                fn rebuilt_steps(dae: bool) {
                    // the first step, steps partway through after order and step-size changes,
                    // and one whose first attempt is shortened by a stop time. A solver that
                    // takes fewer steps to the stop time has them clamped to the last but one
                    let problem = problem(dae);
                    let mut solver = new_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let last = solve_to_stop(&mut solver).len() - 2;
                    for (i, after) in [0, 5, 40, 100].into_iter().enumerate() {
                        rebuilt_step(dae, after.min(last), None, 0.3 + i as f64);
                    }
                    rebuilt_step(dae, 40.min(last), Some(0.4), 7.1);
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
                    for k in 0..3 {
                        let forward = c_y.dot(&lane(s_end, k)) + c_dy.dot(&lane(ds_end, k));
                        let backward = cot.as_ref().y.dot(&lane(s0, k))
                            + cot.as_ref().dy.dot(&lane(ds0, k))
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
                /// a failed solve is retried after, those the schedule makes after each step, and
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
                        solver
                            .state()
                            .y
                            .assert_eq_norm(plain.state().y, &tol(1e-8), 1e-8, 1.0);
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
    );
}
