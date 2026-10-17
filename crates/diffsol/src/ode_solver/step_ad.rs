//! Forward- and reverse-mode derivatives of a single accepted solver step.
//!
//! These differentiate what the solver actually computed ("discretise-then-optimise"), in
//! contrast to the continuous adjoint equations of [crate::AdjointOdeSolverMethod]. Error
//! control is not differentiated: the step size, the order and the accept/reject decisions of
//! each step are treated as fixed, and only the accepted attempt contributes.
//!
//! Tangents and cotangents have the same type as the solver state, `Self::State`. What they
//! hold, and which of their fields are read, depends on the solver; see its implementation.
//! Allocate them with [crate::OdeSolverState::zeros_like].
//!
//! A forward-mode pass interleaves [StepJvp::step_jvp] with [crate::OdeSolverMethod::step]:
//! ```ignore
//! solver.step()?;
//! solver.step_jvp(&mut tangent, &p_dot)?;
//! ```
//! A reverse-mode pass saves a [crate::OdeSolverMethod::state_clone] after every step, then
//! pulls a cotangent back through them in reverse:
//! ```ignore
//! while let Some(end) = checkpoints.pop() {
//!     solver.set_replay_state(end);
//!     solver.step_vjp(&mut cotangent, &mut p_bar)?;
//! }
//! ```

use crate::{
    error::DiffsolError, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeSolverMethod,
};

/// Forward-mode derivative (Jacobian-vector product) of the last accepted step.
pub trait StepJvp<'a, Eqn: OdeEquationsImplicitSens + 'a>: OdeSolverMethod<'a, Eqn> {
    /// Push `tangent`, a tangent state at the start of the last accepted step, through that
    /// step, together with the parameter tangent `p_dot` (of length `nparams`). The step is the
    /// one that ended at [OdeSolverMethod::state]. On return, `tangent` is the tangent state at
    /// its end.
    fn step_jvp(&mut self, tangent: &mut Self::State, p_dot: &Eqn::V) -> Result<(), DiffsolError>;
}

/// Reverse-mode derivative (vector-Jacobian product) of a checkpointed step.
pub trait StepVjp<'a, Eqn: OdeEquationsImplicitAdjoint + 'a>: OdeSolverMethod<'a, Eqn> {
    /// Load `state`, a checkpointed end-of-step state, for a following
    /// [StepVjp::step_vjp]. Unlike [OdeSolverMethod::set_state] this need not refresh the
    /// Jacobian, since the solves in [StepVjp::step_vjp] detect a stale one themselves.
    fn set_replay_state(&mut self, state: Self::State) {
        self.set_state(state)
    }

    /// Pull `cotangent`, a cotangent state at [OdeSolverMethod::state], back through the step
    /// that ended there. On return it is the cotangent state at the start of that step, and the
    /// parameter cotangent has been added to `p_bar` (of length `nparams`).
    fn step_vjp(
        &mut self,
        cotangent: &mut Self::State,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError>;
}

#[cfg(test)]
mod tests {
    /// The step-derivative tests, written once against concrete types: `$M` for the equations
    /// and `$LS` for the linear solver. The state's dense matrix is `$M`'s default.
    macro_rules! step_ad_tests {
        ($mod_name:ident, $M:ty, $LS:ty) => {
            mod $mod_name {
                use crate::ode_equations::test_models::ad_test_model::{
                    ad_test_problem, AdTestEqn,
                };
                use crate::{
                    Bdf, BdfState, DefaultDenseMatrix, DenseMatrix, MatrixCommon, OdeSolverMethod,
                    OdeSolverProblem, OdeSolverState, OdeSolverStopReason, StepJvp, StepVjp,
                    Vector, VectorView,
                };

                type M = $M;
                type LS = $LS;
                type V = <M as MatrixCommon>::V;
                type DM = <V as DefaultDenseMatrix>::M;
                type State = BdfState<V, DM>;
                type Problem = OdeSolverProblem<AdTestEqn<M>>;

                const P: [f64; 3] = [1.5, 3.0, 1.0];
                const T_FINAL: f64 = 5.0;

                fn rand(n: usize, seed: f64) -> Vec<f64> {
                    (0..n).map(|i| ((i as f64 + 1.0) * seed).sin()).collect()
                }

                fn vec(x: &[f64]) -> V {
                    V::from_vec(x.to_vec(), Default::default())
                }

                fn dot_v(a: &V, b: &V) -> f64 {
                    let (a, b) = (a.clone_as_vec(), b.clone_as_vec());
                    a.iter().zip(&b).map(|(x, y)| x * y).sum()
                }

                fn cols(m: &DM) -> Vec<f64> {
                    (0..m.ncols())
                        .flat_map(|j| m.column(j).into_owned().clone_as_vec())
                        .collect()
                }

                fn dot_m(a: &DM, b: &DM) -> f64 {
                    cols(a).iter().zip(&cols(b)).map(|(x, y)| x * y).sum()
                }

                fn dot_state(a: &State, b: &State) -> f64 {
                    dot_m(&a.diff, &b.diff) + dot_v(&a.y, &b.y) + dot_v(&a.dy, &b.dy)
                }

                fn randomise(s: &mut State, seed: f64) {
                    let (n, ncols) = (s.diff.nrows(), s.diff.ncols());
                    s.diff = DM::from_vec(n, ncols, rand(n * ncols, seed), Default::default());
                    s.y = vec(&rand(n, seed * 1.7));
                    s.dy = vec(&rand(n, seed * 2.3));
                }

                fn assert_close(a: f64, b: f64, rtol: f64, what: &str) {
                    let scale = 1.0 + a.abs().max(b.abs());
                    assert!((a - b).abs() <= rtol * scale, "{what}: {a} != {b}");
                }

                fn assert_close_vec(a: &[f64], b: &[f64], rtol: f64, what: &str) {
                    let scale = 1.0 + a.iter().chain(b).fold(0.0_f64, |m, x| m.max(x.abs()));
                    for (i, (x, y)) in a.iter().zip(b).enumerate() {
                        assert!(
                            (x - y).abs() <= rtol * scale,
                            "{what}[{i}]: {x} != {y} (scale {scale})"
                        );
                    }
                }

                fn problem(p: [f64; 3], dae: bool) -> Problem {
                    ad_test_problem::<M>(p, dae, 1e-8)
                }

                fn shifted(p_dot: &[f64], eps: f64) -> [f64; 3] {
                    [
                        P[0] + eps * p_dot[0],
                        P[1] + eps * p_dot[1],
                        P[2] + eps * p_dot[2],
                    ]
                }

                /// Per-step dot test: <C_end, jvp(T, p_dot)> = <vjp(C_end), T> + <p_bar, p_dot>,
                /// for the step that just ended at `solver.state()` and started at `start`.
                fn dot_test(solver: &mut Bdf<'_, AdTestEqn<M>, LS>, start: &State, seed: f64) {
                    let mut t_start = start.zeros_like();
                    randomise(&mut t_start, seed);
                    let p_dot = vec(&rand(3, seed * 0.31));
                    let mut tangent = t_start.clone();
                    solver.step_jvp(&mut tangent, &p_dot).unwrap();

                    let mut c_end = solver.state_clone().zeros_like();
                    randomise(&mut c_end, seed * 0.77);
                    let mut cot = c_end.clone();
                    let mut p_bar = V::zeros(3, Default::default());
                    solver.step_vjp(&mut cot, &mut p_bar).unwrap();

                    let from_init = solver.state_clone().last_step.unwrap().from_init;
                    if from_init {
                        // the history was rebuilt from (y, dy), so only those carry anything
                        assert!(cols(&cot.diff).iter().all(|&x| x == 0.0));
                    } else {
                        assert!(cot.y.clone_as_vec().iter().all(|&x| x == 0.0));
                        assert!(cot.dy.clone_as_vec().iter().all(|&x| x == 0.0));
                    }
                    let lhs = dot_state(&c_end, &tangent);
                    let rhs = dot_state(&cot, &t_start) + dot_v(&p_bar, &p_dot);
                    assert_close(lhs, rhs, 1e-8, "dot test");
                }

                fn per_step_dot_test(dae: bool) {
                    let problem = problem(P, dae);
                    let mut solver = problem.bdf::<LS>().unwrap();
                    // solve tightly, so the test checks that the two products are transposes
                    // rather than how accurate the iterative solves are
                    solver.config_mut().ad_linear_solve_tolerance = 1e-6;
                    solver.config_mut().ad_linear_solve_max_iter = 50;
                    solver.set_stop_time(T_FINAL).unwrap();
                    let (mut max_order, mut rescaled, mut nsteps) = (0, false, 0);
                    loop {
                        let start = solver.state_clone();
                        let reason = solver.step().unwrap();
                        nsteps += 1;
                        let record = solver.state_clone().last_step.unwrap();
                        max_order = max_order.max(record.order);
                        rescaled |= record.h != record.h_ref;
                        dot_test(&mut solver, &start, 0.1 + nsteps as f64);
                        if let OdeSolverStopReason::TstopReached = reason {
                            break;
                        }
                    }
                    assert!(max_order >= 3, "only reached order {max_order}");
                    assert!(rescaled, "no step was rescaled");
                }

                #[test]
                fn per_step_dot_test_ode() {
                    per_step_dot_test(false);
                }

                #[test]
                fn per_step_dot_test_dae() {
                    per_step_dot_test(true);
                }

                /// Take one step from `start` (or with its history rebuilt, if not initialised)
                /// on a fresh solver for parameters `p`, optionally with a stop time.
                fn fresh_step(problem: &Problem, start: &State, tstop: Option<f64>) -> State {
                    let mut solver = problem.bdf_solver::<LS>(start.clone()).unwrap();
                    if let Some(tstop) = tstop {
                        solver.set_stop_time(tstop).unwrap();
                    }
                    solver.step().unwrap();
                    solver.into_state()
                }

                /// Finite differences of a step against `step_jvp`. `start` is the state the
                /// step starts from: one whose history is not initialised starts from an init.
                /// With `whole_state`, `diff`, `y` and `dy` are compared, otherwise only `y`.
                fn fd_test(
                    dae: bool,
                    start: &State,
                    tstop: Option<f64>,
                    whole_state: bool,
                    seed: f64,
                ) {
                    let problem0 = problem(P, dae);
                    let mut solver = problem0.bdf_solver::<LS>(start.clone()).unwrap();
                    if let Some(tstop) = tstop {
                        solver.set_stop_time(tstop).unwrap();
                    }
                    solver.step().unwrap();
                    let end = solver.state_clone();
                    let record = end.last_step.unwrap();
                    assert_eq!(record.from_init, !start.diff_initialised);

                    let mut tangent = start.zeros_like();
                    randomise(&mut tangent, seed);
                    // perturb each history entry relative to its size: the high-order columns
                    // are tiny, and a perturbation that swamps them changes the error estimate
                    let scaled = cols(&tangent.diff)
                        .iter()
                        .zip(cols(&start.diff))
                        .map(|(t, d)| t * d.abs())
                        .collect();
                    tangent.diff = DM::from_vec(
                        start.diff.nrows(),
                        start.diff.ncols(),
                        scaled,
                        Default::default(),
                    );
                    let p_dot = rand(3, seed * 0.31);
                    // keep a perturbed DAE state consistent with y0 + y1 + y2 = p2: an
                    // inconsistent one makes the step jump, whatever its size
                    if dae && record.from_init {
                        let (y, dy) = (tangent.y.clone_as_vec(), tangent.dy.clone_as_vec());
                        tangent.y = vec(&[y[0], y[1], p_dot[2] - y[0] - y[1]]);
                        tangent.dy = vec(&[dy[0], dy[1], -dy[0] - dy[1]]);
                    } else if dae {
                        // column 0 is y, the rest are differences of it
                        let mut d = cols(&tangent.diff);
                        for j in 0..tangent.diff.ncols() {
                            let target = if j == 0 { p_dot[2] } else { 0.0 };
                            d[3 * j + 2] = target - d[3 * j] - d[3 * j + 1];
                        }
                        tangent.diff = DM::from_vec(3, tangent.diff.ncols(), d, Default::default());
                    }
                    let t0 = tangent.clone();
                    solver.step_jvp(&mut tangent, &vec(&p_dot)).unwrap();

                    let eps = 1e-6;
                    let perturbed = |sign: f64| {
                        let mut s = start.clone();
                        let e = sign * eps;
                        if record.from_init {
                            s.y.axpy(e, &t0.y, 1.0);
                            s.dy.axpy(e, &t0.dy, 1.0);
                        } else {
                            let d = cols(&s.diff)
                                .iter()
                                .zip(cols(&t0.diff))
                                .map(|(x, dx)| x + e * dx)
                                .collect();
                            s.diff =
                                DM::from_vec(s.diff.nrows(), s.diff.ncols(), d, Default::default());
                            s.y.copy_from_view(&s.diff.column(0));
                        }
                        let p = problem(shifted(&p_dot, e), dae);
                        let s = fresh_step(&p, &s, tstop);
                        let r = s.last_step.unwrap();
                        assert_eq!((r.h, r.order), (record.h, record.order), "step changed");
                        s
                    };
                    let (plus, minus) = (perturbed(1.0), perturbed(-1.0));
                    let fd = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> {
                        a.iter()
                            .zip(&b)
                            .map(|(x, y)| (x - y) / (2.0 * eps))
                            .collect()
                    };
                    let rtol = 1e-5;
                    assert_close_vec(
                        &tangent.y.clone_as_vec(),
                        &fd(plus.y.clone_as_vec(), minus.y.clone_as_vec()),
                        rtol,
                        "y",
                    );
                    if whole_state {
                        assert_close_vec(
                            &tangent.dy.clone_as_vec(),
                            &fd(plus.dy.clone_as_vec(), minus.dy.clone_as_vec()),
                            rtol,
                            "dy",
                        );
                        assert_close_vec(
                            &cols(&tangent.diff),
                            &fd(cols(&plus.diff), cols(&minus.diff)),
                            rtol,
                            "diff",
                        );
                    }
                }

                /// The states after every step up to `T_FINAL`, starting with the initial one.
                fn trajectory(problem: &Problem) -> Vec<State> {
                    let mut solver = problem.bdf::<LS>().unwrap();
                    solver.set_stop_time(T_FINAL).unwrap();
                    let mut states = vec![problem.bdf_state::<LS>().unwrap()];
                    while let OdeSolverStopReason::InternalTimestep = solver.step().unwrap() {
                        states.push(solver.state_clone());
                    }
                    states
                }

                fn fd_steps(dae: bool) {
                    let states = trajectory(&problem(P, dae));
                    // the first step starts from an init, and does not rescale at its end
                    fd_test(dae, &states[0], None, true, 0.3);
                    // later steps, through order changes and rescales: only y is independent
                    // of the step-size controller
                    for (i, idx) in [5, states.len() / 3, 2 * states.len() / 3]
                        .into_iter()
                        .enumerate()
                    {
                        fd_test(dae, &states[idx], None, false, 0.7 + i as f64);
                    }
                }

                #[test]
                fn fd_steps_ode() {
                    fd_steps(false);
                }

                #[test]
                fn fd_steps_dae() {
                    fd_steps(true);
                }

                /// A reinitialisation partway through, with a stop time that shortens the first
                /// attempt: the step starts from an init with `h != h_ref`.
                fn reinit(dae: bool) {
                    let problem0 = problem(P, dae);
                    let states = trajectory(&problem0);
                    let mut start = states[10].clone();
                    start.as_mut();
                    let tstop = start.t + 0.4 * start.h;
                    fd_test(dae, &start, Some(tstop), true, 1.9);

                    let mut solver = problem0.bdf_solver::<LS>(start.clone()).unwrap();
                    solver.set_stop_time(tstop).unwrap();
                    solver.step().unwrap();
                    let record = solver.state_clone().last_step.unwrap();
                    assert!(record.from_init);
                    assert!(
                        record.h != record.h_ref,
                        "the stop time did not rescale the step"
                    );
                    dot_test(&mut solver, &start, 2.9);

                    // an ordinary step starts from the h the previous one ended with
                    let end = solver.state_clone();
                    solver.set_stop_time(T_FINAL).unwrap();
                    solver.step().unwrap();
                    let record = solver.state_clone().last_step.unwrap();
                    assert!(!record.from_init);
                    assert_eq!(record.h_ref, end.h_end);
                }

                #[test]
                fn reinit_ode() {
                    reinit(false);
                }

                #[test]
                fn reinit_dae() {
                    reinit(true);
                }

                /// The tangent pushed forward through the whole trajectory against the cotangent
                /// pulled back through the checkpoints.
                fn trajectory_test(dae: bool) {
                    let problem = problem(P, dae);
                    let mut solver = problem.bdf::<LS>().unwrap();
                    solver.set_stop_time(T_FINAL).unwrap();
                    let mut t0 = solver.state_clone().zeros_like();
                    randomise(&mut t0, 0.9);
                    let p_dot = vec(&rand(3, 0.4));
                    let mut tangent = t0.clone();
                    let mut checkpoints = Vec::new();
                    loop {
                        let reason = solver.step().unwrap();
                        solver.step_jvp(&mut tangent, &p_dot).unwrap();
                        checkpoints.push(solver.state_clone());
                        if let OdeSolverStopReason::TstopReached = reason {
                            break;
                        }
                    }
                    let w = vec(&rand(3, 1.3));

                    let mut cot = checkpoints.last().unwrap().zeros_like();
                    cot.y.copy_from(&w);
                    let mut p_bar = V::zeros(3, Default::default());
                    while let Some(end) = checkpoints.pop() {
                        solver.set_replay_state(end);
                        solver.step_vjp(&mut cot, &mut p_bar).unwrap();
                    }
                    let forward = dot_v(&w, &tangent.y);
                    let backward =
                        dot_v(&cot.y, &t0.y) + dot_v(&cot.dy, &t0.dy) + dot_v(&p_bar, &p_dot);
                    assert_close(forward, backward, 1e-7, "trajectory");
                }

                #[test]
                fn trajectory_ode() {
                    trajectory_test(false);
                }

                #[test]
                fn trajectory_dae() {
                    trajectory_test(true);
                }

                /// The tangent for p_dot = e_i, started from the solver's own initial
                /// sensitivities, follows the solver's forward sensitivities step by step.
                fn forward_sensitivities(dae: bool) {
                    let problem = problem(P, dae);
                    let mut solver = problem.bdf_sens::<LS>().unwrap();
                    solver.set_stop_time(T_FINAL).unwrap();
                    let n = 3;
                    let lane = |v: &V, i: usize| v.clone_as_vec()[i * n..(i + 1) * n].to_vec();
                    let state0 = solver.state_clone();
                    let mut tangents: Vec<State> = (0..3)
                        .map(|i| {
                            let mut t = state0.zeros_like();
                            t.y = vec(&lane(&state0.s, i));
                            t.dy = vec(&lane(&state0.ds, i));
                            t
                        })
                        .collect();
                    for nstep in 0.. {
                        let reason = solver.step().unwrap();
                        let state = solver.state_clone();
                        for (i, tangent) in tangents.iter_mut().enumerate() {
                            let mut e_i = vec![0.0; 3];
                            e_i[i] = 1.0;
                            solver.step_jvp(tangent, &vec(&e_i)).unwrap();
                            assert_close_vec(
                                &tangent.y.clone_as_vec(),
                                &lane(&state.s, i),
                                1e-6,
                                &format!("s, step {nstep}, parameter {i}"),
                            );
                        }
                        if let OdeSolverStopReason::TstopReached = reason {
                            break;
                        }
                    }
                }

                #[test]
                fn forward_sensitivities_ode() {
                    forward_sensitivities(false);
                }

                #[test]
                fn forward_sensitivities_dae() {
                    forward_sensitivities(true);
                }

                /// With too few iterations to converge on a stale Newton matrix, the
                /// solves refresh it, and the result is unchanged.
                #[test]
                fn refresh_fallback() {
                    let problem = problem(P, false);
                    let run = |max_iter: usize| {
                        let mut solver = problem.bdf::<LS>().unwrap();
                        solver.config_mut().ad_linear_solve_max_iter = max_iter;
                        let mut tangent = solver.state_clone().zeros_like();
                        randomise(&mut tangent, 0.5);
                        let p_dot = vec(&rand(3, 0.2));
                        solver.set_stop_time(T_FINAL).unwrap();
                        loop {
                            let reason = solver.step().unwrap();
                            solver.step_jvp(&mut tangent, &p_dot).unwrap();
                            if let OdeSolverStopReason::TstopReached = reason {
                                break;
                            }
                        }
                        (
                            tangent.y.clone_as_vec(),
                            solver.number_of_ad_jacobian_refreshes(),
                        )
                    };
                    let (y_default, _) = run(10);
                    // two iterations only converge with a (nearly) exact preconditioner
                    let (y_fallback, refreshes) = run(2);
                    assert!(refreshes > 0, "the Newton matrix was never refreshed");
                    assert_close_vec(&y_fallback, &y_default, 1e-6, "y");
                }

                fn assert_no_step(solver: &mut Bdf<'_, AdTestEqn<M>, LS>, what: &str) {
                    let mut t = solver.state_clone().zeros_like();
                    let p = V::zeros(3, Default::default());
                    assert!(solver.step_jvp(&mut t, &p).is_err(), "jvp after {what}");
                    let mut p_bar = V::zeros(3, Default::default());
                    assert!(
                        solver.step_vjp(&mut t, &mut p_bar).is_err(),
                        "vjp after {what}"
                    );
                }

                /// `step_jvp` and `step_vjp` need a step to differentiate, ending at the
                /// solver's current state: none has been taken yet, or the state has since been
                /// modified or moved back within the step.
                #[test]
                fn no_step() {
                    let problem = problem(P, false);
                    let mut solver = problem.bdf::<LS>().unwrap();
                    assert_no_step(&mut solver, "no step");

                    solver.step().unwrap();
                    solver.state_mut();
                    assert_no_step(&mut solver, "state_mut");

                    solver.step().unwrap();
                    let t = solver.state().t - 0.5 * solver.state_clone().last_step.unwrap().h;
                    solver.state_mut_back(t).unwrap();
                    assert_no_step(&mut solver, "state_mut_back");
                }

                /// The AD evaluates the step's Jacobian at `y` but propagates the history
                /// `diff`, so after every step `y` must be its readout `diff[:, 0]`.
                fn y_is_history_readout(dae: bool) {
                    for state in trajectory(&problem(P, dae)).iter().skip(1) {
                        assert_close_vec(
                            &state.y.clone_as_vec(),
                            &state.diff.column(0).into_owned().clone_as_vec(),
                            1e-12,
                            "y",
                        );
                    }
                }

                #[test]
                fn y_is_history_readout_ode() {
                    y_is_history_readout(false);
                }

                #[test]
                fn y_is_history_readout_dae() {
                    y_is_history_readout(true);
                }
            }
        };
    }

    step_ad_tests!(nalgebra, crate::NalgebraMat<f64>, crate::NalgebraLU<f64>);
    step_ad_tests!(faer, crate::FaerMat<f64>, crate::FaerLU<f64>);
}
