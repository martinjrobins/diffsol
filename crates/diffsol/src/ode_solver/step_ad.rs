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
    //! The step derivatives tested through the public API only: tangents and cotangents are
    //! seeded and read through the state's `y` and `dy`, so the tests apply to any solver that
    //! implements [super::StepJvp] and [super::StepVjp]. They are written once in
    //! `step_ad_tests!` and run for each solver and linear solver.

    use crate::Vector;

    pub(super) const P: [f64; 3] = [1.5, 3.0, 1.0];
    pub(super) const T_FINAL: f64 = 5.0;

    pub(super) fn rand(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|i| ((i as f64 + 1.0) * seed).sin()).collect()
    }

    pub(super) fn dot<V: Vector<T = f64>>(a: &V, b: &V) -> f64 {
        let (a, b) = (a.clone_as_vec(), b.clone_as_vec());
        a.iter().zip(&b).map(|(x, y)| x * y).sum()
    }

    pub(super) fn assert_close(a: f64, b: f64, rtol: f64, what: &str) {
        let scale = 1.0 + a.abs().max(b.abs());
        assert!((a - b).abs() <= rtol * scale, "{what}: {a} != {b}");
    }

    pub(super) fn assert_close_vec(a: &[f64], b: &[f64], rtol: f64, what: &str) {
        let scale = 1.0 + a.iter().chain(b).fold(0.0_f64, |m, x| m.max(x.abs()));
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= rtol * scale,
                "{what}[{i}]: {x} != {y} (scale {scale})"
            );
        }
    }

    /// Pseudo-random `(y, dy)` tangent components. For the DAE they satisfy the linearised
    /// constraint `y0 + y1 + y2 = p2` with `p2_dot`, so that a perturbed state stays consistent:
    /// an inconsistent one makes a step jump, whatever its size.
    pub(super) fn tangent_y_dy(seed: f64, dae: bool, p2_dot: f64) -> (Vec<f64>, Vec<f64>) {
        let (mut y, mut dy) = (rand(3, seed), rand(3, 1.7 * seed));
        if dae {
            y[2] = p2_dot - y[0] - y[1];
            dy[2] = -dy[0] - dy[1];
        }
        (y, dy)
    }

    /// The tests for one solver and linear solver: on matrix type `$M`, with linear solver `$LS`,
    /// for the solver built by the problem methods `$new`, `$from_state` (from a state of type
    /// `$State`) and, with forward sensitivities, `$new_sens`.
    macro_rules! step_ad_tests {
        (
            $mod_name:ident,
            M = $M:ty,
            LS = $LS:ty,
            State = $State:ty,
            new = $new:ident,
            from_state = $from_state:ident,
            new_sens = $new_sens:ident $(,)?
        ) => {
            mod $mod_name {
                use super::{assert_close, assert_close_vec, dot, rand, tangent_y_dy, P, T_FINAL};
                use crate::ode_equations::test_models::ad_test_model::{
                    ad_test_problem, AdTestEqn,
                };
                use crate::{
                    MatrixCommon, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
                    OdeSolverStopReason, StepJvp, StepVjp, Vector,
                };

                type M = $M;
                type V = <M as MatrixCommon>::V;
                type Problem = OdeSolverProblem<AdTestEqn<M>>;

                type E = AdTestEqn<M>;

                fn new_solver(
                    p: &Problem,
                ) -> impl OdeSolverMethod<'_, E, State = $State> + StepJvp<'_, E> + StepVjp<'_, E>
                {
                    p.$new::<$LS>().unwrap()
                }

                fn solver_from_state(
                    p: &Problem,
                    state: $State,
                ) -> impl OdeSolverMethod<'_, E, State = $State> + StepJvp<'_, E> + StepVjp<'_, E>
                {
                    p.$from_state::<$LS>(state).unwrap()
                }

                fn new_sens_solver(p: &Problem) -> impl OdeSolverMethod<'_, E> + StepJvp<'_, E> {
                    p.$new_sens::<$LS>().unwrap()
                }

                fn vec(x: &[f64]) -> V {
                    V::from_vec(x.to_vec(), Default::default())
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

                /// A copy of `state` with `y` and `dy` set, which also rebuilds its history.
                fn with_y_dy<S: OdeSolverState<V>>(state: &S, y: &[f64], dy: &[f64]) -> S {
                    let mut s = state.clone();
                    let s_mut = s.as_mut();
                    s_mut.y.copy_from(&vec(y));
                    s_mut.dy.copy_from(&vec(dy));
                    s
                }

                fn y_dy<S: OdeSolverState<V>>(s: &S) -> (Vec<f64>, Vec<f64>) {
                    (s.as_ref().y.clone_as_vec(), s.as_ref().dy.clone_as_vec())
                }

                /// A solver started from `start`, after one step towards `tstop`.
                fn step_from(
                    problem: &Problem,
                    start: $State,
                    tstop: f64,
                ) -> impl OdeSolverMethod<'_, E, State = $State> + StepJvp<'_, E> + StepVjp<'_, E>
                {
                    let mut solver = solver_from_state(problem, start);
                    solver.set_stop_time(tstop).unwrap();
                    solver.step().unwrap();
                    solver
                }

                /// One step, from the state after `after` steps with its history rebuilt from
                /// `(y, dy)` (as after `state_mut`), and with a stop time at `tstop_fraction` of
                /// the step if given. The step is checked two ways: against central
                /// finite differences of `step()` in `(y, dy, p)`, and with a dot test of
                /// `step_jvp` against `step_vjp`.
                fn rebuilt_step(dae: bool, after: usize, tstop_fraction: Option<f64>, seed: f64) {
                    let problem0 = problem(P, dae);
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
                    let h_accepted = step_from(&problem0, start.clone(), T_FINAL).state().t - t0;
                    *start.as_mut().h = 0.9 * h_accepted;
                    let tstop = tstop_fraction
                        .map(|f| t0 + f * start.as_ref().h)
                        .unwrap_or(T_FINAL);

                    let p_dot = rand(3, 0.31 * seed);
                    let (y_dot, dy_dot) = tangent_y_dy(seed, dae, p_dot[2]);
                    // finite differences, on steps that must take the same step size. The Newton
                    // error in dy = (history) / h grows as h shrinks, so eps is not too small
                    let eps = 1e-4;
                    let fd = |a: &[f64], b: &[f64]| -> Vec<f64> {
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| (x - y) / (2.0 * eps))
                            .collect()
                    };
                    // a perturbed restart: parameters p + e p_dot, y + e y_dot and dy + e dy_dot
                    let restart = |e: f64| {
                        let axpy = |a: &[f64], b: &[f64]| -> Vec<f64> {
                            a.iter().zip(b).map(|(a, b)| a + e * b).collect()
                        };
                        (shifted(&p_dot, e), axpy(&y0, &y_dot), axpy(&dy0, &dy_dot))
                    };

                    let mut solver = step_from(&problem0, start.clone(), tstop);
                    let t_end = solver.state().t;
                    if tstop_fraction.is_some() {
                        assert!(
                            t_end < t0 + start.as_ref().h,
                            "the stop time did not shorten the step"
                        );
                    }
                    let mut tangent = with_y_dy(&start.zeros_like(), &y_dot, &dy_dot);
                    solver.step_jvp(&mut tangent, &vec(&p_dot)).unwrap();

                    let perturbed = |e: f64| {
                        let (p, y, dy) = restart(e);
                        let problem = problem(p, dae);
                        let solver = step_from(&problem, with_y_dy(&start, &y, &dy), tstop);
                        assert_eq!(solver.state().t, t_end, "the perturbed step changed size");
                        y_dy(&solver.state_clone())
                    };
                    let ((y_p, dy_p), (y_m, dy_m)) = (perturbed(eps), perturbed(-eps));
                    let (y_jvp, dy_jvp) = y_dy(&tangent);
                    assert_close_vec(&y_jvp, &fd(&y_p, &y_m), 1e-5, "y");
                    assert_close_vec(&dy_jvp, &fd(&dy_p, &dy_m), 1e-5, "dy");

                    // dot test: <c_end, jvp(t)> = <vjp(c_end), t> + <p_bar, p_dot>
                    let (c_y, c_dy) = (rand(3, 2.9 * seed), rand(3, 3.7 * seed));
                    let mut cot = with_y_dy(&solver.state_clone().zeros_like(), &c_y, &c_dy);
                    let mut p_bar = V::zeros(3, Default::default());
                    solver.step_vjp(&mut cot, &mut p_bar).unwrap();
                    let lhs = dot(&vec(&c_y), &vec(&y_jvp)) + dot(&vec(&c_dy), &vec(&dy_jvp));
                    let rhs = dot(cot.as_ref().y, &vec(&y_dot))
                        + dot(cot.as_ref().dy, &vec(&dy_dot))
                        + dot(&p_bar, &vec(&p_dot));
                    assert_close(lhs, rhs, 1e-7, "dot test");
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

                /// The tangent pushed forward through the trajectory against the cotangent pulled
                /// back through the checkpoints, for trajectories ending at many steps along the
                /// way, so that the comparison includes steps whose step size changes at their end.
                fn trajectory_test(dae: bool) {
                    let problem = problem(P, dae);
                    let mut solver = new_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let p_dot = vec(&rand(3, 0.4));
                    let (y_dot, dy_dot) = tangent_y_dy(0.9, dae, 0.0);
                    let mut tangent =
                        with_y_dy(&solver.state_clone().zeros_like(), &y_dot, &dy_dot);
                    // the checkpoint and the tangent's (y, dy) after each step
                    let mut steps = Vec::new();
                    loop {
                        let reason = solver.step().unwrap();
                        solver.step_jvp(&mut tangent, &p_dot).unwrap();
                        steps.push((solver.state_clone(), y_dy(&tangent)));
                        if let OdeSolverStopReason::TstopReached = reason {
                            break;
                        }
                    }

                    let (c_y, c_dy) = (rand(3, 1.3), rand(3, 2.1));
                    let ends = (1..steps.len()).step_by(3).chain([steps.len()]);
                    for end in ends {
                        let (y_end, dy_end) = &steps[end - 1].1;
                        let mut cot = with_y_dy(&steps[end - 1].0.zeros_like(), &c_y, &c_dy);
                        let mut p_bar = V::zeros(3, Default::default());
                        for (checkpoint, _) in steps[..end].iter().rev() {
                            solver.set_replay_state(checkpoint.clone());
                            solver.step_vjp(&mut cot, &mut p_bar).unwrap();
                        }
                        let forward = dot(&vec(&c_y), &vec(y_end)) + dot(&vec(&c_dy), &vec(dy_end));
                        let backward = dot(cot.as_ref().y, &vec(&y_dot))
                            + dot(cot.as_ref().dy, &vec(&dy_dot))
                            + dot(&p_bar, &p_dot);
                        assert_close(
                            forward,
                            backward,
                            1e-7,
                            &format!("trajectory to step {end}"),
                        );
                    }
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
                    let mut solver = new_sens_solver(&problem);
                    solver.set_stop_time(T_FINAL).unwrap();
                    let lane = |v: &V, i: usize| v.clone_as_vec()[3 * i..3 * (i + 1)].to_vec();
                    let state0 = solver.state_clone();
                    let mut tangents: Vec<_> = (0..3)
                        .map(|i| {
                            let s = state0.as_ref();
                            with_y_dy(&state0.zeros_like(), &lane(s.s, i), &lane(s.ds, i))
                        })
                        .collect();
                    for nstep in 0.. {
                        let reason = solver.step().unwrap();
                        let s = solver.state().s.clone();
                        for (i, tangent) in tangents.iter_mut().enumerate() {
                            let mut e_i = vec![0.0; 3];
                            e_i[i] = 1.0;
                            solver.step_jvp(tangent, &vec(&e_i)).unwrap();
                            assert_close_vec(
                                &tangent.as_ref().y.clone_as_vec(),
                                &lane(&s, i),
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

                /// A `set_stop_time` after a step rescales the state for the next step, but not
                /// the step just taken, so its derivative is unchanged.
                #[test]
                fn set_stop_time_after_step() {
                    let problem = problem(P, false);
                    let mut solver = new_solver(&problem);
                    let (y_dot, dy_dot) = tangent_y_dy(0.6, false, 0.0);
                    let mut tangent =
                        with_y_dy(&solver.state_clone().zeros_like(), &y_dot, &dy_dot);
                    let p_dot = vec(&rand(3, 0.8));
                    for _ in 0..9 {
                        solver.step().unwrap();
                        solver.step_jvp(&mut tangent, &p_dot).unwrap();
                    }
                    solver.step().unwrap();
                    let mut stopped = solver.clone();
                    let (t, h) = (solver.state().t, solver.state().h);
                    stopped.set_stop_time(t + 0.3 * h).unwrap();
                    assert!(
                        stopped.state().h != h,
                        "the stop time did not rescale the state"
                    );

                    let mut tangent_stopped = tangent.clone();
                    solver.step_jvp(&mut tangent, &p_dot).unwrap();
                    stopped.step_jvp(&mut tangent_stopped, &p_dot).unwrap();
                    let ((y, dy), (y_s, dy_s)) = (y_dy(&tangent), y_dy(&tangent_stopped));
                    assert_close_vec(&y_s, &y, 1e-8, "y");
                    assert_close_vec(&dy_s, &dy, 1e-8, "dy");
                }

                fn assert_no_step<'a, S>(solver: &mut S, what: &str)
                where
                    S: StepJvp<'a, AdTestEqn<M>> + StepVjp<'a, AdTestEqn<M>>,
                {
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
        new = bdf,
        from_state = bdf_solver,
        new_sens = bdf_sens,
    );
    step_ad_tests!(
        bdf_sparse,
        M = crate::FaerSparseMat<f64>,
        LS = crate::FaerSparseLU<f64>,
        State = crate::BdfState<crate::FaerVec<f64>>,
        new = bdf,
        from_state = bdf_solver,
        new_sens = bdf_sens,
    );
}
