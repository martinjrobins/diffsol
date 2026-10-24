//! The gradient of a loss with respect to the parameters, by the continuous adjoint
//! (`solve_dense_with_checkpointing` then `solve_adjoint_backwards_pass`) and by pulling the
//! loss back through the solver's steps (`StepVjp::solve_dense_vjp`). Each measurement is a
//! whole gradient, forward and backward passes, of the loss `G = 1/2 sum_i |u(t_i)|^2` over
//! the model's reference times. For comparison, `<name>_forward` measures the loss alone: a
//! plain dense solve to the same times, without checkpoints or derivatives.

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use diffsol::{
    error::DiffsolError,
    ode_equations::test_models::{
        exponential_decay::exponential_decay_problem_adjoint, robertson::robertson_adjoint,
        robertson_ode::robertson_ode,
    },
    AdjointOdeSolverMethod, CheckpointingPath, DenseMatrix, MatrixCommon, NalgebraLU, NalgebraMat,
    NalgebraVec, OdeEquationsImplicitAdjoint, OdeSolverMethod, OdeSolverState, StepVjp, Vector,
    VectorView,
};

// only the problem setup of the shared bench helpers is used here
#[allow(dead_code, unused_imports, unused_macros)]
mod common;

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type LS = NalgebraLU<f64>;

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

/// The loss alone, summed over the lanes, from a dense solve.
fn loss_forward<'a, Eqn, S>(t_eval: &[f64], mut solver: S) -> Result<f64, DiffsolError>
where
    Eqn: OdeEquationsImplicitAdjoint<T = f64, V = V, M = M> + 'a,
    S: OdeSolverMethod<'a, Eqn>,
{
    let (soln, _) = solver.solve_dense(t_eval)?;
    Ok((0..soln.ncols())
        .flat_map(|i| soln.column(i).into_owned().clone_as_vec())
        .map(|x| 0.5 * x * x)
        .sum())
}

fn grad_adjoint<'a, Eqn, S, A>(
    t_eval: &[f64],
    mut solver: S,
    adjoint: impl FnOnce(CheckpointingPath<Eqn, S::State>, S) -> Result<A, DiffsolError>,
) -> Result<V, DiffsolError>
where
    Eqn: OdeEquationsImplicitAdjoint<T = f64, V = V, M = M> + 'a,
    S: OdeSolverMethod<'a, Eqn>,
    A: AdjointOdeSolverMethod<'a, Eqn, S>,
{
    let (checkpoints, soln, _) = solver.solve_dense_with_checkpointing(t_eval, None)?;
    // dg/du = u, so the dense solution is the cost gradient
    let (state, _) =
        adjoint(checkpoints, solver)?.solve_adjoint_backwards_pass(t_eval, &[&soln])?;
    Ok(state.into_common().sg)
}

fn grad_step_vjp<'a, Eqn, S>(t_eval: &[f64], mut solver: S) -> Result<V, DiffsolError>
where
    Eqn: OdeEquationsImplicitAdjoint<T = f64, V = V, M = M> + 'a,
    S: StepVjp<'a, Eqn>,
{
    Ok(solver.solve_dense_vjp(t_eval, 1, half_squares, identity)?.1)
}

/// Benchmark `grad`, after one untimed run: one that fails is reported and skipped, as not
/// every method solves every problem yet.
fn bench_gradient<R>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: String,
    grad: impl Fn() -> Result<R, DiffsolError>,
) {
    if let Err(e) = grad() {
        eprintln!("skipping {name}: {e}");
        return;
    }
    g.bench_function(name, |b| b.iter(|| grad().unwrap()));
}

/// Benchmark `<name>_forward` and `<name>_step_vjp` for the problem `$problem`, at its
/// reference times up to `$t_max`, with the solver made by the problem method `$new`, and,
/// with `adjoint`, `<name>_adjoint` with the adjoint solver made by that method.
macro_rules! bench_gradients {
    ($g:expr, $name:expr, $problem:expr, $t_max:expr, new = { $($new:tt)* },
     adjoint = { $($adjoint:tt)* }) => {{
        let (problem, t_evals) = common::setup_problem!($problem);
        let t_evals: Vec<f64> = t_evals.into_iter().filter(|t| *t <= $t_max).collect();
        let name: String = $name;
        bench_gradient($g, format!("{name}_forward"), || {
            loss_forward(&t_evals, problem.$($new)*()?)
        });
        bench_gradient($g, format!("{name}_adjoint"), || {
            grad_adjoint(&t_evals, problem.$($new)*()?, |checkpoints, solver| {
                problem.$($adjoint)*(checkpoints, Some(solver), Some(1))
            })
        });
        bench_gradient($g, format!("{name}_step_vjp"), || {
            grad_step_vjp(&t_evals, problem.$($new)*()?)
        });
    }};
    ($g:expr, $name:expr, $problem:expr, $t_max:expr, new = { $($new:tt)* }) => {{
        let (problem, t_evals) = common::setup_problem!($problem);
        let t_evals: Vec<f64> = t_evals.into_iter().filter(|t| *t <= $t_max).collect();
        let name: String = $name;
        bench_gradient($g, format!("{name}_forward"), || {
            loss_forward(&t_evals, problem.$($new)*()?)
        });
        bench_gradient($g, format!("{name}_step_vjp"), || {
            grad_step_vjp(&t_evals, problem.$($new)*()?)
        });
    }};
}

/// Robertson's reference times run to 4e10, but the continuous adjoint's backward solve fails
/// on the last decades (too many error test failures), so stop where both methods run.
const ROBERTSON_T_MAX: f64 = 4e8;

fn robertson_ode_sizes(g: &mut BenchmarkGroup<'_, WallTime>) {
    for nbatch in [1, 25, 100] {
        bench_gradients!(
            g,
            format!("bdf_{nbatch}"),
            robertson_ode::<M>(nbatch),
            ROBERTSON_T_MAX,
            new = { bdf::<LS> },
            adjoint = { bdf_solver_adjoint::<LS, _> }
        );
        bench_gradients!(
            g,
            format!("esdirk34_{nbatch}"),
            robertson_ode::<M>(nbatch),
            ROBERTSON_T_MAX,
            new = { esdirk34::<LS> },
            adjoint = { esdirk34_solver_adjoint::<LS, _> }
        );
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    *c = std::mem::take(c)
        .measurement_time(Duration::from_secs(3))
        .sample_size(50)
        .warm_up_time(Duration::from_secs(1));

    // -------------------------------------------------------------------------
    // Robertson ODE (stiff, one copy of the system per batch lane)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("robertson_ode_grad");
        robertson_ode_sizes(&mut g);
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Robertson DAE (stiff). The continuous adjoint does not yet solve this DAE (its backward
    // solve fails at the first step once the loss involves a differential state), so only the
    // step derivatives are measured.
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("robertson_grad");
        bench_gradients!(
            &mut g,
            "bdf".to_string(),
            robertson_adjoint::<M>(),
            ROBERTSON_T_MAX,
            new = { bdf::<LS> }
        );
        bench_gradients!(
            &mut g,
            "esdirk34".to_string(),
            robertson_adjoint::<M>(),
            ROBERTSON_T_MAX,
            new = { esdirk34::<LS> }
        );
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Exponential decay (non-stiff)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("exponential_decay_grad");
        bench_gradients!(
            &mut g,
            "tsit45".to_string(),
            exponential_decay_problem_adjoint::<M>(false, false),
            f64::INFINITY,
            new = { tsit45 },
            adjoint = { tsit45_solver_adjoint }
        );
        g.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
