//! The ensemble driver: sampling, the device fold, both solve paths, and the plots.

use crate::swing::swing_problem;

use diffsol::{
    NalgebraMat, NalgebraScalar, NalgebraVec, OdeEquations, OdeSolverMethod, OdeSolverStopReason,
    Op, Scalar, ScalarCuda, Vector,
};
use num_traits::{FromPrimitive, Signed, ToPrimitive, Zero};
use plotly::{
    common::{Mode, Title},
    layout::{Axis, BarMode},
    Histogram, Layout, Plot, Scatter,
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;
use std::{fs, path::PathBuf};

// ANCHOR: types
/// number of samples for input demand parameter
const N_SAMPLES: usize = 1000;
/// Generators per grid (nstates = 2 * N_BUSES)
const N_BUSES: usize = 64;
const T_FINAL: f64 = 10.0;
const SEED: u64 = 42;
/// Mean and standard deviation of the uncertain demand at bus 0, per unit.
const DEMAND_MEAN: f64 = 0.30;
const DEMAND_SD: f64 = 0.08;
/// Timed runs per point in the scaling sweep.
const BENCH_REPEATS: usize = 15;
/// Solver tolerances; f32 cannot resolve the f64 ones (its epsilon is ~1.2e-7).
const RTOL: f64 = 1e-6;
const ATOL: f64 = 1e-8;
const RTOL_F32: f64 = 1e-4;
const ATOL_F32: f64 = 1e-6;
/// Solve paths timed in the scaling sweep, in the order `solve` runs them.
const SERIES: [&str; 4] = ["CPU f64", "CPU f32", "GPU f64", "GPU f32"];
// ANCHOR_END: types

// ANCHOR: reduce
/// Worst speed deviation of any generator over the whole run, one value per grid.
fn max_deviation_hz<'a, Solver, Eqn>(solver: &mut Solver, nbuses: usize, t_final: f64) -> Vec<f64>
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations + 'a,
{
    let ctx = solver.problem().context().clone();
    let mut worst = Eqn::V::zeros(1, ctx.clone());
    let mut next = Eqn::V::zeros(1, ctx);

    // find the maximum speed deviation (rad/s) across all generators
    fn fold<V: Vector>(y: &V, worst: &mut V, next: &mut V, nbuses: usize) {
        V::reduce_elem(
            next,
            [y, worst],
            V::T::zero(),
            move |[x, w], _lane, i| {
                if i >= nbuses {
                    x[i].abs().max(w[0])
                } else {
                    w[0]
                }
            },
            |a: V::T, b: V::T| a.max(b),
        );
        std::mem::swap(worst, next);
    }

    solver
        .set_stop_time(Eqn::T::from_f64(t_final).unwrap())
        .unwrap();
    loop {
        fold(solver.state().y, &mut worst, &mut next, nbuses);
        match solver.step() {
            Ok(OdeSolverStopReason::TstopReached) => break,
            Ok(_) => (),
            Err(e) => panic!("solver failed: {e}"),
        }
    }
    fold(solver.state().y, &mut worst, &mut next, nbuses);
    worst
        .clone_as_vec()
        .into_iter()
        // rad/s in the rotating frame, so Hz is omega / 2*pi
        .map(|w| w.to_f64().unwrap() / (2.0 * PI))
        .collect()
}
// ANCHOR_END: reduce

// ANCHOR: solve_cpu
/// One solve per grid, spread over rayon's threads.
fn solve_cpu<T: NalgebraScalar>(
    nbuses: usize,
    demands: &[f64],
    t_final: f64,
    rtol: f64,
    atol: f64,
) -> Vec<f64> {
    demands
        .par_iter()
        .map_init(
            || swing_problem::<NalgebraMat<T>>(nbuses, &[0.0], rtol, atol),
            |problem, &d| {
                let d = T::from_f64(d).unwrap();
                let p = NalgebraVec::from_vec(vec![d], *problem.eqn.context());
                problem.eqn_mut().set_params(&p);
                let mut solver = problem.tsit45().unwrap();
                max_deviation_hz(&mut solver, nbuses, t_final)[0]
            },
        )
        .collect()
}
// ANCHOR_END: solve_cpu

// ANCHOR: solve_gpu
/// Every grid in one batched solve, in precision `T`.
fn solve_gpu<T: ScalarCuda>(
    nbuses: usize,
    demands: &[f64],
    t_final: f64,
    rtol: f64,
    atol: f64,
) -> Vec<f64> {
    use diffsol::OxideMat;
    let problem = swing_problem::<OxideMat<T>>(nbuses, demands, rtol, atol);
    let mut solver = problem.tsit45().unwrap();
    max_deviation_hz(&mut solver, nbuses, t_final)
}
// ANCHOR_END: solve_gpu

// ANCHOR: plot_histogram
fn plot_histogram(max_dev: &[f64], max_dev_f32: &[f64]) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Histogram::new(max_dev.to_vec())
            .name("GPU f64")
            .opacity(0.6),
    );
    plot.add_trace(
        Histogram::new(max_dev_f32.to_vec())
            .name("GPU f32")
            .opacity(0.6),
    );
    plot.set_layout(
        Layout::new()
            .bar_mode(BarMode::Overlay)
            .x_axis(Axis::new().title("max |frequency deviation| (Hz)"))
            .y_axis(Axis::new().title("grids")),
    );
    plot
}
// ANCHOR_END: plot_histogram

// ANCHOR: plot_scaling
/// One line per entry of `SERIES`, `times[k]` holding its timings at each ensemble size.
fn plot_scaling(grids: &[f64], times: &[Vec<f64>]) -> Plot {
    let mut plot = Plot::new();
    for (name, t) in SERIES.iter().zip(times) {
        let how = if name.starts_with("CPU") {
            "rayon, one solve per grid"
        } else {
            "one batched solve"
        };
        plot.add_trace(
            Scatter::new(grids.to_vec(), t.clone())
                .mode(Mode::LinesMarkers)
                .name(format!("{name} ({how})")),
        );
    }
    plot.set_layout(
        Layout::new()
            .title(Title::with_text(
                "CPU: 2x AMD EPYC 7343 (32 cores) | GPU: NVIDIA A40 (46 GiB)",
            ))
            .x_axis(
                Axis::new()
                    .title("grids in the ensemble")
                    .type_(plotly::layout::AxisType::Log),
            )
            .y_axis(
                Axis::new()
                    .title("elapsed (s)")
                    .type_(plotly::layout::AxisType::Log),
            ),
    );
    plot
}
// ANCHOR_END: plot_scaling

fn write_plot(plot: &Plot, name: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/performance/images")
        .join(format!("{name}.html"));
    fs::write(path, plot.to_inline_html(Some(name))).expect("Unable to write file");
}

pub fn run() {
    // ANCHOR: sample
    // the demand at bus 0 of each grid sample
    let demands: Vec<f64> = (0..N_SAMPLES)
        .map(|index| {
            let mut rng = ChaCha12Rng::seed_from_u64(SEED);
            rng.set_stream(index as u64);
            Normal::new(DEMAND_MEAN, DEMAND_SD)
                .unwrap()
                .sample(&mut rng)
        })
        .collect();
    // ANCHOR_END: sample

    // the first device call pays for context and module setup, which is not what we are
    // measuring
    solve_gpu::<f64>(8, &demands[..1], T_FINAL, RTOL, ATOL);
    solve_gpu::<f32>(8, &demands[..1], T_FINAL, RTOL_F32, ATOL_F32);

    let start = Instant::now();
    let gpu = solve_gpu::<f64>(N_BUSES, &demands, T_FINAL, RTOL, ATOL);
    let gpu_time = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let gpu_f32 = solve_gpu::<f32>(N_BUSES, &demands, T_FINAL, RTOL_F32, ATOL_F32);
    let gpu_f32_time = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let cpu = solve_cpu::<f64>(N_BUSES, &demands, T_FINAL, RTOL, ATOL);
    let cpu_time = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let cpu_f32 = solve_cpu::<f32>(N_BUSES, &demands, T_FINAL, RTOL_F32, ATOL_F32);
    let cpu_f32_time = start.elapsed().as_secs_f64();

    // check that cpu and gpu results agree (will differ slightly since they take different internal steps)
    let worst_diff = |other: &[f64]| {
        other
            .iter()
            .zip(cpu.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0, f64::max)
    };
    // nothing but one scalar per grid comes back from either path, so these are the solves
    println!(
        "{N_SAMPLES} grids x {N_BUSES} generators: GPU f64 {gpu_time:.3}s, \
         GPU f32 {gpu_f32_time:.3}s, CPU f64 {cpu_time:.3}s, CPU f32 {cpu_f32_time:.3}s"
    );

    let mut sorted = gpu.clone();
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let median = at(0.5);
    for (name, other) in [
        ("GPU f64", &gpu),
        ("GPU f32", &gpu_f32),
        ("CPU f32", &cpu_f32),
    ] {
        let diff = worst_diff(other);
        println!(
            "worst {name}/CPU f64 disagreement: {diff:.3e} Hz ({:.2}% of median)",
            100.0 * diff / median
        );
        assert!(diff / median < 0.05, "{name} and CPU f64 results disagree");
    }
    println!(
        "max |df|: median {median:.5} Hz, 5% {:.5} Hz, 95% {:.5} Hz",
        at(0.05),
        at(0.95)
    );
    write_plot(&plot_histogram(&gpu, &gpu_f32), "gpu_ensemble_frequency");

    // Sweep over n_samples and time every solve path, plot results
    let sweep = [1usize, 2, 5, 10, 20, 50, 100, 200, 500, N_SAMPLES];
    let max_threads = rayon::current_num_threads();
    let mut grids = Vec::new();
    let mut times = vec![Vec::new(); SERIES.len()];
    println!(
        "\ngrids {}   (speedup vs CPU f64)",
        SERIES.map(|s| format!("{s:>9} (s)")).join(" ")
    );
    for n in sweep {
        let lanes = &demands[..n];
        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n.min(max_threads))
            .build()
            .unwrap();
        let solve = |k: usize| match k {
            0 => cpu_pool.install(|| solve_cpu::<f64>(N_BUSES, lanes, T_FINAL, RTOL, ATOL)),
            1 => cpu_pool.install(|| solve_cpu::<f32>(N_BUSES, lanes, T_FINAL, RTOL_F32, ATOL_F32)),
            2 => solve_gpu::<f64>(N_BUSES, lanes, T_FINAL, RTOL, ATOL),
            _ => solve_gpu::<f32>(N_BUSES, lanes, T_FINAL, RTOL_F32, ATOL_F32),
        };
        // one untimed pass first: the first run at a new size pays for buffers
        for k in 0..SERIES.len() {
            solve(k);
        }
        // the paths are interleaved so all see the same machine conditions
        let mut runs = vec![Vec::new(); SERIES.len()];
        for _ in 0..BENCH_REPEATS {
            for (k, r) in runs.iter_mut().enumerate() {
                let start = Instant::now();
                assert_eq!(solve(k).len(), n);
                r.push(start.elapsed().as_secs_f64());
            }
        }
        let medians: Vec<f64> = runs
            .into_iter()
            .map(|mut r| {
                r.sort_by(f64::total_cmp);
                r[r.len() / 2]
            })
            .collect();
        println!(
            "{n:5} {}   ({})",
            medians
                .iter()
                .map(|t| format!("{t:13.4}"))
                .collect::<Vec<_>>()
                .join(" "),
            medians
                .iter()
                .map(|t| format!("{:.1}x", medians[0] / t))
                .collect::<Vec<_>>()
                .join(", ")
        );
        grids.push(n as f64);
        for (t, m) in times.iter_mut().zip(medians) {
            t.push(m);
        }
    }
    write_plot(&plot_scaling(&grids, &times), "gpu_ensemble_scaling");
}
