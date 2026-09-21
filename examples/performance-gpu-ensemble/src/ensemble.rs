//! The ensemble driver: sampling, the device fold, both solve paths, and the plots.

use crate::swing::swing_problem;

use diffsol::{
    NalgebraMat, NalgebraVec, OdeEquations, OdeSolverMethod, OdeSolverStopReason, Op, Vector,
};
use plotly::{
    common::{Mode, Title},
    layout::Axis,
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
type CpuM = NalgebraMat<f64>;

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
// ANCHOR_END: types

// ANCHOR: reduce
/// Worst speed deviation of any generator over the whole run, one value per grid.
fn max_deviation_hz<'a, Solver, Eqn>(solver: &mut Solver, nbuses: usize, t_final: f64) -> Vec<f64>
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations<T = f64> + 'a,
{
    let ctx = solver.problem().context().clone();
    let mut worst = Eqn::V::zeros(1, ctx.clone());
    let mut next = Eqn::V::zeros(1, ctx);

    // find the maximum speed deviation in Hz across all generators
    fn fold<V: Vector<T = f64>>(y: &V, worst: &mut V, next: &mut V, nbuses: usize) {
        V::reduce_elem(
            next,
            [y, worst],
            0.0,
            move |[x, w], _lane, i| {
                // rad/s in the rotating frame, so Hz is omega / 2*pi
                if i >= nbuses {
                    f64::max(x[i].abs() / (2.0 * PI), w[0])
                } else {
                    w[0]
                }
            },
            f64::max,
        );
        std::mem::swap(worst, next);
    }

    solver.set_stop_time(t_final).unwrap();
    loop {
        fold(solver.state().y, &mut worst, &mut next, nbuses);
        match solver.step() {
            Ok(OdeSolverStopReason::TstopReached) => break,
            Ok(_) => (),
            Err(e) => panic!("solver failed: {e}"),
        }
    }
    fold(solver.state().y, &mut worst, &mut next, nbuses);
    worst.clone_as_vec()
}
// ANCHOR_END: reduce

// ANCHOR: solve_cpu
/// One solve per grid, spread over rayon's threads.
fn solve_cpu(nbuses: usize, demands: &[f64], t_final: f64) -> Vec<f64> {
    demands
        .par_iter()
        .map_init(
            || swing_problem::<CpuM>(nbuses, &[0.0]),
            |problem, &d| {
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
/// Every grid in one batched solve.
fn solve_gpu(nbuses: usize, demands: &[f64], t_final: f64) -> Vec<f64> {
    use diffsol::OxideMat;
    let problem = swing_problem::<OxideMat>(nbuses, demands);
    let mut solver = problem.tsit45().unwrap();
    max_deviation_hz(&mut solver, nbuses, t_final)
}
// ANCHOR_END: solve_gpu

// ANCHOR: plot_histogram
fn plot_histogram(max_dev: &[f64]) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(Histogram::new(max_dev.to_vec()).name("grids"));
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("max |frequency deviation| (Hz)"))
            .y_axis(Axis::new().title("grids")),
    );
    plot
}
// ANCHOR_END: plot_histogram

// ANCHOR: plot_scaling
fn plot_scaling(grids: &[f64], cpu: &[f64], gpu: &[f64]) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(grids.to_vec(), cpu.to_vec())
            .mode(Mode::LinesMarkers)
            .name("CPU (rayon, one solve per grid)"),
    );
    plot.add_trace(
        Scatter::new(grids.to_vec(), gpu.to_vec())
            .mode(Mode::LinesMarkers)
            .name("GPU (one batched solve)"),
    );
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
    solve_gpu(8, &demands[..1], T_FINAL);

    let start = Instant::now();
    let gpu = solve_gpu(N_BUSES, &demands, T_FINAL);
    let gpu_time = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let cpu = solve_cpu(N_BUSES, &demands, T_FINAL);
    let cpu_time = start.elapsed().as_secs_f64();

    // check that cpu and gpu results agree (will differ slightly since they take different internal steps)
    let worst_diff = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0, f64::max);
    // nothing but one scalar per grid comes back from either path, so these are the solves
    println!("{N_SAMPLES} grids x {N_BUSES} generators: GPU {gpu_time:.3}s, CPU {cpu_time:.3}s");

    let mut sorted = gpu.clone();
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let median = at(0.5);
    println!(
        "worst GPU/CPU disagreement: {worst_diff:.3e} Hz ({:.2}% of median)",
        100.0 * worst_diff / median
    );
    assert!(worst_diff / median < 0.05, "GPU and CPU results disagree");
    println!(
        "max |df|: median {median:.5} Hz, 5% {:.5} Hz, 95% {:.5} Hz",
        at(0.05),
        at(0.95)
    );
    write_plot(&plot_histogram(&gpu), "gpu_ensemble_frequency");

    // Sweep over n_samples and time gpu and cpu, plot results
    let sweep = [1usize, 2, 5, 10, 20, 50, 100, 200, 500, N_SAMPLES];
    let max_threads = rayon::current_num_threads();
    let (mut grids, mut cpu_times, mut gpu_times) = (Vec::new(), Vec::new(), Vec::new());
    println!("\ngrids   CPU (s)   GPU (s)   speedup");
    for n in sweep {
        let lanes = &demands[..n];
        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n.min(max_threads))
            .build()
            .unwrap();
        // one untimed pass first: the first run at a new size pays for buffers
        solve_gpu(N_BUSES, lanes, T_FINAL);
        cpu_pool.install(|| solve_cpu(N_BUSES, lanes, T_FINAL));
        // the two paths are interleaved so both see the same machine conditions
        let (mut cpu_runs, mut gpu_runs) = (Vec::new(), Vec::new());
        for _ in 0..BENCH_REPEATS {
            let start = Instant::now();
            let g = solve_gpu(N_BUSES, lanes, T_FINAL);
            gpu_runs.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let c = cpu_pool.install(|| solve_cpu(N_BUSES, lanes, T_FINAL));
            cpu_runs.push(start.elapsed().as_secs_f64());
            assert_eq!(g.len(), c.len());
        }
        cpu_runs.sort_by(f64::total_cmp);
        gpu_runs.sort_by(f64::total_cmp);
        let ct = cpu_runs[cpu_runs.len() / 2];
        let gt = gpu_runs[gpu_runs.len() / 2];
        println!("{n:5}   {ct:7.4}   {gt:7.4}   {:6.1}x", ct / gt);
        grids.push(n as f64);
        cpu_times.push(ct);
        gpu_times.push(gt);
    }
    write_plot(
        &plot_scaling(&grids, &cpu_times, &gpu_times),
        "gpu_ensemble_scaling",
    );
}
