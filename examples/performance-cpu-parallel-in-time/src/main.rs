use diffsol::{
    DenseMatrix, MatrixCommon, NalgebraMat, OdeBuilder, OdeEquationsImplicitSens, OdeSolverMethod,
    OdeSolverProblem, OdeSolverStopReason, Vector, VectorView,
};
use nalgebra::{Matrix2, Vector2};
use plotly::{
    common::{DashType, Line, Mode},
    layout::{Axis, AxisType},
    Layout, Plot, Scatter,
};
use rayon::prelude::*;
use rayon_scan::ScanParallelIterator;
use std::{fs, path::PathBuf, time::Instant};

// ANCHOR: types
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type T = <M as MatrixCommon>::T;
type C = <M as MatrixCommon>::C;
// ANCHOR_END: types

const BENCH_REPEATS: usize = 100;

// ANCHOR: problem
const DELTA: f64 = 0.5;
const GAMMA: f64 = 0.5;
const OMEGA: f64 = 1.0;
const Y0: [f64; 2] = [1.0, 0.0];

fn problem(
    rtol: f64,
    atol: f64,
) -> OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .rtol(rtol)
        .atol([atol])
        .p(Y0)
        .rhs_sens_implicit(
            |y, _p, t, dy| {
                dy[0] = y[1];
                dy[1] = -DELTA * y[1] - y[0] - y[0] * y[0] * y[0] + GAMMA * (OMEGA * t).cos();
            },
            |y, _p, _t, v, jv| {
                jv[0] = v[1];
                jv[1] = -(1.0 + 3.0 * y[0] * y[0]) * v[0] - DELTA * v[1];
            },
            |_y, _p, _t, _v, jv| jv.fill(0.0),
        )
        .init_sens(
            |p, _t, y| y.copy_from_slice(p),
            |_p, _t, v, y| y.copy_from_slice(v),
            2,
        )
        .build()
        .unwrap()
}
// ANCHOR_END: problem

// ANCHOR: chunk
const T_FINAL: f64 = 3000.0;
const N_CHUNKS: usize = 96;
const DT_CHUNK: f64 = T_FINAL / N_CHUNKS as f64;

fn propagate(
    problems: &mut (
        OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = V, T = T, C = C>>,
        OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = V, T = T, C = C>>,
    ),
    k: usize,
    y_start: &Vector2<f64>,
) -> (Vector2<f64>, Matrix2<f64>) {
    let (accurate, loose) = problems;
    let (t0, t1) = (k as f64 * DT_CHUNK, (k + 1) as f64 * DT_CHUNK);
    let p = V::from_vec(vec![y_start[0], y_start[1]], *accurate.eqn.context());

    accurate.eqn_mut().set_params(&p);
    accurate.t0 = t0;
    let mut solver = accurate.tsit45().unwrap();
    solver.set_stop_time(t1).unwrap();
    while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}
    let y = Vector2::from_fn(|i, _| solver.state().y.get_index(i));

    loose.eqn_mut().set_params(&p);
    loose.t0 = t0;
    let mut solver = loose.tsit45_sens().unwrap();
    solver.set_stop_time(t1).unwrap();
    while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}
    let s = solver.state().s;
    let jac = Matrix2::from_fn(|i, j| s.get_batch(j).get_index(i));
    (y, jac)
}
// ANCHOR_END: chunk

// ANCHOR: coarse
const COARSE_TOL: f64 = 1e-3;

fn coarse() -> Vec<Vector2<f64>> {
    let t: Vec<f64> = (0..=N_CHUNKS).map(|k| k as f64 * DT_CHUNK).collect();
    let (ys, _) = problem(COARSE_TOL, COARSE_TOL)
        .tsit45()
        .unwrap()
        .solve_dense(&t)
        .unwrap();
    (0..t.len())
        .map(|k| Vector2::new(ys.column(k)[0], ys.column(k)[1]))
        .collect()
}
// ANCHOR_END: coarse

// ANCHOR: deer
const RTOL: f64 = 1e-10;
const ATOL: f64 = 1e-12;
const JAC_TOL: f64 = 1e-4;
const MAX_ITERS: usize = 50;
const NEWTON_TOL: f64 = 1e-6;

fn deer() -> Vec<Vec<Vector2<f64>>> {
    let y0 = Vector2::from(Y0);
    let mut x = coarse();
    let mut iterates = vec![x.clone()];

    for _ in 0..MAX_ITERS {
        let maps: Vec<(Matrix2<f64>, Vector2<f64>)> = x[..N_CHUNKS]
            .par_iter()
            .enumerate()
            .map_init(
                || (problem(RTOL, ATOL), problem(JAC_TOL, JAC_TOL)),
                |problems, (k, x_k)| {
                    let (f_k, j_k) = propagate(problems, k, x_k);
                    (j_k, f_k - j_k * x_k)
                },
            )
            .collect();

        let prefixes: Vec<(Matrix2<f64>, Vector2<f64>)> = maps
            .into_par_iter()
            .scan(
                |(a1, b1), (a2, b2)| (a2 * a1, a2 * b1 + b2),
                (Matrix2::identity(), Vector2::zeros()),
            )
            .collect();

        let mut x_new = vec![y0];
        x_new.par_extend(prefixes.par_iter().map(|(a, b)| a * y0 + b));
        let delta = max_error(&x, &x_new);
        x = x_new;
        iterates.push(x.clone());
        if delta < NEWTON_TOL {
            break;
        }
    }
    iterates
}
// ANCHOR_END: deer

/// Sequential reference solution at the chunk boundaries.
fn sequential(t_boundaries: &[f64]) -> Vec<Vector2<f64>> {
    let problem = problem(RTOL, ATOL);
    let (ys, _) = problem.tsit45().unwrap().solve_dense(t_boundaries).unwrap();
    (0..t_boundaries.len())
        .map(|k| Vector2::new(ys.column(k)[0], ys.column(k)[1]))
        .collect()
}

fn max_error(x: &[Vector2<f64>], reference: &[Vector2<f64>]) -> f64 {
    x.iter()
        .zip(reference)
        .map(|(a, b)| (a - b).amax())
        .fold(0.0, f64::max)
}

fn thread_scaling() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    deer(); // warm up the pool and the allocator

    let mut threads = Vec::new();
    let mut elapsed = Vec::new();
    let max_threads = rayon::current_num_threads();
    for n in [1, 2, 4, 8, 16, 24, 30]
        .into_iter()
        .filter(|&n| n <= max_threads)
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        // use the median of a few runs to avoid outliers
        let mut runs: Vec<f64> = (0..BENCH_REPEATS)
            .map(|_| {
                let start = Instant::now();
                pool.install(deer);
                start.elapsed().as_secs_f64()
            })
            .collect();
        runs.sort_by(f64::total_cmp);
        elapsed.push(runs[runs.len() / 2]);
        threads.push(n as f64);
    }
    let speedup = elapsed.iter().map(|t| elapsed[0] / t).collect();
    (threads, elapsed, speedup)
}

/// Plot the maximum error against the sequential solution for each Newton iterate.
fn plot_error(errors: &[f64]) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new((0..errors.len()).collect(), errors.to_vec()).mode(Mode::LinesMarkers),
    );
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("iteration"))
            .y_axis(Axis::new().title("max error").type_(AxisType::Log)),
    );
    plot
}

/// Plot DEER speedup against thread count, relative to both single-threaded DEER and the
/// sequential solve, with the ideal linear speedup for reference.
fn plot_scaling(threads: &[f64], speedup: &[f64], vs_sequential: &[f64]) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(threads.to_vec(), speedup.to_vec())
            .mode(Mode::LinesMarkers)
            .name("vs DEER on 1 thread"),
    );
    plot.add_trace(
        Scatter::new(threads.to_vec(), vs_sequential.to_vec())
            .mode(Mode::LinesMarkers)
            .name("vs sequential"),
    );
    plot.add_trace(
        Scatter::new(threads.to_vec(), threads.to_vec())
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash))
            .name("ideal"),
    );
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("threads"))
            .y_axis(Axis::new().title("speedup")),
    );
    plot
}

fn write_plot(plot: &Plot, name: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/performance/images")
        .join(format!("{name}.html"));
    fs::write(path, plot.to_inline_html(Some(name))).expect("Unable to write file");
}

fn main() {
    let t: Vec<f64> = (0..=N_CHUNKS).map(|k| k as f64 * DT_CHUNK).collect();
    println!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    let reference = sequential(&t);
    let mut runs: Vec<f64> = (0..BENCH_REPEATS)
        .map(|_| {
            let start = Instant::now();
            sequential(&t);
            start.elapsed().as_secs_f64()
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    let t_sequential = runs[runs.len() / 2];

    let iterates = deer();
    let errors: Vec<f64> = iterates.iter().map(|x| max_error(x, &reference)).collect();
    println!("iteration  max error");
    for (i, e) in errors.iter().enumerate() {
        println!("{i:>9}  {e:>9.2e}");
    }
    write_plot(&plot_error(&errors), "deer_error");

    let (threads, elapsed, speedup) = thread_scaling();
    let vs_sequential: Vec<f64> = elapsed.iter().map(|t| t_sequential / t).collect();
    println!("sequential solve: {t_sequential:.4} s");
    println!("threads  elapsed (s)  speedup  vs sequential");
    for i in 0..threads.len() {
        println!(
            "{:>7}  {:>11.4}  {:>7.2}  {:>13.2}",
            threads[i] as usize, elapsed[i], speedup[i], vs_sequential[i]
        );
    }
    write_plot(
        &plot_scaling(&threads, &speedup, &vs_sequential),
        "deer_scaling",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deer_converges_to_the_sequential_solution() {
        let t: Vec<f64> = (0..=N_CHUNKS).map(|k| k as f64 * DT_CHUNK).collect();
        let iterates = deer();
        assert!(iterates.len() <= MAX_ITERS);
        assert!(max_error(iterates.last().unwrap(), &sequential(&t)) < 1e-6);
    }
}
