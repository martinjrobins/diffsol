use diffsol::{
    DenseMatrix, MatrixCommon, NalgebraMat, OdeBuilder, OdeEquations, OdeSolverMethod, Op, Vector,
};
use plotly::{
    common::{Fill, Line, Mode},
    layout::Axis,
    Layout, Plot, Scatter,
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, LogNormal};
use rayon::prelude::*;
use std::{fs, path::PathBuf, time::Instant};

// ANCHOR: types
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;

const N_SAMPLES: usize = 1000;
const N_TIMES: usize = 101;
const T_FINAL: f64 = 20.0;
const K: f64 = 10.0;
const Y0: f64 = 0.1;
const SEED: u64 = 42;
// ANCHOR_END: types

/// Number of samples used for the thread-scaling benchmark. Larger than `N_SAMPLES` so that
/// each timed run is long enough to measure reliably.
const BENCH_SAMPLES: usize = 20000;
const BENCH_REPEATS: usize = 3;

// ANCHOR: ensemble
fn ensemble(n_samples: usize, t_eval: &[f64]) -> Vec<[f64; 3]> {
    let solutions: Vec<M> = (0..n_samples)
        .into_par_iter()
        .map_init(
            || {
                OdeBuilder::<M>::new()
                    .p([1.0, K])
                    .rhs(|y, p, _t, dy| dy[0] = p[0] * y[0] * (1.0 - y[0] / p[1]))
                    .init(|_p, _t, y| y[0] = Y0, 1)
                    .build()
                    .unwrap()
            },
            |problem, i| {
                let mut rng = ChaCha12Rng::seed_from_u64(SEED);
                rng.set_stream(i as u64);
                let r = LogNormal::new(0.5_f64.ln(), 0.3).unwrap().sample(&mut rng);

                let p = V::from_vec(vec![r, K], *problem.eqn.context());
                problem.eqn_mut().set_params(&p);
                problem.tsit45().unwrap().solve_dense(t_eval).unwrap().0
            },
        )
        .collect();

    (0..t_eval.len())
        .into_par_iter()
        .map(|i| {
            let mut ys: Vec<f64> = solutions.iter().map(|s| s.column(i)[0]).collect();
            ys.sort_by(f64::total_cmp);
            [quantile(&ys, 0.05), quantile(&ys, 0.5), quantile(&ys, 0.95)]
        })
        .collect()
}

/// Quantile of an ascending slice, linearly interpolating between order statistics.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}
// ANCHOR_END: ensemble

// ANCHOR: scaling
fn thread_scaling(t_eval: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    ensemble(BENCH_SAMPLES, t_eval); // warm up the pool and the allocator

    let mut threads = Vec::new();
    let mut elapsed = Vec::new();
    for n in 1..=rayon::current_num_threads() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        // use the median of a few runs to avoid outliers
        let mut runs: Vec<f64> = (0..BENCH_REPEATS)
            .map(|_| {
                let start = Instant::now();
                let bands = pool.install(|| ensemble(BENCH_SAMPLES, t_eval));
                assert_eq!(bands.len(), t_eval.len());
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
// ANCHOR_END: scaling

// ANCHOR: plot
/// Plot the median with a shaded 5%-95% band. The band is drawn by filling the 95% trace
/// down to the 5% trace that precedes it.
fn plot_bands(t_eval: &[f64], bands: &[[f64; 3]]) -> Plot {
    let t: Vec<f64> = t_eval.to_vec();
    let lower: Vec<f64> = bands.iter().map(|b| b[0]).collect();
    let median: Vec<f64> = bands.iter().map(|b| b[1]).collect();
    let upper: Vec<f64> = bands.iter().map(|b| b[2]).collect();

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(t.clone(), lower)
            .mode(Mode::Lines)
            .line(Line::new().width(0.0))
            .name("5%"),
    );
    plot.add_trace(
        Scatter::new(t.clone(), upper)
            .mode(Mode::Lines)
            .line(Line::new().width(0.0))
            .fill(Fill::ToNextY)
            .fill_color("rgba(31, 119, 180, 0.25)")
            .name("95%"),
    );
    plot.add_trace(Scatter::new(t, median).mode(Mode::Lines).name("median"));
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("t"))
            .y_axis(Axis::new().title("y")),
    );
    plot
}
// ANCHOR_END: plot

// ANCHOR: scaling_plot
/// Plot measured speedup against thread count, with the ideal linear speedup for reference.
fn plot_scaling(threads: &[f64], speedup: &[f64]) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(threads.to_vec(), speedup.to_vec())
            .mode(Mode::LinesMarkers)
            .name("measured"),
    );
    plot.add_trace(
        Scatter::new(threads.to_vec(), threads.to_vec())
            .mode(Mode::Lines)
            .line(Line::new().dash(plotly::common::DashType::Dash))
            .name("ideal"),
    );
    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title("threads"))
            .y_axis(Axis::new().title("speedup")),
    );
    plot
}
// ANCHOR_END: scaling_plot

fn write_plot(plot: &Plot, name: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../book/src/performance/images")
        .join(format!("{name}.html"));
    fs::write(path, plot.to_inline_html(Some(name))).expect("Unable to write file");
}

fn main() {
    let t_eval: Vec<f64> = (0..N_TIMES)
        .map(|i| i as f64 * T_FINAL / (N_TIMES - 1) as f64)
        .collect();

    println!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    let bands = ensemble(N_SAMPLES, &t_eval);
    write_plot(&plot_bands(&t_eval, &bands), "parallel_ensemble_bands");

    let (threads, elapsed, speedup) = thread_scaling(&t_eval);
    println!("threads  elapsed (s)  speedup");
    for i in 0..threads.len() {
        println!(
            "{:>7}  {:>11.4}  {:>7.2}",
            threads[i] as usize, elapsed[i], speedup[i]
        );
    }
    write_plot(
        &plot_scaling(&threads, &speedup),
        "parallel_ensemble_scaling",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logistic_exact(r: f64, t: f64) -> f64 {
        K / (1.0 + ((K - Y0) / Y0) * (-r * t).exp())
    }

    #[test]
    fn solves_match_the_exact_solution_and_quantiles_are_ordered() {
        let problem = OdeBuilder::<M>::new()
            .p([1.0, K])
            .rhs(|y, p, _t, dy| dy[0] = p[0] * y[0] * (1.0 - y[0] / p[1]))
            .init(|_p, _t, y| y[0] = Y0, 1)
            .build()
            .unwrap();
        let (solution, _) = problem.tsit45().unwrap().solve_dense(&[5.0]).unwrap();
        assert!((solution.column(0)[0] - logistic_exact(1.0, 5.0)).abs() < 1e-5);

        let t_eval: Vec<f64> = (0..N_TIMES)
            .map(|i| i as f64 * T_FINAL / (N_TIMES - 1) as f64)
            .collect();
        let bands = ensemble(N_SAMPLES, &t_eval);
        assert_eq!(bands.len(), N_TIMES);
        for [lower, median, upper] in bands {
            assert!(lower <= median && median <= upper);
        }
    }
}
