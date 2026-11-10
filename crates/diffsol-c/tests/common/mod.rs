#![allow(dead_code)]

use diffsol_c::{host_array::HostArray, JitBackendType, OdeSolverType, SolutionWrapper};

pub const ASSERT_TOL: f64 = 1e-5;
pub const LOGISTIC_X0: f64 = 0.1;

pub fn all_ode_solvers() -> [OdeSolverType; 4] {
    [
        OdeSolverType::Bdf,
        OdeSolverType::Esdirk34,
        OdeSolverType::TrBdf2,
        OdeSolverType::Tsit45,
    ]
}

pub fn available_jit_backends() -> Vec<JitBackendType> {
    let backends: &[JitBackendType] = &[
        #[cfg(feature = "diffsl-cranelift")]
        JitBackendType::Cranelift,
        #[cfg(feature = "diffsl-llvm")]
        JitBackendType::Llvm,
    ];
    backends.to_vec()
}

/// An `[n, 1]` array: one lane of values, for example `params`, `y` or `t_eval`.
pub fn vector_host(values: &[f64]) -> HostArray {
    batched_vector_host(&[values])
}

/// An `[n, nlanes]` array from `nlanes` lanes of `n` values.
pub fn batched_vector_host(lanes: &[&[f64]]) -> HostArray {
    let n = lanes.first().map_or(0, |lane| lane.len());
    assert!(
        lanes.iter().all(|lane| lane.len() == n),
        "lanes must have the same length"
    );
    HostArray::from_vec(lanes.concat(), vec![n, lanes.len()])
}

/// A `[rows, cols, 1]` array (one lane) from column-major values.
pub fn matrix_host(rows: usize, cols: usize, values_col_major: &[f64]) -> HostArray {
    // one lane: `[rows, cols, 1]`, in the column-major order of the values
    HostArray::from_vec(values_col_major.to_vec(), vec![rows, cols, 1])
}

pub fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{label}: expected {expected:.8}, got {actual:.8}, abs err {err:.8} > {tol:.8}"
    );
}

pub fn logistic_state(x0: f64, r: f64, t: f64) -> f64 {
    let exp_rt = (r * t).exp();
    (x0 * exp_rt) / (1.0 - x0 + x0 * exp_rt)
}

pub fn logistic_state_dr(x0: f64, r: f64, t: f64) -> f64 {
    let x = logistic_state(x0, r, t);
    t * x * (1.0 - x)
}

pub fn logistic_integral(x0: f64, r: f64, t: f64) -> f64 {
    let a = (1.0 - x0) / x0;
    t + ((1.0 + a * (-r * t).exp()).ln() - (1.0 + a).ln()) / r
}

pub fn logistic_integral_dr(x0: f64, r: f64, t: f64) -> f64 {
    let a = (1.0 - x0) / x0;
    let exp_term = (-r * t).exp();
    let numerator = (1.0 + a * exp_term).ln() - (1.0 + a).ln();
    let numerator_dr = -a * t * exp_term / (1.0 + a * exp_term);
    (r * numerator_dr - numerator) / (r * r)
}

pub fn hybrid_logistic_period(r: f64) -> f64 {
    81.0_f64.ln() / r
}

pub fn hybrid_logistic_state(r: f64, t: f64) -> f64 {
    let tau = hybrid_logistic_period(r);
    let cycles = (t / tau).floor();
    let local_t = t - cycles * tau;
    logistic_state(LOGISTIC_X0, r, local_t)
}

pub fn hybrid_logistic_state_dr(r: f64, t: f64) -> f64 {
    let tau = hybrid_logistic_period(r);
    let cycles = (t / tau).floor();
    let local_t = t - cycles * tau;
    let x = hybrid_logistic_state(r, t);
    (local_t + cycles * tau) * x * (1.0 - x)
}

/// The value of `array` at `index` as `f64`. The array has the scalar type of the solver, `f32`
/// or `f64`.
pub fn value_f64(array: &HostArray, index: &[usize]) -> f64 {
    match array.index::<f64>(index) {
        Ok(&value) => value,
        Err(_) => *array.index::<f32>(index).unwrap() as f64,
    }
}

/// Lane `lane` of a `[n, nlanes]` array, or the values of a 1D array (lane 0), as `f64` values.
pub fn lane_f64(array: &HostArray, lane: usize) -> Vec<f64> {
    match array.shape() {
        [n] => (0..*n).map(|i| value_f64(array, &[i])).collect(),
        [n, _] => (0..*n).map(|i| value_f64(array, &[i, lane])).collect(),
        shape => panic!("expected a 1D or 2D array, got the shape {shape:?}"),
    }
}

/// Lane `lane` of a `[nrows, ncols, nlanes]` array as an `f64` matrix.
pub fn lane_matrix(array: &HostArray, lane: usize) -> ndarray::Array2<f64> {
    let shape = array.shape();
    ndarray::Array2::from_shape_fn((shape[0], shape[1]), |(i, j)| {
        value_f64(array, &[i, j, lane])
    })
}

pub fn assert_solution_tail(
    solution: &SolutionWrapper,
    expected_ts: &[f64],
    x0: f64,
    r: f64,
    tol: f64,
) {
    let ys_array = solution.get_ys().unwrap();
    let ys = lane_matrix(&ys_array, 0);
    let ts = lane_f64(&solution.get_ts().unwrap(), 0);

    assert_eq!(ys.nrows(), 1, "expected a single state/output row");
    assert!(
        ys.ncols() >= expected_ts.len(),
        "expected at least {} columns, got {}",
        expected_ts.len(),
        ys.ncols()
    );
    assert!(
        ts.len() >= expected_ts.len(),
        "expected at least {} time points, got {}",
        expected_ts.len(),
        ts.len()
    );

    let start = ts
        .windows(expected_ts.len())
        .enumerate()
        .filter_map(|(start, window)| {
            window
                .iter()
                .zip(expected_ts.iter())
                .all(|(&actual, &expected)| (actual - expected).abs() <= tol)
                .then_some(start)
        })
        .next_back()
        .unwrap_or_else(|| {
            panic!(
                "could not find expected time window {:?} inside actual times {:?}",
                expected_ts, ts
            )
        });

    for (i, &t) in expected_ts.iter().enumerate() {
        assert_close(ts[start + i], t, tol, "solution time");
        assert_close(
            ys[(0, start + i)],
            logistic_state(x0, r, t),
            tol,
            &format!("solution value[{i}]"),
        );
    }
}
