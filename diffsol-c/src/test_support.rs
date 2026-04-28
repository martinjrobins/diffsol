#![allow(clippy::missing_safety_doc, dead_code)]

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
use std::ffi::CStr;
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use std::ffi::CString;
#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
use std::os::raw::c_char;
#[cfg(feature = "diffsl-external-dynamic")]
use std::path::PathBuf;
#[cfg(feature = "diffsl-external-dynamic")]
use std::process::Command;
#[cfg(feature = "diffsl-external-dynamic")]
use std::sync::OnceLock;

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-llvm"))]
use nalgebra::DMatrix;

use crate::error_c::diffsol_clear_last_error;
#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
use crate::error_c::{
    diffsol_error_code, diffsol_last_error_file, diffsol_last_error_line,
    diffsol_last_error_message,
};
use crate::host_array::{FromHostArray, HostArray, ToHostArray};
use crate::host_array_c::{
    diffsol_host_array_dim, diffsol_host_array_dtype, diffsol_host_array_free,
    diffsol_host_array_ndim, diffsol_host_array_ptr,
};
#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-llvm"))]
use crate::ode_c::diffsol_host_array_list_free;
use crate::scalar_type::ScalarType;
use crate::scalar_type_c::scalar_type_to_i32;
use crate::solution_wrapper::SolutionWrapper;
#[cfg(any(
    feature = "diffsl-external-f64",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm"
))]
use crate::solution_wrapper_c::diffsol_solution_wrapper_free;

pub(crate) const ASSERT_TOL: f64 = 1e-5;
#[cfg(any(
    feature = "diffsl-external-f64",
    feature = "diffsl-external-dynamic",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm"
))]
pub(crate) const LOGISTIC_X0: f64 = 0.1;

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
pub(crate) fn rhs_state_deps() -> Vec<(usize, usize)> {
    vec![(0, 0)]
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
pub(crate) fn rhs_input_deps() -> Vec<(usize, usize)> {
    vec![(0, 0)]
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
pub(crate) fn mass_state_deps() -> Vec<(usize, usize)> {
    Vec::new()
}

#[cfg(feature = "diffsl-external-dynamic")]
fn external_dynamic_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/external-dynamic-logistic")
}

#[cfg(feature = "diffsl-external-dynamic")]
fn external_dynamic_fixture_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "diffsol_c_external_dynamic_fixture.dll"
    } else if cfg!(target_os = "macos") {
        "libdiffsol_c_external_dynamic_fixture.dylib"
    } else {
        "libdiffsol_c_external_dynamic_fixture.so"
    }
}

#[cfg(feature = "diffsl-external-dynamic")]
pub(crate) fn external_dynamic_fixture_path() -> PathBuf {
    static FIXTURE_PATH: OnceLock<PathBuf> = OnceLock::new();

    FIXTURE_PATH
        .get_or_init(|| {
            let fixture_dir = external_dynamic_fixture_dir();
            let manifest_path = fixture_dir.join("Cargo.toml");
            let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
            let output = Command::new(cargo)
                .arg("build")
                .arg("--manifest-path")
                .arg(&manifest_path)
                .current_dir(&fixture_dir)
                .output()
                .expect("failed to build external dynamic fixture");
            if !output.status.success() {
                panic!(
                    "failed to build external dynamic fixture\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            }

            let profile_dir = if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            };
            let library_path = fixture_dir
                .join("target")
                .join(profile_dir)
                .join(external_dynamic_fixture_filename());
            assert!(
                library_path.is_file(),
                "expected built fixture library at {}",
                library_path.display()
            );
            library_path
        })
        .clone()
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn logistic_diffsl_code() -> &'static str {
    r#"
        in_i { r = 1 }
        u_i { y = 0.1 }
        dudt_i { dydt = 0 }
        F_i { (r * y) * (1 - y) }
        out_i { y }
    "#
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn hybrid_logistic_diffsl_code() -> &'static str {
    r#"
        in_i { r = 1 }
        u_i { y = 0.1 }
        dudt_i { dydt = 0 }
        F_i { (r * y) * (1 - y) }
        stop_i { y - 0.9 }
        reset_i { 0.1 }
        out_i { y }
    "#
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn logistic_diffsl_code_cstring() -> CString {
    CString::new(logistic_diffsl_code()).unwrap()
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn available_jit_backends() -> Vec<crate::jit::JitBackendType> {
    [
        #[cfg(feature = "diffsl-cranelift")]
        Some(crate::jit::JitBackendType::Cranelift),
        #[cfg(feature = "diffsl-llvm")]
        Some(crate::jit::JitBackendType::Llvm),
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub(crate) fn vector_host(values: &[f64]) -> HostArray {
    values.to_vec().to_host_array()
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-llvm"))]
pub(crate) fn matrix_host(rows: usize, cols: usize, values_col_major: &[f64]) -> HostArray {
    DMatrix::from_column_slice(rows, cols, values_col_major).to_host_array()
}

pub(crate) fn logistic_state(x0: f64, r: f64, t: f64) -> f64 {
    let exp_rt = (r * t).exp();
    (x0 * exp_rt) / (1.0 - x0 + x0 * exp_rt)
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn hybrid_logistic_period(r: f64) -> f64 {
    81.0_f64.ln() / r
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn hybrid_logistic_state(r: f64, t: f64) -> f64 {
    let tau = hybrid_logistic_period(r);
    let cycles = (t / tau).floor();
    let local_t = t - cycles * tau;
    logistic_state(LOGISTIC_X0, r, local_t)
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-llvm"))]
pub(crate) fn logistic_state_dr(x0: f64, r: f64, t: f64) -> f64 {
    let x = logistic_state(x0, r, t);
    t * x * (1.0 - x)
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn hybrid_logistic_state_dr(r: f64, t: f64) -> f64 {
    let tau = hybrid_logistic_period(r);
    let cycles = (t / tau).floor();
    let local_t = t - cycles * tau;
    let x = hybrid_logistic_state(r, t);
    // On each segment, the fixed-local-time sensitivity is local_t * x * (1 - x).
    // For the hybrid solution at fixed global time, the stop time also moves with r.
    // With stop(y) = y - 0.9, we have s_y = 1 and s_r = 0, so implicit differentiation
    // of stop(y(tau(r), r)) = 0 gives d tau / d r = -tau / r. Each completed cycle
    // therefore contributes tau * x * (1 - x), so the total is:
    //   (local_t + cycles * tau) * x * (1 - x) = t * x * (1 - x).
    (local_t + cycles * tau) * x * (1.0 - x)
}

#[cfg(feature = "diffsl-llvm")]
pub(crate) fn logistic_integral(x0: f64, r: f64, t: f64) -> f64 {
    let a = (1.0 - x0) / x0;
    t + ((1.0 + a * (-r * t).exp()).ln() - (1.0 + a).ln()) / r
}

pub(crate) fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{label}: expected {expected:.8}, got {actual:.8}, abs err {err:.8} > {tol:.8}"
    );
}

pub(crate) fn find_time_window(actual_ts: &[f64], expected_ts: &[f64], tol: f64) -> usize {
    actual_ts
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
                expected_ts, actual_ts
            )
        })
}

pub(crate) fn assert_solution_tail(
    solution: &SolutionWrapper,
    expected_ts: &[f64],
    x0: f64,
    r: f64,
    tol: f64,
) {
    let ys_array = solution.get_ys().unwrap();
    let ys = Vec::<Vec<f64>>::from_host_array(ys_array).unwrap();
    let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

    assert_eq!(ys.len(), 1, "expected a single state/output row");
    assert!(
        ys[0].len() >= expected_ts.len(),
        "expected at least {} columns, got {}",
        expected_ts.len(),
        ys[0].len()
    );
    assert!(
        ts.len() >= expected_ts.len(),
        "expected at least {} time points, got {}",
        expected_ts.len(),
        ts.len()
    );

    let start = find_time_window(&ts, expected_ts, tol);
    for (i, &t) in expected_ts.iter().enumerate() {
        assert_close(ts[start + i], t, tol, "solution time");
        assert_close(
            ys[0][start + i],
            logistic_state(x0, r, t),
            tol,
            "solution value",
        );
    }
}

#[cfg(any(
    feature = "diffsl-external-f64",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm"
))]
pub(crate) unsafe fn ffi_free_solution(ptr: *mut SolutionWrapper) {
    if !ptr.is_null() {
        unsafe {
            diffsol_solution_wrapper_free(ptr);
        }
    }
}

pub(crate) unsafe fn ffi_read_host_array_vector(ptr: *mut HostArray) -> Vec<f64> {
    assert!(!ptr.is_null(), "host array pointer must not be null");
    assert_eq!(unsafe { diffsol_host_array_ndim(ptr) }, 1);
    assert_eq!(
        unsafe { diffsol_host_array_dtype(ptr) },
        scalar_type_to_i32(ScalarType::F64)
    );
    let len = unsafe { diffsol_host_array_dim(ptr, 0) };
    let data_ptr = unsafe { diffsol_host_array_ptr(ptr) } as *const f64;
    let values = unsafe { std::slice::from_raw_parts(data_ptr, len) }.to_vec();
    unsafe {
        diffsol_host_array_free(ptr);
    }
    values
}

#[cfg(any(
    feature = "diffsl-external-f64",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm"
))]
pub(crate) unsafe fn ffi_read_host_array_matrix(ptr: *mut HostArray) -> (usize, usize, Vec<f64>) {
    assert!(!ptr.is_null(), "host array pointer must not be null");
    assert_eq!(unsafe { diffsol_host_array_ndim(ptr) }, 2);
    assert_eq!(
        unsafe { diffsol_host_array_dtype(ptr) },
        scalar_type_to_i32(ScalarType::F64)
    );
    let rows = unsafe { diffsol_host_array_dim(ptr, 0) };
    let cols = unsafe { diffsol_host_array_dim(ptr, 1) };
    let row_stride = unsafe { crate::host_array_c::diffsol_host_array_stride(ptr, 0) };
    let col_stride = unsafe { crate::host_array_c::diffsol_host_array_stride(ptr, 1) };
    let base_ptr = unsafe { diffsol_host_array_ptr(ptr) };
    let mut values = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            let byte_offset = row * row_stride + col * col_stride;
            let value_ptr = unsafe { base_ptr.add(byte_offset) } as *const f64;
            values.push(unsafe { *value_ptr });
        }
    }
    unsafe {
        diffsol_host_array_free(ptr);
    }
    (rows, cols, values)
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-llvm"))]
pub(crate) unsafe fn ffi_read_host_array_list_matrices(
    list: *mut *mut HostArray,
    len: usize,
) -> Vec<(usize, usize, Vec<f64>)> {
    assert!(!list.is_null(), "host array list pointer must not be null");
    let mut arrays = Vec::with_capacity(len);
    for index in 0..len {
        let array_ptr = unsafe { *list.add(index) };
        arrays.push(unsafe { ffi_read_host_array_matrix(array_ptr) });
    }
    unsafe {
        diffsol_host_array_list_free(list, len);
    }
    arrays
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
pub(crate) unsafe fn c_string(ptr: *const c_char) -> String {
    assert!(!ptr.is_null(), "expected non-null C string");
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned()
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
pub(crate) unsafe fn assert_last_error_set() {
    assert_eq!(
        unsafe { diffsol_error_code() },
        1,
        "expected last error to be set"
    );
    let message_ptr = unsafe { diffsol_last_error_message() };
    assert!(
        !message_ptr.is_null(),
        "expected last error message to be set"
    );
    let message = unsafe { c_string(message_ptr) };
    assert!(
        !message.is_empty(),
        "expected last error message to be non-empty"
    );
    let file_ptr = unsafe { diffsol_last_error_file() };
    assert!(!file_ptr.is_null(), "expected last error file to be set");
    assert!(
        unsafe { diffsol_last_error_line() } > 0,
        "expected last error line to be > 0"
    );
}

#[cfg(any(feature = "diffsl-external-f64", feature = "diffsl-external-dynamic"))]
pub(crate) unsafe fn assert_last_error_contains(expected_substring: &str) {
    unsafe { assert_last_error_set() };
    let message_ptr = unsafe { diffsol_last_error_message() };
    let message = unsafe { c_string(message_ptr) };
    assert!(
        message.contains(expected_substring),
        "expected last error message to contain {expected_substring:?}, got {message:?}"
    );
    let file_ptr = unsafe { diffsol_last_error_file() };
    assert!(!file_ptr.is_null(), "expected last error file to be set");
    assert!(
        unsafe { diffsol_last_error_line() } > 0,
        "expected last error line to be > 0"
    );
}

pub(crate) fn clear_last_error() {
    unsafe {
        diffsol_clear_last_error();
    }
}

#[cfg(feature = "diffsl-external-f64")]
const STATES: u32 = 1;
#[cfg(feature = "diffsl-external-f64")]
const INPUTS: u32 = 1;
#[cfg(feature = "diffsl-external-f64")]
const OUTPUTS: u32 = 1;
#[cfg(feature = "diffsl-external-f64")]
const DATA: u32 = 1;
#[cfg(feature = "diffsl-external-f64")]
const STOP: u32 = 1;

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn barrier_init() {}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_constants(_thread_id: u32, _thread_dim: u32) {}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_u0(u: *mut f64, _data: *mut f64, _thread_id: u32, _thread_dim: u32) {
    if !u.is_null() {
        unsafe {
            *u = LOGISTIC_X0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reset(
    _time: f64,
    u: *const f64,
    _data: *mut f64,
    reset: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || reset.is_null() {
        return;
    }
    unsafe {
        *reset = 2.0 * *u;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reset_grad(
    _time: f64,
    _u: *const f64,
    du: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _reset: *const f64,
    dreset: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || ddata.is_null() || dreset.is_null() {
        return;
    }
    unsafe {
        *dreset = 2.0 * *du;
        *ddata = 0.0;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reset_rgrad(
    _time: f64,
    _u: *const f64,
    du: *mut f64,
    _data: *const f64,
    ddata: *mut f64,
    _reset: *const f64,
    dreset: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || ddata.is_null() || dreset.is_null() {
        return;
    }
    unsafe {
        *du += 2.0 * *dreset;
        *ddata += 0.0;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reset_sgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _reset: *const f64,
    dreset: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if ddata.is_null() || dreset.is_null() {
        return;
    }
    unsafe {
        *dreset = 0.0;
        *ddata = 0.0;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reset_srgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _reset: *const f64,
    dreset: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if ddata.is_null() || dreset.is_null() {
        return;
    }
    unsafe {
        *dreset = 0.0;
        *ddata = 0.0;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhs(
    _time: f64,
    u: *const f64,
    data: *mut f64,
    rr: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || data.is_null() || rr.is_null() {
        return;
    }
    unsafe {
        let x = *u;
        let r = *data;
        *rr = r * x * (1.0 - x);
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhs_grad(
    _time: f64,
    u: *const f64,
    du: *const f64,
    data: *const f64,
    ddata: *mut f64,
    _rr: *const f64,
    drr: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || du.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
        return;
    }
    unsafe {
        let x = *u;
        let dx = *du;
        let r = *data;
        *drr = r * (1.0 - 2.0 * x) * dx;
        *ddata = x * (1.0 - x);
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhs_rgrad(
    _time: f64,
    u: *const f64,
    du: *mut f64,
    data: *const f64,
    ddata: *mut f64,
    _rr: *const f64,
    drr: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || du.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
        return;
    }
    unsafe {
        let x = *u;
        let r = *data;
        *du += r * (1.0 - 2.0 * x) * *drr;
        *ddata += x * (1.0 - x) * *drr;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhs_sgrad(
    _time: f64,
    u: *const f64,
    data: *const f64,
    ddata: *mut f64,
    _rr: *const f64,
    drr: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || data.is_null() || ddata.is_null() || drr.is_null() {
        return;
    }
    unsafe {
        let x = *u;
        *drr = x * (1.0 - x);
        *ddata = x * (1.0 - x);
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rhs_srgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _rr: *const f64,
    drr: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !drr.is_null() {
        unsafe {
            *drr = 0.0;
        }
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mass(
    _time: f64,
    v: *const f64,
    _data: *mut f64,
    mv: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if v.is_null() || mv.is_null() {
        return;
    }
    unsafe {
        *mv = *v;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mass_rgrad(
    _time: f64,
    _v: *const f64,
    dv: *mut f64,
    _data: *const f64,
    _ddata: *mut f64,
    _mv: *const f64,
    dmv: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if dv.is_null() || dmv.is_null() {
        return;
    }
    unsafe {
        *dv += *dmv;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_u0_grad(
    _u: *const f64,
    _du: *mut f64,
    _data: *const f64,
    _ddata: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_u0_rgrad(
    _u: *const f64,
    _du: *mut f64,
    _data: *const f64,
    _ddata: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_u0_sgrad(
    _u: *const f64,
    _du: *mut f64,
    _data: *const f64,
    _ddata: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_out(
    _time: f64,
    u: *const f64,
    _data: *mut f64,
    out: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || out.is_null() {
        return;
    }
    unsafe {
        *out = *u;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_out_grad(
    _time: f64,
    _u: *const f64,
    du: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _out: *const f64,
    dout: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || ddata.is_null() || dout.is_null() {
        return;
    }
    unsafe {
        *dout = *du;
        *ddata = 0.0;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_out_rgrad(
    _time: f64,
    _u: *const f64,
    du: *mut f64,
    _data: *const f64,
    _ddata: *mut f64,
    _out: *const f64,
    dout: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || dout.is_null() {
        return;
    }
    unsafe {
        *du += *dout;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_out_sgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _out: *const f64,
    dout: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !dout.is_null() {
        unsafe {
            *dout = 0.0;
        }
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_out_srgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _out: *const f64,
    dout: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !dout.is_null() {
        unsafe {
            *dout = 0.0;
        }
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_stop(
    _time: f64,
    u: *const f64,
    _data: *mut f64,
    root: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if u.is_null() || root.is_null() {
        return;
    }
    unsafe {
        *root = *u - 0.5;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_stop_grad(
    _time: f64,
    _u: *const f64,
    du: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _root: *const f64,
    droot: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || droot.is_null() {
        return;
    }
    unsafe {
        *droot = *du;
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_stop_rgrad(
    _time: f64,
    _u: *const f64,
    du: *mut f64,
    _data: *const f64,
    ddata: *mut f64,
    _root: *const f64,
    droot: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if du.is_null() || droot.is_null() {
        return;
    }
    unsafe {
        *du += *droot;
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_stop_sgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _root: *const f64,
    droot: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !droot.is_null() {
        unsafe {
            *droot = 0.0;
        }
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calc_stop_srgrad(
    _time: f64,
    _u: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _root: *const f64,
    droot: *mut f64,
    _thread_id: u32,
    _thread_dim: u32,
) {
    if !droot.is_null() {
        unsafe {
            *droot = 0.0;
        }
    }
    if !ddata.is_null() {
        unsafe {
            *ddata = 0.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_id(id: *mut f64) {
    if !id.is_null() {
        unsafe {
            *id = 42.0;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_dims(
    states: *mut u32,
    inputs: *mut u32,
    outputs: *mut u32,
    data: *mut u32,
    stop: *mut u32,
    has_mass: *mut u32,
    has_reset: *mut u32,
) {
    if !states.is_null() {
        unsafe {
            *states = STATES;
        }
    }
    if !inputs.is_null() {
        unsafe {
            *inputs = INPUTS;
        }
    }
    if !outputs.is_null() {
        unsafe {
            *outputs = OUTPUTS;
        }
    }
    if !data.is_null() {
        unsafe {
            *data = DATA;
        }
    }
    if !stop.is_null() {
        unsafe {
            *stop = STOP;
        }
    }
    if !has_mass.is_null() {
        unsafe {
            *has_mass = 0;
        }
    }
    if !has_reset.is_null() {
        unsafe {
            *has_reset = 1;
        }
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_inputs(inputs: *const f64, data: *mut f64, _model_index: u32) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    unsafe {
        *data = *inputs;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_inputs(inputs: *mut f64, data: *const f64) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    unsafe {
        *inputs = *data;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_inputs_grad(
    _inputs: *const f64,
    dinputs: *const f64,
    _data: *const f64,
    ddata: *mut f64,
    _model_index: u32,
) {
    if dinputs.is_null() || ddata.is_null() {
        return;
    }
    unsafe {
        *ddata = *dinputs;
    }
}

#[cfg(feature = "diffsl-external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_inputs_rgrad(
    _inputs: *const f64,
    dinputs: *mut f64,
    _data: *const f64,
    ddata: *mut f64,
    _model_index: u32,
) {
    if dinputs.is_null() || ddata.is_null() {
        return;
    }
    unsafe {
        *dinputs += *ddata;
    }
}
