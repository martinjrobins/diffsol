use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use nalgebra::DMatrix;

use crate::error_c::{
    diffsol_clear_last_error, diffsol_error_code, diffsol_last_error_file, diffsol_last_error_line,
    diffsol_last_error_message,
};
use crate::host_array::{FromHostArray, HostArray, ToHostArray};
use crate::host_array_c::{
    diffsol_host_array_dim, diffsol_host_array_dtype, diffsol_host_array_free,
    diffsol_host_array_ndim, diffsol_host_array_ptr, diffsol_host_array_stride,
};
use crate::ode_c::diffsol_host_array_list_free;
use crate::scalar_type::ScalarType;
use crate::scalar_type_c::scalar_type_to_i32;
use crate::solution_wrapper::SolutionWrapper;
use crate::solution_wrapper_c::diffsol_solution_wrapper_free;

pub(crate) const ASSERT_TOL: f64 = 1e-5;

pub(crate) fn rhs_state_deps() -> Vec<(usize, usize)> {
    vec![(0, 0)]
}

pub(crate) fn rhs_input_deps() -> Vec<(usize, usize)> {
    vec![(0, 0)]
}

pub(crate) fn mass_state_deps() -> Vec<(usize, usize)> {
    Vec::new()
}

pub(crate) fn dummy_code() -> CString {
    CString::new("ignored-by-diffsol-c").unwrap()
}

pub(crate) fn vector_host(values: &[f64]) -> HostArray {
    values.to_vec().to_host_array()
}

pub(crate) fn matrix_host(rows: usize, cols: usize, values_col_major: &[f64]) -> HostArray {
    DMatrix::from_column_slice(rows, cols, values_col_major).to_host_array()
}

pub(crate) fn logistic_state(x0: f64, r: f64, t: f64) -> f64 {
    let exp_rt = (r * t).exp();
    (x0 * exp_rt) / (1.0 - x0 + x0 * exp_rt)
}

pub(crate) fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{label}: expected {expected:.8}, got {actual:.8}, abs err {err:.8} > {tol:.8}"
    );
}

pub(crate) fn host_array_to_matrix_f64(array: &HostArray) -> (usize, usize, Vec<f64>) {
    let view = array.as_array::<f64>().unwrap();
    let mut values = Vec::with_capacity(view.nrows() * view.ncols());
    for col in 0..view.ncols() {
        for row in 0..view.nrows() {
            values.push(view[(row, col)]);
        }
    }
    (view.nrows(), view.ncols(), values)
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
        .last()
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
    let (rows, cols, ys) = host_array_to_matrix_f64(&ys_array);
    let ts = Vec::<f64>::from_host_array(solution.get_ts().unwrap()).unwrap();

    assert_eq!(rows, 1, "expected a single state/output row");
    assert!(
        cols >= expected_ts.len(),
        "expected at least {} columns, got {cols}",
        expected_ts.len()
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
            ys[start + i],
            logistic_state(x0, r, t),
            tol,
            "solution value",
        );
    }
}

pub(crate) fn assert_current_state(solution: &SolutionWrapper, expected: &[f64], tol: f64) {
    let actual = Vec::<f64>::from_host_array(solution.get_current_state().unwrap()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(actual, expected, tol, &format!("current_state[{i}]"));
    }
}

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

pub(crate) unsafe fn ffi_read_host_array_matrix(ptr: *mut HostArray) -> (usize, usize, Vec<f64>) {
    assert!(!ptr.is_null(), "host array pointer must not be null");
    assert_eq!(unsafe { diffsol_host_array_ndim(ptr) }, 2);
    assert_eq!(
        unsafe { diffsol_host_array_dtype(ptr) },
        scalar_type_to_i32(ScalarType::F64)
    );
    let rows = unsafe { diffsol_host_array_dim(ptr, 0) };
    let cols = unsafe { diffsol_host_array_dim(ptr, 1) };
    let row_stride = unsafe { diffsol_host_array_stride(ptr, 0) };
    let col_stride = unsafe { diffsol_host_array_stride(ptr, 1) };
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

pub(crate) unsafe fn c_string(ptr: *const c_char) -> String {
    assert!(!ptr.is_null(), "expected non-null C string");
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned()
}

pub(crate) unsafe fn assert_last_error_contains(expected_substring: &str) {
    assert_eq!(
        unsafe { diffsol_error_code() },
        1,
        "expected last error to be set"
    );
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

#[cfg(feature = "external-f64")]
const STATES: u32 = 1;
#[cfg(feature = "external-f64")]
const INPUTS: u32 = 1;
#[cfg(feature = "external-f64")]
const OUTPUTS: u32 = 1;
#[cfg(feature = "external-f64")]
const DATA: u32 = 1;
#[cfg(feature = "external-f64")]
const STOP: u32 = 1;

#[cfg(feature = "external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn barrier_init() {}

#[cfg(feature = "external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_constants(_thread_id: u32, _thread_dim: u32) {}

#[cfg(feature = "external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_u0(u: *mut f64, _data: *mut f64, _thread_id: u32, _thread_dim: u32) {
    if !u.is_null() {
        unsafe {
            *u = 1.0;
        }
    }
}

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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
        *drr = *data * x * (1.0 - x);
        *ddata = x * (1.0 - x);
    }
}

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_id(id: *mut f64) {
    if !id.is_null() {
        unsafe {
            *id = 42.0;
        }
    }
}

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_inputs(inputs: *const f64, data: *mut f64, _model_index: u32) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    unsafe {
        *data = *inputs;
    }
}

#[cfg(feature = "external-f64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_inputs(inputs: *mut f64, data: *const f64) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    unsafe {
        *inputs = *data;
    }
}

#[cfg(feature = "external-f64")]
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

#[cfg(feature = "external-f64")]
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
