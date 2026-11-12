//! The logistic model `dy/dt = r y (1 - y)`, `y(0) = 0.1`, `out = y`, defined with C callbacks
//! (see `diffsol_c::ode_callbacks`) rather than DiffSL, solved through the runtime API.

mod common;
use common::{batched_vector_tensor, lane_matrix, value_f64};

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::{
    all_ode_solvers, assert_close, assert_solution_tail, hybrid_logistic_state,
    hybrid_logistic_state_dr, lane_f64, logistic_integral, logistic_integral_dr, logistic_state,
    logistic_state_dr, matrix_tensor, vector_tensor, ASSERT_TOL, LOGISTIC_X0,
};
use diffsol_c::ode_c::diffsol_ode_free;
use diffsol_c::ode_callbacks::for_each_batch;
use diffsol_c::ode_callbacks::DiffsolOdeCallbacksF32;
use diffsol_c::ode_callbacks_c::{diffsol_ode_new_callbacks_f32, diffsol_ode_new_callbacks_f64};
use diffsol_c::scalar_type::Scalar;
use diffsol_c::tensor::{Tensor, TensorView, TensorViewMut};
use diffsol_c::{
    LinearSolverType, MatrixType, OdeCallbacksDesc, OdeSolverType, OdeWrapper, ScalarType,
};

const ALL_MATRIX_TYPES: [MatrixType; 3] = [
    MatrixType::NalgebraDense,
    MatrixType::FaerDense,
    MatrixType::FaerSparse,
];

/// What the callbacks see through `user_data`.
struct UserData {
    /// If non-zero, `rhs` returns this status instead of evaluating.
    rhs_status: i32,
    /// If non-zero, `rhs` fails (with status 1) on this call only, counting from 1.
    rhs_fail_call: usize,
    /// Counts the calls to `rhs`.
    rhs_calls: AtomicUsize,
    /// Counts the calls to `free_user_data`. The test keeps a clone, so that it can read the
    /// count after the library frees `user_data`.
    frees: Arc<AtomicUsize>,
}

fn user_data<'a>(ud: *mut c_void) -> &'a UserData {
    unsafe { &*(ud as *const UserData) }
}

type View<'a> = *const TensorView<'a>;
type ViewMut<'a> = *mut TensorViewMut<'a>;

/// Set `y[0] = f(inputs[0][0], ...)` in each lane, and return success.
unsafe fn set<T: Scalar, const N: usize>(
    inputs: [View<'_>; N],
    y: ViewMut<'_>,
    f: impl Fn([T; N]) -> T,
) -> i32 {
    unsafe {
        for_each_batch(inputs, y, |ins: [&[T]; N], y: &mut [T], _| {
            y[0] = f(ins.map(|v| v[0]))
        })
    };
    0
}

/// Set all the values of `y` to zero, and return success.
unsafe fn zero<T: Scalar>(y: ViewMut<'_>) -> i32 {
    unsafe { for_each_batch([], y, |_, y: &mut [T], _| y.fill(T::zero())) };
    0
}

/// Define the logistic callbacks for scalar type `$t` in a module `$m`.
macro_rules! logistic_callbacks {
    ($m:ident, $t:ty) => {
        #[allow(dead_code)]
        mod $m {
            use super::*;
            type T = $t;

            pub unsafe extern "C" fn rhs(
                x: View<'_>,
                p: View<'_>,
                _t: T,
                y: ViewMut<'_>,
                ud: *mut c_void,
            ) -> i32 {
                let ud = user_data(ud);
                if ud.rhs_status != 0 {
                    return ud.rhs_status;
                }
                let call = ud.rhs_calls.fetch_add(1, Ordering::SeqCst) + 1;
                if call == ud.rhs_fail_call {
                    return 1;
                }
                unsafe { set([x, p], y, |[x, p]: [T; 2]| p * x * (1.0 - x)) }
            }

            pub unsafe extern "C" fn rhs_jac(
                x: View<'_>,
                p: View<'_>,
                _t: T,
                v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([x, p, v], y, |[x, p, v]: [T; 3]| p * (1.0 - 2.0 * x) * v) }
            }

            pub unsafe extern "C" fn rhs_sens(
                x: View<'_>,
                _p: View<'_>,
                _t: T,
                v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([x, v], y, |[x, v]: [T; 2]| x * (1.0 - x) * v) }
            }

            pub unsafe extern "C" fn rhs_adjoint(
                x: View<'_>,
                p: View<'_>,
                _t: T,
                v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                // the Jacobian is 1x1, so it is its own transpose
                unsafe { set([x, p, v], y, |[x, p, v]: [T; 3]| -p * (1.0 - 2.0 * x) * v) }
            }

            pub unsafe extern "C" fn rhs_sens_adjoint(
                x: View<'_>,
                _p: View<'_>,
                _t: T,
                v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([x, v], y, |[x, v]: [T; 2]| -x * (1.0 - x) * v) }
            }

            pub unsafe extern "C" fn init(
                _p: View<'_>,
                _t: T,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([], y, |[]: [T; 0]| LOGISTIC_X0 as T) }
            }

            pub unsafe extern "C" fn init_zero_mul(
                _p: View<'_>,
                _t: T,
                _v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { zero::<T>(y) }
            }

            pub unsafe extern "C" fn identity(
                x: View<'_>,
                _p: View<'_>,
                _t: T,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([x], y, |[x]: [T; 1]| x) }
            }

            pub unsafe extern "C" fn identity_mul(
                _x: View<'_>,
                _p: View<'_>,
                _t: T,
                v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([v], y, |[v]: [T; 1]| v) }
            }

            pub unsafe extern "C" fn neg_identity_mul(
                _x: View<'_>,
                _p: View<'_>,
                _t: T,
                v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([v], y, |[v]: [T; 1]| -v) }
            }

            pub unsafe extern "C" fn zero_mul(
                _x: View<'_>,
                _p: View<'_>,
                _t: T,
                _v: View<'_>,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { zero::<T>(y) }
            }

            /// root: `y - 0.9`
            pub unsafe extern "C" fn root(
                x: View<'_>,
                _p: View<'_>,
                _t: T,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([x], y, |[x]: [T; 1]| x - 0.9) }
            }

            /// reset: `y = 0.1`
            pub unsafe extern "C" fn reset(
                _x: View<'_>,
                _p: View<'_>,
                _t: T,
                y: ViewMut<'_>,
                _ud: *mut c_void,
            ) -> i32 {
                unsafe { set([], y, |[]: [T; 0]| 0.1) }
            }

            /// The logistic model with every callback needed for forward sensitivities and
            /// adjoints.
            pub fn desc(ud: *mut UserData) -> OdeCallbacksDesc<T> {
                let mut desc = OdeCallbacksDesc::new(1, 1, rhs, init);
                desc.user_data = ud as *mut c_void;
                desc.free_user_data = Some(free_user_data);
                desc.rhs_jac = Some(rhs_jac);
                desc.rhs_sens = Some(rhs_sens);
                desc.rhs_adjoint = Some(rhs_adjoint);
                desc.rhs_sens_adjoint = Some(rhs_sens_adjoint);
                desc.init_sens = Some(init_zero_mul);
                desc.init_sens_adjoint = Some(init_zero_mul);
                desc.out = Some(identity);
                desc.nout = 1;
                desc.out_jac = Some(identity_mul);
                desc.out_sens = Some(zero_mul);
                desc.out_adjoint = Some(neg_identity_mul);
                desc.out_sens_adjoint = Some(zero_mul);
                desc
            }

            /// [`desc`] plus a root at `y = 0.9` that resets the state to `y = 0.1`.
            pub fn hybrid_desc(ud: *mut UserData) -> OdeCallbacksDesc<T> {
                let mut desc = desc(ud);
                desc.root = Some(root);
                desc.nroots = 1;
                desc.root_jac = Some(identity_mul);
                desc.root_sens = Some(zero_mul);
                desc.reset = Some(reset);
                desc.reset_jac = Some(zero_mul);
                desc.reset_sens = Some(zero_mul);
                desc
            }
        }
    };
}

logistic_callbacks!(f64_callbacks, f64);
logistic_callbacks!(f32_callbacks, f32);

unsafe extern "C" fn free_user_data(ud: *mut c_void) {
    let ud = unsafe { Box::from_raw(ud as *mut UserData) };
    ud.frees.fetch_add(1, Ordering::SeqCst);
}

/// A fresh `user_data`, owned by whoever it is passed to, and its free counter.
fn new_user_data(rhs_status: i32) -> (*mut UserData, Arc<AtomicUsize>) {
    let frees = Arc::new(AtomicUsize::new(0));
    let ud = Box::into_raw(Box::new(UserData {
        rhs_status,
        rhs_fail_call: 0,
        rhs_calls: AtomicUsize::new(0),
        frees: frees.clone(),
    }));
    (ud, frees)
}

fn configure(ode: &OdeWrapper) {
    ode.set_rtol(1e-8).unwrap();
    ode.set_atol(1e-8).unwrap();
    ode.set_sens_rtol(Some(1e-6)).unwrap();
    ode.set_sens_atol(Some(1e-6)).unwrap();
    ode.set_param_rtol(Some(1e-6)).unwrap();
    ode.set_param_atol(Some(1e-6)).unwrap();
    ode.set_out_rtol(Some(1e-6)).unwrap();
    ode.set_out_atol(Some(1e-6)).unwrap();
}

fn make_ode(matrix_type: MatrixType, ode_solver: OdeSolverType) -> OdeWrapper {
    let (ud, _) = new_user_data(0);
    let ode = new_callbacks(
        f64_callbacks::desc(ud),
        matrix_type,
        LinearSolverType::Default,
        ode_solver,
    )
    .unwrap();
    configure(&ode);
    ode
}

/// Construct a solver with one lane from the test callbacks.
fn new_callbacks<T: diffsol_c::CallbackScalar>(
    desc: OdeCallbacksDesc<T>,
    matrix_type: MatrixType,
    linear_solver: LinearSolverType,
    ode_solver: OdeSolverType,
) -> Result<OdeWrapper, diffsol_c::DiffsolRtError> {
    new_callbacks_batched(desc, matrix_type, linear_solver, ode_solver, 1)
}

/// Construct a solver with `nbatch` lanes from the test callbacks.
fn new_callbacks_batched<T: diffsol_c::CallbackScalar>(
    desc: OdeCallbacksDesc<T>,
    matrix_type: MatrixType,
    linear_solver: LinearSolverType,
    ode_solver: OdeSolverType,
    nbatch: usize,
) -> Result<OdeWrapper, diffsol_c::DiffsolRtError> {
    // SAFETY: the test callbacks are valid, use the documented lengths, and each test gives
    // its own `user_data` to one solver only
    unsafe { OdeWrapper::new_callbacks(desc, matrix_type, linear_solver, ode_solver, nbatch) }
}

fn err_string<T>(result: Result<T, diffsol_c::DiffsolRtError>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn runtime_dispatch_matches_requested_matrix_type() {
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_ode(matrix_type, OdeSolverType::Bdf);
        assert_eq!(ode.get_matrix_type().unwrap(), matrix_type);
        assert_eq!(ode.get_scalar_type().unwrap(), ScalarType::F64);
        assert_eq!(ode.get_code().unwrap(), "");
        assert_eq!(ode.get_jit_backend().unwrap(), None);
        assert_eq!(ode.get_nstates().unwrap(), 1);
        assert_eq!(ode.get_nparams().unwrap(), 1);
        assert_eq!(ode.get_nout().unwrap(), 1);
        assert!(!ode.has_stop().unwrap());

        let y0 = ode.y0(vector_tensor(&[2.0]).view()).unwrap();
        assert_eq!(lane_f64(&y0, 0), vec![LOGISTIC_X0]);

        let rhs = ode
            .rhs(
                vector_tensor(&[2.0]).view(),
                0.0,
                vector_tensor(&[0.25]).view(),
            )
            .unwrap();
        assert_close(lane_f64(&rhs, 0)[0], 0.375, ASSERT_TOL, "rhs(0.25)");

        let rhs_jac_mul = ode
            .rhs_jac_mul(
                vector_tensor(&[2.0]).view(),
                0.0,
                vector_tensor(&[0.25]).view(),
                vector_tensor(&[3.0]).view(),
            )
            .unwrap();
        assert_close(
            lane_f64(&rhs_jac_mul, 0)[0],
            3.0,
            ASSERT_TOL,
            "rhs_jac_mul(0.25, 3.0)",
        );
    }
}

#[test]
fn dense_solution_matches_logistic_solution_for_every_solver_and_matrix_type() {
    let t_eval = [0.25, 0.5, 1.0];
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let ode = make_ode(matrix_type, ode_solver);
            let solution = ode
                .solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
                .unwrap();
            assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
        }
    }
}

#[test]
fn adaptive_solution_matches_logistic_solution() {
    let r = 2.0;
    let final_time = 1.0;
    let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
    let solution = ode.solve(vector_tensor(&[r]).view(), final_time).unwrap();
    let ys = solution.get_ys().unwrap();
    let ys = lane_matrix(&ys, 0);
    let ts = lane_f64(&solution.get_ts().unwrap(), 0);
    assert_close(*ts.last().unwrap(), final_time, ASSERT_TOL, "final time");
    for (i, &t) in ts.iter().enumerate() {
        assert_close(
            ys[(0, i)],
            logistic_state(LOGISTIC_X0, r, t),
            5e-4,
            &format!("solve value[{i}]"),
        );
    }
}

#[test]
fn forward_sensitivities_match_analytic_derivative() {
    let t_eval = [0.25, 0.5, 1.0];
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let ode = make_ode(matrix_type, ode_solver);
            let solution = ode
                .solve_fwd_sens(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
                .unwrap();
            assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
            let sens = solution.get_sens().unwrap();
            assert_eq!(sens.len(), 1);
            let sens = lane_matrix(&sens[0], 0);
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    sens[(0, i)],
                    logistic_state_dr(LOGISTIC_X0, 2.0, t),
                    1e-4,
                    &format!("{matrix_type:?}/{ode_solver:?} sensitivity[{i}]"),
                );
            }
        }
    }
}

#[test]
fn continuous_adjoint_matches_analytic_gradient() {
    let r = 2.0;
    let final_time = 1.0;
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in [OdeSolverType::Bdf, OdeSolverType::Esdirk34] {
            let ode = make_ode(matrix_type, ode_solver);
            let (integral, gradient) = ode
                .solve_continuous_adjoint(vector_tensor(&[r]).view(), final_time)
                .unwrap();
            let integral = lane_f64(&integral, 0);
            let gradient = lane_matrix(&gradient, 0);
            assert_close(
                integral[0],
                logistic_integral(LOGISTIC_X0, r, final_time),
                5e-5,
                "integral",
            );
            assert_close(
                gradient[(0, 0)],
                logistic_integral_dr(LOGISTIC_X0, r, final_time),
                5e-4,
                &format!("{matrix_type:?}/{ode_solver:?} gradient"),
            );
        }
    }
}

#[test]
fn split_adjoint_matches_analytic_gradient() {
    // g = sum_i (y(t_i) - data_i)^2, so dg/dr = sum_i 2 (y(t_i) - data_i) dy/dr(t_i)
    let t_eval = [0.25, 0.5, 0.75, 1.0];
    let data = [0.2, 0.3, 0.4, 0.5];
    let r = 2.0;
    let expected: f64 = t_eval
        .iter()
        .zip(data)
        .map(|(&t, d)| {
            2.0 * (logistic_state(LOGISTIC_X0, r, t) - d) * logistic_state_dr(LOGISTIC_X0, r, t)
        })
        .sum();
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_ode(matrix_type, OdeSolverType::Bdf);
        let (solution, checkpoint) = ode
            .solve_adjoint_fwd(vector_tensor(&[r]).view(), vector_tensor(&t_eval).view())
            .unwrap();
        let ys = solution.get_ys().unwrap();
        let ys = lane_matrix(&ys, 0);
        let dgdu: Vec<f64> = (0..t_eval.len())
            .map(|i| 2.0 * (ys[(0, i)] - data[i]))
            .collect();
        let gradient = ode
            .solve_adjoint_bkwd(
                &solution,
                &checkpoint,
                matrix_tensor(1, t_eval.len(), &dgdu).view(),
            )
            .unwrap();
        let gradient = lane_matrix(&gradient, 0);
        assert_close(
            gradient[(0, 0)],
            expected,
            1e-4,
            &format!("{matrix_type:?} split adjoint gradient"),
        );
    }
}

#[test]
fn root_and_reset_callbacks_drive_hybrid_solve_and_sensitivities() {
    let r = 2.0;
    // spans two resets: the period is ln(81) / r ~ 2.2
    let t_eval = [0.5, 1.5, 2.5, 3.5, 4.75];
    for matrix_type in ALL_MATRIX_TYPES {
        let (ud, _) = new_user_data(0);
        let ode = new_callbacks(
            f64_callbacks::hybrid_desc(ud),
            matrix_type,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
        )
        .unwrap();
        configure(&ode);
        assert!(ode.has_stop().unwrap());

        let solution = ode
            .solve_dense(vector_tensor(&[r]).view(), vector_tensor(&t_eval).view())
            .unwrap();
        let ys = solution.get_ys().unwrap();
        let ys = lane_matrix(&ys, 0);
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                ys[(0, i)],
                hybrid_logistic_state(r, t),
                5e-4,
                &format!("{matrix_type:?} hybrid value[{i}]"),
            );
        }

        let solution = ode
            .solve_fwd_sens(vector_tensor(&[r]).view(), vector_tensor(&t_eval).view())
            .unwrap();
        let sens = solution.get_sens().unwrap();
        let sens = lane_matrix(&sens[0], 0);
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                sens[(0, i)],
                hybrid_logistic_state_dr(r, t),
                1e-3,
                &format!("{matrix_type:?} hybrid sensitivity[{i}]"),
            );
        }
    }
}

#[test]
fn single_precision_callbacks_solve_logistic_model() {
    let (ud, frees) = new_user_data(0);
    let ode = new_callbacks(
        f32_callbacks::desc(ud),
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    assert_eq!(ode.get_scalar_type().unwrap(), ScalarType::F32);
    ode.set_rtol(1e-5).unwrap();
    ode.set_atol(1e-5).unwrap();
    let t_eval = [0.25, 0.5, 1.0];
    let solution = ode
        .solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
        .unwrap();
    assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 1e-3);
    drop(solution);
    drop(ode);
    assert_eq!(frees.load(Ordering::SeqCst), 1);
}

#[test]
fn missing_callbacks_are_reported_before_solving() {
    let t_eval = [0.5, 1.0];

    // no rhs_jac: implicit methods are rejected, the explicit method is fine
    let (ud, _) = new_user_data(0);
    let mut desc = f64_callbacks::desc(ud);
    desc.rhs_jac = None;
    let ode = new_callbacks(
        desc,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    configure(&ode);
    let err =
        err_string(ode.solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view()));
    assert!(err.contains("requires the 'rhs_jac' callback"), "{err}");
    let err = err_string(ode.rhs_jac_mul(
        vector_tensor(&[2.0]).view(),
        0.0,
        vector_tensor(&[0.25]).view(),
        vector_tensor(&[1.0]).view(),
    ));
    assert!(
        err.contains("OdeCallbacks: evaluating the equations requires the 'rhs_jac' callback"),
        "{err}"
    );
    ode.set_ode_solver(OdeSolverType::Tsit45).unwrap();
    let solution = ode
        .solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
        .unwrap();
    assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);

    // no rhs_sens: plain solves work, forward sensitivities are rejected
    let (ud, _) = new_user_data(0);
    let mut desc = f64_callbacks::desc(ud);
    desc.rhs_sens = None;
    let ode = new_callbacks(
        desc,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    configure(&ode);
    ode.solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
        .unwrap();
    let err =
        err_string(ode.solve_fwd_sens(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view()));
    assert!(err.contains("requires the 'rhs_sens' callback"), "{err}");

    // no out_adjoint: adjoint solves are rejected
    let (ud, _) = new_user_data(0);
    let mut desc = f64_callbacks::desc(ud);
    desc.out_adjoint = None;
    let ode = new_callbacks(
        desc,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    configure(&ode);
    let err = err_string(ode.solve_continuous_adjoint(vector_tensor(&[2.0]).view(), 1.0));
    assert!(err.contains("requires the 'out_adjoint' callback"), "{err}");
}

/// With no output function, the output is the state, so the continuous adjoint gives the
/// integral of `y` and its gradient, the same as with `out = y`.
#[test]
fn continuous_adjoint_without_out_integrates_the_state() {
    let (r, final_time) = (2.0, 1.0);
    for ode_solver in [OdeSolverType::Bdf, OdeSolverType::Tsit45] {
        let (ud, _) = new_user_data(0);
        let mut desc = f64_callbacks::desc(ud);
        desc.nout = 0;
        desc.out = None;
        desc.out_jac = None;
        desc.out_sens = None;
        desc.out_adjoint = None;
        desc.out_sens_adjoint = None;
        let ode = new_callbacks(
            desc,
            MatrixType::NalgebraDense,
            LinearSolverType::Default,
            ode_solver,
        )
        .unwrap();
        configure(&ode);
        let (integral, gradient) = ode
            .solve_continuous_adjoint(vector_tensor(&[r]).view(), final_time)
            .unwrap();
        let integral = lane_f64(&integral, 0);
        let gradient = lane_matrix(&gradient, 0);
        let label = format!("{ode_solver:?}");
        let expect = logistic_integral(LOGISTIC_X0, r, final_time);
        assert_close(integral[0], expect, 5e-5, &format!("{label} integral"));
        let expect = logistic_integral_dr(LOGISTIC_X0, r, final_time);
        assert_close(gradient[(0, 0)], expect, 5e-4, &format!("{label} gradient"));
    }
}

/// With root and reset events, a continuous adjoint without an output function gives the same
/// integral and gradient as the same model with `out = y`. At each reset this needs the jump of
/// the integrated output, `y- - y+`.
#[test]
fn continuous_adjoint_without_out_matches_identity_out_with_resets() {
    let (r, final_time) = (2.0, 5.0);
    let solve = |with_out: bool| {
        let (ud, _) = new_user_data(0);
        let mut desc = f64_callbacks::hybrid_desc(ud);
        desc.root_adjoint = Some(f64_callbacks::neg_identity_mul);
        desc.root_sens_adjoint = Some(f64_callbacks::zero_mul);
        desc.reset_adjoint = Some(f64_callbacks::zero_mul);
        desc.reset_sens_adjoint = Some(f64_callbacks::zero_mul);
        if !with_out {
            desc.nout = 0;
            desc.out = None;
            desc.out_jac = None;
            desc.out_sens = None;
            desc.out_adjoint = None;
            desc.out_sens_adjoint = None;
        }
        let ode = new_callbacks(
            desc,
            MatrixType::NalgebraDense,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
        )
        .unwrap();
        configure(&ode);
        let (integral, gradient) = ode
            .solve_continuous_adjoint(vector_tensor(&[r]).view(), final_time)
            .unwrap();
        let integral = lane_f64(&integral, 0)[0];
        let gradient = lane_matrix(&gradient, 0)[(0, 0)];
        (integral, gradient)
    };
    // with `out`, the output is in the error control, so the steps (and the small errors)
    // are not the same. A missing reset jump changes the gradient by much more.
    let (integral, gradient) = solve(false);
    let (expect_integral, expect_gradient) = solve(true);
    assert_close(integral, expect_integral, 1e-3, "integral");
    assert_close(gradient, expect_gradient, 1e-3, "gradient");
}

#[test]
fn callback_error_codes_fail_the_solve() {
    let (ud, _) = new_user_data(7);
    let ode = new_callbacks(
        f64_callbacks::desc(ud),
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    // the solver's error, with the callback failure as context
    let err = err_string(ode.solve_dense(
        vector_tensor(&[2.0]).view(),
        vector_tensor(&[0.5, 1.0]).view(),
    ));
    assert!(
        err.contains("(a callback failed during the solve: callback 'rhs' returned error code 7)"),
        "{err}"
    );
    let err = err_string(ode.rhs(
        vector_tensor(&[2.0]).view(),
        0.0,
        vector_tensor(&[0.25]).view(),
    ));
    assert!(
        err.contains("callback 'rhs' returned error code 7"),
        "{err}"
    );
}

/// A callback failure that the solver recovers from (here by rejecting the step) does not fail
/// the solve.
#[test]
fn solver_recovers_from_a_callback_failure() {
    let r = 2.0;
    let t_eval = [0.5, 1.0];
    for fail_call in [10, 20, 30] {
        let (ud, _) = new_user_data(0);
        // SAFETY: `ud` is valid, and nothing else uses it yet
        unsafe { (*ud).rhs_fail_call = fail_call };
        let ode = new_callbacks(
            f64_callbacks::desc(ud),
            MatrixType::NalgebraDense,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
        )
        .unwrap();
        configure(&ode);
        let solution = ode
            .solve_dense(vector_tensor(&[r]).view(), vector_tensor(&t_eval).view())
            .unwrap_or_else(|err| panic!("failure at rhs call {fail_call}: {err}"));
        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, r, 1e-5);
    }
}

#[test]
fn rhs_rejects_arguments_of_the_wrong_length() {
    let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
    let err = err_string(ode.rhs(
        vector_tensor(&[2.0]).view(),
        0.0,
        vector_tensor(&[0.25, 0.5]).view(),
    ));
    assert!(
        err.contains("y: Expected 1 values in each lane, got 2"),
        "{err}"
    );
    let err = err_string(ode.rhs_jac_mul(
        vector_tensor(&[2.0]).view(),
        0.0,
        vector_tensor(&[0.25]).view(),
        vector_tensor(&[]).view(),
    ));
    assert!(
        err.contains("v: Expected 1 values in each lane, got 0"),
        "{err}"
    );
}

#[test]
fn invalid_callbacks_are_rejected_and_user_data_is_freed() {
    let (ud, frees) = new_user_data(0);
    let mut desc = f64_callbacks::desc(ud);
    desc.nout = 0;
    let err = err_string(new_callbacks(
        desc,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    ));
    assert!(err.contains("nout"), "{err}");
    assert_eq!(frees.load(Ordering::SeqCst), 1);
}

#[test]
fn c_api_constructs_solves_and_frees_user_data_once() {
    // the C enum values of MatrixType::FaerSparse, LinearSolverType::Default and
    // OdeSolverType::Bdf
    let (matrix_type, linear_solver, ode_solver) = (2, 0, 0);

    let (ud, frees) = new_user_data(0);
    let desc = f64_callbacks::desc(ud);
    let ode =
        unsafe { diffsol_ode_new_callbacks_f64(&desc, matrix_type, linear_solver, ode_solver, 1) };
    assert!(!ode.is_null());
    {
        let ode = unsafe { &*ode };
        configure(ode);
        let t_eval = [0.25, 0.5, 1.0];
        let solution = ode
            .solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
            .unwrap();
        assert_solution_tail(&solution, &t_eval, LOGISTIC_X0, 2.0, 5e-4);
    }
    assert_eq!(frees.load(Ordering::SeqCst), 0);
    unsafe { diffsol_ode_free(ode) };
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    // an invalid argument is rejected before the equations are built, and still frees
    let (ud, frees) = new_user_data(0);
    let desc = f64_callbacks::desc(ud);
    let ode = unsafe { diffsol_ode_new_callbacks_f64(&desc, -1, linear_solver, ode_solver, 1) };
    assert!(ode.is_null());
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    // f32
    let (ud, frees) = new_user_data(0);
    let desc: DiffsolOdeCallbacksF32 = f32_callbacks::desc(ud);
    let ode =
        unsafe { diffsol_ode_new_callbacks_f32(&desc, matrix_type, linear_solver, ode_solver, 1) };
    assert!(!ode.is_null());
    assert_eq!(unsafe { &*ode }.get_scalar_type().unwrap(), ScalarType::F32);
    unsafe { diffsol_ode_free(ode) };
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    // a null descriptor
    let ode = unsafe {
        diffsol_ode_new_callbacks_f64(std::ptr::null(), matrix_type, linear_solver, ode_solver, 1)
    };
    assert!(ode.is_null());
}

// Batched solves: one ODE with `nbatch` lanes, each lane with its own growth rate `r`.

const BATCH_RATES: [f64; 3] = [1.0, 2.0, 3.0];

fn make_batched_ode(
    matrix_type: MatrixType,
    ode_solver: OdeSolverType,
    nbatch: usize,
) -> OdeWrapper {
    let (ud, _) = new_user_data(0);
    let ode = new_callbacks_batched(
        f64_callbacks::desc(ud),
        matrix_type,
        LinearSolverType::Default,
        ode_solver,
        nbatch,
    )
    .unwrap();
    configure(&ode);
    ode
}

/// The batch rates as a `[1, nbatch]` params array.
fn batch_params() -> Tensor {
    let lanes: Vec<&[f64]> = BATCH_RATES.iter().map(std::slice::from_ref).collect();
    batched_vector_tensor(&lanes)
}

#[test]
fn batched_dense_solution_matches_each_lane() {
    let t_eval = [0.25, 0.5, 1.0];
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_batched_ode(matrix_type, ode_solver, nbatch);
            assert_eq!(ode.get_nbatch().unwrap(), nbatch);
            let solution = ode
                .solve_dense(batch_params().view(), vector_tensor(&t_eval).view())
                .unwrap();
            let ts = lane_f64(&solution.get_ts().unwrap(), 0);
            assert_eq!(ts, t_eval.to_vec());
            let ys = solution.get_ys().unwrap();
            assert_eq!(ys.shape(), &[1, t_eval.len(), nbatch], "{label}");
            for (b, &r) in BATCH_RATES.iter().enumerate() {
                for (i, &t) in t_eval.iter().enumerate() {
                    assert_close(
                        value_f64(&ys, &[0, i, b]),
                        logistic_state(LOGISTIC_X0, r, t),
                        5e-4,
                        &format!("{label} lane {b} y[{i}]"),
                    );
                }
            }
        }
    }
}

#[test]
fn batched_adaptive_solution_shares_the_time_points() {
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_batched_ode(matrix_type, OdeSolverType::Bdf, nbatch);
        let solution = ode.solve(batch_params().view(), 1.0).unwrap();
        let ts = lane_f64(&solution.get_ts().unwrap(), 0);
        let ys = solution.get_ys().unwrap();
        assert_eq!(ys.shape(), &[1, ts.len(), nbatch]);
        assert_close(*ts.last().unwrap(), 1.0, 1e-12, "final time");
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            for (i, &t) in ts.iter().enumerate() {
                assert_close(
                    value_f64(&ys, &[0, i, b]),
                    logistic_state(LOGISTIC_X0, r, t),
                    5e-4,
                    &format!("{matrix_type:?} lane {b} y(t = {t})"),
                );
            }
        }
    }
}

#[test]
fn solution_tensors_share_the_solution_and_outlive_it() {
    let t_eval = [0.25, 0.5, 1.0];
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_batched_ode(matrix_type, OdeSolverType::Bdf, nbatch);
        // a solve to a final time, which grows `ys` with spare columns
        let adaptive = ode.solve(batch_params().view(), 1.0).unwrap();
        let sens_solution = ode
            .solve_fwd_sens(batch_params().view(), vector_tensor(&t_eval).view())
            .unwrap();
        let (ts, ys) = (adaptive.get_ts().unwrap(), adaptive.get_ys().unwrap());
        let sens = sens_solution.get_sens().unwrap();

        // the tensors show the data of the solution, with no copy
        assert_eq!(adaptive.get_ts().unwrap().data_ptr(), ts.data_ptr());
        assert_eq!(adaptive.get_ys().unwrap().data_ptr(), ys.data_ptr());
        assert_eq!(
            sens_solution.get_sens().unwrap()[0].data_ptr(),
            sens[0].data_ptr()
        );

        // the tensors keep the solution alive after the solution and the ODE are dropped
        drop((adaptive, sens_solution, ode));
        let ts = lane_f64(&ts, 0);
        assert_eq!(ys.shape(), &[1, ts.len(), nbatch]);
        assert_eq!(sens[0].shape(), &[1, t_eval.len(), nbatch]);
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            for (i, &t) in ts.iter().enumerate() {
                assert_close(
                    value_f64(&ys, &[0, i, b]),
                    logistic_state(LOGISTIC_X0, r, t),
                    5e-4,
                    &format!("{matrix_type:?} lane {b} y(t = {t})"),
                );
            }
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    value_f64(&sens[0], &[0, i, b]),
                    logistic_state_dr(LOGISTIC_X0, r, t),
                    5e-3,
                    &format!("{matrix_type:?} lane {b} dy/dr(t = {t})"),
                );
            }
        }
    }
}

#[test]
fn batched_forward_sensitivities_match_each_lane() {
    let t_eval = [0.25, 0.5, 1.0];
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_batched_ode(matrix_type, ode_solver, nbatch);
            let solution = ode
                .solve_fwd_sens(batch_params().view(), vector_tensor(&t_eval).view())
                .unwrap();
            let sens = solution.get_sens().unwrap();
            assert_eq!(sens.len(), 1);
            let sens = &sens[0];
            assert_eq!(sens.shape(), &[1, t_eval.len(), nbatch], "{label}");
            for (b, &r) in BATCH_RATES.iter().enumerate() {
                for (i, &t) in t_eval.iter().enumerate() {
                    assert_close(
                        value_f64(sens, &[0, i, b]),
                        logistic_state_dr(LOGISTIC_X0, r, t),
                        1e-4,
                        &format!("{label} lane {b} sensitivity[{i}]"),
                    );
                }
            }
        }
    }
}

/// The logistic model with the rate `r = p[0] + p[1]`. In a sensitivity solve with more than
/// one batch lane, the callbacks get the state and the parameters with one lane for each batch
/// lane, and `v` and `y` with one lane for each batch lane and each parameter (`P = 2`).
mod two_param_callbacks {
    use super::*;
    type T = f64;

    unsafe extern "C" fn rhs(
        x: View<'_>,
        p: View<'_>,
        _t: T,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([x, p], y, |[x, p]: [&[T]; 2], y: &mut [T], _| {
                y[0] = (p[0] + p[1]) * x[0] * (1.0 - x[0])
            })
        };
        0
    }

    unsafe extern "C" fn rhs_jac(
        x: View<'_>,
        p: View<'_>,
        _t: T,
        v: View<'_>,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([x, p, v], y, |[x, p, v]: [&[T]; 3], y: &mut [T], _| {
                y[0] = (p[0] + p[1]) * (1.0 - 2.0 * x[0]) * v[0]
            })
        };
        0
    }

    unsafe extern "C" fn rhs_sens(
        x: View<'_>,
        _p: View<'_>,
        _t: T,
        v: View<'_>,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([x, v], y, |[x, v]: [&[T]; 2], y: &mut [T], _| {
                y[0] = x[0] * (1.0 - x[0]) * (v[0] + v[1])
            })
        };
        0
    }

    unsafe extern "C" fn rhs_adjoint(
        x: View<'_>,
        p: View<'_>,
        _t: T,
        v: View<'_>,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([x, p, v], y, |[x, p, v]: [&[T]; 3], y: &mut [T], _| {
                y[0] = -(p[0] + p[1]) * (1.0 - 2.0 * x[0]) * v[0]
            })
        };
        0
    }

    unsafe extern "C" fn rhs_sens_adjoint(
        x: View<'_>,
        _p: View<'_>,
        _t: T,
        v: View<'_>,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([x, v], y, |[x, v]: [&[T]; 2], y: &mut [T], _| {
                y.fill(-x[0] * (1.0 - x[0]) * v[0])
            })
        };
        0
    }

    pub fn desc() -> OdeCallbacksDesc<T> {
        use f64_callbacks as base;
        let mut desc = OdeCallbacksDesc::new(1, 2, rhs, base::init);
        desc.rhs_jac = Some(rhs_jac);
        desc.rhs_sens = Some(rhs_sens);
        desc.rhs_adjoint = Some(rhs_adjoint);
        desc.rhs_sens_adjoint = Some(rhs_sens_adjoint);
        desc.init_sens = Some(base::init_zero_mul);
        desc.init_sens_adjoint = Some(base::init_zero_mul);
        desc.out = Some(base::identity);
        desc.nout = 1;
        desc.out_jac = Some(base::identity_mul);
        desc.out_sens = Some(base::zero_mul);
        desc.out_adjoint = Some(base::neg_identity_mul);
        desc.out_sens_adjoint = Some(base::zero_mul);
        desc
    }
}

#[test]
fn batched_sensitivities_with_two_parameters_match_each_lane() {
    let t_eval = [0.25, 0.5, 1.0];
    let final_time = 1.0;
    let nbatch = BATCH_RATES.len();
    // lane `b` has the parameters `[r_b - 0.5, 0.5]`, so its rate is `r_b`
    let lanes: Vec<[f64; 2]> = BATCH_RATES.iter().map(|&r| [r - 0.5, 0.5]).collect();
    let lanes: Vec<&[f64]> = lanes.iter().map(|lane| &lane[..]).collect();
    let params = batched_vector_tensor(&lanes);
    let make_ode = |matrix_type, ode_solver| {
        let ode = new_callbacks_batched(
            two_param_callbacks::desc(),
            matrix_type,
            LinearSolverType::Default,
            ode_solver,
            nbatch,
        )
        .unwrap();
        configure(&ode);
        ode
    };
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_ode(matrix_type, ode_solver);
            let solution = ode
                .solve_fwd_sens(params.view(), vector_tensor(&t_eval).view())
                .unwrap();
            let sens = solution.get_sens().unwrap();
            assert_eq!(sens.len(), 2, "{label}");
            // dy/dp_j = dy/dr for both parameters
            for (j, sens) in sens.iter().enumerate() {
                for (b, &r) in BATCH_RATES.iter().enumerate() {
                    for (i, &t) in t_eval.iter().enumerate() {
                        assert_close(
                            value_f64(sens, &[0, i, b]),
                            logistic_state_dr(LOGISTIC_X0, r, t),
                            1e-4,
                            &format!("{label} lane {b} dy/dp{j}[{i}]"),
                        );
                    }
                }
            }
        }

        let ode = make_ode(matrix_type, OdeSolverType::Bdf);
        let (integral, gradient) = ode
            .solve_continuous_adjoint(params.view(), final_time)
            .unwrap();
        assert_eq!(gradient.shape(), &[2, 1, nbatch], "{matrix_type:?}");
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            assert_close(
                lane_f64(&integral, b)[0],
                logistic_integral(LOGISTIC_X0, r, final_time),
                5e-5,
                &format!("{matrix_type:?} lane {b} integral"),
            );
            for j in 0..2 {
                assert_close(
                    value_f64(&gradient, &[j, 0, b]),
                    logistic_integral_dr(LOGISTIC_X0, r, final_time),
                    5e-4,
                    &format!("{matrix_type:?} lane {b} gradient {j}"),
                );
            }
        }
    }
}

#[test]
fn batched_continuous_adjoint_matches_each_lane() {
    let final_time = 1.0;
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in [OdeSolverType::Bdf, OdeSolverType::Esdirk34] {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_batched_ode(matrix_type, ode_solver, nbatch);
            let (integral, gradient) = ode
                .solve_continuous_adjoint(batch_params().view(), final_time)
                .unwrap();
            assert_eq!(integral.shape(), &[1, nbatch], "{label}");
            assert_eq!(gradient.shape(), &[1, 1, nbatch], "{label}");
            for (b, &r) in BATCH_RATES.iter().enumerate() {
                assert_close(
                    lane_f64(&integral, b)[0],
                    logistic_integral(LOGISTIC_X0, r, final_time),
                    5e-5,
                    &format!("{label} lane {b} integral"),
                );
                assert_close(
                    value_f64(&gradient, &[0, 0, b]),
                    logistic_integral_dr(LOGISTIC_X0, r, final_time),
                    5e-4,
                    &format!("{label} lane {b} gradient"),
                );
            }
        }
    }
}

#[test]
fn batched_split_adjoint_matches_each_lane() {
    // g_b = sum_i (y_b(t_i) - data_i)^2 for each lane b
    let t_eval = [0.25, 0.5, 0.75, 1.0];
    let data = [0.2, 0.3, 0.4, 0.5];
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_batched_ode(matrix_type, OdeSolverType::Bdf, nbatch);
        let (solution, checkpoint) = ode
            .solve_adjoint_fwd(batch_params().view(), vector_tensor(&t_eval).view())
            .unwrap();
        let ys = solution.get_ys().unwrap();
        // dgdu as `[1, nt, nbatch]`, in column-major order
        let dgdu: Vec<f64> = (0..nbatch)
            .flat_map(|b| (0..t_eval.len()).map(move |i| (b, i)))
            .map(|(b, i)| 2.0 * (value_f64(&ys, &[0, i, b]) - data[i]))
            .collect();
        let dgdu = Tensor::from_vec(dgdu, &[1, t_eval.len(), nbatch]);
        let gradient = ode
            .solve_adjoint_bkwd(&solution, &checkpoint, dgdu.view())
            .unwrap();
        assert_eq!(gradient.shape(), &[1, 1, nbatch]);
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            let expected: f64 = t_eval
                .iter()
                .zip(data)
                .map(|(&t, d)| {
                    2.0 * (logistic_state(LOGISTIC_X0, r, t) - d)
                        * logistic_state_dr(LOGISTIC_X0, r, t)
                })
                .sum();
            assert_close(
                value_f64(&gradient, &[0, 0, b]),
                expected,
                1e-4,
                &format!("{matrix_type:?} lane {b} split adjoint gradient"),
            );
        }
    }
}

#[test]
fn batched_split_adjoint_copies_a_one_lane_dgdu_to_each_lane() {
    // the same dgdu for each lane: g_b = sum_i y_b(t_i)
    let t_eval = [0.25, 0.5, 0.75, 1.0];
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_batched_ode(matrix_type, OdeSolverType::Bdf, nbatch);
        let (solution, checkpoint) = ode
            .solve_adjoint_fwd(batch_params().view(), vector_tensor(&t_eval).view())
            .unwrap();
        let dgdu = matrix_tensor(1, t_eval.len(), &[1.0; 4]);
        let gradient = ode
            .solve_adjoint_bkwd(&solution, &checkpoint, dgdu.view())
            .unwrap();
        assert_eq!(gradient.shape(), &[1, 1, nbatch]);
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            let expected: f64 = t_eval
                .iter()
                .map(|&t| logistic_state_dr(LOGISTIC_X0, r, t))
                .sum();
            assert_close(
                value_f64(&gradient, &[0, 0, b]),
                expected,
                1e-4,
                &format!("{matrix_type:?} lane {b} split adjoint gradient"),
            );
        }
    }
}

#[test]
fn batched_evaluations_use_each_lane() {
    let ys = [0.25, 0.5, 0.75];
    let vs = [1.0, 2.0, 3.0];
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        let ode = make_batched_ode(matrix_type, OdeSolverType::Bdf, nbatch);

        let y0 = ode.y0(batch_params().view()).unwrap();
        assert_eq!(y0.shape(), &[1, nbatch]);
        for b in 0..nbatch {
            assert_eq!(lane_f64(&y0, b), vec![LOGISTIC_X0]);
        }

        let y_lanes: Vec<&[f64]> = ys.iter().map(std::slice::from_ref).collect();
        let v_lanes: Vec<&[f64]> = vs.iter().map(std::slice::from_ref).collect();
        let rhs = ode
            .rhs(
                batch_params().view(),
                0.0,
                batched_vector_tensor(&y_lanes).view(),
            )
            .unwrap();
        let jac_mul = ode
            .rhs_jac_mul(
                batch_params().view(),
                0.0,
                batched_vector_tensor(&y_lanes).view(),
                batched_vector_tensor(&v_lanes).view(),
            )
            .unwrap();
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            let y = ys[b];
            assert_close(
                lane_f64(&rhs, b)[0],
                r * y * (1.0 - y),
                1e-12,
                &format!("{matrix_type:?} lane {b} rhs"),
            );
            assert_close(
                lane_f64(&jac_mul, b)[0],
                r * (1.0 - 2.0 * y) * vs[b],
                1e-12,
                &format!("{matrix_type:?} lane {b} rhs_jac_mul"),
            );
        }

        // one lane of `y` broadcasts to all lanes
        let rhs = ode
            .rhs(batch_params().view(), 0.0, vector_tensor(&[0.5]).view())
            .unwrap();
        for (b, &r) in BATCH_RATES.iter().enumerate() {
            assert_close(
                lane_f64(&rhs, b)[0],
                r * 0.25,
                1e-12,
                &format!("{matrix_type:?} lane {b} rhs of a broadcast y"),
            );
        }
    }
}

#[test]
fn batched_tolerance_getters_read_the_value_of_each_lane() {
    for matrix_type in ALL_MATRIX_TYPES {
        // `configure` sets the tolerances after the construction, for all the lanes
        let ode = make_batched_ode(matrix_type, OdeSolverType::Bdf, 3);
        assert_eq!(ode.get_atol().unwrap(), 1e-8);
        assert_eq!(ode.get_sens_atol().unwrap(), Some(1e-6));
        assert_eq!(ode.get_out_atol().unwrap(), Some(1e-6));
        assert_eq!(ode.get_param_atol().unwrap(), Some(1e-6));
    }
}

#[test]
fn one_lane_of_params_broadcasts_to_all_lanes() {
    let t_eval = [0.5, 1.0];
    let nbatch = 3;
    let ode = make_batched_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf, nbatch);
    let solution = ode
        .solve_dense(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
        .unwrap();
    let ys = solution.get_ys().unwrap();
    assert_eq!(ys.shape(), &[1, t_eval.len(), nbatch]);
    for b in 0..nbatch {
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                value_f64(&ys, &[0, i, b]),
                logistic_state(LOGISTIC_X0, 2.0, t),
                5e-4,
                &format!("lane {b} y[{i}]"),
            );
        }
    }
}

#[test]
fn batched_inputs_with_the_wrong_number_of_lanes_are_rejected() {
    let ode = make_batched_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf, 3);
    let err = err_string(ode.solve(batched_vector_tensor(&[&[1.0], &[2.0]]).view(), 1.0));
    assert!(
        err.contains("params: Expected 1 or 3 lanes, got 2"),
        "{err}"
    );
    let err = err_string(ode.rhs(
        batch_params().view(),
        0.0,
        batched_vector_tensor(&[&[0.1], &[0.2]]).view(),
    ));
    assert!(err.contains("y: Expected 1 or 3 lanes, got 2"), "{err}");
}

#[test]
fn t_eval_is_a_1d_array_or_one_lane() {
    let ode = make_batched_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf, 3);
    let times = [0.5, 1.0];
    // all the lanes use the same times, so `[nt]` and `[nt, 1]` give the same solution
    let one_lane = ode
        .solve_dense(batch_params().view(), vector_tensor(&times).view())
        .unwrap();
    let flat = TensorView::from_slice(&times, &[times.len()]);
    let flat = ode.solve_dense(batch_params().view(), flat).unwrap();
    assert_eq!(
        lane_f64(&one_lane.get_ts().unwrap(), 0),
        lane_f64(&flat.get_ts().unwrap(), 0)
    );
    // more than one lane of times is an error
    let two_lanes = batched_vector_tensor(&[&times, &times]);
    let err = err_string(ode.solve_dense(batch_params().view(), two_lanes.view()));
    assert!(
        err.contains("t_eval: Expected an [nt] or an [nt, 1] array, got the shape [2, 2]"),
        "{err}"
    );
}

#[test]
fn invalid_nbatch_or_batched_root_callbacks_are_rejected_and_user_data_is_freed() {
    let (ud, frees) = new_user_data(0);
    let err = err_string(new_callbacks_batched(
        f64_callbacks::desc(ud),
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
        0,
    ));
    assert!(err.contains("nbatch must be greater than 0"), "{err}");
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    let (ud, frees) = new_user_data(0);
    let err = err_string(new_callbacks_batched(
        f64_callbacks::desc(ud),
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
        usize::MAX,
    ));
    assert!(err.contains("is too large for this ODE"), "{err}");
    assert_eq!(frees.load(Ordering::SeqCst), 1);

    for matrix_type in ALL_MATRIX_TYPES {
        let (ud, frees) = new_user_data(0);
        let err = err_string(new_callbacks_batched(
            f64_callbacks::hybrid_desc(ud),
            matrix_type,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
            2,
        ));
        assert!(
            err.contains("not available with more than one lane"),
            "{err}"
        );
        assert_eq!(frees.load(Ordering::SeqCst), 1);
    }
}

// VJP: reverse mode back through the steps of the solver.

/// The loss `g = sum_i (y(t_i) - data_i)^2` of the VJP tests, and its analytic gradient
/// with respect to `r`.
const VJP_T_EVAL: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const VJP_DATA: [f64; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];

fn vjp_expected_gradient(r: f64) -> f64 {
    VJP_T_EVAL
        .iter()
        .zip(VJP_DATA)
        .map(|(&t, d)| {
            2.0 * (logistic_state(LOGISTIC_X0, r, t) - d) * logistic_state_dr(LOGISTIC_X0, r, t)
        })
        .sum()
}

/// `dg/du` of the VJP loss, as a `[1, nt, nbatch]` array, from the `[1, nt, nbatch]`
/// solution values `ys`.
fn vjp_dgdu(ys: &Tensor) -> Tensor {
    let (nt, nbatch) = (ys.shape()[1], ys.shape()[2]);
    let values: Vec<f64> = (0..nbatch)
        .flat_map(|b| (0..nt).map(move |i| (b, i)))
        .map(|(b, i)| 2.0 * (value_f64(ys, &[0, i, b]) - VJP_DATA[i]))
        .collect();
    Tensor::from_vec(values, &[1, nt, nbatch])
}

#[test]
fn vjp_matches_analytic_gradient() {
    let r = 2.0;
    let t_eval = VJP_T_EVAL;
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_ode(matrix_type, ode_solver);
            let (solution, checkpoint) = ode
                .solve_vjp_fwd(vector_tensor(&[r]).view(), vector_tensor(&t_eval).view())
                .unwrap();
            assert_eq!(lane_f64(&solution.get_ts().unwrap(), 0), t_eval.to_vec());
            let ys = solution.get_ys().unwrap();
            assert_eq!(ys.shape(), &[1, t_eval.len(), 1], "{label}");

            // the same solution as solve_dense
            let dense = ode
                .solve_dense(vector_tensor(&[r]).view(), vector_tensor(&t_eval).view())
                .unwrap()
                .get_ys()
                .unwrap();
            for i in 0..t_eval.len() {
                assert_close(
                    value_f64(&ys, &[0, i, 0]),
                    value_f64(&dense, &[0, i, 0]),
                    1e-12,
                    &format!("{label} y[{i}]"),
                );
            }

            let dgdu = vjp_dgdu(&ys);
            let gradient = ode.solve_vjp_bkwd(&checkpoint, dgdu.view()).unwrap();
            assert_eq!(gradient.shape(), &[1, 1, 1], "{label}");
            assert_close(
                value_f64(&gradient, &[0, 0, 0]),
                vjp_expected_gradient(r),
                1e-4,
                &format!("{label} VJP gradient"),
            );

            // the checkpoint does not change, so a second backward pass gives the same gradient
            let again = ode.solve_vjp_bkwd(&checkpoint, dgdu.view()).unwrap();
            assert_close(
                value_f64(&again, &[0, 0, 0]),
                value_f64(&gradient, &[0, 0, 0]),
                1e-8,
                &format!("{label} second VJP gradient"),
            );
        }
    }
}

/// The backward pass uses the method of the forward pass, also if the solver changes between
/// the two passes.
#[test]
fn vjp_backward_uses_the_forward_method() {
    let r = 2.0;
    let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Tsit45);
    let (solution, checkpoint) = ode
        .solve_vjp_fwd(
            vector_tensor(&[r]).view(),
            vector_tensor(&VJP_T_EVAL).view(),
        )
        .unwrap();
    let dgdu = vjp_dgdu(&solution.get_ys().unwrap());
    ode.set_ode_solver(OdeSolverType::Bdf).unwrap();
    let gradient = ode.solve_vjp_bkwd(&checkpoint, dgdu.view()).unwrap();
    assert_close(
        value_f64(&gradient, &[0, 0, 0]),
        vjp_expected_gradient(r),
        1e-4,
        "VJP gradient after a change of solver",
    );
}

#[test]
fn batched_vjp_matches_each_lane() {
    let nbatch = BATCH_RATES.len();
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_batched_ode(matrix_type, ode_solver, nbatch);
            let (solution, checkpoint) = ode
                .solve_vjp_fwd(batch_params().view(), vector_tensor(&VJP_T_EVAL).view())
                .unwrap();
            let ys = solution.get_ys().unwrap();
            assert_eq!(ys.shape(), &[1, VJP_T_EVAL.len(), nbatch], "{label}");
            let gradient = ode
                .solve_vjp_bkwd(&checkpoint, vjp_dgdu(&ys).view())
                .unwrap();
            assert_eq!(gradient.shape(), &[1, 1, nbatch], "{label}");
            for (b, &r) in BATCH_RATES.iter().enumerate() {
                assert_close(
                    value_f64(&gradient, &[0, 0, b]),
                    vjp_expected_gradient(r),
                    1e-4,
                    &format!("{label} lane {b} VJP gradient"),
                );
            }
        }
    }
}

/// Two directions in one backward pass: direction 0 is the VJP loss, direction 1 is
/// `g = sum_i y(t_i)`, with the gradient `sum_i dy/dr(t_i)`. The `[1, nt, 2 * nbatch]` input has
/// lane `b * 2 + k` for direction `k` of batch lane `b`.
#[test]
fn multi_direction_vjp_matches_each_direction() {
    let nbatch = BATCH_RATES.len();
    let nt = VJP_T_EVAL.len();
    let sum_gradient = |r: f64| -> f64 {
        VJP_T_EVAL
            .iter()
            .map(|&t| logistic_state_dr(LOGISTIC_X0, r, t))
            .sum()
    };
    for matrix_type in ALL_MATRIX_TYPES {
        for ode_solver in all_ode_solvers() {
            let label = format!("{matrix_type:?}/{ode_solver:?}");
            let ode = make_batched_ode(matrix_type, ode_solver, nbatch);
            let (solution, checkpoint) = ode
                .solve_vjp_fwd(batch_params().view(), vector_tensor(&VJP_T_EVAL).view())
                .unwrap();
            let ys = solution.get_ys().unwrap();
            let loss = vjp_dgdu(&ys);
            let ones = Tensor::from_vec(vec![1.0; nt * nbatch], &[1, nt, nbatch]);
            let values: Vec<f64> = (0..nbatch)
                .flat_map(|b| {
                    let loss = &loss;
                    (0..nt)
                        .map(move |i| value_f64(loss, &[0, i, b]))
                        .chain(std::iter::repeat_n(1.0, nt))
                })
                .collect();
            let dgdu = Tensor::from_vec(values, &[1, nt, 2 * nbatch]);

            let gradient = ode.solve_vjp_bkwd(&checkpoint, dgdu.view()).unwrap();
            assert_eq!(gradient.shape(), &[1, 2, nbatch], "{label}");
            let loss_gradient = ode.solve_vjp_bkwd(&checkpoint, loss.view()).unwrap();
            let sum_gradients = ode.solve_vjp_bkwd(&checkpoint, ones.view()).unwrap();
            for (b, &r) in BATCH_RATES.iter().enumerate() {
                for (k, single, expected) in [
                    (0, &loss_gradient, vjp_expected_gradient(r)),
                    (1, &sum_gradients, sum_gradient(r)),
                ] {
                    let got = value_f64(&gradient, &[0, k, b]);
                    let label = format!("{label} lane {b} direction {k}");
                    assert_close(got, expected, 1e-4, &format!("{label} analytic"));
                    assert_close(
                        got,
                        value_f64(single, &[0, 0, b]),
                        1e-6,
                        &format!("{label} single direction"),
                    );
                }
            }

            // the lanes must be 1 or a multiple of nbatch
            let bad = Tensor::from_vec(vec![0.0; nt * (nbatch + 1)], &[1, nt, nbatch + 1]);
            let err = err_string(ode.solve_vjp_bkwd(&checkpoint, bad.view()));
            assert!(
                err.contains(&format!(
                    "dgdu_eval: Expected 1 lane or a multiple of {nbatch} lanes"
                )),
                "{label}: {err}"
            );
        }
    }
}

#[test]
fn vjp_rejects_unsupported_models_and_bad_inputs() {
    let t_eval = [0.5, 1.0];

    // a missing adjoint callback is reported before the solve
    let (ud, _) = new_user_data(0);
    let mut desc = f64_callbacks::desc(ud);
    desc.out_adjoint = None;
    let ode = new_callbacks(
        desc,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    configure(&ode);
    let err =
        err_string(ode.solve_vjp_fwd(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view()));
    assert!(
        err.contains("a VJP solve requires the 'out_adjoint' callback"),
        "{err}"
    );

    // roots and resets are not supported
    let (ud, _) = new_user_data(0);
    let ode = new_callbacks(
        f64_callbacks::hybrid_desc(ud),
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    configure(&ode);
    let err =
        err_string(ode.solve_vjp_fwd(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view()));
    assert!(err.contains("stop or reset condition"), "{err}");

    // dgdu_eval must have a column for each time of the forward pass
    let ode = make_ode(MatrixType::NalgebraDense, OdeSolverType::Bdf);
    let (_, checkpoint) = ode
        .solve_vjp_fwd(vector_tensor(&[2.0]).view(), vector_tensor(&t_eval).view())
        .unwrap();
    let err = err_string(ode.solve_vjp_bkwd(&checkpoint, matrix_tensor(1, 3, &[0.0; 3]).view()));
    assert!(
        err.contains("dgdu_eval: Expected 1 rows and 2 columns"),
        "{err}"
    );

    // a checkpoint of another matrix type
    let other = make_ode(MatrixType::FaerDense, OdeSolverType::Bdf);
    let err = err_string(other.solve_vjp_bkwd(&checkpoint, matrix_tensor(1, 2, &[0.0; 2]).view()));
    assert!(err.contains("incompatible"), "{err}");
}

#[test]
fn c_api_vjp_solves_and_frees() {
    use diffsol_c::ode_c::{diffsol_ode_solve_vjp_bkwd, diffsol_ode_solve_vjp_fwd};
    use diffsol_c::solution_wrapper_c::{
        diffsol_solution_wrapper_free, diffsol_solution_wrapper_get_ys,
    };
    use diffsol_c::tensor_c::diffsol_tensor_free;
    use diffsol_c::vjp_checkpoint_c::diffsol_vjp_checkpoint_free;
    use std::ptr;

    // the C enum values of MatrixType::NalgebraDense, LinearSolverType::Default and
    // OdeSolverType::Bdf
    let (matrix_type, linear_solver, ode_solver) = (0, 0, 0);
    let (ud, _) = new_user_data(0);
    let desc = f64_callbacks::desc(ud);
    let ode =
        unsafe { diffsol_ode_new_callbacks_f64(&desc, matrix_type, linear_solver, ode_solver, 1) };
    assert!(!ode.is_null());
    configure(unsafe { &*ode });

    let r = [2.0];
    let t_eval = VJP_T_EVAL;
    let nt = t_eval.len();
    let mut solution = ptr::null_mut();
    let mut checkpoint = ptr::null_mut();
    let status = unsafe {
        diffsol_ode_solve_vjp_fwd(
            ode,
            r.as_ptr(),
            1,
            1,
            t_eval.as_ptr(),
            nt,
            &mut solution,
            &mut checkpoint,
        )
    };
    assert_eq!(status, 0);
    assert!(!solution.is_null() && !checkpoint.is_null());

    let mut ys = ptr::null_mut();
    assert_eq!(
        unsafe { diffsol_solution_wrapper_get_ys(solution, &mut ys) },
        0
    );
    let dgdu = vjp_dgdu(unsafe { &*ys });
    let dgdu_values: Vec<f64> = (0..nt).map(|i| value_f64(&dgdu, &[0, i, 0])).collect();
    unsafe { diffsol_tensor_free(ys) };

    let mut gradient = ptr::null_mut();
    let status = unsafe {
        diffsol_ode_solve_vjp_bkwd(
            ode,
            checkpoint,
            dgdu_values.as_ptr(),
            1,
            nt,
            1,
            &mut gradient,
        )
    };
    assert_eq!(status, 0);
    let gradient_ref = unsafe { &*gradient };
    assert_eq!(gradient_ref.shape(), &[1, 1, 1]);
    assert_close(
        value_f64(gradient_ref, &[0, 0, 0]),
        vjp_expected_gradient(r[0]),
        1e-4,
        "C API VJP gradient",
    );
    unsafe { diffsol_tensor_free(gradient) };

    // null and inconsistent arguments are rejected
    let mut gradient = ptr::null_mut();
    let bad = [
        unsafe {
            diffsol_ode_solve_vjp_fwd(
                ode,
                r.as_ptr(),
                1,
                1,
                t_eval.as_ptr(),
                nt,
                &mut solution,
                ptr::null_mut(),
            )
        },
        unsafe {
            diffsol_ode_solve_vjp_bkwd(
                ode,
                ptr::null(),
                dgdu_values.as_ptr(),
                1,
                nt,
                1,
                &mut gradient,
            )
        },
        unsafe {
            diffsol_ode_solve_vjp_bkwd(ode, checkpoint, ptr::null(), 1, nt, 1, &mut gradient)
        },
        unsafe {
            diffsol_ode_solve_vjp_bkwd(
                ode,
                checkpoint,
                dgdu_values.as_ptr(),
                usize::MAX,
                2,
                1,
                &mut gradient,
            )
        },
    ];
    for status in bad {
        assert_ne!(status, 0);
    }
    // a wrong number of times is an error of the solve
    let status = unsafe {
        diffsol_ode_solve_vjp_bkwd(
            ode,
            checkpoint,
            dgdu_values.as_ptr(),
            1,
            nt - 1,
            1,
            &mut gradient,
        )
    };
    assert_ne!(status, 0);
    assert!(gradient.is_null());

    unsafe {
        diffsol_vjp_checkpoint_free(checkpoint);
        diffsol_solution_wrapper_free(solution);
        diffsol_ode_free(ode);
    }
}
