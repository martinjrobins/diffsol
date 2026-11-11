//! Checks that the callbacks [`OdeCallbacks`] requires before a solve (its requirement table)
//! are exactly the callbacks that the solve calls.
//!
//! For each model, operation and solver method, each optional derivative callback is removed
//! in turn. The operation is then run directly (not through `GenericSolve`, so without the
//! check before the solve), and the result is compared with the requirement table:
//! - a callback the table does not require must never be found missing during the solve, and
//! - a callback the table requires must be found missing during the solve.

use std::ffi::c_void;
use std::panic::AssertUnwindSafe;

use diffsol::{
    DefaultSolver, DenseMatrix, NalgebraContext, NalgebraMat, OdeBuilder, OdeSolverProblem, Op,
};

use crate::linear_solver_type::LinearSolverType;
use crate::ode_callbacks::{for_each_batch, OdeCallbackError, OdeCallbacks, OdeCallbacksDesc};
use crate::ode_solver_type::OdeSolverType;
use crate::solve_eqn::{EqnRequirement, SolveEqn};
use crate::tensor::{TensorView, TensorViewMut};

type M = NalgebraMat<f64>;
type LS = <M as DefaultSolver>::LS;
type Eqn = OdeCallbacks<M>;

// The logistic model `dy/dt = r y (1 - y)`, `y(0) = 0.1`, with `out = y`, a root at `y = 0.9`
// that resets the state to `y = 0.1`, and an identity mass matrix. Every adjoint callback
// computes the negative transpose product.

type View<'a> = *const TensorView<'a>;
type ViewMut<'a> = *mut TensorViewMut<'a>;

/// Set `y[0] = f(inputs[0][0], ...)` in each lane.
unsafe fn set<const N: usize>(
    inputs: [View<'_>; N],
    y: ViewMut<'_>,
    f: impl Fn([f64; N]) -> f64,
) -> i32 {
    unsafe {
        for_each_batch(inputs, y, |ins: [&[f64]; N], y: &mut [f64], _| {
            y[0] = f(ins.map(|v| v[0]))
        })
    };
    0
}

unsafe extern "C" fn rhs(x: View<'_>, p: View<'_>, _: f64, y: ViewMut<'_>, _: *mut c_void) -> i32 {
    unsafe { set([x, p], y, |[x, p]| p * x * (1.0 - x)) }
}

unsafe extern "C" fn rhs_jac(
    x: View<'_>,
    p: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([x, p, v], y, |[x, p, v]| p * (1.0 - 2.0 * x) * v) }
}

unsafe extern "C" fn rhs_sens(
    x: View<'_>,
    _: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([x, v], y, |[x, v]| x * (1.0 - x) * v) }
}

unsafe extern "C" fn rhs_adjoint(
    x: View<'_>,
    p: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([x, p, v], y, |[x, p, v]| -p * (1.0 - 2.0 * x) * v) }
}

unsafe extern "C" fn rhs_sens_adjoint(
    x: View<'_>,
    _: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([x, v], y, |[x, v]| -x * (1.0 - x) * v) }
}

unsafe extern "C" fn init(_: View<'_>, _: f64, y: ViewMut<'_>, _: *mut c_void) -> i32 {
    unsafe { set([], y, |[]| 0.1) }
}

unsafe extern "C" fn init_zero(
    _: View<'_>,
    _: f64,
    _: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([], y, |[]| 0.0) }
}

unsafe extern "C" fn identity(
    x: View<'_>,
    _: View<'_>,
    _: f64,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([x], y, |[x]| x) }
}

unsafe extern "C" fn root(x: View<'_>, _: View<'_>, _: f64, y: ViewMut<'_>, _: *mut c_void) -> i32 {
    unsafe { set([x], y, |[x]| x - 0.9) }
}

unsafe extern "C" fn reset(
    _: View<'_>,
    _: View<'_>,
    _: f64,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([], y, |[]| 0.1) }
}

unsafe extern "C" fn identity_mul(
    _: View<'_>,
    _: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([v], y, |[v]| v) }
}

unsafe extern "C" fn neg_identity_mul(
    _: View<'_>,
    _: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([v], y, |[v]| -v) }
}

unsafe extern "C" fn zero_mul(
    _: View<'_>,
    _: View<'_>,
    _: f64,
    _: View<'_>,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    unsafe { set([], y, |[]| 0.0) }
}

unsafe extern "C" fn mass(
    v: View<'_>,
    _: View<'_>,
    _: f64,
    beta: f64,
    y: ViewMut<'_>,
    _: *mut c_void,
) -> i32 {
    // `y` is also an input: `y = v + beta y`
    unsafe { for_each_batch([v], y, |[v], y: &mut [f64], _| y[0] = v[0] + beta * y[0]) };
    0
}

/// The logistic model with every derivative callback. `out`, `hybrid` (root and reset) and
/// `mass` add those parts of the model.
fn model(out: bool, hybrid: bool, with_mass: bool) -> OdeCallbacksDesc<f64> {
    let mut d = OdeCallbacksDesc::new(1, 1, rhs, init);
    d.rhs_jac = Some(rhs_jac);
    d.rhs_sens = Some(rhs_sens);
    d.rhs_adjoint = Some(rhs_adjoint);
    d.rhs_sens_adjoint = Some(rhs_sens_adjoint);
    d.init_sens = Some(init_zero);
    d.init_sens_adjoint = Some(init_zero);
    if out {
        d.nout = 1;
        d.out = Some(identity);
        d.out_jac = Some(identity_mul);
        d.out_sens = Some(zero_mul);
        d.out_adjoint = Some(neg_identity_mul);
        d.out_sens_adjoint = Some(zero_mul);
    }
    if hybrid {
        d.nroots = 1;
        d.root = Some(root);
        d.root_jac = Some(identity_mul);
        d.root_sens = Some(zero_mul);
        d.root_adjoint = Some(neg_identity_mul);
        d.root_sens_adjoint = Some(zero_mul);
        d.reset = Some(reset);
        d.reset_jac = Some(zero_mul);
        d.reset_sens = Some(zero_mul);
        d.reset_adjoint = Some(zero_mul);
        d.reset_sens_adjoint = Some(zero_mul);
    }
    if with_mass {
        d.mass = Some(mass);
        d.mass_adjoint = Some(mass);
    }
    d
}

/// The optional derivative callbacks, which are removed in turn.
const OPTIONAL: [&str; 19] = [
    "rhs_jac",
    "rhs_sens",
    "rhs_adjoint",
    "rhs_sens_adjoint",
    "init_sens",
    "init_sens_adjoint",
    "mass_adjoint",
    "out_jac",
    "out_sens",
    "out_adjoint",
    "out_sens_adjoint",
    "root_jac",
    "root_sens",
    "root_adjoint",
    "root_sens_adjoint",
    "reset_jac",
    "reset_sens",
    "reset_adjoint",
    "reset_sens_adjoint",
];

/// `desc` without the callback `name`, or `None` if `desc` does not have it.
fn without(mut d: OdeCallbacksDesc<f64>, name: &str) -> Option<OdeCallbacksDesc<f64>> {
    let present = match name {
        "rhs_jac" => d.rhs_jac.take().is_some(),
        "rhs_sens" => d.rhs_sens.take().is_some(),
        "rhs_adjoint" => d.rhs_adjoint.take().is_some(),
        "rhs_sens_adjoint" => d.rhs_sens_adjoint.take().is_some(),
        "init_sens" => d.init_sens.take().is_some(),
        "init_sens_adjoint" => d.init_sens_adjoint.take().is_some(),
        "mass_adjoint" => d.mass_adjoint.take().is_some(),
        "out_jac" => d.out_jac.take().is_some(),
        "out_sens" => d.out_sens.take().is_some(),
        "out_adjoint" => d.out_adjoint.take().is_some(),
        "out_sens_adjoint" => d.out_sens_adjoint.take().is_some(),
        "root_jac" => d.root_jac.take().is_some(),
        "root_sens" => d.root_sens.take().is_some(),
        "root_adjoint" => d.root_adjoint.take().is_some(),
        "root_sens_adjoint" => d.root_sens_adjoint.take().is_some(),
        "reset_jac" => d.reset_jac.take().is_some(),
        "reset_sens" => d.reset_sens.take().is_some(),
        "reset_adjoint" => d.reset_adjoint.take().is_some(),
        "reset_sens_adjoint" => d.reset_sens_adjoint.take().is_some(),
        _ => unreachable!("unknown callback {name}"),
    };
    present.then_some(d)
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Solve,
    FwdSens,
    ContinuousAdjoint,
    SplitAdjoint,
}

impl Operation {
    fn requirement(self, method: OdeSolverType) -> EqnRequirement {
        match self {
            Operation::Solve => EqnRequirement::Solve(method),
            Operation::FwdSens => EqnRequirement::FwdSens,
            Operation::ContinuousAdjoint | Operation::SplitAdjoint => {
                EqnRequirement::Adjoint(method)
            }
        }
    }
}

const FINAL_TIME: f64 = 5.0;
const T_EVAL: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

fn problem(desc: OdeCallbacksDesc<f64>) -> OdeSolverProblem<Eqn> {
    // SAFETY: the test callbacks are valid and use the documented lengths
    let eqn = unsafe { OdeCallbacks::<M>::new(desc, NalgebraContext::default()) }.unwrap();
    OdeBuilder::<M>::new()
        .p(vec![2.0])
        .build_from_eqn(eqn)
        .unwrap()
}

/// Run `op` with `method` directly, without the check before the solve. Return whether the
/// solve succeeded, and the error the callbacks recorded (if any).
fn run(
    desc: OdeCallbacksDesc<f64>,
    op: Operation,
    method: OdeSolverType,
) -> (bool, Option<OdeCallbackError>) {
    let mut problem = problem(desc);
    let ok = match op {
        Operation::Solve => method
            .solve_dense::<M, Eqn, LS>(&mut problem, &T_EVAL)
            .is_ok(),
        Operation::FwdSens => method
            .solve_fwd_sens::<M, Eqn, LS>(&mut problem, &T_EVAL)
            .is_ok(),
        Operation::ContinuousAdjoint => {
            problem.integrate_out = true;
            method
                .solve_continuous_adjoint::<M, Eqn, LS>(&mut problem, FINAL_TIME)
                .is_ok()
        }
        Operation::SplitAdjoint => {
            let fwd = method.solve_adjoint_fwd::<M, Eqn, LS>(
                &mut problem,
                &T_EVAL,
                LinearSolverType::Default,
            );
            let error = problem.eqn.take_callback_error();
            let checkpoint = match (fwd, error) {
                (Ok((_, checkpoint)), None) => checkpoint,
                (fwd, error) => return (fwd.is_ok(), error),
            };
            let nout = problem.eqn.nout();
            let dgdu = M::from_vec(
                nout,
                T_EVAL.len(),
                vec![1.0; nout * T_EVAL.len()],
                NalgebraContext::default(),
            );
            method
                .solve_adjoint_bkwd::<M, Eqn, LS>(&mut problem, checkpoint.as_ref(), &dgdu, &T_EVAL)
                .is_ok()
        }
    };
    (ok, problem.eqn.take_callback_error())
}

const METHODS: [OdeSolverType; 4] = [
    OdeSolverType::Bdf,
    OdeSolverType::Esdirk34,
    OdeSolverType::TrBdf2,
    OdeSolverType::Tsit45,
];

const OPERATIONS: [Operation; 4] = [
    Operation::Solve,
    Operation::FwdSens,
    Operation::ContinuousAdjoint,
    Operation::SplitAdjoint,
];

#[test]
fn requirement_table_matches_the_callbacks_each_solve_calls() {
    let models = [
        ("plain", model(false, false, false)),
        ("out", model(true, false, false)),
        ("out + root/reset", model(true, true, false)),
        ("out + mass", model(true, false, true)),
        ("out + root/reset + mass", model(true, true, true)),
    ];
    let mut mismatches = Vec::new();
    let mut checked = 0;
    for (model_name, full) in models {
        for op in OPERATIONS {
            for method in METHODS {
                // only compare where the solve works with every callback provided
                let baseline = std::panic::catch_unwind(AssertUnwindSafe(|| run(full, op, method)));
                match baseline {
                    Ok((true, None)) => {}
                    Ok(_) => continue,
                    Err(_) => {
                        mismatches.push(format!(
                            "{model_name}, {op:?}, {method:?}: panics with every callback"
                        ));
                        continue;
                    }
                }
                for name in OPTIONAL {
                    let Some(desc) = without(full, name) else {
                        continue;
                    };
                    let required = problem(desc)
                        .eqn
                        .check_requirement(op.requirement(method))
                        .is_err();
                    let Ok((_, error)) =
                        std::panic::catch_unwind(AssertUnwindSafe(|| run(desc, op, method)))
                    else {
                        mismatches.push(format!(
                            "{model_name}, {op:?}, {method:?}, without '{name}': panics"
                        ));
                        continue;
                    };
                    let called = matches!(
                        error,
                        Some(OdeCallbackError::Missing { callback }) if callback == name
                    );
                    checked += 1;
                    if required != called {
                        mismatches.push(format!(
                            "{model_name}, {op:?}, {method:?}, without '{name}': \
                             the table {} it, the solve {} it",
                            if required {
                                "requires"
                            } else {
                                "does not require"
                            },
                            if called { "calls" } else { "does not call" },
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 0);
    assert!(
        mismatches.is_empty(),
        "{} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
