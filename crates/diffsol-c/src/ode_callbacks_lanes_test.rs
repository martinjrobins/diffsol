//! Check the lane layout (`P`, and which inputs are broadcast) that each solve gives to
//! each callback. The JAX/PJRT wrapper compiles one executable for each layout, so it must know
//! all the layouts before the solve.
//!
//! The model is `dx_i/dt = -p0 x_i + p1` with 2 states, 2 parameters and `out = x0 + x1`. It
//! runs through the C API with `nbatch` lanes, the same path as the future XLA FFI handlers.
//!

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::ptr;

use crate::ode_c::{
    diffsol_ode_free, diffsol_ode_solve_dense, diffsol_ode_solve_fwd_sens,
    diffsol_ode_solve_vjp_bkwd, diffsol_ode_solve_vjp_fwd,
};
use crate::ode_callbacks::{for_each_batch, OdeCallbacksDesc};
use crate::ode_callbacks_c::diffsol_ode_new_callbacks_f64;
use crate::solution_wrapper_c::diffsol_solution_wrapper_free;
use crate::tensor::{TensorView, TensorViewMut};
use crate::tensor_c::diffsol_tensor_free;
use crate::vjp_checkpoint_c::diffsol_vjp_checkpoint_free;

type View<'a> = *const TensorView<'a>;
type ViewMut<'a> = *mut TensorViewMut<'a>;

/// One record: callback name, `P`, `B`, and for each input whether its `P` axis is broadcast
/// (`strides[1] == 0`) and whether its `B` axis is broadcast.
type Record = (&'static str, usize, usize, Vec<(bool, bool)>);

struct Log(RefCell<BTreeSet<Record>>);

unsafe fn log(ud: *mut c_void, name: &'static str, inputs: &[View<'_>], y: ViewMut<'_>) {
    let log = unsafe { &*(ud as *const Log) };
    let y = unsafe { &*y };
    let (p, b) = (y.shape()[1], y.shape()[2]);
    assert_eq!(y.strides()[0], 8, "{name}: y lane not contiguous");
    assert_ne!(y.strides()[1], 0, "{name}: y broadcast on P");
    let bc = inputs
        .iter()
        .map(|v| {
            let v = unsafe { &**v };
            assert_eq!(
                &v.shape()[1..],
                &[p, b],
                "{name}: input lanes differ from y"
            );
            (p > 1 && v.strides()[1] == 0, b > 1 && v.strides()[2] == 0)
        })
        .collect();
    log.0.borrow_mut().insert((name, p, b, bc));
}

// The model. Each adjoint callback is the negative transposed product.

unsafe extern "C" fn rhs(x: View<'_>, p: View<'_>, _: f64, y: ViewMut<'_>, ud: *mut c_void) -> i32 {
    unsafe {
        log(ud, "rhs", &[x, p], y);
        for_each_batch([x, p], y, |[x, p], y: &mut [f64], _| {
            for i in 0..2 {
                y[i] = -p[0] * x[i] + p[1];
            }
        });
    }
    0
}

macro_rules! mul_cb {
    ($fname:ident, $name:literal, |$x:ident, $p:ident, $v:ident, $y:ident| $body:block) => {
        unsafe extern "C" fn $fname(
            x: View<'_>,
            p: View<'_>,
            _: f64,
            v: View<'_>,
            y: ViewMut<'_>,
            ud: *mut c_void,
        ) -> i32 {
            unsafe {
                log(ud, $name, &[x, p, v], y);
                for_each_batch([x, p, v], y, |[$x, $p, $v], $y: &mut [f64], _| $body);
            }
            0
        }
    };
}

mul_cb!(rhs_jac, "rhs_jac", |x, p, v, y| {
    let _ = x;
    for i in 0..2 {
        y[i] = -p[0] * v[i];
    }
});
mul_cb!(rhs_sens, "rhs_sens", |x, p, v, y| {
    let _ = p;
    for i in 0..2 {
        y[i] = -x[i] * v[0] + v[1];
    }
});
mul_cb!(rhs_adjoint, "rhs_adjoint", |x, p, v, y| {
    let _ = x;
    for i in 0..2 {
        y[i] = p[0] * v[i];
    }
});
mul_cb!(rhs_sens_adjoint, "rhs_sens_adjoint", |x, p, v, y| {
    let _ = p;
    y[0] = x[0] * v[0] + x[1] * v[1];
    y[1] = -(v[0] + v[1]);
});
mul_cb!(out_jac, "out_jac", |x, p, v, y| {
    let _ = (x, p);
    y[0] = v[0] + v[1];
});
mul_cb!(out_sens, "out_sens", |x, p, v, y| {
    let _ = (x, p, v);
    y[0] = 0.0;
});
mul_cb!(out_adjoint, "out_adjoint", |x, p, v, y| {
    let _ = (x, p);
    y[0] = -v[0];
    y[1] = -v[0];
});
mul_cb!(out_sens_adjoint, "out_sens_adjoint", |x, p, v, y| {
    let _ = (x, p, v);
    y[0] = 0.0;
    y[1] = 0.0;
});

unsafe extern "C" fn out(x: View<'_>, p: View<'_>, _: f64, y: ViewMut<'_>, ud: *mut c_void) -> i32 {
    unsafe {
        log(ud, "out", &[x, p], y);
        for_each_batch([x], y, |[x], y: &mut [f64], _| y[0] = x[0] + x[1]);
    }
    0
}

unsafe extern "C" fn init(p: View<'_>, _: f64, y: ViewMut<'_>, ud: *mut c_void) -> i32 {
    unsafe {
        log(ud, "init", &[p], y);
        for_each_batch([], y, |[], y: &mut [f64], _| y.fill(1.0));
    }
    0
}

unsafe extern "C" fn init_sens(
    p: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    ud: *mut c_void,
) -> i32 {
    unsafe {
        log(ud, "init_sens", &[p, v], y);
        for_each_batch([], y, |[], y: &mut [f64], _| y.fill(0.0));
    }
    0
}

unsafe extern "C" fn init_sens_adjoint(
    p: View<'_>,
    _: f64,
    v: View<'_>,
    y: ViewMut<'_>,
    ud: *mut c_void,
) -> i32 {
    unsafe {
        log(ud, "init_sens_adjoint", &[p, v], y);
        for_each_batch([], y, |[], y: &mut [f64], _| y.fill(0.0));
    }
    0
}

fn desc(log: &Log, with_out: bool) -> OdeCallbacksDesc<f64> {
    let mut d = OdeCallbacksDesc::new(2, 2, rhs, init);
    d.user_data = log as *const Log as *mut c_void;
    d.rhs_jac = Some(rhs_jac);
    d.rhs_sens = Some(rhs_sens);
    d.rhs_adjoint = Some(rhs_adjoint);
    d.rhs_sens_adjoint = Some(rhs_sens_adjoint);
    d.init_sens = Some(init_sens);
    d.init_sens_adjoint = Some(init_sens_adjoint);
    if with_out {
        d.nout = 1;
        d.out = Some(out);
        d.out_jac = Some(out_jac);
        d.out_sens = Some(out_sens);
        d.out_adjoint = Some(out_adjoint);
        d.out_sens_adjoint = Some(out_sens_adjoint);
    }
    d
}

const T_EVAL: [f64; 4] = [0.0, 0.5, 1.0, 2.0];
const SOLVERS: [(&str, i32); 4] = [("bdf", 0), ("esdirk34", 1), ("tr_bdf2", 2), ("tsit45", 3)];

/// Run one solve kind and return the records.
fn run(kind: &str, solver: i32, nbatch: usize, ndir: usize, with_out: bool) -> BTreeSet<Record> {
    let log = Log(RefCell::new(BTreeSet::new()));
    let d = desc(&log, with_out);
    let params: Vec<f64> = (0..nbatch).flat_map(|b| [1.0 + b as f64, 0.5]).collect();
    unsafe {
        let ode = diffsol_ode_new_callbacks_f64(&d, 0, 0, solver, nbatch);
        assert!(!ode.is_null());
        // only record the solve itself, not the construction
        log.0.borrow_mut().clear();
        let mut soln = ptr::null_mut();
        let (pp, t) = (params.as_ptr(), T_EVAL.as_ptr());
        let nt = T_EVAL.len();
        let rc = match kind {
            "dense" => diffsol_ode_solve_dense(ode, pp, 2, nbatch, t, nt, &mut soln),
            "fwd_sens" => diffsol_ode_solve_fwd_sens(ode, pp, 2, nbatch, t, nt, &mut soln),
            "vjp_fwd" | "vjp_bkwd" => {
                let mut ckpt = ptr::null_mut();
                let rc = diffsol_ode_solve_vjp_fwd(ode, pp, 2, nbatch, t, nt, &mut soln, &mut ckpt);
                assert_eq!(rc, 0, "vjp_fwd failed");
                if kind == "vjp_bkwd" {
                    log.0.borrow_mut().clear();
                    let nout = if with_out { 1 } else { 2 };
                    let dgdu = vec![1.0; nout * nt * nbatch * ndir];
                    let mut grad = ptr::null_mut();
                    let rc = diffsol_ode_solve_vjp_bkwd(
                        ode,
                        ckpt,
                        dgdu.as_ptr(),
                        nout,
                        nt,
                        nbatch * ndir,
                        &mut grad,
                    );
                    assert_eq!(rc, 0, "vjp_bkwd failed");
                    diffsol_tensor_free(grad);
                }
                diffsol_vjp_checkpoint_free(ckpt);
                0
            }
            _ => unreachable!(),
        };
        assert_eq!(rc, 0, "{kind} failed");
        diffsol_solution_wrapper_free(soln);
        diffsol_ode_free(ode);
    }
    log.0.into_inner()
}

/// The lane layout rule that the JAX/PJRT wrapper uses to select an executable:
/// - the `B` axis of an input is never broadcast;
/// - if `P == 1`, no input is broadcast;
/// - if `P > 1`, the inputs `x` and `p` are broadcast on `P`, and `v` is not;
/// - `P > 1` only for `rhs_jac`/`out_jac` in a forward sensitivity solve (`P == nparams`), and
///   for the adjoint callbacks in a VJP backward solve (`P == ndirections`).
#[test]
fn lane_layout_rule() {
    const NPARAMS: usize = 2;
    let mut seen_p_gt_1 = BTreeSet::new();
    for with_out in [false, true] {
        for kind in ["dense", "fwd_sens", "vjp_fwd", "vjp_bkwd"] {
            for (sname, solver) in SOLVERS {
                for (nbatch, ndir) in [(1, 1), (1, 3), (3, 1), (3, 3)] {
                    if ndir > 1 && kind != "vjp_bkwd" {
                        continue;
                    }
                    for (name, p, b, bc) in run(kind, solver, nbatch, ndir, with_out) {
                        let ctx = format!("{kind} {sname} B={nbatch} D={ndir} {name}");
                        assert_eq!(b, nbatch, "{ctx}: B");
                        assert!(bc.iter().all(|&(_, bb)| !bb), "{ctx}: B axis broadcast");
                        if p == 1 {
                            continue;
                        }
                        // the inputs are (x, p, v) or (p, v): all but the last are broadcast
                        let (v, xp) = bc.split_last().unwrap();
                        assert!(xp.iter().all(|&(bp, _)| bp), "{ctx}: x/p not broadcast");
                        assert!(!v.0, "{ctx}: v broadcast");
                        let expected = match kind {
                            "fwd_sens" => NPARAMS,
                            "vjp_bkwd" => ndir,
                            _ => panic!("{ctx}: P={p} > 1"),
                        };
                        assert_eq!(p, expected, "{ctx}: P");
                        seen_p_gt_1.insert((kind, name));
                    }
                }
            }
        }
    }
    let expected: BTreeSet<_> = [
        ("fwd_sens", "out_jac"),
        ("fwd_sens", "rhs_jac"),
        ("vjp_bkwd", "init_sens_adjoint"),
        ("vjp_bkwd", "out_adjoint"),
        ("vjp_bkwd", "out_sens_adjoint"),
        ("vjp_bkwd", "rhs_adjoint"),
        ("vjp_bkwd", "rhs_sens_adjoint"),
    ]
    .into_iter()
    .collect();
    assert_eq!(seen_p_gt_1, expected);
}
