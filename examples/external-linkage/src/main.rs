use diffsol::{
    ode_equations::external_linkage::{RealType, UIntType},
    DenseMatrix, OdeBuilder, OdeSolverMethod,
};
type M = diffsol::NalgebraMat<f64>;
type LS = diffsol::NalgebraLU<f64>;

#[no_mangle]
extern "C" fn stop(
    _time: RealType,
    _u: *const RealType,
    _data: *mut RealType,
    _root: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
}

#[no_mangle]
extern "C" fn rhs(
    _time: RealType,
    u: *const RealType,
    data: *mut RealType,
    rr: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
    let r = unsafe { *data.add(0) };
    let k = unsafe { *data.add(1) };
    let u = unsafe { *u.add(0) };
    let f = r * u * (1.0 - u / k);
    unsafe {
        *rr.add(0) = f;
    }
}

#[no_mangle]
extern "C" fn rhs_grad(
    _time: RealType,
    u: *const RealType,
    du: *const RealType,
    data: *const RealType,
    _ddata: *mut RealType,
    _rr: *const RealType,
    drr: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
    let r = unsafe { *data.add(0) };
    let k = unsafe { *data.add(1) };
    let u = unsafe { *u.add(0) };
    let du = unsafe { *du.add(0) };

    let df_du = r * (1.0 - 2.0 * u / k);
    unsafe {
        *drr.add(0) = df_du * du;
    }
}

#[no_mangle]
extern "C" fn mass(
    _time: RealType,
    _u: *const RealType,
    _data: *mut RealType,
    _mv: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
}

#[no_mangle]
extern "C" fn set_u0(
    u: *mut RealType,
    _data: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
    unsafe {
        *u.add(0) = 0.1;
    }
}

#[no_mangle]
extern "C" fn set_u0_grad(
    _u: *const RealType,
    du: *mut RealType,
    _data: *const RealType,
    _ddata: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
    unsafe {
        *du.add(0) = 0.0;
    }
}

#[no_mangle]
extern "C" fn calc_out(
    _time: RealType,
    _u: *const RealType,
    _data: *mut RealType,
    _out: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
}

#[no_mangle]
extern "C" fn calc_out_grad(
    _time: RealType,
    _u: *const RealType,
    _du: *const RealType,
    _data: *const RealType,
    _ddata: *mut RealType,
    _out: *const RealType,
    _dout: *mut RealType,
    _thread_id: UIntType,
    _thread_dim: UIntType,
) {
}

#[no_mangle]
extern "C" fn get_dims(
    states: *mut UIntType,
    inputs: *mut UIntType,
    outputs: *mut UIntType,
    data: *mut UIntType,
    stop: *mut UIntType,
    has_mass: *mut UIntType,
) {
    unsafe {
        *states = 1;
        *inputs = 2;
        *outputs = 0;
        *data = 2;
        *stop = 0;
        *has_mass = 0;
    }
}
#[no_mangle]
extern "C" fn set_inputs(inputs: *const RealType, data: *mut RealType) {
    unsafe {
        *data.add(0) = *inputs.add(0);
        *data.add(1) = *inputs.add(1);
    }
}
#[no_mangle]
extern "C" fn get_inputs(inputs: *mut RealType, data: *const RealType) {
    unsafe {
        *inputs.add(0) = *data.add(0);
        *inputs.add(1) = *data.add(1);
    }
}
#[no_mangle]
extern "C" fn set_inputs_grad(
    _inputs: *const RealType,
    dinputs: *const RealType,
    _data: *const RealType,
    ddata: *mut RealType,
) {
    unsafe {
        *ddata.add(0) = *dinputs.add(0);
        *ddata.add(1) = *dinputs.add(1);
    }
}

#[no_mangle]
extern "C" fn set_id(id: *mut RealType) {
    unsafe {
        *id.add(0) = 1.0;
    }
}

#[no_mangle]
extern "C" fn set_constants(_thread_id: UIntType, _thread_dim: UIntType) {}

fn main() {
    let r = 1.0;
    let k = 10.0;
    let y0 = 0.1;
    let problem = OdeBuilder::<M>::new()
        .rtol(1e-6)
        .p([r, k])
        .build_from_external_linkage()
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let t = 0.4;
    let (ys, ts) = solver.solve(t).unwrap();
    for (i, t) in ts.iter().enumerate() {
        let y = ys.column(i);
        let expect_y = k / (1.0 + (k - y0) * (-r * t).exp() / y0);
        assert!(
            (y[0] - expect_y).abs() < 1e-6,
            "at t={:.3}, got {}, expected {}",
            t,
            y[0],
            expect_y
        );
    }
}
