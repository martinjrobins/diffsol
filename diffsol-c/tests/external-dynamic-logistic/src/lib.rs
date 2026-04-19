const LOGISTIC_X0: f64 = 0.1;
const STATES: u32 = 1;
const INPUTS: u32 = 1;
const OUTPUTS: u32 = 1;
const DATA: u32 = 1;
const STOP: u32 = 1;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn barrier_init() {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_constants(_thread_id: u32, _thread_dim: u32) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_u0(u: *mut f64, _data: *mut f64, _thread_id: u32, _thread_dim: u32) {
    if !u.is_null() {
        unsafe {
            *u = LOGISTIC_X0;
        }
    }
}

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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_id(id: *mut f64) {
    if !id.is_null() {
        unsafe {
            *id = 42.0;
        }
    }
}

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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_inputs(inputs: *const f64, data: *mut f64, _model_index: u32) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    unsafe {
        *data = *inputs;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_inputs(inputs: *mut f64, data: *const f64) {
    if inputs.is_null() || data.is_null() {
        return;
    }
    unsafe {
        *inputs = *data;
    }
}

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
