use diffsol_c::host_array::HostArray;
use diffsol_c::{
    JitBackendType, LinearSolverType, MatrixType, OdeSolverType, OdeWrapper, ScalarType,
};

fn main() {
    // ANCHOR: logistic_code
    let code = r#"
        in_i { r = 2.0 }
        u_i { y = 0.1 }
        dudt_i { dydt = 0 }
        F_i { (r * y) * (1 - y) }
        out_i { y }
    "#;
    // ANCHOR_END: logistic_code

    // ANCHOR: create_wrapper
    let ode = OdeWrapper::new_jit(
        code,
        JitBackendType::Cranelift,
        ScalarType::F64,
        MatrixType::NalgebraDense,
        LinearSolverType::Default,
        OdeSolverType::Bdf,
    )
    .unwrap();
    // ANCHOR_END: create_wrapper

    // ANCHOR: configure
    ode.set_rtol(1e-6).unwrap();
    ode.set_atol(1e-8).unwrap();
    // ANCHOR_END: configure

    // ANCHOR: solve_dense
    let t_eval = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    // params is a `[nparams, nlanes]` array: here one lane with one parameter
    let params = HostArray::from_vec(vec![2.0], vec![1, 1]);
    // t_eval is a `[nt]` array that owns the times
    let t_eval = HostArray::from_vec(t_eval.clone(), vec![t_eval.len()]);
    let solution = ode.solve_dense(&params, &t_eval).unwrap();
    // ANCHOR_END: solve_dense

    // ANCHOR: extract
    // ys is a `[nout, nt, nbatch]` array; ts is a `[nt]` array that all lanes share.
    let ys = solution.get_ys().unwrap();
    let ts = solution.get_ts().unwrap();
    // ANCHOR_END: extract

    // ANCHOR: check
    let r = 2.0f64;
    let y0 = 0.1f64;
    let tol = 1e-5;
    for i in 0..ts.shape()[0] {
        // `index` gives a reference to the value at an index, with one value for each axis
        let t = *ts.index::<f64>([i]).unwrap();
        let y_computed = *ys.index::<f64>([0, i, 0]).unwrap();
        let y_analytical = y0 * (r * t).exp() / (1.0 - y0 + y0 * (r * t).exp());
        assert!(
            (y_computed - y_analytical).abs() < tol,
            "at t={t:.4}: computed={y_computed:.6}, analytical={y_analytical:.6}"
        );
    }
    // ANCHOR_END: check
}
