#[cfg(feature = "diffsl-llvm")]
use diffsol::{
    FaerSparseLU, FaerSparseMat, LlvmModule, OdeBuilder, OdeSolverMethod, OdeSolverStopReason,
};

#[cfg(feature = "diffsl-llvm")]
#[test]
fn pybamm_dfn_exp_solve() {
    let _ = colog::init();
    
    type M = FaerSparseMat<f64>;
    type LS = FaerSparseLU<f64>;
    type CG = LlvmModule;

    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("pybamm_dfn_exp.diffsl");
    let full_text = std::fs::read_to_string(model_path).unwrap();

    let mut problem = OdeBuilder::<M>::new()
        .p([298.15]) // Set ambienttemperaturek parameter to 298.15 K
        .build_from_diffsl::<CG>(full_text.as_str())
        .unwrap();
    // Try very tolerant init options
    problem.ic_options.use_linesearch = true;
    problem.ic_options.armijo_constant = 0.5;  // Very lenient linesearch
    problem.ic_options.step_reduction_factor = 0.9;  // Smaller step reduction
    problem.ic_options.max_newton_iterations = 200;
    problem.ic_options.max_linesearch_iterations = 50;

    let t0 = 0.0;
    let tf = 3600.0;
    let t_interp = (0..100)
        .map(|i| t0 + (tf - t0) * (i as f64) / 99.0)
        .collect::<Vec<_>>();

    let mut solver = problem.bdf::<LS>().unwrap();
    let (_solution, stop_reason) = solver.solve_dense(t_interp.as_slice()).unwrap();
    assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
}
