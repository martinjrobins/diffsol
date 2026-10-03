use crate::{problem_implicit, LS};

pub fn create_solvers() {
    let problem = problem_implicit();

    // BDF method using
    let _bdf = problem.bdf::<LS>().unwrap();

    // Create a tr_bdf2 or esdirk34 solvers directly (both are SDIRK solvers with different tableaus)
    let _tr_bdf2 = problem.tr_bdf2::<LS>();
    let _esdirk34 = problem.esdirk34::<LS>();

    // Rodas5P for stiff identity-mass ODEs with a Jacobian action.
    let _rodas5p = problem.rodas5p::<LS>();

    // Create a TSIT45 solver (a ERK method), this does not require a linear solver
    let _tsit45 = problem.tsit45();
}
