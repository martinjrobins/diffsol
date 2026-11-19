use crate::{problem_implicit, LS};

pub fn create_solvers() {
    let problem = problem_implicit();

    // Create a BDF solver
    let _bdf = problem.bdf::<LS>().unwrap();

    // Create tr_bdf2 and esdirk34 solvers (both are SDIRK solvers with different tableaus)
    let _tr_bdf2 = problem.tr_bdf2::<LS>();
    let _esdirk34 = problem.esdirk34::<LS>();

    // Create a TSIT45 solver (a ERK method), this does not require a linear solver
    let _tsit45 = problem.tsit45();
}
