use crate::{problem_implicit, LS, M};
use diffsol::Tableau;

pub fn create_solvers_tableau() {
    let problem = problem_implicit();

    // Create a SDIRK solver with a pre-defined tableau
    let tableau = Tableau::tr_bdf2();
    let state = problem.rk_state(&tableau).unwrap();
    let _solver = problem.sdirk_solver::<LS, M>(state, tableau);

    // Create an ERK solver with a pre-defined tableau
    let tableau = Tableau::tsit45();
    let state = problem.rk_state(&tableau).unwrap();
    let _solver = problem.explicit_rk_solver::<M>(state, tableau);
}
