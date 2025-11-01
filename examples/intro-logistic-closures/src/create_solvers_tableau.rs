use crate::{problem_implicit, LS, M};
use diffsol::Tableau;

pub fn create_solvers_tableau() {
    let problem = problem_implicit();

    // Create a SDIRK solver with a pre-defined tableau
    let tableau = Tableau::<M>::tr_bdf2(*problem.context());
    let state = problem.rk_state(&tableau).unwrap();
    let _solver = problem.sdirk_solver::<LS, _>(state, tableau);

    // Create an ERK solver with a pre-defined tableau
    let tableau = Tableau::<M>::tsit45(*problem.context());
    let state = problem.rk_state(&tableau).unwrap();
    let _solver = problem.explicit_rk_solver(state, tableau);
}
