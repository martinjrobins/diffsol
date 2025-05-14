use crate::{problem_implicit, LS};
use diffsol::{BdfState, OdeSolverState, RkState};

pub fn create_solvers_uninit() {
    let problem = problem_implicit();

    // Create a non-initialised state and manually set the values before
    // creating the solver
    let mut state = RkState::new_without_initialise(&problem).unwrap();
    // ... set the state values manually
    state.as_mut().y[0] = 0.1;
    let _solver = problem.tr_bdf2_solver::<LS>(state);

    // Do the same for a BDF solver
    let mut state = BdfState::new_without_initialise(&problem).unwrap();
    state.as_mut().y[0] = 0.1;
    let _solver = problem.bdf_solver::<LS>(state);
}
