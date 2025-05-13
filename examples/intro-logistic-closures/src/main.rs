use diffsol::{BdfState, OdeSolverState, RkState, Tableau};
use diffsol::{NalgebraLU, NalgebraMat, OdeSolverMethod};
mod problem_implicit;
use problem_implicit::problem_implicit;
mod problem_explicit;
use problem_explicit::problem_explicit;
mod problem_mass;
use problem_mass::problem_mass;
mod problem_root;
use problem_root::problem_root;
mod problem_fwd_sens;
use problem_fwd_sens::problem_fwd_sens;
mod problem_sparse;
use problem_sparse::problem_sparse;
mod solve_match_step;
use solve_match_step::solve_match_step;
mod solve_fwd_sens;
use solve_fwd_sens::solve_fwd_sens;
mod print_jacobian;
use print_jacobian::print_jacobian;
type M = NalgebraMat<f64>;
type LS = NalgebraLU<f64>;

fn main() {
    //
    // SPECIFYING THE PROBLEM
    //
    let problem = problem_fwd_sens();
    let mut solver = problem.bdf::<LS>().unwrap();
    solve_fwd_sens(&mut solver);
    let _problem = problem_root();
    let _problem = problem_mass();
    let _problem = problem_explicit();
    let problem = problem_sparse();
    print_jacobian(&problem);
    let problem = problem_implicit();
    let mut solver = problem.bdf::<LS>().unwrap();
    solve_match_step(&mut solver);

    //
    // CHOOSING A SOLVER
    //

    // Create a BDF solver with an initial state
    let _solver = problem.bdf::<LS>();

    // Create a non-initialised state and manually set the values before
    // creating the solver
    let state = BdfState::new_without_initialise(&problem).unwrap();
    // ... set the state values manually
    let _solver = problem.bdf_solver::<LS>(state);

    // Create a SDIRK solver with a pre-defined tableau
    let tableau = Tableau::<M>::tr_bdf2(problem.context().clone());
    let state = problem.rk_state(&tableau).unwrap();
    let _solver = problem.sdirk_solver::<LS, _>(state, tableau);

    // Create a tr_bdf2 or esdirk34 solvers directly (both are SDIRK solvers with different tableaus)
    let _solver = problem.tr_bdf2::<LS>();
    let _solver = problem.esdirk34::<LS>();

    // Create a non-initialised state and manually set the values before
    // creating the solver
    let mut state = RkState::new_without_initialise(&problem).unwrap();
    // ... set the state values manually
    state.as_mut().y[0] = 0.1;
    let _solver = problem.tr_bdf2_solver::<LS>(state);

    //
    // SOLVING THE PROBLEM
    //

    // Solve the problem return solution at solver times
    let mut solver = problem.bdf::<LS>().unwrap();
    let (_ys, _ts) = solver.solve(10.0).unwrap();

    // Solve the problem return solution at specified times
    let mut solver = problem.bdf::<LS>().unwrap();
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let _soln = solver.solve_dense(&times).unwrap();

    // Manually step the solver until given time
    let mut solver = problem.bdf::<LS>().unwrap();
    while solver.state().t < 10.0 {
        if solver.step().is_err() {
            break;
        }
    }

    // Manually step the solver, get solution at specified time
    let mut solver = problem.bdf::<LS>().unwrap();
    let t_o = 10.0;
    while solver.state().t < t_o {
        solver.step().unwrap();
    }
    let _soln = solver.interpolate(t_o).unwrap();

}
