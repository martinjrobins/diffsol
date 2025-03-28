use diffsol::OdeBuilder;
use diffsol::{BdfState, OdeSolverState, SdirkState, Tableau};
use diffsol::{NalgebraLU, NalgebraMat, OdeSolverMethod, OdeSolverStopReason, Vector};
type M = NalgebraMat<f64>;
type LS = NalgebraLU<f64>;

fn main() {
    //
    // SPECIFYING THE PROBLEM
    //
    let problem = OdeBuilder::<M>::new()
        .p(vec![1.0, 10.0])
        .rhs_implicit(
            |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
            |x, p, _t, v, y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
        )
        .init(|_p, _t, y| y.fill(0.1))
        .build()
        .unwrap();

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
    let state = problem.sdirk_state::<LS, _>(&tableau).unwrap();
    let _solver = problem.sdirk_solver::<LS, _>(state, tableau);

    // Create a tr_bdf2 or esdirk34 solvers directly (both are SDIRK solvers with different tableaus)
    let _solver = problem.tr_bdf2::<LS>();
    let _solver = problem.esdirk34::<LS>();

    // Create a non-initialised state and manually set the values before
    // creating the solver
    let mut state = SdirkState::new_without_initialise(&problem).unwrap();
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

    // Manually step the solver, stop solver at specified time
    let mut solver = problem.bdf::<LS>().unwrap();
    solver.set_stop_time(10.0).unwrap();
    loop {
        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => continue,
            Ok(OdeSolverStopReason::TstopReached) => break,
            Ok(OdeSolverStopReason::RootFound(_)) => panic!("Root finding not used"),
            Err(e) => panic!("Solver failed to converge: {}", e),
        }
    }
    let _soln = &solver.state().y;
}
