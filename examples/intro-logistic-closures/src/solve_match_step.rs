use diffsol::{OdeEquations, OdeSolverMethod, OdeSolverStopReason};

pub fn solve_match_step<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations<T = f64> + 'a,
{
    solver.set_stop_time(10.0).unwrap();
    loop {
        match solver.step() {
            Ok(OdeSolverStopReason::InternalTimestep) => continue,
            Ok(OdeSolverStopReason::TstopReached) => break,
            Ok(OdeSolverStopReason::RootFound(_t)) => break,
            Err(e) => panic!("Solver failed to converge: {e}"),
        }
    }
    println!("Solver stopped at time: {}", solver.state().t);
}
