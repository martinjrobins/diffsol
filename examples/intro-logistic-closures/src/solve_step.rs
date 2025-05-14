use diffsol::OdeSolverMethod;

pub fn solve_step<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: diffsol::OdeEquations<T = f64> + 'a,
{
    while solver.state().t < 10.0 {
        if solver.step().is_err() {
            break;
        }
    }
}
