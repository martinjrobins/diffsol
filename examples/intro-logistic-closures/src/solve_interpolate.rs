use crate::{T, V};
use diffsol::{OdeEquations, OdeSolverMethod};

pub fn solve_interpolate<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations<V = V, T = T> + 'a,
{
    let t_o = 10.0;
    while solver.state().t < t_o {
        solver.step().unwrap();
    }
    let _soln = solver.interpolate(t_o).unwrap();
}
