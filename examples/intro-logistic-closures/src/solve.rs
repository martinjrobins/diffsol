use crate::{T, V};
use diffsol::{OdeEquations, OdeSolverMethod};

pub fn solve<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations<T = T, V = V> + 'a,
{
    let (_ys, _ts) = solver.solve(10.0).unwrap();
}
