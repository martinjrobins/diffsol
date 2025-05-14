use crate::{T, V};
use diffsol::{OdeEquations, OdeSolverMethod};

pub fn solve_dense<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations<T = T, V = V> + 'a,
{
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let _soln = solver.solve_dense(&times).unwrap();
}
