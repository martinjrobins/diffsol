use crate::{M, T, V};
use diffsol::{OdeEquationsImplicitSens, SensitivitiesOdeSolverMethod};

pub fn solve_fwd_sens<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: SensitivitiesOdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitSens<T = T, V = V, M = M> + 'a,
{
    let t_evals = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let (y, sens) = solver
        .solve_dense_sensitivities(t_evals.as_slice())
        .unwrap();
    println!("solution: {y:?}");
    for (i, dydp_i) in sens.iter().enumerate() {
        println!("sens wrt parameter {i}: {dydp_i:?}");
    }
}
