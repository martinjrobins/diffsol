use crate::{C, LS, M, T, V};
use diffsol::{
    AdjointOdeSolverMethod, OdeEquationsImplicitAdjoint, OdeSolverMethod, OdeSolverState, Vector,
};

pub fn solve_adjoint_sens<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitAdjoint<T = T, V = V, C = C, M = M> + 'a,
{
    // forward pass: solve to t = 10 and save checkpoints
    let (checkpointing, _soln, _times, _stop_reason) =
        solver.solve_with_checkpointing(10.0, None).unwrap();

    // backward pass: G is the integral of the output (the problem must set integrate_out)
    let adjoint_solver = solver
        .problem()
        .bdf_solver_adjoint::<LS, _>(checkpointing, Some(solver.clone()), None)
        .unwrap();
    let (final_state, _) = adjoint_solver
        .solve_adjoint_backwards_pass(&[], &[])
        .unwrap();
    println!("dG/dp = {:?}", final_state.as_ref().sg.clone_as_vec());
}
