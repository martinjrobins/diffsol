use crate::{C, LS, M, T, V};
use diffsol::{
    AdjointOdeSolverMethod, OdeEquationsImplicitAdjoint, OdeSolverMethod, OdeSolverState,
};

#[allow(dead_code)]
pub fn solve_adjoint_sens<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitAdjoint<T = T, V = V, C = C, M = M> + 'a,
{
    let (checkpointing, _soln, _times) = solver.solve_with_checkpointing(10.0, None).unwrap();
    let adjoint_solver = solver
        .problem()
        .bdf_solver_adjoint::<LS, _>(checkpointing, Some(1))
        .unwrap();
    let final_state = adjoint_solver
        .solve_adjoint_backwards_pass(&[], &[])
        .unwrap();
    for (i, dgdp_i) in final_state.as_ref().sg.iter().enumerate() {
        println!("sens wrt parameter {i}: {dgdp_i:?}");
    }
}
