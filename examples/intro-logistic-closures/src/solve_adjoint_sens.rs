use crate::{C, LS, M, T, V};
use diffsol::{
    AdjointOdeSolverMethod, Context, OdeEquationsImplicitAdjoint, OdeSolverMethod, OdeSolverState,
    Vector, VectorView,
};

#[allow(dead_code)]
pub fn solve_adjoint_sens<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitAdjoint<T = T, V = V, C = C, M = M> + 'a,
{
    let (checkpointing, _soln, _times, _stop_reason) =
        solver.solve_with_checkpointing(10.0, None).unwrap();
    let adjoint_solver = solver
        .problem()
        .bdf_solver_adjoint::<LS, _>(checkpointing, Some(solver.clone()), Some(1))
        .unwrap();
    let (final_state, _) = adjoint_solver
        .solve_adjoint_backwards_pass(&[], &[])
        .unwrap();
    // the adjoint state holds one output channel per batch lane
    let sg = final_state.as_ref().sg;
    for i in 0..sg.context().nbatch() {
        let dgdp_i = sg.get_batch(i).into_owned();
        println!("sens of output {i}: {dgdp_i:?}");
    }
}
