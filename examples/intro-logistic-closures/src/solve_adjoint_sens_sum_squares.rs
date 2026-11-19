use crate::{C, LS, M, T, V};
use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, Matrix, OdeEquationsImplicitAdjoint, OdeSolverMethod,
    OdeSolverState, Vector,
};

pub fn solve_adjoint_sens_sum_squares<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitAdjoint<T = T, V = V, C = C, M = M> + 'a,
{
    // measured data d_i at times t_i (here the logistic solution with r = 1.2)
    let t_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = [0.1, 0.3245, 1.0019, 2.699, 5.5104, 8.0296];

    // forward pass: solve at the data times and save checkpoints
    let (checkpointing, soln, _stop_reason) = solver
        .solve_dense_with_checkpointing(&t_data, None)
        .unwrap();

    // G = sum_i (u(t_i) - d_i)^2, so column i of dgdu is 2 (u(t_i) - d_i)
    let mut dgdu = M::zeros(1, t_data.len(), *solver.problem().eqn().context());
    for (i, d) in y_data.iter().enumerate() {
        dgdu.set_index(0, i, 2.0 * (soln.get_index(0, i) - d));
    }

    // backward pass: one set of g_i functions, so nout_override = 1
    let adjoint_solver = solver
        .problem()
        .bdf_solver_adjoint::<LS, _>(checkpointing, Some(solver.clone()), Some(1))
        .unwrap();
    let (final_state, _) = adjoint_solver
        .solve_adjoint_backwards_pass(&t_data, &[&dgdu])
        .unwrap();
    println!("dG/dp = {:?}", final_state.as_ref().sg.clone_as_vec());
}
