use crate::{C, LS, M, T, V};
use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, Matrix, MatrixCommon, OdeEquationsImplicitAdjoint,
    OdeSolverMethod, OdeSolverState, Scale, VectorViewMut,
};

#[allow(dead_code)]
pub fn solve_adjoint_sens_sum_squares<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitAdjoint<T = T, V = V, C = C, M = M> + 'a,
{
    let t_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = solver.solve_dense(t_data.as_slice()).unwrap();
    let problem = solver.problem();

    let (checkpointing, soln) = solver
        .solve_dense_with_checkpointing(t_data.as_slice(), None)
        .unwrap();

    let mut g_m = M::zeros(2, t_data.len(), *solver.problem().eqn().context());
    for j in 0..g_m.ncols() {
        let g_m_i = (soln.column(j) - y_data.column(j)) * Scale(2.0);
        g_m.column_mut(j).copy_from(&g_m_i);
    }

    let adjoint_solver = problem
        .bdf_solver_adjoint::<LS, _>(checkpointing, Some(1))
        .unwrap();
    let final_state = adjoint_solver
        .solve_adjoint_backwards_pass(t_data.as_slice(), &[&g_m])
        .unwrap();
    for (i, dgdp_i) in final_state.as_ref().sg.iter().enumerate() {
        println!("sens wrt parameter {i}: {dgdp_i:?}");
    }
}
