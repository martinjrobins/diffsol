use crate::{C, M, T, V};
use diffsol::{OdeEquationsImplicitAdjoint, StepVjp, Vector};

pub fn solve_discrete_adjoint_sens<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: StepVjp<'a, Eqn>,
    Eqn: OdeEquationsImplicitAdjoint<T = T, V = V, C = C, M = M> + 'a,
{
    // measured data d_i at times t_i (here the logistic solution with r = 1.2)
    let t_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = [0.1, 0.3245, 1.0019, 2.699, 5.5104, 8.0296];

    // G = sum_i (u(t_i) - d_i)^2, with one loss (ndirections = 1)
    let (loss, dgdp) = solver
        .solve_dense_vjp(
            &t_data,
            1,
            // add g_i(u_i) to the loss
            |i, _t, u, g| g[0] += (u[0] - y_data[i]).powi(2),
            // write dg_i/du_i
            |i, _t, u, dgdu| dgdu[0] = 2.0 * (u[0] - y_data[i]),
        )
        .unwrap();
    println!("G = {}, dG/dp = {:?}", loss[0], dgdp.clone_as_vec());
}
