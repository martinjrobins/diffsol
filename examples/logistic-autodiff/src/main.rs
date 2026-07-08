use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, Matrix, NalgebraLU, NalgebraMat, NalgebraVec, OdeBuilder,
    OdeSolverMethod, OdeSolverState, Op, VectorCommon,
};

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type LS = NalgebraLU<f64>;

fn main() {
    println!("=== Logistic ODE with rhs_autodiff ===\n");

    let r_val = 1.0;
    let k_val = 1.0;
    let y0_val = 0.1;
    let t_final = 5.0;

    let problem = OdeBuilder::<M>::new()
        .p([r_val, k_val, y0_val])
        .rhs_autodiff(|x: &V, p: &V, y: &mut V| {
            y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
        })
        .init_autodiff(
            |p: &V, y: &mut V| {
                y[0] = p[2];
            },
            1,
        )
        .build()
        .unwrap();

    let mut solver = problem.bdf::<LS>().unwrap();
    let (checkpointing, _soln, _times, _stop_reason) =
        solver.solve_with_checkpointing(t_final, None).unwrap();

    let y_final = solver.interpolate(t_final).unwrap();
    println!(
        "Forward solution at t={}: y = {:.6}",
        t_final,
        y_final.inner()[0]
    );

    let ctx = *problem.eqn().context();
    let mut dgdu = NalgebraMat::<f64>::zeros(1, 1, ctx);
    dgdu.set_index(0, 0, 1.0);

    let adjoint_solver = problem
        .bdf_solver_adjoint::<LS, _>(checkpointing, Some(solver), Some(1))
        .unwrap();
    let (final_state, _) = adjoint_solver
        .solve_adjoint_backwards_pass(&[t_final], &[&dgdu])
        .unwrap();

    let sg = final_state.as_ref();
    let adjoint_grad = [
        sg.sg[0].inner()[0],
        sg.sg[0].inner()[1],
        sg.sg[0].inner()[2],
    ];
    println!("\nGradient of y({}) w.r.t. parameters (adjoint):", t_final);
    println!("  dy/dr  = {:.6}", adjoint_grad[0]);
    println!("  dy/dk  = {:.6}", adjoint_grad[1]);
    println!("  dy/dy0 = {:.6}", adjoint_grad[2]);

    println!("\nFinite difference verification (eps = 1e-6):");
    let eps = 1e-6;

    let y_base = solve_forward(r_val, k_val, y0_val, t_final);

    for (i, name) in [(0usize, "dy/dr"), (1, "dy/dk"), (2, "dy/dy0")] {
        let mut params = [r_val, k_val, y0_val];
        params[i] += eps;
        let y_perturb = solve_forward(params[0], params[1], params[2], t_final);
        let fd = (y_perturb - y_base) / eps;
        println!("  {} = {:.6} (adjoint: {:.6})", name, fd, adjoint_grad[i]);
    }
}

fn solve_forward(r: f64, k: f64, y0: f64, t_final: f64) -> f64 {
    let p = OdeBuilder::<M>::new()
        .p([r, k, y0])
        .rhs_autodiff(|x: &V, p: &V, y: &mut V| {
            y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
        })
        .init_autodiff(|p: &V, y: &mut V| y[0] = p[2], 1)
        .build()
        .unwrap();
    let mut solver = p.bdf::<LS>().unwrap();
    while solver.state().t <= t_final {
        solver.step().unwrap();
    }
    solver.interpolate(t_final).unwrap().inner()[0]
}
