use diffsol::{
    AdjointOdeSolverMethod, ClosureAutodiff, ConstantAutodiff, ConstantClosureAutodiff,
    DenseMatrix, Matrix, NalgebraContext, NalgebraLU, NalgebraMat, NalgebraVec, NonLinearAutodiff,
    OdeBuilder, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverState, Op, VectorCommon,
};

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type T = f64;
type LS = NalgebraLU<f64>;
type C = NalgebraContext;

struct Logistic;

impl NonLinearAutodiff<M> for Logistic {
    type T = f64;

    fn rhs_inplace(x: &[f64], p: &[f64], y: &mut [f64]) {
        y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
    }
}

impl ConstantAutodiff<M> for Logistic {
    type T = f64;

    fn init_inplace(p: &[f64], y: &mut [f64]) {
        y[0] = p[2];
    }
}

fn main() {
    println!("=== Logistic ODE with NonLinearAutodiff trait ===\n");

    let r_val = 1.0;
    let k_val = 1.0;
    let y0_val = 0.1;
    let t_final = 5.0;

    let rhs_op = ClosureAutodiff::<M, Logistic>::new(0, 0, 0, C::default());
    let init_op = ConstantClosureAutodiff::<M, Logistic>::new(1, 3, C::default());

    let problem = OdeBuilder::<M>::new()
        .p([r_val, k_val, y0_val])
        .rhs_autodiff(rhs_op)
        .init_autodiff(init_op)
        .build()
        .unwrap();

    let mut solver = problem.bdf::<LS>().unwrap();

    let (checkpointing, _soln, _times, _stop_reason) =
        solver.solve_with_checkpointing(t_final, None).unwrap();

    let y_final = solver.interpolate(t_final).unwrap();
    println!("Forward solution at t={}: y = {:.6}", t_final, y_final.inner()[0]);

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

    let y_base = {
        let p = OdeBuilder::<M>::new()
            .p([r_val, k_val, y0_val])
            .rhs_autodiff(ClosureAutodiff::<M, Logistic>::new(0, 0, 0, C::default()))
            .init_autodiff(ConstantClosureAutodiff::<M, Logistic>::new(1, 3, C::default()))
            .build()
            .unwrap();
        let mut solver = p.bdf::<LS>().unwrap();
        while solver.state().t <= t_final {
            solver.step().unwrap();
        }
        solver.interpolate(t_final).unwrap().inner()[0]
    };

    for (i, name) in [(0usize, "dy/dr"), (1, "dy/dk"), (2, "dy/dy0")] {
        let mut params = [r_val, k_val, y0_val];
        params[i] += eps;
        let p = OdeBuilder::<M>::new()
            .p(params)
            .rhs_autodiff(ClosureAutodiff::<M, Logistic>::new(0, 0, 0, C::default()))
            .init_autodiff(ConstantClosureAutodiff::<M, Logistic>::new(1, 3, C::default()))
            .build()
            .unwrap();
        let mut solver = p.bdf::<LS>().unwrap();
        while solver.state().t <= t_final {
            solver.step().unwrap();
        }
        let y_perturb = solver.interpolate(t_final).unwrap().inner()[0];
        let fd = (y_perturb - y_base) / eps;
        println!("  {} = {:.6} (adjoint: {:.6})", name, fd, adjoint_grad[i]);
    }
}
