#![feature(autodiff)]

use std::autodiff::*;

use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, Matrix, NalgebraContext, NalgebraLU, NalgebraMat,
    NalgebraVec, OdeBuilder, OdeEquationsImplicit, OdeEquationsImplicitAdjoint, OdeSolverMethod,
    OdeSolverProblem, OdeSolverState, Op, VectorCommon,
};

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type T = f64;
type LS = NalgebraLU<f64>;
type C = NalgebraContext;

#[autodiff_forward(d_rhs, Dual, Const, Const, Dual)]
#[autodiff_reverse(d_rhs_adj, Active, Const, Const, Active)]
#[autodiff_reverse(d_rhs_sens, Const, Active, Active, Active)]
fn logistic_rhs(x: f64, r: f64, k: f64) -> f64 {
    r * x * (1.0 - x / k)
}

#[autodiff_reverse(d_init, Const, Const, Active, Active)]
fn logistic_init(_r: f64, _k: f64, y0: f64) -> f64 {
    y0
}

fn build_implicit_problem(
    r: f64,
    k: f64,
) -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    let rhs = |x: &V, p: &V, _t: T, y: &mut V| {
        y[0] = logistic_rhs(x[0], p[0], p[1]);
    };
    let rhs_jac = |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
        let (_f, jvp) = d_rhs(x[0], v[0], p[0], p[1]);
        y[0] = jvp;
    };

    OdeBuilder::<M>::new()
        .p([r, k])
        .rhs_implicit(rhs, rhs_jac)
        .init(|_p, _t, y| y[0] = 0.1, 1)
        .build()
        .unwrap()
}

fn part_a_implicit_solver() {
    println!("=== Part (a): Implicit BDF solver with autodiff-generated Jacobian ===");

    let problem = build_implicit_problem(1.0, 10.0);
    let t_final = 5.0;

    let mut solver = problem.bdf::<LS>().unwrap();
    while solver.state().t <= t_final {
        solver.step().unwrap();
    }
    let y = solver.interpolate(t_final).unwrap().inner()[0];

    println!("Logistic ODE: dy/dt = r*y*(1 - y/k)");
    println!("  r = 1.0, k = 10.0, y0 = 0.1");
    println!("  Solution at t={}: y = {:.6}", t_final, y);
}

fn part_b_adjoint() {
    println!(
        "\n=== Part (b): Adjoint sensitivity analysis with autodiff-generated derivatives ==="
    );

    let r_val = 1.0;
    let k_val = 1.0;
    let y0_val = 0.1;
    let t_final = 5.0;

    println!("Logistic ODE: dy/dt = r*y*(1 - y/k)");
    println!("  r = {r_val}, k = {k_val}, y0 = {y0_val}");
    println!("  Computing d(y({}))/dp via adjoint method", t_final);

    let rhs = |x: &V, p: &V, _t: T, y: &mut V| {
        y[0] = logistic_rhs(x[0], p[0], p[1]);
    };
    let rhs_jac = |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
        let (_f, jvp) = d_rhs(x[0], v[0], p[0], p[1]);
        y[0] = jvp;
    };
    let rhs_adjoint = |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
        let (_f, adj) = d_rhs_adj(x[0], p[0], p[1], 1.0);
        y[0] = -adj * v[0];
    };
    let rhs_sens_adjoint = |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
        let (_f, d_dr, d_dk) = d_rhs_sens(x[0], p[0], p[1], 1.0);
        y[0] = -d_dr * v[0];
        y[1] = -d_dk * v[0];
    };
    let init = |p: &V, _t: T, y: &mut V| {
        y[0] = logistic_init(p[0], p[1], p[2]);
    };
    let init_sens_adjoint = |p: &V, _t: T, v: &V, y: &mut V| {
        let (_out, d_dy0) = d_init(p[0], p[1], p[2], 1.0);
        y[0] = 0.0;
        y[1] = 0.0;
        y[2] = -d_dy0 * v[0];
    };

    let problem = OdeBuilder::<M>::new()
        .p([r_val, k_val, y0_val])
        .rhs_adjoint_implicit(rhs, rhs_jac, rhs_adjoint, rhs_sens_adjoint)
        .init_adjoint(init, init_sens_adjoint, 1)
        .build()
        .unwrap();

    let mut solver = problem.bdf::<LS>().unwrap();
    let (checkpointing, _soln, _times, _stop_reason) =
        solver.solve_with_checkpointing(t_final, None).unwrap();

    let y_final = solver.interpolate(t_final).unwrap();
    println!(
        "\n  Forward solution at t={}: y = {:.6}",
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
    println!("\n  Adjoint gradients (dy/dp):");
    println!("    dy/dr  = {:.6}", adjoint_grad[0]);
    println!("    dy/dk  = {:.6}", adjoint_grad[1]);
    println!("    dy/dy0 = {:.6}", adjoint_grad[2]);

    println!("\n  Finite difference verification (eps = 1e-6):");
    let eps = 1e-6;

    let y_base = {
        let p = build_implicit_problem(r_val, k_val);
        let mut solver = p.bdf::<LS>().unwrap();
        while solver.state().t <= t_final {
            solver.step().unwrap();
        }
        solver.interpolate(t_final).unwrap().inner()[0]
    };

    let fd_r = {
        let p = build_implicit_problem(r_val + eps, k_val);
        let mut solver = p.bdf::<LS>().unwrap();
        while solver.state().t <= t_final {
            solver.step().unwrap();
        }
        (solver.interpolate(t_final).unwrap().inner()[0] - y_base) / eps
    };
    let fd_k = {
        let p = build_implicit_problem(r_val, k_val + eps);
        let mut solver = p.bdf::<LS>().unwrap();
        while solver.state().t <= t_final {
            solver.step().unwrap();
        }
        (solver.interpolate(t_final).unwrap().inner()[0] - y_base) / eps
    };

    println!("    dy/dr  = {:.6} (adjoint: {:.6})", fd_r, adjoint_grad[0]);
    println!("    dy/dk  = {:.6} (adjoint: {:.6})", fd_k, adjoint_grad[1]);
    println!("    dy/dy0 = {:.6}", adjoint_grad[2]);
}

fn main() {
    part_a_implicit_solver();
    part_b_adjoint();
}
