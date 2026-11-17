// ANCHOR: problem
use argmin::{
    core::{CostFunction, Executor, Gradient, State},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, DiffSl, Matrix, MatrixCommon, NalgebraMat, NalgebraVec,
    OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, Scale,
    StepVjp, Vector, VectorView, VectorViewMut,
};
use std::cell::RefCell;
use std::time::{Duration, Instant};

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type T = f64;
type LS = diffsol::NalgebraLU<f64>;
type CG = diffsol::LlvmModule;
type Eqn = DiffSl<M, CG>;

#[derive(Clone, Copy, Debug)]
enum GradientMethod {
    ContinuousAdjoint,
    DiscreteAdjoint,
}

struct Problem {
    ys_data: M,
    ts_data: Vec<T>,
    problem: RefCell<OdeSolverProblem<Eqn>>,
    method: GradientMethod,
}
// ANCHOR_END: problem

// ANCHOR: cost_function
impl CostFunction for Problem {
    type Output = T;
    type Param = Vec<T>;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let mut problem = self.problem.borrow_mut();
        let context = *problem.eqn().context();
        problem
            .eqn_mut()
            .set_params(&V::from_vec(param.clone(), context));
        let mut solver = problem.bdf::<LS>().unwrap();
        let ys = match solver.solve_dense(&self.ts_data) {
            Ok((ys, _stop_reason)) => ys,
            Err(_) => return Ok(f64::MAX / 1000.),
        };
        let loss = ys
            .inner()
            .column_iter()
            .zip(self.ys_data.inner().column_iter())
            .map(|(a, b)| (a - b).norm_squared())
            .sum::<f64>();
        Ok(loss)
    }
}
// ANCHOR_END: cost_function

impl Problem {
    /// The gradient of the loss with respect to the model outputs: column `j` is
    /// `2 (y(t_j) - y_data(t_j))`.
    fn dloss_dy(&self, ys: &M) -> M {
        let mut g_m = M::zeros(ys.nrows(), ys.ncols(), *ys.context());
        for j in 0..g_m.ncols() {
            let g_m_i = (ys.column(j) - self.ys_data.column(j)) * Scale(2.0);
            g_m.column_mut(j).copy_from(&g_m_i);
        }
        g_m
    }

    // ANCHOR: gradient_continuous
    fn gradient_continuous(&self, param: &[T]) -> Vec<T> {
        let mut problem = self.problem.borrow_mut();
        let context = *problem.eqn().context();
        problem
            .eqn_mut()
            .set_params(&V::from_vec(param.to_vec(), context));
        let mut solver = problem.bdf::<LS>().unwrap();
        let (c, ys, _stop_reason) = match solver.solve_dense_with_checkpointing(&self.ts_data, None)
        {
            Ok(ys) => ys,
            Err(_) => return vec![f64::MAX / 1000.; param.len()],
        };
        let g_m = self.dloss_dy(&ys);
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(c, Some(solver), Some(1))
            .unwrap();
        match adjoint_solver.solve_adjoint_backwards_pass(self.ts_data.as_slice(), &[&g_m]) {
            Ok((soln, _)) => soln.as_ref().sg.get_batch(0).into_owned().clone_as_vec(),
            Err(_) => vec![f64::MAX / 1000.; param.len()],
        }
    }
    // ANCHOR_END: gradient_continuous

    // ANCHOR: gradient_discrete
    fn gradient_discrete(&self, param: &[T]) -> Vec<T> {
        let mut problem = self.problem.borrow_mut();
        let context = *problem.eqn().context();
        problem
            .eqn_mut()
            .set_params(&V::from_vec(param.to_vec(), context));
        let mut solver = problem.bdf::<LS>().unwrap();
        let (ys, tape) = match solver.solve_dense_vjp_fwd(&self.ts_data) {
            Ok(res) => res,
            Err(_) => return vec![f64::MAX / 1000.; param.len()],
        };
        let g_m = self.dloss_dy(&ys);
        match solver.solve_dense_vjp_bkwd(&tape, &g_m) {
            Ok(grad) => grad.clone_as_vec(),
            Err(_) => vec![f64::MAX / 1000.; param.len()],
        }
    }
    // ANCHOR_END: gradient_discrete
}

// ANCHOR: gradient
impl Gradient for Problem {
    type Gradient = Vec<T>;
    type Param = Vec<T>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin_math::Error> {
        Ok(match self.method {
            GradientMethod::ContinuousAdjoint => self.gradient_continuous(param),
            GradientMethod::DiscreteAdjoint => self.gradient_discrete(param),
        })
    }
}
// ANCHOR_END: gradient

// ANCHOR: main
fn build_problem(k: T, c: T) -> OdeSolverProblem<Eqn> {
    OdeBuilder::<M>::new()
        .p([k, c])
        .sens_atol([1e-6])
        .sens_rtol(1e-6)
        .out_atol([1e-6])
        .out_rtol(1e-6)
        .build_from_diffsl(
            "
        in_i { k = 1.0, c = 0.1 }
        m { 1.0 }
        u_i {
            x = 1,
            v = 0,
        }
        F_i {
            v,
            -k/m * x - c/m * v,
        }
        ",
        )
        .unwrap()
}

/// The mean wall time of one call to `f`, over `n` calls after one warm-up call.
fn time_per_call(n: u32, mut f: impl FnMut()) -> Duration {
    f();
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    start.elapsed() / n
}

pub fn main() {
    let (k_true, c_true) = (1.0, 0.1);
    let t_data = (0..101)
        .map(|i| f64::from(i) * 40. / 100.)
        .collect::<Vec<f64>>();
    let ys_data = {
        let problem = build_problem(k_true, c_true);
        let mut solver = problem.bdf::<LS>().unwrap();
        solver.solve_dense(&t_data).unwrap().0
    };
    let make_cost = |method| Problem {
        ys_data: ys_data.clone(),
        ts_data: t_data.clone(),
        problem: RefCell::new(build_problem(k_true, c_true)),
        method,
    };
    let init_param = vec![k_true - 0.1, c_true - 0.01];
    let methods = [
        GradientMethod::ContinuousAdjoint,
        GradientMethod::DiscreteAdjoint,
    ];

    // benchmark one gradient evaluation at the initial parameters
    println!("Gradient at {init_param:?}:");
    let n = 1000;
    let mut times = Vec::new();
    let mut grads = Vec::new();
    for method in methods {
        let cost = make_cost(method);
        let time = time_per_call(n, || {
            cost.gradient(&init_param).unwrap();
        });
        let grad = cost.gradient(&init_param).unwrap();
        println!("  {method:?}: {grad:?}, {time:?} per gradient");
        times.push(time);
        grads.push(grad);
    }
    let rel_diff = grads[0]
        .iter()
        .zip(&grads[1])
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max);
    println!("  Largest relative difference: {rel_diff:.2e}");
    println!(
        "  Speed-up of discrete over continuous: {:.2}x",
        times[0].as_secs_f64() / times[1].as_secs_f64()
    );

    // benchmark a full fit with each method
    for method in methods {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
        let solver = LBFGS::new(linesearch, 7);
        let start = Instant::now();
        let res = Executor::new(make_cost(method), solver)
            .configure(|state| state.param(init_param.clone()).counting(true))
            .run()
            .unwrap();
        let time = start.elapsed();
        let state = res.state();
        let counts = state.get_func_counts();
        println!("\nFit with {method:?}:");
        println!("  time:           {time:?}");
        println!("  iterations:     {}", state.get_iter());
        println!("  cost evals:     {}", counts["cost_count"]);
        println!("  gradient evals: {}", counts["gradient_count"]);
        println!("  best cost:      {:e}", state.get_best_cost());
        println!("  best params:    {:?}", state.get_best_param().unwrap());
        println!("  termination:    {}", state.get_termination_status());
    }
    println!("\nTrue parameters: {:?}", vec![k_true, c_true]);
}
// ANCHOR_END: main
