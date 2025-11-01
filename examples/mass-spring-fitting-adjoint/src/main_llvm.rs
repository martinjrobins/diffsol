use argmin::{
    core::{observers::ObserverMode, CostFunction, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_observer_slog::SlogLogger;
use diffsol::{
    AdjointOdeSolverMethod, DenseMatrix, DiffSl, Matrix, MatrixCommon, NalgebraMat, NalgebraVec,
    OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, Scale, Vector,
    VectorCommon, VectorViewMut,
};
use std::cell::RefCell;

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type T = f64;
type LS = diffsol::NalgebraLU<f64>;
type CG = diffsol::LlvmModule;
type Eqn = DiffSl<M, CG>;

struct Problem {
    ys_data: M,
    ts_data: Vec<T>,
    problem: RefCell<OdeSolverProblem<Eqn>>,
}

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
            Ok(ys) => ys,
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

impl Gradient for Problem {
    type Gradient = Vec<T>;
    type Param = Vec<T>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin_math::Error> {
        let mut problem = self.problem.borrow_mut();
        let context = *problem.eqn().context();
        problem
            .eqn_mut()
            .set_params(&V::from_vec(param.clone(), context));
        let mut solver = problem.bdf::<LS>().unwrap();
        let (c, ys) = match solver.solve_dense_with_checkpointing(&self.ts_data, None) {
            Ok(ys) => ys,
            Err(_) => return Ok(vec![f64::MAX / 1000.; param.len()]),
        };
        let mut g_m = M::zeros(2, self.ts_data.len(), *problem.eqn().context());
        for j in 0..g_m.ncols() {
            let g_m_i = (ys.column(j) - self.ys_data.column(j)) * Scale(2.0);
            g_m.column_mut(j).copy_from(&g_m_i);
        }
        let adjoint_solver = problem.bdf_solver_adjoint::<LS, _>(c, Some(1)).unwrap();
        match adjoint_solver.solve_adjoint_backwards_pass(self.ts_data.as_slice(), &[&g_m]) {
            Ok(soln) => Ok(soln.as_ref().sg[0]
                .inner()
                .iter()
                .copied()
                .collect::<Vec<_>>()),
            Err(_) => Ok(vec![f64::MAX / 1000.; param.len()]),
        }
    }
}

pub fn main() {
    let (k_true, c_true) = (1.0, 0.1);
    let t_data = (0..101)
        .map(|i| f64::from(i) * 40. / 100.)
        .collect::<Vec<f64>>();
    let problem = OdeBuilder::<M>::new()
        .p([k_true, c_true])
        .sens_atol([1e-6])
        .sens_rtol(1e-6)
        .out_atol([1e-6])
        .out_rtol(1e-6)
        .build_from_diffsl(
            "
        in = [k, c]
        k { 1.0 } m { 1.0 } c { 0.1 }
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
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let ys_data = solver.solve_dense(&t_data).unwrap();

    let cost = Problem {
        ys_data,
        ts_data: t_data,
        problem: RefCell::new(problem),
    };

    let init_param = vec![k_true - 0.1, c_true - 0.01];

    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
    let solver = LBFGS::new(linesearch, 7);
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    // print result
    println!("{res}");
    // Best parameter vector
    let best = res.state().best_param.as_ref().unwrap();
    println!("Best parameter vector: {best:?}");
    println!("True parameter vector: {:?}", vec![k_true, c_true]);
}
