use argmin::{
    core::{observers::ObserverMode, CostFunction, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_observer_slog::SlogLogger;
use diffsol::{
    DiffSl, MatrixCommon, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem, Op,
    SensitivitiesOdeSolverMethod, Vector,
};
use std::cell::RefCell;

type M = diffsol::NalgebraMat<f64>;
type V = diffsol::NalgebraVec<f64>;
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
        let ctx = *problem.eqn().context();
        problem
            .eqn_mut()
            .set_params(&V::from_vec(param.clone(), ctx));
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
        let ctx = *problem.eqn().context();
        problem
            .eqn_mut()
            .set_params(&V::from_vec(param.clone(), ctx));
        let mut solver = problem.bdf_sens::<LS>().unwrap();
        let (ys, sens) = match solver.solve_dense_sensitivities(&self.ts_data) {
            Ok((ys, sens)) => (ys, sens),
            Err(_) => return Ok(vec![f64::MAX / 1000.; param.len()]),
        };
        let dlossdp = sens
            .into_iter()
            .map(|s| {
                s.inner()
                    .column_iter()
                    .zip(
                        ys.inner()
                            .column_iter()
                            .zip(self.ys_data.inner().column_iter()),
                    )
                    .map(|(si, (yi, di))| 2.0 * (yi - di).dot(&si))
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        Ok(dlossdp)
    }
}

pub fn main() {
    let code = "
            in_i { b = 4.0/3.0, d = 1.0 }
            a { 2.0/3.0 } c { 1.0 } x0 { 1.0 } y0 { 1.0 }
            u_i {
                y1 = x0,
                y2 = y0,
            }
            F_i {
                a * y1 - b * y1 * y2,
                c * y1 * y2 - d * y2,
            }
        ";

    let (b_true, d_true) = (4.0 / 3.0, 1.0);
    let t_data = (0..101)
        .map(|i| f64::from(i) * 40. / 100.)
        .collect::<Vec<f64>>();
    let problem = OdeBuilder::<M>::new()
        .p([b_true, d_true])
        .sens_atol([1e-6])
        .sens_rtol(1e-6)
        .build_from_diffsl(code)
        .unwrap();
    let ys_data = {
        let mut solver = problem.bdf::<LS>().unwrap();
        solver.solve_dense(&t_data).unwrap()
    };

    let cost = Problem {
        ys_data,
        ts_data: t_data,
        problem: RefCell::new(problem),
    };

    let init_param = vec![b_true - 0.1, d_true - 0.1];

    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
    let solver = LBFGS::new(linesearch, 7);
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    // print result
    println!("{}", res);
    // Best parameter vector
    let best = res.state().best_param.as_ref().unwrap();
    println!("Best parameter vector: {:?}", best);
    println!("True parameter vector: {:?}", vec![b_true, d_true]);
}
