use diffsol::{ode_solver::test_models::heat2d::head2d_problem, OdeEquations, NonLinearOp, ConstantOp};
use nalgebra_sparse::{io::save_to_matrix_market_file, CscMatrix};

fn main() {
    let (problem, _soln) = head2d_problem::<CscMatrix<f64>, 5>();
    let t0 = 0.0;
    let y0 = problem.eqn.init().call(t0);
    let mat = problem.eqn.rhs().jacobian(&y0, t0);
    save_to_matrix_market_file(&mat, "heat2d_5.mtx").expect("Failed to save matrix to file");
    
   let (problem, _soln) = head2d_problem::<CscMatrix<f64>, 10>();
    let t0 = 0.0;
    let y0 = problem.eqn.init().call(t0);
    let mat = problem.eqn.rhs().jacobian(&y0, t0);
    save_to_matrix_market_file(&mat, "heat2d_10.mtx").expect("Failed to save matrix to file");

    let (problem, _soln) = head2d_problem::<CscMatrix<f64>, 20>();
    let t0 = 0.0;
    let y0 = problem.eqn.init().call(t0);
    let mat = problem.eqn.rhs().jacobian(&y0, t0);
    save_to_matrix_market_file(&mat, "heat2d_20.mtx").expect("Failed to save matrix to file");
    
    let (problem, _soln) = head2d_problem::<CscMatrix<f64>, 30>();
    let t0 = 0.0;
    let y0 = problem.eqn.init().call(t0);
    let mat = problem.eqn.rhs().jacobian(&y0, t0);
    save_to_matrix_market_file(&mat, "heat2d_30.mtx").expect("Failed to save matrix to file");
}
