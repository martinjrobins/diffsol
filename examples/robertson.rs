use diffsol::{ode_solver::test_models::robertson::robertson, Bdf, OdeSolverMethod};

fn main() {
    let mut s = Bdf::default();
    let (problem, _soln) = robertson::<nalgebra::DMatrix<f64>>(false);
    for _ in 0..100000 {
        let _y = s.solve(&problem, 4.0000e+10);
    }
}
