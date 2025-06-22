use diffsol::{ConstantOp, Matrix, NonLinearOpJacobian, OdeEquationsImplicit, OdeSolverProblem};

pub fn print_jacobian(problem: &OdeSolverProblem<impl OdeEquationsImplicit>) {
    let t0 = problem.t0;
    let y0 = problem.eqn.init().call(t0);
    let jacobian = problem.eqn.rhs().jacobian(&y0, t0);
    for (i, j, v) in jacobian.triplet_iter() {
        println!("({i}, {j}) = {v}");
    }
}
