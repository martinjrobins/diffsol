use diffsol::{ConstantOp, Matrix, NonLinearOpJacobian, OdeEquationsImplicit, OdeSolverProblem};

pub fn print_jacobian(problem: &OdeSolverProblem<impl OdeEquationsImplicit>) {
    let t0 = problem.t0;
    let y0 = problem.eqn.init().call(t0);
    let jacobian = problem.eqn.rhs().jacobian(&y0, t0);
    let (idx, vals) = jacobian.triplet_iter();
    for ((i, j), v) in idx.zip(vals) {
        println!("({i}, {j}) = {v}");
    }
}
