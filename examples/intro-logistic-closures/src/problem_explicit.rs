use diffsol::{MatrixCommon, OdeBuilder};
use diffsol::{NalgebraMat, OdeEquations, OdeSolverProblem, Vector};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

pub fn problem_explicit() -> OdeSolverProblem<impl OdeEquations<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .p(vec![1.0, 10.0])
        .rhs(|x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]))
        .init(|_p, _t, y| y.fill(0.1), 1)
        .build()
        .unwrap()
}
