use diffsol::{MatrixCommon, OdeBuilder};
use diffsol::{NalgebraMat, OdeEquationsImplicit, OdeSolverProblem};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

pub fn problem_root() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .t0(0.0)
        .rtol(1e-6)
        .atol([1e-6])
        .p(vec![1.0, 10.0])
        .rhs_implicit(
            |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
            |x, p, _t, v, y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
        )
        .init(|_p, _t, y| y[0] = 0.1, 1)
        .root(|x, _p, _t, y| y[0] = x[0] - 0.5, 1)
        .build()
        .unwrap()
}
