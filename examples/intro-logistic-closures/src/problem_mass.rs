use diffsol::{MatrixCommon, OdeBuilder};
use diffsol::{NalgebraMat, OdeEquationsImplicit, OdeSolverProblem, Vector};
type M = NalgebraMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

pub fn problem_mass() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .t0(0.0)
        .rtol(1e-6)
        .atol([1e-6])
        .p(vec![1.0, 10.0])
        .rhs_implicit(
            |x, p, _t, y| {
                y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
                y[1] = x[0] - x[1];
            },
            |x, p, _t, v, y| {
                y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]);
                y[1] = v[0] - v[1];
            },
        )
        .mass(|v, _p, _t, beta, y| {
            y[0] = v[0] + beta * y[0];
            y[1] *= beta;
        })
        .init(|_p, _t, y| y.fill(0.1), 2)
        .build()
        .unwrap()
}
