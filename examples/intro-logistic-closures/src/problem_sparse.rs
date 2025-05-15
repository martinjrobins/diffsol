use diffsol::{FaerSparseMat, MatrixCommon, OdeBuilder};
use diffsol::{OdeEquationsImplicit, OdeSolverProblem, Vector};
type M = FaerSparseMat<f64>;
type V = <M as MatrixCommon>::V;
type C = <M as MatrixCommon>::C;
type T = <M as MatrixCommon>::T;

pub fn problem_sparse() -> OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .t0(0.0)
        .rtol(1e-6)
        .atol([1e-6])
        .p(vec![1.0, 10.0])
        .rhs_implicit(
            |x, p, _t, y| {
                for i in 0..10 {
                    y[i] = p[0] * x[i] * (1.0 - x[i] / p[1]);
                }
            },
            |x, p, _t, v, y| {
                for i in 0..10 {
                    y[i] = p[0] * v[i] * (1.0 - 2.0 * x[i] / p[1]);
                }
            },
        )
        .init(|_p, _t, y| y.fill(0.1), 10)
        .build()
        .unwrap()
}
