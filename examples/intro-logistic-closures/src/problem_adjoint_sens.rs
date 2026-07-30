use crate::{C, M, T, V};
use diffsol::{OdeBuilder, OdeEquationsImplicitAdjoint, OdeSolverProblem};

pub fn problem_adjoint_sens(
) -> OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .p(vec![1.0, 10.0])
        .rhs_adjoint_implicit(
            |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
            |x, p, _t, v, y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
            |x, p, _t, v, y| y[0] = -p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
            |x, p, _t, v, y| {
                y[0] = -v[0] * x[0] * (1.0 - x[0] / p[1]);
                y[1] = -v[0] * p[0] * x[0] * x[0] / (p[1] * p[1]);
            },
        )
        .init_adjoint(
            |_p, _t, y| y[0] = 0.1,
            |_p, _t, _v, y| {
                y[0] = 0.0;
                y[1] = 0.0;
            },
            1,
        )
        .build()
        .unwrap()
}
