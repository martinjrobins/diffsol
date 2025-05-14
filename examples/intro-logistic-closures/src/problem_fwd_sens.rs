use crate::{C, M, T, V};
use diffsol::{OdeBuilder, OdeEquationsImplicitSens, OdeSolverProblem};

pub fn problem_fwd_sens(
) -> OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .p(vec![1.0, 10.0])
        .rhs_sens_implicit(
            |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
            |x, p, _t, v, y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
            |x, p, _t, v, y| {
                y[0] = v[0] * x[0] * (1.0 - x[0] / p[1]) + v[1] * p[0] * x[0] * x[0] / (p[1] * p[1])
            },
        )
        .init_sens(|_p, _t, y| y[0] = 0.1, |_p, _t, _v, y| y[0] = 0.0, 1)
        .build()
        .unwrap()
}
