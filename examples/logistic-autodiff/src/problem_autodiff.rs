use super::{C, M, T, V};
use diffsol::{OdeBuilder, OdeEquationsImplicitAdjoint, OdeSolverProblem};

pub fn problem_autodiff(
    r: f64,
    k: f64,
    y0: f64,
) -> OdeSolverProblem<impl OdeEquationsImplicitAdjoint<M = M, V = V, T = T, C = C>> {
    OdeBuilder::<M>::new()
        .p([r, k, y0])
        .rhs_autodiff(|x: &[T], p: &[T], _t, y: &mut [T]| {
            y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
        })
        .init_autodiff(
            |p: &[T], _t, y: &mut [T]| {
                y[0] = p[2];
            },
            1,
        )
        .build()
        .unwrap()
}
