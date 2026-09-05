use crate::ode_solver::problem::OdeSolverSolution;
use crate::OdeSolverProblem;
use crate::{
    scalar::Scalar, ConstantOp, DenseMatrix, OdeBuilder, OdeEquations, OdeEquationsImplicit, Vector,
};
use num_traits::{FromPrimitive, Pow};
use num_traits::{One, Zero};

// dy/dt = -aty (p = [a])
fn gaussian_decay<M: DenseMatrix>(x: &[M::T], p: &[M::T], t: M::T, y: &mut [M::T]) {
    for ((y, x), p) in y.iter_mut().zip(x.iter()).zip(p.iter()) {
        *y = -t * *p * *x;
    }
}

// Jv = -atv
fn gaussian_decay_jacobian<M: DenseMatrix>(
    _x: &[M::T],
    p: &[M::T],
    t: M::T,
    v: &[M::T],
    y: &mut [M::T],
) {
    for ((y, p), v) in y.iter_mut().zip(p.iter()).zip(v.iter()) {
        *y = -t * *p * *v;
    }
}

#[allow(clippy::type_complexity)]
pub fn gaussian_decay_problem<M: DenseMatrix + 'static>(
    use_coloring: bool,
    size: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.1].repeat(size))
        .use_coloring(use_coloring)
        .rhs_implicit(gaussian_decay::<M>, gaussian_decay_jacobian::<M>)
        .init(move |_p, _t, y| y.fill(M::T::one()), size)
        .build()
        .unwrap();
    let p = [M::T::from_f64(0.1).unwrap()].repeat(size);
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from_f64(i as f64 / 1.0).unwrap();
        let mut y: M::V = problem.eqn.init().call(M::T::zero());
        let px = M::V::from_vec(
            p.iter()
                .map(|&x| (x * t.pow(2) / M::T::from_f64(-2.0).unwrap()).exp())
                .collect::<Vec<_>>(),
            problem.context().clone(),
        );
        y.component_mul_assign(&px);
        soln.push(y, t);
    }
    (problem, soln)
}
