use crate::{Scalar, vector::Vector, nonlinear_solver::{NonLinearSolver, self, newton::NewtonNonlinearSolver}, IndexType, callable::{ode::BdfCallable, Callable}, matrix::Matrix};

use super::OdeSolverState;



pub struct Bdf<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> {
    state: &'a OdeSolverState<T, V, M, C>,
    nlinsol: Box<dyn NonLinearSolver<T, V, BdfCallable<C, T, V, M>>>,
    t: T,
    order: IndexType,
    h: T,
    n_equal_steps: IndexType,
    diff: V,
    kappa: T,
    gamma: T,
    alpha: T,
    c: T,
    error_const: T,
    psi: T,
}

// implement OdeSolverMethod for Bdf

impl<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> Bdf<'a, T, V, M, C> {
    const MAX_ORDER: IndexType = 5;
    const NEWTON_MAXITER: IndexType = 4;
    const ROOT_SOLVE_MAXITER: IndexType = 15;
    const MIN_FACTOR: T = T::from(0.2);
    const MAX_FACTOR: T = T::from(10.0);

    fn new(state: &'a OdeSolverState<T, V, M, C>) -> Self {
        let nstates = callable.nstates();
        let order = 1;
        let nonlinear_solver = NewtonNonlinearSolver::new(Self::NEWTON_MAXITER, Self::ROOT_SOLVE_MAXITER, Self::MIN_FACTOR, Self::MAX_FACTOR);
        let h = Self::select_initial_step(atol, rtol, callable, t0, y0.clone(), f0, h0);
        let n_equal_steps = 0;
        let diff = V::zeros(nstates);
        let kappa = T::from(1.0);
        let gamma = T::from(1.0);
        let alpha = T::from(1.0);
        let c = T::from(1.0);
        let error_const = T::from(1.0);
        let psi = T::from(1.0);
        let ode_callable = BdfCallable::new(callable, mass, psi, c);
        let nlinsol = nlinsol.new(ode_callable, y0.clone(), p, t0);
        Self { t: t0, nlinsol, order, h, n_equal_steps, diff, y0, kappa, gamma, alpha, c, error_const, psi }
    }

    fn select_initial_step(atol: T, rtol: T, fun: C, t0: T, y0: &V, f0: &V, h0: T) -> T {
        let scale = atol + jnp.abs(y0) * rtol;
        let y1 = y0 + h0 * f0;
        let f1 = fun.call(y1, t0 + h0);
        let d2 = jnp.sqrt(jnp.mean(((f1 - f0) / scale) ** 2));
        let order = 1;
        let h1 = h0 * d2 ** (-1 / (order + 1));
        jnp.minimum(100 * h0, h1)
    }

    pub fn step(&mut self, t: T) -> T {
    }
}