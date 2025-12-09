use crate::{
    error::{DiffsolError, NonLinearSolverError},
    non_linear_solver_error,
    nonlinear_solver::convergence::ConvergenceStatus,
    Convergence, Scalar, Scale, Vector,
};
use num_traits::{FromPrimitive, One, Pow};

/// Line search trait for nonlinear solvers
/// The line search is used to find an optimal step size for the Newton iteration.
/// The line search modifies the delta vector in place to scale it by the optimal step size
/// The line search returns the norm of the modified delta vector
/// The x vector is also modified in place to take the optimal step
pub trait LineSearch<V: Vector>: Default {
    /// Take the optimal step for the current iteration
    fn take_optimal_step(
        &mut self,
        x: &mut V,
        norm: &mut V::T,
        delta: &mut V,
        error_y: &V,
        fun: &impl Fn(&V, &mut V),
        linear_solver: &impl Fn(&mut V) -> Result<(), DiffsolError>,
        convergence: &mut Convergence<V>,
    ) -> Result<(), DiffsolError>;

    /// Reset the line search state
    fn reset(&mut self);
}

#[derive(Default)]
pub struct NoLineSearch;

impl<V: Vector> LineSearch<V> for NoLineSearch {
    /// No line search implementation, simply takes the full step
    fn take_optimal_step(
        &mut self,
        x: &mut V,
        norm: &mut V::T,
        delta: &mut V,
        error_y: &V,
        fun: &impl Fn(&V, &mut V),
        linear_solver: &impl Fn(&mut V) -> Result<(), DiffsolError>,
        convergence: &mut Convergence<V>,
    ) -> Result<(), DiffsolError> {
        //delta = f_at_n
        fun(x, delta);

        //delta = -delta_n
        linear_solver(delta)?;

        // xn = xn + delta_n
        x.sub_assign(&*delta);

        // norm
        *norm = convergence.norm(delta, error_y);
        Ok(())
    }

    fn reset(&mut self) {}
}

///
/// Backtracking line search implementation
///
/// Parameters:
/// - tau: step size reduction factor (0 < tau < 1), default 0.5
/// - c: Armijo condition constant (0 < c < 1), default 1e-4
/// - steptol: minimum step size, default eps^(2/3)
/// - max_iter: maximum number of line search iterations, default 100
/// - n_iters: number of line search iterations performed during last call
///
pub struct BacktrackingLineSearch<V: Vector> {
    pub tau: V::T,
    pub c: V::T,
    pub steptol: V::T,
    pub max_iter: usize,
    pub n_iters: usize,
    delta0: V,
    x0: V,
}

impl<V: Vector> Default for BacktrackingLineSearch<V> {
    fn default() -> Self {
        Self {
            tau: V::T::from_f64(0.5).unwrap(),
            c: V::T::from_f64(1e-4).unwrap(),
            steptol: V::T::EPSILON.pow(V::T::from_f64(2.0 / 3.0).unwrap()),
            max_iter: 100,
            n_iters: 0,
            delta0: V::zeros(0, Default::default()),
            x0: V::zeros(0, Default::default()),
        }
    }
}

/// Backtracking line search implementation, derived from backtracking line search algorithm
/// in Sundials IDA solver (https://github.com/LLNL/sundials/blob/main/src/ida/ida_ic.c#L480)
impl<V: Vector> LineSearch<V> for BacktrackingLineSearch<V> {
    fn reset(&mut self) {
        self.n_iters = 0;
    }
    fn take_optimal_step(
        &mut self,
        x: &mut V,
        rnorm: &mut V::T,
        delta: &mut V,
        error_y: &V,
        fun: &impl Fn(&V, &mut V),
        linear_solver: &impl Fn(&mut V) -> Result<(), DiffsolError>,
        convergence: &mut Convergence<V>,
    ) -> Result<(), DiffsolError> {
        // on the first iteration, we need to init delta and norm
        if convergence.niter() == 0 {
            //delta = f_at_n
            fun(x, delta);

            //delta = -delta_n
            linear_solver(delta)?;

            *rnorm = convergence.norm(delta, error_y);

            // if we've already converged, take the step and return
            if let ConvergenceStatus::Converged = convergence.check_norm(*rnorm) {
                x.sub_assign(&*delta);
                return Ok(());
            }
        }

        if self.x0.len() == 0 {
            self.x0 = V::zeros(x.len(), x.context().clone());
            self.delta0 = V::zeros(delta.len(), delta.context().clone());
        }
        self.x0.copy_from(x);
        self.delta0.copy_from(delta);
        let norm = *rnorm;
        let half = V::T::from_f64(0.5).unwrap();

        // backtracking line search on phi = 0.5 ||F||^2
        let phi0 = norm * norm * half;
        let two_phi0 = norm * norm;
        let min_alpha = self.steptol / norm;
        let mut alpha = V::T::one();
        for i in 0..self.max_iter {
            // take the step and recompute the norm
            x.axpy(-alpha, &self.delta0, V::T::one());
            // xi = x0 + alpha * delta_n

            fun(x, delta);
            //delta_p = f_at_n

            linear_solver(delta)?;
            //delta_p = -delta_n

            let new_norm = convergence.norm(delta, error_y);
            self.n_iters = i;

            // directional derivative for phi along p is: grad_phi^T p = F^T J p = -||F||^2
            // so the Armijo condition reduces to phi(u+α p) <= phi0 - c * α * ||F||^2``
            let phi1 = new_norm * new_norm * half;
            if phi1 <= phi0 - self.c * alpha * two_phi0 {
                *rnorm = new_norm;
                return Ok(());
            }
            alpha *= self.tau;
            if alpha < min_alpha {
                return Err(non_linear_solver_error!(LinesearchFailedMinStep));
            }

            // reset x
            x.copy_from(&self.x0);
        }
        Err(non_linear_solver_error!(LinesearchFailedMaxIterations))
    }
}
