use crate::{
    linear_solver::NalgebraLU, op::bdf::BdfCallable, Bdf, NewtonNonlinearSolver, NonLinearSolver,
    OdeEquations, Scalar, VectorRef,
};

use nalgebra::{DMatrix, DVector};

impl<T: Scalar, Eqn: OdeEquations<T = T, V = DVector<T>, M = DMatrix<T>> + 'static> Default
    for Bdf<DMatrix<T>, Eqn>
{
    fn default() -> Self {
        let n = 1;
        let linear_solver = NalgebraLU::default();
        let mut nonlinear_solver = Box::new(NewtonNonlinearSolver::<BdfCallable<Eqn>>::new(
            linear_solver,
        ));
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        Self {
            ode_problem: None,
            nonlinear_solver,
            order: 1,
            n_equal_steps: 0,
            diff: DMatrix::<T>::zeros(n, Self::MAX_ORDER + 3),
            diff_tmp: DMatrix::<T>::zeros(n, Self::MAX_ORDER + 3),
            gamma: vec![T::from(1.0); Self::MAX_ORDER + 1],
            alpha: vec![T::from(1.0); Self::MAX_ORDER + 1],
            error_const: vec![T::from(1.0); Self::MAX_ORDER + 1],
            u: DMatrix::<T>::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
            statistics: super::BdfStatistics::default(),
            state: None,
        }
    }
}

// implement clone for bdf
impl<T: Scalar, Eqn: OdeEquations<T = T, V = DVector<T>, M = DMatrix<T>> + 'static> Clone
    for Bdf<DMatrix<T>, Eqn>
where
    for<'b> &'b DVector<T>: VectorRef<DVector<T>>,
{
    fn clone(&self) -> Self {
        let n = self.diff.nrows();
        let linear_solver = NalgebraLU::default();
        let mut nonlinear_solver = Box::new(NewtonNonlinearSolver::<BdfCallable<Eqn>>::new(
            linear_solver,
        ));
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        Self {
            ode_problem: self.ode_problem.clone(),
            nonlinear_solver,
            order: self.order,
            n_equal_steps: self.n_equal_steps,
            diff: DMatrix::zeros(n, Self::MAX_ORDER + 3),
            diff_tmp: DMatrix::zeros(n, Self::MAX_ORDER + 3),
            gamma: self.gamma.clone(),
            alpha: self.alpha.clone(),
            error_const: self.error_const.clone(),
            u: DMatrix::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
            statistics: self.statistics.clone(),
            state: self.state.clone(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::tests::{test_interpolate, test_no_set_problem, test_take_state},
        Bdf,
    };

    type M = nalgebra::DMatrix<f64>;
    #[test]
    fn bdf_no_set_problem() {
        test_no_set_problem::<M, _>(Bdf::default())
    }
    #[test]
    fn bdf_take_state() {
        test_take_state::<M, _>(Bdf::default())
    }
    #[test]
    fn bdf_test_interpolate() {
        test_interpolate::<M, _>(Bdf::default())
    }
}
