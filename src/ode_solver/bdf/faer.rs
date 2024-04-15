use faer::{Col, Mat};

use crate::{
    linear_solver::FaerLU, op::bdf::BdfCallable, Bdf, NewtonNonlinearSolver, NonLinearSolver,
    OdeEquations, Scalar, VectorRef,
};

impl<T: Scalar, Eqn: OdeEquations<T = T, V = Col<T>, M = Mat<T>> + 'static> Default
    for Bdf<Mat<T>, Eqn>
{
    fn default() -> Self {
        let n = 1;
        let linear_solver = FaerLU::default();
        let mut nonlinear_solver = Box::new(NewtonNonlinearSolver::<BdfCallable<Eqn>>::new(
            linear_solver,
        ));
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        Self {
            ode_problem: None,
            nonlinear_solver,
            order: 1,
            n_equal_steps: 0,
            diff: Mat::zeros(n, Self::MAX_ORDER + 3),
            diff_tmp: Mat::zeros(n, Self::MAX_ORDER + 3),
            gamma: vec![T::from(1.0); Self::MAX_ORDER + 1],
            alpha: vec![T::from(1.0); Self::MAX_ORDER + 1],
            error_const: vec![T::from(1.0); Self::MAX_ORDER + 1],
            u: Mat::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
            statistics: super::BdfStatistics::default(),
            state: None,
        }
    }
}

// implement clone for bdf
impl<T: Scalar, Eqn: OdeEquations<T = T, V = Col<T>, M = Mat<T>> + 'static> Clone
    for Bdf<Mat<T>, Eqn>
where
    for<'b> &'b Col<T>: VectorRef<Col<T>>,
{
    fn clone(&self) -> Self {
        let n = self.diff.nrows();
        let linear_solver = FaerLU::default();
        let mut nonlinear_solver = Box::new(NewtonNonlinearSolver::<BdfCallable<Eqn>>::new(
            linear_solver,
        ));
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        Self {
            ode_problem: self.ode_problem.clone(),
            nonlinear_solver,
            order: self.order,
            n_equal_steps: self.n_equal_steps,
            diff: Mat::zeros(n, Self::MAX_ORDER + 3),
            diff_tmp: Mat::zeros(n, Self::MAX_ORDER + 3),
            gamma: self.gamma.clone(),
            alpha: self.alpha.clone(),
            error_const: self.error_const.clone(),
            u: Mat::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
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

    type M = faer::Mat<f64>;
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
