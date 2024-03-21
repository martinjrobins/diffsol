use std::ops::AddAssign;
use std::rc::Rc;

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use num_traits::{One, Pow, Zero};
use serde::Serialize;

use crate::{
    matrix::MatrixRef, op::ode::BdfCallable, DenseMatrix, IndexType, MatrixViewMut,
    NewtonNonlinearSolver, NonLinearSolver, Scalar, SolverProblem, Vector, VectorRef, VectorView,
    VectorViewMut, LU,
};

use super::{equations::OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState};

#[derive(Clone, Debug, Serialize)]
pub struct BdfStatistics<T: Scalar> {
    pub number_of_rhs_jac_evals: usize,
    pub number_of_rhs_evals: usize,
    pub number_of_linear_solver_setups: usize,
    pub number_of_jac_mul_evals: usize,
    pub number_of_steps: usize,
    pub number_of_error_test_failures: usize,
    pub number_of_nonlinear_solver_iterations: usize,
    pub number_of_nonlinear_solver_fails: usize,
    pub initial_step_size: T,
    pub final_step_size: T,
}

impl<T: Scalar> Default for BdfStatistics<T> {
    fn default() -> Self {
        Self {
            number_of_rhs_jac_evals: 0,
            number_of_rhs_evals: 0,
            number_of_linear_solver_setups: 0,
            number_of_jac_mul_evals: 0,
            number_of_steps: 0,
            number_of_error_test_failures: 0,
            number_of_nonlinear_solver_iterations: 0,
            number_of_nonlinear_solver_fails: 0,
            initial_step_size: T::zero(),
            final_step_size: T::zero(),
        }
    }
}

pub struct Bdf<M: DenseMatrix<T = Eqn::T, V = Eqn::V>, Eqn: OdeEquations> {
    nonlinear_solver: Box<dyn NonLinearSolver<BdfCallable<Eqn>>>,
    ode_problem: Option<OdeSolverProblem<Eqn>>,
    order: usize,
    n_equal_steps: usize,
    diff: M,
    diff_tmp: M,
    u: M,
    alpha: Vec<Eqn::T>,
    gamma: Vec<Eqn::T>,
    error_const: Vec<Eqn::T>,
    statistics: BdfStatistics<Eqn::T>,
}

impl<T: Scalar, Eqn: OdeEquations<T = T, V = DVector<T>, M = DMatrix<T>> + 'static> Default
    for Bdf<DMatrix<T>, Eqn>
{
    fn default() -> Self {
        let n = 1;
        let linear_solver = LU::default();
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
            statistics: BdfStatistics::default(),
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
        let linear_solver = LU::default();
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
        }
    }
}

impl<M: DenseMatrix<T = Eqn::T, V = Eqn::V>, Eqn: OdeEquations> Bdf<M, Eqn>
where
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    const MAX_ORDER: IndexType = 5;
    const NEWTON_MAXITER: IndexType = 4;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn get_statistics(&self) -> &BdfStatistics<Eqn::T> {
        &self.statistics
    }

    fn nonlinear_problem_op(&self) -> Option<&Rc<BdfCallable<Eqn>>> {
        Some(&self.nonlinear_solver.as_ref().problem()?.f)
    }

    fn _compute_r(order: usize, factor: Eqn::T) -> M {
        //computes the R matrix with entries
        //given by the first equation on page 8 of [1]
        //
        //This is used to update the differences matrix when step size h is varied
        //according to factor = h_{n+1} / h_n
        //
        //Note that the U matrix also defined in the same section can be also be
        //found using factor = 1, which corresponds to R with a constant step size
        let mut r = M::zeros(order + 1, order + 1);

        // r[0, 0:order] = 1
        for j in 0..=order {
            r[(0, j)] = M::T::one();
        }
        // r[i, j] = r[i, j-1] * (j - 1 - factor * i) / j
        for i in 1..=order {
            for j in 1..=order {
                let i_t = M::T::from(i as f64);
                let j_t = M::T::from(j as f64);
                r[(i, j)] = r[(i - 1, j)] * (i_t - M::T::one() - factor * j_t) / i_t;
            }
        }
        r
    }

    fn _update_step_size(&mut self, factor: Eqn::T, state: &mut OdeSolverState<Eqn::M>) {
        //If step size h is changed then also need to update the terms in
        //the first equation of page 9 of [1]:
        //
        //- constant c = h / (1-kappa) gamma_k term
        //- lu factorisation of (M - c * J) used in newton iteration (same equation)

        state.h *= factor;
        self.n_equal_steps = 0;

        // update D using equations in section 3.2 of [1]
        self.u = Self::_compute_r(self.order, Eqn::T::one());
        let r = Self::_compute_r(self.order, factor);
        let ru = r.mat_mul(&self.u);
        // D[0:order+1] = R * U * D[0:order+1]
        {
            let d_zero_order = self.diff.columns(0, self.order + 1);
            let mut d_zero_order_tmp = self.diff_tmp.columns_mut(0, self.order + 1);
            d_zero_order_tmp.gemm_vo(Eqn::T::one(), &d_zero_order, &ru, Eqn::T::zero());
            // diff_sub = diff * RU
        }
        std::mem::swap(&mut self.diff, &mut self.diff_tmp);

        self.nonlinear_problem_op()
            .unwrap()
            .set_c(state.h, &self.alpha, self.order);

        // reset nonlinear's linear solver problem as lu factorisation has changed
        self.nonlinear_solver.as_mut().reset();
    }

    fn _update_differences(&mut self, d: &Eqn::V) {
        //update of difference equations can be done efficiently
        //by reusing d and D.
        //
        //From first equation on page 4 of [1]:
        //d = y_n - y^0_n = D^{k + 1} y_n
        //
        //Standard backwards difference gives
        //D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}
        //
        //Combining these gives the following algorithm
        let order = self.order;
        let d_minus_order_plus_one = d - self.diff.column(order + 1);
        self.diff
            .column_mut(order + 2)
            .copy_from(&d_minus_order_plus_one);
        self.diff.column_mut(order + 1).copy_from(d);
        for i in (0..=order).rev() {
            let tmp = self.diff.column(i + 1).into_owned();
            self.diff.column_mut(i).add_assign(&tmp);
        }
    }

    fn _predict_forward(&mut self, state: &OdeSolverState<Eqn::M>) -> Eqn::V {
        let nstates = self.diff.nrows();
        // predict forward to new step (eq 2 in [1])
        let y_predict = {
            let mut y_predict = <Eqn::V as Vector>::zeros(nstates);
            for i in 0..=self.order {
                y_predict += self.diff.column(i);
            }
            y_predict
        };

        // update psi and c (h, D, y0 has changed)
        {
            let callable = self.nonlinear_problem_op().unwrap();
            callable.set_psi_and_y0(&self.diff, &self.gamma, &self.alpha, self.order, &y_predict);
        }

        // update time
        let t_new = state.t + state.h;
        self.nonlinear_solver.as_mut().set_time(t_new).unwrap();
        y_predict
    }
}

impl<M: DenseMatrix<T = Eqn::T, V = Eqn::V>, Eqn: OdeEquations> OdeSolverMethod<Eqn> for Bdf<M, Eqn>
where
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn interpolate(&self, state: &OdeSolverState<Eqn::M>, t: Eqn::T) -> Eqn::V {
        //interpolate solution at time values t* where t-h < t* < t
        //
        //definition of the interpolating polynomial can be found on page 7 of [1]
        let mut time_factor = Eqn::T::from(1.0);
        let mut order_summation = self.diff.column(0).into_owned();
        for i in 0..self.order {
            let i_t = Eqn::T::from(i as f64);
            time_factor *= (t - (state.t - state.h * i_t)) / (state.h * (Eqn::T::one() + i_t));
            order_summation += self.diff.column(i + 1) * time_factor;
        }
        order_summation
    }

    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.ode_problem.as_ref()
    }

    fn set_problem(&mut self, state: &mut OdeSolverState<Eqn::M>, problem: OdeSolverProblem<Eqn>) {
        self.ode_problem = Some(problem);
        let problem = self.ode_problem.as_ref().unwrap();
        let nstates = problem.eqn.nstates();
        self.order = 1usize;
        self.n_equal_steps = 0;
        self.diff = M::zeros(nstates, Self::MAX_ORDER + 3);
        self.diff_tmp = M::zeros(nstates, Self::MAX_ORDER + 3);
        self.diff.column_mut(0).copy_from(&state.y);

        // kappa values for difference orders, taken from Table 1 of [1]
        let kappa = [
            Eqn::T::from(0.0),
            Eqn::T::from(-0.1850),
            Eqn::T::from(-1.0) / Eqn::T::from(9.0),
            Eqn::T::from(-0.0823),
            Eqn::T::from(-0.0415),
            Eqn::T::from(0.0),
        ];
        self.alpha = vec![Eqn::T::zero()];
        self.gamma = vec![Eqn::T::zero()];
        self.error_const = vec![Eqn::T::one()];

        #[allow(clippy::needless_range_loop)]
        for i in 1..=Self::MAX_ORDER {
            let i_t = Eqn::T::from(i as f64);
            let one_over_i = Eqn::T::one() / i_t;
            let one_over_i_plus_one = Eqn::T::one() / (i_t + Eqn::T::one());
            self.gamma.push(self.gamma[i - 1] + one_over_i);
            self.alpha
                .push(Eqn::T::one() / ((Eqn::T::one() - kappa[i]) * self.gamma[i]));
            self.error_const
                .push(kappa[i] * self.gamma[i] + one_over_i_plus_one);
        }

        // update initial step size based on function
        let mut scale = state.y.abs();
        scale *= problem.rtol;
        scale += problem.atol.as_ref();

        let f0 = problem.eqn.rhs(state.t, &state.y);
        let hf0 = &f0 * state.h;
        let y1 = &state.y + &hf0;
        let t1 = state.t + state.h;
        let f1 = problem.eqn.rhs(t1, &y1);

        // store f1 in diff[1] for use in step size control
        self.diff.column_mut(1).copy_from(&hf0);

        let mut df = f1 - f0;
        df.component_div_assign(&scale);
        let d2 = df.norm();

        let one_over_order_plus_one =
            Eqn::T::one() / (Eqn::T::from(self.order as f64) + Eqn::T::one());
        let mut new_h = state.h * d2.pow(-one_over_order_plus_one);
        if new_h > Eqn::T::from(100.0) * state.h {
            new_h = Eqn::T::from(100.0) * state.h;
        }
        state.h = new_h;

        // setup linear solver for first step
        let bdf_callable = Rc::new(BdfCallable::new(problem));
        let nonlinear_problem = SolverProblem::new_from_ode_problem(bdf_callable, problem);
        self.nonlinear_solver
            .as_mut()
            .set_problem(nonlinear_problem);
        let _test = self.nonlinear_problem_op().unwrap();

        // setup U
        self.u = Self::_compute_r(self.order, Eqn::T::one());

        // update statistics
        self.statistics.initial_step_size = state.h;
    }

    fn step(&mut self, state: &mut OdeSolverState<Eqn::M>) -> Result<()> {
        // we will try and use the old jacobian unless convergence of newton iteration
        // fails
        // tells callable to update rhs jacobian if the jacobian is requested (by nonlinear solver)
        // initialise step size and try to make the step,
        // iterate, reducing step size until error is in bounds
        let mut d: Eqn::V;
        let mut safety: Eqn::T;
        let mut error_norm: Eqn::T;
        let mut scale_y: Eqn::V;
        let mut updated_jacobian = false;
        let mut y_predict = self._predict_forward(state);

        // loop until step is accepted
        let y_new = loop {
            let mut y_new = y_predict.clone();

            // solve BDF equation using y0 as starting point
            let solver_result = self.nonlinear_solver.solve_in_place(&mut y_new);
            // update statistics
            self.statistics.number_of_nonlinear_solver_iterations += self.nonlinear_solver.niter();
            match solver_result {
                Ok(()) => {
                    // test error is within tolerance
                    {
                        let ode_problem = self.ode_problem.as_ref().unwrap();
                        scale_y = y_new.abs() * ode_problem.rtol;
                        scale_y += ode_problem.atol.as_ref();
                    }

                    // combine eq 3, 4 and 6 from [1] to obtain error
                    // Note that error = C_k * h^{k+1} y^{k+1}
                    // and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
                    d = &y_new - &y_predict;

                    let mut error = &d * self.error_const[self.order];
                    error.component_div_assign(&scale_y);
                    error_norm = error.norm();
                    let maxiter = self.nonlinear_solver.max_iter() as f64;
                    let niter = self.nonlinear_solver.niter() as f64;
                    safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));

                    if error_norm <= Eqn::T::from(1.0) {
                        // step is accepted
                        break y_new;
                    } else {
                        // step is rejected
                        // calculate optimal step size factor as per eq 2.46 of [2]
                        // and reduce step size and try again
                        let order = self.order as f64;
                        let mut factor =
                            safety * error_norm.pow(Eqn::T::from(-1.0 / (order + 1.0)));
                        if factor < Eqn::T::from(Self::MIN_FACTOR) {
                            factor = Eqn::T::from(Self::MIN_FACTOR);
                        }
                        // todo, do we need to update the linear solver problem here since we converged?
                        self._update_step_size(factor, state);

                        // if step size too small, then fail
                        if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                            return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
                        }

                        // new prediction
                        y_predict = self._predict_forward(state);

                        // update statistics
                        self.statistics.number_of_error_test_failures += 1;
                    }
                }
                Err(_e) => {
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    if updated_jacobian {
                        // newton iteration did not converge, but jacobian has already been
                        // evaluated so reduce step size by 0.3 (as per [1]) and try again
                        self._update_step_size(Eqn::T::from(0.3), state);

                        // new prediction
                        y_predict = self._predict_forward(state);

                        // update statistics
                    } else {
                        // newton iteration did not converge, so update jacobian and try again
                        {
                            let callable = self.nonlinear_problem_op().unwrap();
                            callable.set_rhs_jacobian_is_stale();
                        }
                        self.nonlinear_solver.as_mut().reset();
                        updated_jacobian = true;
                        // same prediction as last time
                    }
                }
            };
        };

        // take the accepted step
        state.t += state.h;
        state.y = y_new;

        // update statistics
        self.statistics.number_of_jac_mul_evals = self
            .nonlinear_problem_op()
            .unwrap()
            .number_of_jac_mul_evals();
        self.statistics.number_of_rhs_evals =
            self.nonlinear_problem_op().unwrap().number_of_rhs_evals();
        self.statistics.number_of_rhs_jac_evals = self
            .nonlinear_problem_op()
            .unwrap()
            .number_of_rhs_jac_evals();
        self.statistics.number_of_linear_solver_setups =
            self.nonlinear_problem_op().unwrap().number_of_jac_evals();
        self.statistics.number_of_steps += 1;
        self.statistics.final_step_size = state.h;

        self._update_differences(&d);

        // a change in order is only done after running at order k for k + 1 steps
        // (see page 83 of [2])
        self.n_equal_steps += 1;

        if self.n_equal_steps > self.order {
            let order = self.order;
            // similar to the optimal step size factor we calculated above for the current
            // order k, we need to calculate the optimal step size factors for orders
            // k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
            let error_m_norm = if order > 1 {
                let mut error_m = self.diff.column(order) * self.error_const[order - 1];
                error_m.component_div_assign(&scale_y);
                error_m.norm()
            } else {
                Eqn::T::INFINITY
            };
            let error_p_norm = if order < Self::MAX_ORDER {
                let mut error_p = self.diff.column(order + 2) * self.error_const[order + 1];
                error_p.component_div_assign(&scale_y);
                error_p.norm()
            } else {
                Eqn::T::INFINITY
            };

            let error_norms = [error_m_norm, error_norm, error_p_norm];
            let factors = error_norms
                .into_iter()
                .enumerate()
                .map(|(i, error_norm)| {
                    error_norm.pow(Eqn::T::from(-1.0 / (i as f64 + order as f64)))
                })
                .collect::<Vec<_>>();

            // now we have the three factors for orders k-1, k and k+1, pick the maximum in
            // order to maximise the resultant step size
            let max_index = factors
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            if max_index == 0 {
                self.order -= 1;
            } else {
                self.order += max_index - 1;
            }

            let mut factor = safety * factors[max_index];
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }
            self._update_step_size(factor, state);
        }
        Ok(())
    }
}
