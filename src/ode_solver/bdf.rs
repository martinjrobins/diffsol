use std::rc::Rc;
use std::{ops::AddAssign, ops::MulAssign, panic};

use anyhow::{anyhow, Result};

use num_traits::{abs, One, Pow, Zero};
use serde::Serialize;

use crate::{
    matrix::{default_solver::DefaultSolver, Matrix, MatrixRef},
    nonlinear_solver::root::RootFinder,
    op::bdf::BdfCallable,
    scalar::scale,
    vector::DefaultDenseMatrix,
    DenseMatrix, IndexType, MatrixViewMut, NewtonNonlinearSolver, NonLinearSolver, OdeSolverMethod,
    OdeSolverProblem, OdeSolverState, OdeSolverStopReason, Op, Scalar, SolverProblem, Vector,
    VectorRef, VectorView, VectorViewMut,
};

use super::equations::OdeEquations;

#[derive(Clone, Debug, Serialize)]
pub struct BdfStatistics<T: Scalar> {
    pub number_of_linear_solver_setups: usize,
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
            number_of_linear_solver_setups: 0,
            number_of_steps: 0,
            number_of_error_test_failures: 0,
            number_of_nonlinear_solver_iterations: 0,
            number_of_nonlinear_solver_fails: 0,
            initial_step_size: T::zero(),
            final_step_size: T::zero(),
        }
    }
}

/// Implements a Backward Difference formula (BDF) implicit multistep integrator.
/// The basic algorithm is derived in \[1\]. This
/// particular implementation follows that implemented in the Matlab routine ode15s
/// described in \[2\] and the SciPy implementation
/// /[3/], which features the NDF formulas for improved
/// stability with associated differences in the error constants, and calculates
/// the jacobian at J(t_{n+1}, y^0_{n+1}). This implementation was based on that
/// implemented in the SciPy library \[3\], which also mainly
/// follows \[2\] but uses the more standard Jacobian update.
///
/// # References
///
/// \[1\] Byrne, G. D., & Hindmarsh, A. C. (1975). A polyalgorithm for the numerical solution of ordinary differential equations. ACM Transactions on Mathematical Software (TOMS), 1(1), 71-96.
/// \[2\] Shampine, L. F., & Reichelt, M. W. (1997). The matlab ode suite. SIAM journal on scientific computing, 18(1), 1-22.
/// \[3\] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & Van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.
pub struct Bdf<
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
> {
    nonlinear_solver: Nls,
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
    state: Option<OdeSolverState<Eqn::V>>,
    tstop: Option<Eqn::T>,
    root_finder: Option<RootFinder<Eqn::V>>,
    is_state_modified: bool,
}

impl<Eqn> Default
    for Bdf<
        <Eqn::V as DefaultDenseMatrix>::M,
        Eqn,
        NewtonNonlinearSolver<BdfCallable<Eqn>, <Eqn::M as DefaultSolver>::LS<BdfCallable<Eqn>>>,
    >
where
    Eqn: OdeEquations,
    Eqn::M: DefaultSolver,
    Eqn::V: DefaultDenseMatrix,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn default() -> Self {
        let n = 1;
        let linear_solver = Eqn::M::default_solver();
        let mut nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        type M<V> = <V as DefaultDenseMatrix>::M;

        // kappa values for difference orders, taken from Table 1 of [1]
        let kappa = [
            Eqn::T::from(0.0),
            Eqn::T::from(-0.1850),
            Eqn::T::from(-1.0) / Eqn::T::from(9.0),
            Eqn::T::from(-0.0823),
            Eqn::T::from(-0.0415),
            Eqn::T::from(0.0),
        ];
        let mut alpha = vec![Eqn::T::zero()];
        let mut gamma = vec![Eqn::T::zero()];
        let mut error_const = vec![Eqn::T::one()];

        #[allow(clippy::needless_range_loop)]
        for i in 1..=Self::MAX_ORDER {
            let i_t = Eqn::T::from(i as f64);
            let one_over_i = Eqn::T::one() / i_t;
            let one_over_i_plus_one = Eqn::T::one() / (i_t + Eqn::T::one());
            gamma.push(gamma[i - 1] + one_over_i);
            alpha.push(Eqn::T::one() / ((Eqn::T::one() - kappa[i]) * gamma[i]));
            error_const.push(kappa[i] * gamma[i] + one_over_i_plus_one);
        }

        Self {
            ode_problem: None,
            nonlinear_solver,
            order: 1,
            n_equal_steps: 0,
            diff: <M<Eqn::V> as Matrix>::zeros(n, Self::MAX_ORDER + 3), //DMatrix::<T>::zeros(n, Self::MAX_ORDER + 3),
            diff_tmp: <M<Eqn::V> as Matrix>::zeros(n, Self::MAX_ORDER + 3),
            gamma,
            alpha,
            error_const,
            u: <M<Eqn::V> as Matrix>::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
            statistics: BdfStatistics::default(),
            state: None,
            tstop: None,
            root_finder: None,
            is_state_modified: false,
        }
    }
}

impl<M: DenseMatrix<T = Eqn::T, V = Eqn::V>, Eqn: OdeEquations, Nls> Bdf<M, Eqn, Nls>
where
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
{
    const MAX_ORDER: IndexType = 5;
    const NEWTON_MAXITER: IndexType = 4;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn get_statistics(&self) -> &BdfStatistics<Eqn::T> {
        &self.statistics
    }

    fn nonlinear_problem_op(&self) -> &Rc<BdfCallable<Eqn>> {
        &self.nonlinear_solver.problem().f
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

    fn _update_step_size(&mut self, factor: Eqn::T) {
        //If step size h is changed then also need to update the terms in
        //the first equation of page 9 of [1]:
        //
        //- constant c = h / (1-kappa) gamma_k term
        //- lu factorisation of (M - c * J) used in newton iteration (same equation)

        self.state.as_mut().unwrap().h *= factor;
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
            .set_c(self.state.as_ref().unwrap().h, self.alpha[self.order]);

        // reset nonlinear's linear solver problem as lu factorisation has changed
        // use any x and t as they won't be used
        let t = self.state.as_ref().unwrap().t;
        let x = &self.state.as_ref().unwrap().y;
        self.nonlinear_solver.reset_jacobian(x, t);
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

    fn _predict_forward(&mut self) -> (Eqn::V, Eqn::T) {
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
            // update psi term as defined in second equation on page 9 of [1]
            let mut new_psi = self.diff.column(1) * scale(self.gamma[1]);
            for (i, &gamma_i) in self.gamma.iter().enumerate().take(self.order + 1).skip(2) {
                new_psi += self.diff.column(i) * scale(gamma_i)
            }
            new_psi *= scale(self.alpha[self.order]);

            self.nonlinear_problem_op()
                .set_psi_and_y0(new_psi, &y_predict);
        }

        // update time
        let t_new = {
            let state = self.state.as_ref().unwrap();
            state.t + state.h
        };
        (y_predict, t_new)
    }

    fn handle_tstop(&mut self, tstop: Eqn::T) -> Result<Option<OdeSolverStopReason<Eqn::T>>> {
        // check if the we are at tstop
        let state = self.state.as_ref().unwrap();
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if tstop < state.t - troundoff {
            self.tstop = None;
            return Err(anyhow!("tstop is before current time"));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if state.t + state.h > tstop + troundoff {
            let factor = (tstop - state.t) / state.h;
            self._update_step_size(factor);
        }
        Ok(None)
    }

    fn initialise_to_first_order(&mut self) {
        if self.state.as_ref().unwrap().y.len() != self.problem().unwrap().eqn.rhs().nstates() {
            panic!("State vector length does not match number of states in problem");
        }
        let nstates = self.ode_problem.as_ref().unwrap().eqn.rhs().nstates();
        let state = self.state.as_ref().unwrap();
        self.order = 1usize;
        self.n_equal_steps = 0;
        if self.diff.nrows() != nstates {
            self.diff = M::zeros(nstates, Self::MAX_ORDER + 3);
            self.diff_tmp = M::zeros(nstates, Self::MAX_ORDER + 3);
        }
        self.diff.column_mut(0).copy_from(&state.y);
        self.diff.column_mut(1).copy_from(&state.dy);
        self.diff.column_mut(1).mul_assign(scale(state.h));

        // setup U
        self.u = Self::_compute_r(self.order, Eqn::T::one());

        // update statistics
        self.statistics.initial_step_size = state.h;

        self.is_state_modified = false;
    }
}

impl<M: DenseMatrix<T = Eqn::T, V = Eqn::V>, Eqn: OdeEquations, Nls> OdeSolverMethod<Eqn>
    for Bdf<M, Eqn, Nls>
where
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn order(&self) -> usize {
        self.order
    }

    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V> {
        //interpolate solution at time values t* where t-h < t* < t
        //
        //definition of the interpolating polynomial can be found on page 7 of [1]

        // state must be set
        let state = self.state.as_ref().ok_or(anyhow!("State not set"))?;

        if self.is_state_modified {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(anyhow::anyhow!("Interpolation time is not within the current step. Step size is zero after calling state_mut()"));
            }
        }

        // check that t is before the current time
        if t > state.t {
            return Err(anyhow!("Interpolation time is after current time"));
        }

        let mut time_factor = Eqn::T::from(1.0);
        let mut order_summation = self.diff.column(0).into_owned();
        for i in 0..self.order {
            let i_t = Eqn::T::from(i as f64);
            time_factor *= (t - (state.t - state.h * i_t)) / (state.h * (Eqn::T::one() + i_t));
            order_summation += self.diff.column(i + 1) * scale(time_factor);
        }
        Ok(order_summation)
    }

    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.ode_problem.as_ref()
    }

    fn state(&self) -> Option<&OdeSolverState<Eqn::V>> {
        self.state.as_ref()
    }
    fn take_state(&mut self) -> Option<OdeSolverState<Eqn::V>> {
        Option::take(&mut self.state)
    }

    fn state_mut(&mut self) -> Option<&mut OdeSolverState<Eqn::V>> {
        self.is_state_modified = true;
        self.state.as_mut()
    }

    fn set_problem(&mut self, state: OdeSolverState<Eqn::V>, problem: &OdeSolverProblem<Eqn>) {
        self.ode_problem = Some(problem.clone());

        // setup linear solver for first step
        let bdf_callable = Rc::new(BdfCallable::new(problem));
        bdf_callable.set_c(state.h, self.alpha[self.order]);

        let nonlinear_problem = SolverProblem::new_from_ode_problem(bdf_callable, problem);
        self.nonlinear_solver.set_problem(&nonlinear_problem);

        // store state and setup root solver
        self.state = Some(state);
        if let Some(root_fn) = problem.eqn.root() {
            let state = self.state.as_ref().unwrap();
            self.root_finder = Some(RootFinder::new(root_fn.nout()));
            self.root_finder
                .as_ref()
                .unwrap()
                .init(root_fn.as_ref(), &state.y, state.t);
        }

        // initialise solver to first order
        self.initialise_to_first_order();
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>> {
        let mut d: Eqn::V;
        let mut safety: Eqn::T;
        let mut error_norm: Eqn::T;
        let mut scale_y: Eqn::V;
        let mut updated_jacobian = false;
        if self.state.is_none() {
            return Err(anyhow!("State not set"));
        }

        if self.is_state_modified {
            self.initialise_to_first_order();
        }

        let (mut y_predict, mut t_new) = self._predict_forward();

        // loop until step is accepted
        let y_new = loop {
            let mut y_new = y_predict.clone();

            // solve BDF equation using y0 as starting point
            let solver_result = self.nonlinear_solver.solve_in_place(&mut y_new, t_new);
            // update statistics
            self.statistics.number_of_nonlinear_solver_iterations += self.nonlinear_solver.niter();
            match solver_result {
                Ok(()) => {
                    // test error is within tolerance
                    {
                        let ode_problem = self.ode_problem.as_ref().unwrap();
                        scale_y = y_new.abs() * scale(ode_problem.rtol);
                        scale_y += ode_problem.atol.as_ref();
                    }

                    // combine eq 3, 4 and 6 from [1] to obtain error
                    // Note that error = C_k * h^{k+1} y^{k+1}
                    // and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
                    d = &y_new - &y_predict;

                    let mut error = &d * scale(self.error_const[self.order]);
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
                        self._update_step_size(factor);

                        // if step size too small, then fail
                        let state = self.state.as_ref().unwrap();
                        if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                            return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
                        }

                        // new prediction
                        (y_predict, t_new) = self._predict_forward();

                        // update statistics
                        self.statistics.number_of_error_test_failures += 1;
                    }
                }
                Err(_e) => {
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    if updated_jacobian {
                        // newton iteration did not converge, but jacobian has already been
                        // evaluated so reduce step size by 0.3 (as per [1]) and try again
                        self._update_step_size(Eqn::T::from(0.3));

                        // new prediction
                        (y_predict, t_new) = self._predict_forward();

                        // update statistics
                    } else {
                        // newton iteration did not converge, so update jacobian and try again
                        self.nonlinear_problem_op().set_jacobian_is_stale();
                        self.nonlinear_solver
                            .reset_jacobian(&y_predict, self.state.as_ref().unwrap().t);
                        updated_jacobian = true;
                        // same prediction as last time
                    }
                }
            };
        };

        // take the accepted step
        {
            let state = self.state.as_mut().unwrap();
            state.y = y_new;
            state.t += state.h;
        }

        // update statistics
        self.statistics.number_of_linear_solver_setups =
            self.nonlinear_problem_op().number_of_jac_evals();
        self.statistics.number_of_steps += 1;
        self.statistics.final_step_size = self.state.as_ref().unwrap().h;

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
                let mut error_m = self.diff.column(order) * scale(self.error_const[order - 1]);
                error_m.component_div_assign(&scale_y);
                error_m.norm()
            } else {
                Eqn::T::INFINITY
            };
            let error_p_norm = if order < Self::MAX_ORDER {
                let mut error_p = self.diff.column(order + 2) * scale(self.error_const[order + 1]);
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
            self._update_step_size(factor);
        }

        // check for root within accepted step
        if let Some(root_fn) = self.problem().as_ref().unwrap().eqn.root() {
            let ret = self.root_finder.as_ref().unwrap().check_root(
                &|t| self.interpolate(t),
                root_fn.as_ref(),
                &self.state.as_ref().unwrap().y,
                self.state.as_ref().unwrap().t,
            );
            if let Some(root) = ret {
                return Ok(OdeSolverStopReason::RootFound(root));
            }
        }

        if let Some(tstop) = self.tstop {
            if let Some(reason) = self.handle_tstop(tstop).unwrap() {
                return Ok(reason);
            }
        }

        // just a normal step, no roots or tstop reached
        Ok(OdeSolverStopReason::InternalTimestep)
    }

    fn set_stop_time(&mut self, tstop: <Eqn as OdeEquations>::T) -> Result<()> {
        self.tstop = Some(tstop);
        if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
            self.tstop = None;
            return Err(anyhow!(
                "tstop is at or before current time t = {}",
                self.state.as_ref().unwrap().t
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::{
            test_models::dydt_y2::dydt_y2_problem,
            test_models::exponential_decay::exponential_decay_problem,
            test_models::exponential_decay::exponential_decay_problem_with_root,
            test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem,
            test_models::gaussian_decay::gaussian_decay_problem,
            test_models::robertson::robertson,
            test_models::robertson_ode::robertson_ode,
            tests::{
                test_interpolate, test_no_set_problem, test_ode_solver, test_state_mut,
                test_state_mut_on_problem,
            },
        },
        Bdf, FaerLU, NalgebraLU, NewtonNonlinearSolver, OdeEquations, Op,
    };

    use num_traits::abs;

    type M = nalgebra::DMatrix<f64>;
    #[test]
    fn bdf_no_set_problem() {
        test_no_set_problem::<M, _>(Bdf::default())
    }
    #[test]
    fn bdf_state_mut() {
        test_state_mut::<M, _>(Bdf::default())
    }
    #[test]
    fn bdf_test_interpolate() {
        test_interpolate::<M, _>(Bdf::default())
    }

    #[test]
    fn bdf_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = Bdf::default();
        test_state_mut_on_problem(s, p, soln);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 8
        number_of_steps: 25
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 50
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.001189207115002721
        final_step_size: 0.9861196765479318
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 52
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn bdf_test_faer_exponential_decay() {
        type M = faer::Mat<f64>;
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(FaerLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 8
        number_of_steps: 25
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 50
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.001189207115002721
        final_step_size: 0.9861196765889989
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 52
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = exponential_decay_with_algebraic_problem::<M>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 11
        number_of_steps: 17
        number_of_error_test_failures: 4
        number_of_nonlinear_solver_iterations: 42
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.00014907855910877986
        final_step_size: 0.2008052778053449
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 46
        number_of_jac_muls: 4
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = robertson::<M>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 103
        number_of_steps: 352
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 1003
        number_of_nonlinear_solver_fails: 21
        initial_step_size: 0.000000005427827356796531
        final_step_size: 5943224095.574959
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1006
        number_of_jac_muls: 55
        number_of_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = robertson::<M>(true);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 103
        number_of_steps: 352
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 1003
        number_of_nonlinear_solver_fails: 21
        initial_step_size: 0.000000005427827356796531
        final_step_size: 5943224095.574959
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1006
        number_of_jac_muls: 58
        number_of_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = robertson_ode::<M>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 94
        number_of_steps: 340
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 950
        number_of_nonlinear_solver_fails: 15
        initial_step_size: 0.000000004564240566951627
        final_step_size: 6155729544.745563
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 952
        number_of_jac_muls: 48
        number_of_matrix_evals: 16
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = dydt_y2_problem::<M>(false, 10);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 45
        number_of_steps: 192
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 538
        number_of_nonlinear_solver_fails: 3
        initial_step_size: 0.00000028403960645516395
        final_step_size: 1.0749050435964294
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 540
        number_of_jac_muls: 40
        number_of_matrix_evals: 4
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = dydt_y2_problem::<M>(true, 10);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 45
        number_of_steps: 192
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 538
        number_of_nonlinear_solver_fails: 3
        initial_step_size: 0.00000028403960645516395
        final_step_size: 1.0749050435964294
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 540
        number_of_jac_muls: 14
        number_of_matrix_evals: 4
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = gaussian_decay_problem::<M>(false, 10);
        test_ode_solver(&mut s, rs, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 16
        number_of_steps: 60
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 165
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.00009999999999999999
        final_step_size: 0.19565537798887184
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 167
        number_of_jac_muls: 10
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_tstop_bdf() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, rs, &problem, soln, None, true);
    }

    #[test]
    fn test_root_finder_bdf() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::new(NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let y = test_ode_solver(&mut s, rs, &problem, soln, None, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }
}
