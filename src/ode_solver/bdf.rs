use nalgebra::ComplexField;
use std::ops::AddAssign;
use std::rc::Rc;
use std::cell::RefCell;

use crate::{
    error::{DiffsolError, OdeSolverError}, AdjointContext, AdjointEquations, SensEquations, StateRef, StateRefMut
};

use num_traits::{abs, One, Pow, Zero};
use serde::Serialize;

use crate::ode_solver_error;
use crate::{
    matrix::{default_solver::DefaultSolver, MatrixRef},
    nonlinear_solver::root::RootFinder,
    op::bdf::BdfCallable,
    scalar::scale,
    vector::DefaultDenseMatrix,
    AugmentedOdeEquations, BdfState, Checkpointing, DenseMatrix, IndexType, InitOp, JacobianUpdate,
    MatrixViewMut, NewtonNonlinearSolver, NonLinearOp, NonLinearSolver, OdeSolverMethod,
    OdeSolverProblem, OdeSolverState, OdeSolverStopReason, Op, Scalar, SolverProblem, Vector,
    VectorRef, VectorView, VectorViewMut,
};

use super::jacobian_update::SolverState;
use super::{
    equations::OdeEquations,
    method::{AdjointOdeSolverMethod, AugmentedOdeSolverMethod, SensitivitiesOdeSolverMethod},
};

#[derive(Clone, Debug, Serialize, Default)]
pub struct BdfStatistics {
    pub number_of_linear_solver_setups: usize,
    pub number_of_steps: usize,
    pub number_of_error_test_failures: usize,
    pub number_of_nonlinear_solver_iterations: usize,
    pub number_of_nonlinear_solver_fails: usize,
}

pub type Bdf<M, Eqn, Nls> = BdfAug<M, Eqn, Nls, SensEquations<Eqn>>;
pub type BdfAdj<M, Eqn, Nls> = BdfAug<
    M,
    AdjointEquations<Eqn, Bdf<M, Eqn, Nls>>,
    Nls,
    AdjointEquations<Eqn, Bdf<M, Eqn, Nls>>,
>;
impl<M, Eqn, Nls> SensitivitiesOdeSolverMethod<Eqn> for Bdf<M, Eqn, Nls>
where
    Eqn: OdeEquations,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
{
}

// notes quadrature.
// ndf formula rearranged to [2]:
// (1 - kappa) * gamma_k * (y_{n+1} - y^0_{n+1}) + (\sum_{m=1}^k gamma_m * y^m_n) - h * F(t_{n+1}, y_{n+1}) = 0 (1)
// where d = y_{n+1} - y^0_{n+1}
// and y^0_{n+1} = \sum_{m=0}^k y^m_n
//
// 1. use (1) to calculate d explicitly
// 2. use d to update the differences matrix
// 3. use d to calculate the predicted solution y_{n+1}

/// Implements a Backward Difference formula (BDF) implicit multistep integrator.
///
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
pub struct BdfAug<
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
> {
    nonlinear_solver: Nls,
    ode_problem: Option<OdeSolverProblem<Eqn>>,
    n_equal_steps: usize,
    y_delta: Eqn::V,
    g_delta: Eqn::V,
    y_predict: Eqn::V,
    t_predict: Eqn::T,
    s_predict: Eqn::V,
    s_op: Option<BdfCallable<AugmentedEqn>>,
    s_deltas: Vec<Eqn::V>,
    sg_deltas: Vec<Eqn::V>,
    diff_tmp: M,
    u: M,
    alpha: Vec<Eqn::T>,
    gamma: Vec<Eqn::T>,
    error_const2: Vec<Eqn::T>,
    statistics: BdfStatistics,
    state: Option<BdfState<Eqn::V, M>>,
    tstop: Option<Eqn::T>,
    root_finder: Option<RootFinder<Eqn::V>>,
    is_state_modified: bool,
    jacobian_update: JacobianUpdate<Eqn::T>,
}

impl<Eqn, AugmentedEqn> Default
    for BdfAug<
        <Eqn::V as DefaultDenseMatrix>::M,
        Eqn,
        NewtonNonlinearSolver<BdfCallable<Eqn>, <Eqn::M as DefaultSolver>::LS<BdfCallable<Eqn>>>,
        AugmentedEqn,
    >
where
    Eqn: OdeEquations,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::M: DefaultSolver,
    Eqn::V: DefaultDenseMatrix,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn default() -> Self {
        let linear_solver = Eqn::M::default_solver();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        Self::new(nonlinear_solver)
    }
}

impl<M, Eqn, Nls, AugmentedEqn> BdfAug<M, Eqn, Nls, AugmentedEqn>
where
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn: OdeEquations,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
{
    const NEWTON_MAXITER: IndexType = 4;
    const MIN_FACTOR: f64 = 0.5;
    const MAX_FACTOR: f64 = 2.1;
    const MAX_THRESHOLD: f64 = 2.0;
    const MIN_THRESHOLD: f64 = 0.9;
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn new(nonlinear_solver: Nls) -> Self {
        let n = 1;

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
        let mut error_const2 = vec![Eqn::T::one()];

        let max_order: usize = BdfState::<Eqn::V, M>::MAX_ORDER;

        #[allow(clippy::needless_range_loop)]
        for i in 1..=max_order {
            let i_t = Eqn::T::from(i as f64);
            let one_over_i = Eqn::T::one() / i_t;
            let one_over_i_plus_one = Eqn::T::one() / (i_t + Eqn::T::one());
            gamma.push(gamma[i - 1] + one_over_i);
            alpha.push(Eqn::T::one() / ((Eqn::T::one() - kappa[i]) * gamma[i]));
            error_const2.push((kappa[i] * gamma[i] + one_over_i_plus_one).powi(2));
        }

        Self {
            s_op: None,
            ode_problem: None,
            nonlinear_solver,
            n_equal_steps: 0,
            diff_tmp: M::zeros(n, max_order + 3),
            y_delta: Eqn::V::zeros(n),
            y_predict: Eqn::V::zeros(n),
            t_predict: Eqn::T::zero(),
            s_predict: Eqn::V::zeros(n),
            s_deltas: Vec::new(),
            sg_deltas: Vec::new(),
            g_delta: Eqn::V::zeros(n),
            gamma,
            alpha,
            error_const2,
            u: M::zeros(max_order + 1, max_order + 1),
            statistics: BdfStatistics::default(),
            state: None,
            tstop: None,
            root_finder: None,
            is_state_modified: false,
            jacobian_update: JacobianUpdate::default(),
        }
    }

    pub fn get_statistics(&self) -> &BdfStatistics {
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

    fn _jacobian_updates(&mut self, c: Eqn::T, state: SolverState) {
        let y = &self.state.as_ref().unwrap().y;
        let t = self.state.as_ref().unwrap().t;
        //let y = &self.y_predict;
        //let t = self.t_predict;
        if self.jacobian_update.check_rhs_jacobian_update(c, &state) {
            self.nonlinear_solver.problem().f.set_jacobian_is_stale();
            self.nonlinear_solver.reset_jacobian(y, t);
            self.jacobian_update.update_rhs_jacobian();
            self.jacobian_update.update_jacobian(c);
        } else if self.jacobian_update.check_jacobian_update(c, &state) {
            self.nonlinear_solver.reset_jacobian(y, t);
            self.jacobian_update.update_jacobian(c);
        }
    }

    fn _update_step_size(&mut self, factor: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        //If step size h is changed then also need to update the terms in
        //the first equation of page 9 of [1]:
        //
        //- constant c = h / (1-kappa) gamma_k term
        //- lu factorisation of (M - c * J) used in newton iteration (same equation)

        let new_h = factor * self.state.as_ref().unwrap().h;
        self.n_equal_steps = 0;

        // update D using equations in section 3.2 of [1]
        let order = self.state.as_ref().unwrap().order;
        let r = Self::_compute_r(order, factor);
        let ru = r.mat_mul(&self.u);
        {
            let state = self.state.as_mut().unwrap();
            Self::_update_diff_for_step_size(&ru, &mut state.diff, &mut self.diff_tmp, order);
            for diff in state.sdiff.iter_mut() {
                Self::_update_diff_for_step_size(&ru, diff, &mut self.diff_tmp, order);
            }
            if self.ode_problem.as_ref().unwrap().integrate_out {
                Self::_update_diff_for_step_size(&ru, &mut state.gdiff, &mut self.diff_tmp, order);
            }
            for diff in state.sgdiff.iter_mut() {
                Self::_update_diff_for_step_size(&ru, diff, &mut self.diff_tmp, order);
            }
        }

        self.nonlinear_problem_op().set_c(new_h, self.alpha[order]);

        self.state.as_mut().unwrap().h = new_h;

        // if step size too small, then fail
        let state = self.state.as_ref().unwrap();
        if state.h.abs() < Eqn::T::from(Self::MIN_TIMESTEP) {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: state.t.into(),
            }));
        }
        Ok(new_h)
    }

    fn _update_diff_for_step_size(ru: &M, diff: &mut M, diff_tmp: &mut M, order: usize) {
        // D[0:order+1] = R * U * D[0:order+1]
        {
            let d_zero_order = diff.columns(0, order + 1);
            let mut d_zero_order_tmp = diff_tmp.columns_mut(0, order + 1);
            d_zero_order_tmp.gemm_vo(Eqn::T::one(), &d_zero_order, ru, Eqn::T::zero());
            // diff_sub = diff * RU
        }
        std::mem::swap(diff, diff_tmp);
    }

    fn update_differences_and_integrate_out(&mut self) {
        let order = self.state.as_ref().unwrap().order;
        let state = self.state.as_mut().unwrap();

        // update differences
        Self::_update_diff(order, &self.y_delta, &mut state.diff);

        // integrate output function
        if self.ode_problem.as_ref().unwrap().integrate_out {
            let out = self.ode_problem.as_ref().unwrap().eqn.out().unwrap();
            out.call_inplace(&self.y_predict, self.t_predict, &mut state.dg);
            self.nonlinear_solver.problem().f.integrate_out(
                &state.dg,
                &state.gdiff,
                self.gamma.as_slice(),
                self.alpha.as_slice(),
                state.order,
                &mut self.g_delta,
            );
            Self::_predict_using_diff(&mut state.g, &state.gdiff, order);
            state.g.axpy(Eqn::T::one(), &self.g_delta, Eqn::T::one());

            // update output difference
            Self::_update_diff(order, &self.g_delta, &mut state.gdiff);
        }

        // do the same for sensitivities
        if self.s_op.is_some() {
            for i in 0..self.s_op.as_ref().unwrap().eqn().max_index() {
                Rc::get_mut(self.s_op.as_mut().unwrap().eqn_mut()).unwrap().set_index(i);
                let op = self.s_op.as_ref().unwrap();

                // update sensitivity differences
                Self::_update_diff(order, &self.s_deltas[i], &mut state.sdiff[i]);

                // integrate sensitivity output equations
                if op.eqn().integrate_out() {
                    let out = op.eqn().out().unwrap();
                    out.call_inplace(&state.s[i], self.t_predict, &mut state.dsg[i]);
                    self.nonlinear_solver.problem().f.integrate_out(
                        &state.dsg[i],
                        &state.sgdiff[i],
                        self.gamma.as_slice(),
                        self.alpha.as_slice(),
                        state.order,
                        &mut self.sg_deltas[i],
                    );
                    Self::_predict_using_diff(&mut state.sg[i], &state.sgdiff[i], order);
                    state.sg[i].axpy(Eqn::T::one(), &self.sg_deltas[i], Eqn::T::one());

                    // update sensitivity output difference
                    Self::_update_diff(order, &self.sg_deltas[i], &mut state.sgdiff[i]);
                }

            }
        }
    }

    fn _update_diff(order: usize, d: &Eqn::V, diff: &mut M) {
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
        let d_minus_order_plus_one = d - diff.column(order + 1);
        diff.column_mut(order + 2)
            .copy_from(&d_minus_order_plus_one);
        diff.column_mut(order + 1).copy_from(d);
        for i in (0..=order).rev() {
            diff.column_axpy(Eqn::T::one(), i + 1, Eqn::T::one(), i);
        }
    }

    // predict forward to new step (eq 2 in [1])
    fn _predict_using_diff(y_predict: &mut Eqn::V, diff: &M, order: usize) {
        y_predict.fill(Eqn::T::zero());
        for i in 0..=order {
            y_predict.add_assign(diff.column(i));
        }
    }

    fn _predict_forward(&mut self) {
        let state = self.state.as_ref().unwrap();
        Self::_predict_using_diff(&mut self.y_predict, &state.diff, state.order);

        // update psi and c (h, D, y0 has changed)
        self.nonlinear_problem_op().set_psi_and_y0(
            &state.diff,
            self.gamma.as_slice(),
            self.alpha.as_slice(),
            state.order,
            &self.y_predict,
        );

        // update time
        let t_new = state.t + state.h;
        self.t_predict = t_new;
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        // check if the we are at tstop
        let state = self.state.as_ref().unwrap();
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if (state.h > M::T::zero() && tstop < state.t - troundoff)
            || (state.h < M::T::zero() && tstop > state.t + troundoff)
        {
            let error = OdeSolverError::StopTimeBeforeCurrentTime {
                stop_time: self.tstop.unwrap().into(),
                state_time: state.t.into(),
            };
            self.tstop = None;

            return Err(DiffsolError::from(error));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if (state.h > M::T::zero() && state.t + state.h > tstop + troundoff)
            || (state.h < M::T::zero() && state.t + state.h < tstop - troundoff)
        {
            let factor = (tstop - state.t) / state.h;
            // update step size ignoring the possible "step size too small" error
            _ = self._update_step_size(factor);
        }
        Ok(None)
    }

    fn initialise_to_first_order(&mut self) {
        self.n_equal_steps = 0;
        self.state
            .as_mut()
            .unwrap()
            .initialise_diff_to_first_order();

        if self.ode_problem.as_ref().unwrap().integrate_out {
            self.state
                .as_mut()
                .unwrap()
                .initialise_gdiff_to_first_order();
        }
        if self.s_op.is_some() {
            self.state
                .as_mut()
                .unwrap()
                .initialise_sdiff_to_first_order();
            if self.s_op.as_ref().unwrap().eqn().integrate_out() {
                self.state
                    .as_mut()
                    .unwrap()
                    .initialise_sgdiff_to_first_order();
            }
        }
        

        self.u = Self::_compute_r(1, Eqn::T::one());
        self.is_state_modified = false;
    }

    //interpolate solution at time values t* where t-h < t* < t
    //definition of the interpolating polynomial can be found on page 7 of [1]
    fn interpolate_from_diff(t: Eqn::T, diff: &M, t1: Eqn::T, h: Eqn::T, order: usize) -> Eqn::V {
        let mut time_factor = Eqn::T::from(1.0);
        let mut order_summation = diff.column(0).into_owned();
        for i in 0..order {
            let i_t = Eqn::T::from(i as f64);
            time_factor *= (t - (t1 - h * i_t)) / (h * (Eqn::T::one() + i_t));
            order_summation += diff.column(i + 1) * scale(time_factor);
        }
        order_summation
    }

    fn sensitivity_solve(
        &mut self,
        t_new: Eqn::T,
        mut error_norm: Eqn::T,
    ) -> Result<Eqn::T, DiffsolError> {
        let h = self.state.as_ref().unwrap().h;
        let order = self.state.as_ref().unwrap().order;
        let op = self.s_op.as_mut().unwrap();

        // update for new state
        {
            let dy_new = self.nonlinear_solver.problem().f.tmp();
            let y_new = &self.y_predict;
            Rc::get_mut(op.eqn_mut())
                .unwrap()
                .update_rhs_out_state(y_new, &dy_new, t_new);

            // construct bdf discretisation of sensitivity equations
            op.set_c(h, self.alpha[order]);
        }

        // solve for sensitivities equations discretised using BDF
        let naug = op.eqn().max_index();
        for i in 0..naug {
            // setup
            {
                let state = self.state.as_ref().unwrap();
                // predict forward to new step
                Self::_predict_using_diff(&mut self.s_predict, &state.sdiff[i], order);

                // setup op
                op.set_psi_and_y0(
                    &state.sdiff[i],
                    self.gamma.as_slice(),
                    self.alpha.as_slice(),
                    order,
                    &self.s_predict,
                );
                Rc::get_mut(op.eqn_mut()).unwrap().set_index(i);
            }

            // solve
            {
                let s_new = &mut self.state.as_mut().unwrap().s[i];
                s_new.copy_from(&self.s_predict);
                self.nonlinear_solver
                    .solve_other_in_place(&*op, s_new, t_new, &self.s_predict)?;
                self.statistics.number_of_nonlinear_solver_iterations +=
                    self.nonlinear_solver.convergence().niter();
                let s_new = &*s_new;
                self.s_deltas[i].copy_from(s_new);
                self.s_deltas[i] -= &self.s_predict;
            }

            let s_new = &self.state.as_ref().unwrap().s[i];

            if op.eqn().include_in_error_control() {
                let rtol = self.ode_problem.as_ref().unwrap().rtol;
                let atol = self.ode_problem.as_ref().unwrap().atol.as_ref();
                error_norm += self.s_deltas[i].squared_norm(s_new, atol, rtol);
            }
        }
        if op.eqn().include_in_error_control() {
            error_norm /= Eqn::T::from(naug as f64 + 1.0);
        }
        Ok(error_norm)
    }
}

impl<M, Eqn, Nls, AugmentedEqn> OdeSolverMethod<Eqn> for BdfAug<M, Eqn, Nls, AugmentedEqn>
where
    Eqn: OdeEquations,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    type State = BdfState<Eqn::V, M>;

    fn order(&self) -> usize {
        self.state.as_ref().map_or(1, |state| state.order)
    }

    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        // state must be set
        let state = self.state.as_ref().ok_or(ode_solver_error!(StateNotSet))?;
        if self.is_state_modified {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }
        // check that t is before/after the current time depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }
        Ok(Self::interpolate_from_diff(
            t,
            &state.diff,
            state.t,
            state.h,
            state.order,
        ))
    }

    fn interpolate_out(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        // state must be set
        let state = self.state.as_ref().ok_or(ode_solver_error!(StateNotSet))?;
        if self.is_state_modified {
            if t == state.t {
                return Ok(state.g.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }
        // check that t is before/after the current time depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }
        Ok(Self::interpolate_from_diff(
            t,
            &state.gdiff,
            state.t,
            state.h,
            state.order,
        ))
    }

    fn interpolate_sens(&self, t: <Eqn as OdeEquations>::T) -> Result<Vec<Eqn::V>, DiffsolError> {
        // state must be set
        let state = self.state.as_ref().ok_or(ode_solver_error!(StateNotSet))?;
        if self.is_state_modified {
            if t == state.t {
                return Ok(state.s.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }
        // check that t is before/after the current time depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }

        let mut s = Vec::with_capacity(state.s.len());
        for i in 0..state.s.len() {
            s.push(Self::interpolate_from_diff(
                t,
                &state.sdiff[i],
                state.t,
                state.h,
                state.order,
            ));
        }
        Ok(s)
    }

    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.ode_problem.as_ref()
    }

    fn state(&self) -> Option<StateRef<Eqn::V>> {
        self.state.as_ref().map(|state| state.as_ref())
    }
    fn take_state(&mut self) -> Option<BdfState<Eqn::V, M>> {
        Option::take(&mut self.state)
    }

    fn state_mut(&mut self) -> Option<StateRefMut<Eqn::V>> {
        self.is_state_modified = true;
        self.state.as_mut().map(|state| state.as_mut())
    }

    fn checkpoint(&mut self) -> Result<Self::State, DiffsolError> {
        if self.state.is_none() {
            return Err(ode_solver_error!(StateNotSet));
        }
        self._jacobian_updates(
            self.state.as_ref().unwrap().h * self.alpha[self.state.as_ref().unwrap().order],
            SolverState::Checkpoint,
        );

        Ok(self.state.as_ref().unwrap().clone())
    }

    fn set_problem(
        &mut self,
        mut state: BdfState<Eqn::V, M>,
        problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        self.ode_problem = Some(problem.clone());

        state.check_consistent_with_problem(problem)?;

        // setup linear solver for first step
        let bdf_callable = Rc::new(BdfCallable::new(problem));
        bdf_callable.set_c(state.h, self.alpha[state.order]);

        let nonlinear_problem = SolverProblem::new_from_ode_problem(bdf_callable, problem);
        self.nonlinear_solver.set_problem(&nonlinear_problem);
        self.nonlinear_solver
            .convergence_mut()
            .set_max_iter(Self::NEWTON_MAXITER);
        self.nonlinear_solver.reset_jacobian(&state.y, state.t);

        // setup root solver
        if let Some(root_fn) = problem.eqn.root() {
            self.root_finder = Some(RootFinder::new(root_fn.nout()));
            self.root_finder
                .as_ref()
                .unwrap()
                .init(root_fn.as_ref(), &state.y, state.t);
        }

        // (re)allocate internal state
        let nstates = problem.eqn.rhs().nstates();
        if self.diff_tmp.nrows() != nstates {
            self.diff_tmp = M::zeros(nstates, BdfState::<Eqn::V, M>::MAX_ORDER + 3);
            self.y_delta = <Eqn::V as Vector>::zeros(nstates);
            self.y_predict = <Eqn::V as Vector>::zeros(nstates);
        }

        let nout = if let Some(out) = problem.eqn.out() {
            out.nout()
        } else {
            0
        };
        if self.g_delta.len() != nout {
            self.g_delta = <Eqn::V as Vector>::zeros(nout);
        }

        // init U matrix
        self.u = Self::_compute_r(state.order, Eqn::T::one());
        self.is_state_modified = false;

        // initialise state and store it
        state.set_problem(problem)?;
        self.state = Some(state);
        Ok(())
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut safety: Eqn::T;
        let mut error_norm: Eqn::T;
        if self.state.is_none() {
            return Err(ode_solver_error!(StateNotSet));
        }

        let mut convergence_fail = false;

        if self.is_state_modified {
            self.initialise_to_first_order();
        }

        self._predict_forward();

        // loop until step is accepted
        loop {
            let order = self.state.as_ref().unwrap().order;
            self.y_delta.copy_from(&self.y_predict);

            // initialise error_norm to quieten the compiler
            error_norm = Eqn::T::from(2.0);

            // solve BDF equation using y0 as starting point
            let mut solve_result = self.nonlinear_solver.solve_in_place(
                &mut self.y_delta,
                self.t_predict,
                &self.y_predict,
            );
            // update statistics
            self.statistics.number_of_nonlinear_solver_iterations +=
                self.nonlinear_solver.convergence().niter();

            // only calculate norm and sensitivities if solve was successful
            if solve_result.is_ok() {
                // test error is within tolerance
                // combine eq 3, 4 and 6 from [1] to obtain error
                // Note that error = C_k * h^{k+1} y^{k+1}
                // and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
                self.y_delta -= &self.y_predict;

                // calculate error norm
                {
                    let rtol = self.problem().as_ref().unwrap().rtol;
                    let atol = self.ode_problem.as_ref().unwrap().atol.as_ref();
                    error_norm =
                        self.y_delta
                            .squared_norm(&self.state.as_mut().unwrap().y, atol, rtol)
                            * self.error_const2[order];
                }

                // only bother doing sensitivity calculations if we might keep the step
                if self.s_op.is_some() && error_norm <= Eqn::T::from(1.0) {
                    error_norm = match self.sensitivity_solve(self.t_predict, error_norm) {
                        Ok(en) => en,
                        Err(_) => {
                            solve_result = Err(ode_solver_error!(SensitivitySolveFailed));
                            Eqn::T::from(2.0)
                        }
                    }
                }
            }

            // handle case where either nonlinear solve failed
            if solve_result.is_err() {
                self.statistics.number_of_nonlinear_solver_fails += 1;
                if convergence_fail {
                    // newton iteration did not converge, but jacobian has already been
                    // evaluated so reduce step size by 0.3 (as per [1]) and try again
                    let new_h = self._update_step_size(Eqn::T::from(0.3))?;
                    self._jacobian_updates(
                        new_h * self.alpha[order],
                        SolverState::SecondConvergenceFail,
                    );

                    // new prediction
                    self._predict_forward();

                    // update statistics
                } else {
                    // newton iteration did not converge, so update jacobian and try again
                    self._jacobian_updates(
                        self.state.as_ref().unwrap().h * self.alpha[order],
                        SolverState::FirstConvergenceFail,
                    );
                    convergence_fail = true;
                    // same prediction as last time
                }
                continue;
            }

            // need to caulate safety even if step is accepted
            let maxiter = self.nonlinear_solver.convergence().max_iter() as f64;
            let niter = self.nonlinear_solver.convergence().niter() as f64;
            safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));

            // do the error test
            if error_norm <= Eqn::T::from(1.0) {
                // step is accepted
                break;
            } else {
                // step is rejected
                // calculate optimal step size factor as per eq 2.46 of [2]
                // and reduce step size and try again
                let mut factor = safety * error_norm.pow(Eqn::T::from(-0.5 / (order as f64 + 1.0)));
                if factor < Eqn::T::from(Self::MIN_FACTOR) {
                    factor = Eqn::T::from(Self::MIN_FACTOR);
                }
                let new_h = self._update_step_size(factor)?;
                self._jacobian_updates(new_h * self.alpha[order], SolverState::ErrorTestFail);

                // new prediction
                self._predict_forward();

                // update statistics
                self.statistics.number_of_error_test_failures += 1;
            }
        }

        // take the accepted step
        self.update_differences_and_integrate_out();

        {
            let state = self.state.as_mut().unwrap();
            state.y.copy_from(&self.y_predict);
            state.t = self.t_predict;
            state.dy.copy_from_view(&state.diff.column(1));
            state.dy *= scale(Eqn::T::one() / state.h);
        }

        // update statistics
        self.statistics.number_of_linear_solver_setups =
            self.nonlinear_problem_op().number_of_jac_evals();
        self.statistics.number_of_steps += 1;
        self.jacobian_update.step();

        // a change in order is only done after running at order k for k + 1 steps
        // (see page 83 of [2])
        self.n_equal_steps += 1;

        if self.n_equal_steps > self.state.as_ref().unwrap().order {
            let factors = {
                let state = self.state.as_mut().unwrap();
                let atol = self.ode_problem.as_ref().unwrap().atol.as_ref();
                let rtol = self.ode_problem.as_ref().unwrap().rtol;
                let order = state.order;
                // similar to the optimal step size factor we calculated above for the current
                // order k, we need to calculate the optimal step size factors for orders
                // k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
                let error_m_norm = if order > 1 {
                    let mut error_m_norm =
                        state.diff.column(order).squared_norm(&state.y, atol, rtol)
                            * self.error_const2[order - 1];
                    for i in 0..state.sdiff.len() {
                        error_m_norm +=
                            state.sdiff[i]
                                .column(order)
                                .squared_norm(&state.s[i], atol, rtol)
                                * self.error_const2[order - 1];
                    }
                    error_m_norm / Eqn::T::from((state.sdiff.len() + 1) as f64)
                } else {
                    Eqn::T::INFINITY
                };
                let error_p_norm = if order < BdfState::<Eqn::V, M>::MAX_ORDER {
                    let mut error_p_norm = state
                        .diff
                        .column(order + 2)
                        .squared_norm(&state.y, atol, rtol)
                        * self.error_const2[order + 1];
                    for i in 0..state.sdiff.len() {
                        error_p_norm =
                            state.sdiff[i]
                                .column(order + 2)
                                .squared_norm(&state.s[i], atol, rtol)
                                * self.error_const2[order + 1];
                    }
                    error_p_norm / Eqn::T::from((state.sdiff.len() + 1) as f64)
                } else {
                    Eqn::T::INFINITY
                };

                let error_norms = [error_m_norm, error_norm, error_p_norm];
                error_norms
                    .into_iter()
                    .enumerate()
                    .map(|(i, error_norm)| {
                        error_norm.pow(Eqn::T::from(-0.5 / (i as f64 + order as f64)))
                    })
                    .collect::<Vec<_>>()
            };

            // now we have the three factors for orders k-1, k and k+1, pick the maximum in
            // order to maximise the resultant step size
            let max_index = factors
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            // update order and update the U matrix
            let order = {
                let old_order = self.state.as_ref().unwrap().order;
                let new_order = match max_index {
                    0 => old_order - 1,
                    1 => old_order,
                    2 => old_order + 1,
                    _ => unreachable!(),
                };
                self.state.as_mut().unwrap().order = new_order;
                if max_index != 1 {
                    self.u = Self::_compute_r(new_order, Eqn::T::one());
                }
                new_order
            };

            let mut factor = safety * factors[max_index];
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }
            if factor < Eqn::T::from(Self::MIN_FACTOR) {
                factor = Eqn::T::from(Self::MIN_FACTOR);
            }
            if factor >= Eqn::T::from(Self::MAX_THRESHOLD)
                || factor < Eqn::T::from(Self::MIN_THRESHOLD)
                || max_index == 0
                || max_index == 2
            {
                let new_h = self._update_step_size(factor)?;
                self._jacobian_updates(new_h * self.alpha[order], SolverState::StepSuccess);
            }
        }

        // check for root within accepted step
        if let Some(root_fn) = self.problem().as_ref().unwrap().eqn.root() {
            let ret = self.root_finder.as_ref().unwrap().check_root(
                &|t: <Eqn as OdeEquations>::T| self.interpolate(t),
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

    fn set_stop_time(&mut self, tstop: <Eqn as OdeEquations>::T) -> Result<(), DiffsolError> {
        self.tstop = Some(tstop);
        if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
            let error = OdeSolverError::StopTimeBeforeCurrentTime {
                stop_time: tstop.into(),
                state_time: self.state.as_ref().unwrap().t.into(),
            };
            self.tstop = None;
            return Err(DiffsolError::from(error));
        }
        Ok(())
    }
}

impl<M, Eqn, Nls, AugmentedEqn> AugmentedOdeSolverMethod<Eqn, AugmentedEqn>
    for BdfAug<M, Eqn, Nls, AugmentedEqn>
where
    Eqn: OdeEquations,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn set_augmented_problem(
        &mut self,
        state: BdfState<Eqn::V, M>,
        problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: AugmentedEqn,
    ) -> Result<(), DiffsolError> {
        state.check_sens_consistent_with_problem(problem, &augmented_eqn)?;

        self.set_problem(state, problem)?;

        self.state
            .as_mut()
            .unwrap()
            .set_augmented_problem(problem, &augmented_eqn)?;

        // allocate internal state for sensitivities
        let naug = augmented_eqn.max_index();
        let nstates = problem.eqn.rhs().nstates();
        let augmented_eqn = Rc::new(augmented_eqn);
        self.s_op = Some(BdfCallable::from_sensitivity_eqn(&augmented_eqn));

        if self.s_deltas.len() != naug || self.s_deltas[0].len() != nstates {
            self.s_deltas = vec![<Eqn::V as Vector>::zeros(nstates); naug];
        }
        if self.s_predict.len() != nstates {
            self.s_predict = <Eqn::V as Vector>::zeros(nstates);
        }
        if let Some(out) = self.s_op.as_ref().unwrap().eqn().out() {
            if self.sg_deltas.len() != naug || self.sg_deltas[0].len() != out.nout() {
                self.sg_deltas = vec![<Eqn::V as Vector>::zeros(out.nout()); naug];
            }
        }
        Ok(())
    }
}

impl<M, Eqn, Nls, AugmentedEqn> AdjointOdeSolverMethod<Eqn> for BdfAug<M, Eqn, Nls, AugmentedEqn>
where
    Eqn: OdeEquations,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Nls: NonLinearSolver<BdfCallable<Eqn>>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    type AdjointSolver = BdfAug<
        M,
        AdjointEquations<Eqn, Self>,
        Nls::SelfNewOp<BdfCallable<AdjointEquations<Eqn, Self>>>,
        AdjointEquations<Eqn, Self>,
    >;

    fn new_adjoint_solver(
        &self,
        checkpoints: Vec<Self::State>,
        include_in_error_control: bool,
    ) -> Result<Self::AdjointSolver, DiffsolError> {
        // construct checkpointing
        let checkpointer_nls = Nls::default();
        let checkpointer_solver = Self::new(checkpointer_nls);
        let problem = self.ode_problem.as_ref().unwrap();
        let checkpointer = Checkpointing::new(
            problem,
            checkpointer_solver,
            checkpoints.len() - 2,
            checkpoints,
        );

        // construct adjoint equations and problem
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer)));
        let new_eqn = AdjointEquations::new(&problem.eqn, context.clone(), false);
        let mut new_augmented_eqn = AdjointEquations::new(&problem.eqn, context, true);
        new_augmented_eqn.set_include_in_error_control(include_in_error_control);
        let adj_problem = OdeSolverProblem {
            eqn: Rc::new(new_eqn),
            rtol: problem.rtol,
            atol: problem.atol.clone(),
            t0: self.state.as_ref().unwrap().t,
            h0: -self.state.as_ref().unwrap().h,
            integrate_out: false,
        };

        // initialise adjoint state
        let mut state =
            Self::State::new_without_initialise_augmented(&adj_problem, &mut new_augmented_eqn)?;
        let mut init_nls = Nls::SelfNewOp::<InitOp<AdjointEquations<Eqn, Self>>>::default();
        let new_augmented_eqn =
            state.set_consistent_augmented(&adj_problem, new_augmented_eqn, &mut init_nls)?;

        // create adjoint solver
        let adjoint_nls = Nls::SelfNewOp::<BdfCallable<AdjointEquations<Eqn, Self>>>::default();
        let mut adjoint_solver = Self::AdjointSolver::new(adjoint_nls);

        // setup the solver
        adjoint_solver.set_augmented_problem(state, &adj_problem, new_augmented_eqn)?;
        Ok(adjoint_solver)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::{
            test_models::{
                dydt_y2::dydt_y2_problem,
                exponential_decay::{
                    exponential_decay_problem, exponential_decay_problem_adjoint,
                    exponential_decay_problem_sens, exponential_decay_problem_with_root,
                    negative_exponential_decay_problem,
                },
                exponential_decay_with_algebraic::{
                    exponential_decay_with_algebraic_problem,
                    exponential_decay_with_algebraic_problem_sens,
                },
                foodweb::{foodweb_problem, FoodWebContext},
                gaussian_decay::gaussian_decay_problem,
                heat2d::head2d_problem,
                robertson::robertson,
                robertson::robertson_sens,
                robertson_ode::robertson_ode,
                robertson_ode_with_sens::robertson_ode_with_sens,
            },
            tests::{
                test_checkpointing, test_interpolate, test_no_set_problem, test_ode_solver,
                test_ode_solver_adjoint, test_state_mut, test_state_mut_on_problem,
            },
        },
        Bdf, FaerSparseLU, NewtonNonlinearSolver, OdeEquations, OdeSolverMethod, Op, SparseColMat,
    };

    use faer::Mat;
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
    fn bdf_test_nalgebra_negative_exponential_decay() {
        let mut s = Bdf::default();
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 11
        number_of_steps: 41
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 82
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 84
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay() {
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        let (problem, soln) = exponential_decay_problem::<SparseColMat<f64>>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[test]
    fn bdf_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_checkpointing(Bdf::default(), Bdf::default(), problem, soln);
    }

    #[test]
    fn bdf_test_faer_exponential_decay() {
        type M = faer::Mat<f64>;
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 11
        number_of_steps: 41
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 82
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 84
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_sens() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 11
        number_of_steps: 55
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 273
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 110
        number_of_jac_muls: 169
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_problem_adjoint::<M>();
        let adjoint_solver = test_ode_solver_adjoint(&mut s, &problem, soln);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 12
        number_of_steps: 41
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 82
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(s.problem().as_ref().unwrap().eqn.rhs().statistics(), @r###"
        ---
        number_of_calls: 166
        number_of_jac_muls: 8
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 254
        "###);
        insta::assert_yaml_snapshot!(adjoint_solver.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 18
        number_of_steps: 41
        number_of_error_test_failures: 9
        number_of_nonlinear_solver_iterations: 250
        number_of_nonlinear_solver_fails: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_with_algebraic_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 16
        number_of_steps: 36
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 71
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 75
        number_of_jac_muls: 6
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay_algebraic() {
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        let (problem, soln) = exponential_decay_with_algebraic_problem::<SparseColMat<f64>>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic_sens() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_with_algebraic_problem_sens::<M>();
        test_ode_solver(&mut s, &problem, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 21
        number_of_steps: 49
        number_of_error_test_failures: 5
        number_of_nonlinear_solver_iterations: 163
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 71
        number_of_jac_muls: 108
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let mut s = Bdf::default();
        let (problem, soln) = robertson::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 79
        number_of_steps: 330
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 748
        number_of_nonlinear_solver_fails: 19
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 751
        number_of_jac_muls: 60
        number_of_matrix_evals: 20
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_robertson() {
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        let (problem, soln) = robertson::<SparseColMat<f64>>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[cfg(feature = "suitesparse")]
    #[test]
    fn bdf_test_faer_sparse_ku_robertson() {
        let linear_solver = crate::KLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        let (problem, soln) = robertson::<SparseColMat<f64>>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_diffsl_robertson() {
        use diffsl::LlvmModule;

        use crate::ode_solver::test_models::robertson;
        let mut context = crate::DiffSlContext::default();
        let mut s = Bdf::default();
        robertson::robertson_diffsl_compile(&mut context);
        let (problem, soln) =
            robertson::robertson_diffsl_problem::<M, LlvmModule>(&context, false);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_sens() {
        let mut s = Bdf::default();
        let (problem, soln) = robertson_sens::<M>();
        test_ode_solver(&mut s, &problem, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 191
        number_of_steps: 506
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 3747
        number_of_nonlinear_solver_fails: 101
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1203
        number_of_jac_muls: 3016
        number_of_matrix_evals: 87
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let mut s = Bdf::default();
        let (problem, soln) = robertson::<M>(true);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 79
        number_of_steps: 330
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 748
        number_of_nonlinear_solver_fails: 19
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 751
        number_of_jac_muls: 63
        number_of_matrix_evals: 20
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let mut s = Bdf::default();
        let (problem, soln) = robertson_ode::<M>(false, 3);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 90
        number_of_steps: 412
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 908
        number_of_nonlinear_solver_fails: 15
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 910
        number_of_jac_muls: 162
        number_of_matrix_evals: 18
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode_sens() {
        let mut s = Bdf::default();
        let (problem, soln) = robertson_ode_with_sens::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 171
        number_of_steps: 463
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 3497
        number_of_nonlinear_solver_fails: 88
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1123
        number_of_jac_muls: 2771
        number_of_matrix_evals: 74
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let mut s = Bdf::default();
        let (problem, soln) = dydt_y2_problem::<M>(false, 10);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 30
        number_of_steps: 160
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 356
        number_of_nonlinear_solver_fails: 2
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 358
        number_of_jac_muls: 40
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let mut s = Bdf::default();
        let (problem, soln) = dydt_y2_problem::<M>(true, 10);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 30
        number_of_steps: 160
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 356
        number_of_nonlinear_solver_fails: 2
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 358
        number_of_jac_muls: 14
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let mut s = Bdf::default();
        let (problem, soln) = gaussian_decay_problem::<M>(false, 10);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 14
        number_of_steps: 60
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 124
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 126
        number_of_jac_muls: 20
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_faer_sparse_heat2d() {
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 21
        number_of_steps: 173
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 343
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 346
        number_of_jac_muls: 135
        number_of_matrix_evals: 5
        number_of_jac_adj_muls: 0
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_bdf_faer_sparse_heat2d_diffsl() {
        use diffsl::LlvmModule;

        use crate::ode_solver::test_models::heat2d::{self, heat2d_diffsl_compile};
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut context = crate::DiffSlContext::default();
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        heat2d_diffsl_compile::<SparseColMat<f64>, LlvmModule, 10>(&mut context);
        let (problem, soln) = heat2d::heat2d_diffsl_problem(&context);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[test]
    fn test_bdf_faer_sparse_foodweb() {
        let foodweb_context = FoodWebContext::default();
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        let (problem, soln) = foodweb_problem::<SparseColMat<f64>, 10>(&foodweb_context);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 41
        number_of_steps: 149
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 332
        number_of_nonlinear_solver_fails: 14
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_bdf_faer_sparse_foodweb_diffsl() {
        use diffsl::LlvmModule;

        use crate::ode_solver::test_models::foodweb;
        let mut context = crate::DiffSlContext::default();
        let linear_solver = FaerSparseLU::default();
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        let mut s = Bdf::<Mat<f64>, _, _>::new(nonlinear_solver);
        foodweb::foodweb_diffsl_compile::<SparseColMat<f64>, LlvmModule, 10>(&mut context);
        let (problem, soln) = foodweb::foodweb_diffsl_problem(&context);
        test_ode_solver(&mut s, &problem, soln, None, false, false);
    }

    #[test]
    fn test_tstop_bdf() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, true, false);
    }

    #[test]
    fn test_root_finder_bdf() {
        let mut s = Bdf::default();
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let y = test_ode_solver(&mut s, &problem, soln, None, false, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }
}
