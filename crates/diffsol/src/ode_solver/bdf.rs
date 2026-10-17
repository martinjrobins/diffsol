use log::{debug, info, trace};
use std::cell::Ref;

use crate::{
    error::{DiffsolError, OdeSolverError},
    AugmentedOdeEquationsImplicit, Convergence, DefaultDenseMatrix, LinearSolver,
    NewtonNonlinearSolver, NoAug, NoLineSearch, StateRef, StateRefMut,
};

use num_traits::{abs, FromPrimitive, One, Signed, ToPrimitive, Zero};

use crate::ode_solver_error;
use crate::{
    matrix::MatrixRef, nonlinear_solver::root::RootFinder, op::bdf::BdfCallable,
    AugmentedOdeEquations, BdfState, Context, DenseMatrix, JacobianUpdate, NonLinearOp,
    NonLinearSolver, OdeEquationsImplicit, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason, Op, Scalar,
    SensEquations, Vector, VectorRef, VectorView,
};

use super::adjoint::AdjointOdeSolverMethod;
use super::bdf_state::{BdfStepRecord, MAX_ORDER};
use crate::small::{SmallMat, SmallVec};

use super::config::BdfConfig;
use super::jacobian_update::SolverState;
use super::method::{check_interpolate_shape, check_interpolate_time, AugmentedOdeSolverMethod};
use super::pi_controller::pi_controller_raw;
use super::sensitivities::SensitivitiesOdeSolverMethod;
use super::step_ad::StepVjp;
use super::OdeSolverStatistics;
use crate::{Scale, VectorViewMut};
use std::ops::MulAssign;

/// Length of a per-order BDF coefficient list — `gamma`, `alpha`, `error_const2`, the
/// interpolation weights.
const COEFFS_LEN: usize = MAX_ORDER + 1;

/// Capacity of a square per-order BDF coefficient block — the `R`, `U` and `R * U` matrices of
/// section 3.2 of [1].
const BLOCK_LEN: usize = COEFFS_LEN * COEFFS_LEN;

/// A per-order BDF coefficient list, sized once for the maximum order.
type BdfVec<T> = SmallVec<T, COEFFS_LEN>;

/// A square per-order BDF coefficient block, held at the current order.
type BdfMat<T> = SmallMat<T, BLOCK_LEN>;

impl<'a, M, Eqn, LS, AugEqn> AugmentedOdeSolverMethod<'a, Eqn, AugEqn>
    for Bdf<'a, Eqn, LS, M, AugEqn>
where
    Eqn: OdeEquationsImplicit,
    AugEqn: AugmentedOdeEquationsImplicit<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
{
    fn into_state_and_eqn(mut self) -> (Self::State, Option<AugEqn>) {
        let callable = std::mem::take(&mut self.s_op);
        let state = self.into_state();
        (state, callable.map(|op| op.eqn))
    }
    fn augmented_eqn(&self) -> Option<&AugEqn> {
        self.s_op.as_ref().map(|op| op.eqn())
    }
    fn augmented_eqn_mut(&mut self) -> Option<&mut AugEqn> {
        self.s_op.as_mut().map(|op| &mut op.eqn)
    }
    fn state_and_augmented_eqn_mut(&mut self) -> Option<(StateRefMut<'_, Eqn::V>, &mut AugEqn)> {
        self.is_state_modified = true;
        let state = self.state.as_mut();
        let augmented_eqn = self.s_op.as_mut().map(|op| &mut op.eqn)?;
        Some((state, augmented_eqn))
    }
}

impl<'a, M, Eqn, LS> SensitivitiesOdeSolverMethod<'a, Eqn>
    for Bdf<'a, Eqn, LS, M, SensEquations<'a, Eqn>>
where
    Eqn: OdeEquationsImplicitSens + 'a,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
{
}

impl<'a, M, Eqn, LS, Solver> AdjointOdeSolverMethod<'a, Eqn, Solver>
    for Bdf<'a, Eqn, LS, M, crate::AdjointEquations<'a, Eqn, Solver>>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    Solver: OdeSolverMethod<'a, Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
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
pub struct Bdf<
    'a,
    Eqn: OdeEquationsImplicit,
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C> = <<Eqn as Op>::V as DefaultDenseMatrix>::M,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn> = NoAug<Eqn>,
> where
    Eqn::V: DefaultDenseMatrix,
{
    nonlinear_solver: NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
    convergence: Convergence<'a, Eqn::V>,
    /// Convergence of the sensitivity solves alongside the main equations, and of the linear
    /// solves of [StepVjp::step_vjp]: derivative solves, to `derivative_solve_tolerance`, kept
    /// apart from the Newton solves so that each keeps its own convergence-rate estimate and
    /// leaves the Newton solves (and so the steps) alone.
    sens_convergence: Convergence<'a, Eqn::V>,
    vjp_convergence: Convergence<'a, Eqn::V>,
    ode_problem: &'a OdeSolverProblem<Eqn>,
    op: Option<BdfCallable<&'a Eqn>>,
    n_equal_steps: usize,
    y_delta: Eqn::V,
    g_delta: Eqn::V,
    y_predict: Eqn::V,
    /// The corrector's iterate, which is the corrected solution `y_{n+1}` once the solve
    /// succeeds.
    y_new: Eqn::V,
    t_predict: Eqn::T,
    s_predict: Eqn::V,
    s_op: Option<BdfCallable<AugmentedEqn>>,
    /// step deltas of the augmented state, one channel per batch lane
    s_deltas: Eqn::V,
    sg_deltas: Eqn::V,
    /// The U matrix of section 3.2 of [1], `(order+1) x (order+1)` on the host.
    /// Rebuilt only when the order changes.
    u: BdfMat<Eqn::T>,
    alpha: BdfVec<Eqn::T>,
    /// The `gamma` coefficients of section 2 of [1].
    gamma: BdfVec<Eqn::T>,
    error_const2: BdfVec<Eqn::T>,
    statistics: OdeSolverStatistics,
    state: BdfState<Eqn::V, M>,
    tstop: Option<Eqn::T>,
    root_finder: Option<RootFinder<Eqn::V>>,
    is_state_modified: bool,
    jacobian_update: JacobianUpdate<Eqn::T>,
    config: BdfConfig<Eqn::T>,
    prev_error_norm: Option<Eqn::T>,
}

impl<M, Eqn, LS, AugmentedEqn> Clone for Bdf<'_, Eqn, LS, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicit,
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
    Eqn::V: DefaultDenseMatrix,
{
    fn clone(&self) -> Self {
        let problem = self.ode_problem;
        let mut nonlinear_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
        let op = if let Some(op) = self.op.as_ref() {
            let op = op.clone_state(&self.ode_problem.eqn);
            nonlinear_solver.set_problem(&op);
            Some(op)
        } else {
            None
        };
        let s_op = self.s_op.as_ref().map(|op| {
            let op = op.clone_state(op.eqn().clone());
            op
        });
        let mut convergence = self.convergence.clone();
        convergence.reset_eta();
        Self {
            sens_convergence: convergence.clone(),
            vjp_convergence: convergence.clone(),
            nonlinear_solver,
            ode_problem: problem,
            convergence,
            op,
            s_op,
            n_equal_steps: self.n_equal_steps,
            y_delta: self.y_delta.clone(),
            g_delta: self.g_delta.clone(),
            y_predict: self.y_predict.clone(),
            y_new: self.y_new.clone(),
            t_predict: self.t_predict,
            s_predict: self.s_predict.clone(),
            s_deltas: self.s_deltas.clone(),
            sg_deltas: self.sg_deltas.clone(),
            u: self.u,
            alpha: self.alpha,
            gamma: self.gamma,
            error_const2: self.error_const2,
            statistics: OdeSolverStatistics::default(),
            state: self.state.clone(),
            tstop: self.tstop,
            root_finder: self.root_finder.clone(),
            is_state_modified: self.is_state_modified,
            jacobian_update: self.jacobian_update.clone(),
            config: self.config.clone(),
            prev_error_norm: self.prev_error_norm,
        }
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> Drop for Bdf<'a, Eqn, LS, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicit,
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
    Eqn::V: DefaultDenseMatrix,
{
    fn drop(&mut self) {
        info!("BDF Solver Statistics: {}", self.statistics);
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> Bdf<'a, Eqn, LS, M, AugmentedEqn>
where
    AugmentedEqn: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    LS: LinearSolver<Eqn::M>,
{
    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: BdfState<Eqn::V, M>,
        linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        Self::_new(
            problem,
            state,
            linear_solver,
            true,
            BdfConfig::new(&problem.ode_options),
        )
    }

    fn _new(
        problem: &'a OdeSolverProblem<Eqn>,
        mut state: BdfState<Eqn::V, M>,
        linear_solver: LS,
        integrate_main_eqn: bool,
        config: BdfConfig<Eqn::T>,
    ) -> Result<Self, DiffsolError> {
        let mut nonlinear_solver = NewtonNonlinearSolver::new(linear_solver, NoLineSearch);
        // kappa values for difference orders, taken from Table 1 of [1]
        let kappa: [Eqn::T; 6] = [
            Eqn::T::zero(),
            <Eqn::T as FromPrimitive>::from_f64(-0.1850).unwrap(),
            -Eqn::T::one() / <Eqn::T as FromPrimitive>::from_f64(9.0).unwrap(),
            <Eqn::T as FromPrimitive>::from_f64(-0.0823).unwrap(),
            <Eqn::T as FromPrimitive>::from_f64(-0.0415).unwrap(),
            Eqn::T::zero(),
        ];

        let mut alpha = BdfVec::zeros(COEFFS_LEN);
        let mut gamma = BdfVec::zeros(COEFFS_LEN);
        let mut error_const2 = BdfVec::from_slice(&[Eqn::T::one(); COEFFS_LEN]);

        #[allow(clippy::needless_range_loop)]
        for i in 1..=MAX_ORDER {
            let i_t = <Eqn::T as FromPrimitive>::from_f64(i as f64).unwrap();
            let one_over_i = Eqn::T::one() / i_t;
            let one_over_i_plus_one = Eqn::T::one() / (i_t + Eqn::T::one());
            gamma[i] = gamma[i - 1] + one_over_i;
            alpha[i] = Eqn::T::one() / ((Eqn::T::one() - kappa[i]) * gamma[i]);
            let error_const2_i = kappa[i] * gamma[i] + one_over_i_plus_one;
            error_const2[i] = error_const2_i * error_const2_i;
        }

        state.check_consistent_with_problem(problem)?;

        let mut convergence = Convergence::with_tolerance(
            problem.rtol,
            &problem.atol,
            problem.ode_options.nonlinear_solver_tolerance,
        );
        convergence.set_max_iter(config.maximum_newton_iterations);
        let sens_convergence = convergence.clone();
        let vjp_convergence = convergence.clone();

        let op = if integrate_main_eqn {
            // setup linear solver for first step
            let mut bdf_callable = BdfCallable::new(&problem.eqn, problem.context().clone());
            bdf_callable.set_c(state.h, alpha[state.order]);
            nonlinear_solver.set_problem(&bdf_callable);
            nonlinear_solver.reset_jacobian(&bdf_callable, &state.y, state.t);
            Some(bdf_callable)
        } else {
            None
        };

        state.set_problem(problem)?;

        let ctx = problem.eqn.context();
        let root_finder = if integrate_main_eqn {
            problem.eqn.root().map(|root_fn| {
                let root_finder =
                    RootFinder::new(root_fn.nout(), problem.eqn.nstates(), ctx.clone());
                root_finder.init(&root_fn, &state.y, state.t);
                root_finder
            })
        } else {
            None
        };

        // (re)allocate internal state
        let nstates = problem.eqn.rhs().nstates();

        let y_delta = <Eqn::V as Vector>::zeros(nstates, ctx.clone());
        let y_predict = <Eqn::V as Vector>::zeros(nstates, ctx.clone());
        let y_new = <Eqn::V as Vector>::zeros(nstates, ctx.clone());

        let nout = if problem.integrate_out {
            problem.eqn.nout()
        } else {
            0
        };
        let g_delta = <Eqn::V as Vector>::zeros(nout, ctx.clone());

        // init U matrix (deps only on order so need to update whenever order changes)
        let u = Self::_compute_r(state.order, Eqn::T::one());
        let is_state_modified = false;

        Ok(Self {
            convergence,
            sens_convergence,
            vjp_convergence,
            s_op: None,
            op,
            ode_problem: problem,
            nonlinear_solver,
            n_equal_steps: 0,
            y_delta,
            y_predict,
            y_new,
            t_predict: Eqn::T::zero(),
            s_predict: Eqn::V::zeros(0, ctx.clone()),
            s_deltas: Eqn::V::zeros(0, ctx.clone()),
            sg_deltas: Eqn::V::zeros(0, ctx.clone()),
            g_delta,
            gamma,
            alpha,
            error_const2,
            u,
            statistics: OdeSolverStatistics {
                number_of_linear_solver_setups: if integrate_main_eqn { 1 } else { 0 },
                number_of_linear_solver_setups_from_checkpoint: if integrate_main_eqn {
                    1
                } else {
                    0
                },
                ..OdeSolverStatistics::default()
            },
            state,
            tstop: None,
            root_finder,
            is_state_modified,
            jacobian_update: JacobianUpdate::new(&problem.ode_options),
            config,
            prev_error_norm: None,
        })
    }

    pub fn new_augmented(
        state: BdfState<Eqn::V, M>,
        problem: &'a OdeSolverProblem<Eqn>,
        augmented_eqn: AugmentedEqn,
        linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        Self::new_augmented_with_config(
            state,
            problem,
            augmented_eqn,
            linear_solver,
            BdfConfig::new(&problem.ode_options),
        )
    }

    pub fn new_augmented_with_config(
        state: BdfState<Eqn::V, M>,
        problem: &'a OdeSolverProblem<Eqn>,
        augmented_eqn: AugmentedEqn,
        linear_solver: LS,
        config: BdfConfig<Eqn::T>,
    ) -> Result<Self, DiffsolError> {
        state.check_sens_consistent_with_problem(problem, &augmented_eqn)?;

        let integrate_main_eqn = augmented_eqn.integrate_main_eqn();
        let mut ret = Self::_new(problem, state, linear_solver, integrate_main_eqn, config)?;

        ret.state.set_augmented_problem(problem, &augmented_eqn)?;

        // allocate internal state for augmented equations (sensitivities or adjoints).
        // The augmented equations have B * C batches, where B is the number of batches
        // and C is the number of channels per batch (C = nparams for sensitivities, C = nout for adjoints).
        let nstates = problem.eqn.rhs().nstates();
        let aug_ctx = augmented_eqn.aug_context().clone();
        ret.s_op = if integrate_main_eqn {
            let mut bdf_callable = BdfCallable::new_no_jacobian(augmented_eqn, aug_ctx.clone());
            bdf_callable.set_c(ret.state.h, ret.alpha[ret.state.order]);
            Some(bdf_callable)
        } else {
            let mut bdf_callable = BdfCallable::new(augmented_eqn, aug_ctx.clone());
            bdf_callable.set_c(ret.state.h, ret.alpha[ret.state.order]);
            ret.nonlinear_solver.set_problem(&bdf_callable);
            ret.nonlinear_solver
                .reset_jacobian(&bdf_callable, &ret.state.s, ret.state.t);
            ret.statistics
                .record_linear_solver_setup(SolverState::Checkpoint);
            Some(bdf_callable)
        };

        ret.s_deltas = <Eqn::V as Vector>::zeros(nstates, aug_ctx.clone());
        ret.s_predict = <Eqn::V as Vector>::zeros(nstates, aug_ctx.clone());
        if let Some(out) = ret.s_op.as_ref().unwrap().eqn().out() {
            ret.sg_deltas = <Eqn::V as Vector>::zeros(out.nout(), aug_ctx);
        }
        Ok(ret)
    }

    /// The `(order+1) x (order+1)` R matrix.
    ///
    /// Entries are given by the first equation on page 8 of [1]. This is used to update the
    /// differences matrix when step size h is varied according to `factor = h_{n+1} / h_n`.
    ///
    /// Note that the U matrix also defined in the same section can also be found using
    /// `factor = 1`, which corresponds to R with a constant step size.
    fn _compute_r(order: usize, factor: Eqn::T) -> BdfMat<Eqn::T> {
        let n = order + 1;
        let mut r = BdfMat::zeros(n, n);
        // one column at a time, so the recurrence walks a slice rather than assuming how the
        // block is packed
        for j in 0..n {
            let j_t = <Eqn::T as FromPrimitive>::from_f64(j as f64).unwrap();
            let col = r.as_col_slice_mut(j);

            // r[0, j] = 1
            col[0] = Eqn::T::one();

            // r[i, j] = r[i-1, j] * (i - 1 - factor * j) / i
            for i in 1..n {
                let i_t = <Eqn::T as FromPrimitive>::from_f64(i as f64).unwrap();
                col[i] = col[i - 1] * (i_t - Eqn::T::one() - factor * j_t) / i_t;
            }
        }
        r
    }

    /// The Newton matrix has changed, so forget the convergence rates measured with the old one.
    fn reset_etas(&mut self) {
        self.convergence.reset_eta();
        self.sens_convergence.reset_eta();
        self.vjp_convergence.reset_eta();
    }

    /// Refactorise the Newton matrix at `(y, t)`, with the `c` the op holds, for a derivative
    /// solve (of the sensitivities, or of [StepVjp::step_vjp]) that the current matrix is too poor
    /// a preconditioner for. The forward solve carries on with the new matrix, so this only
    /// refreshes it earlier than its own schedule would.
    fn refresh_jacobian_for_derivative(&mut self, y: &Eqn::V, t: Eqn::T) {
        let op = self.op.as_ref().unwrap();
        op.set_jacobian_is_stale();
        self.nonlinear_solver.reset_jacobian(op, y, t);
        self.jacobian_update.update_rhs_jacobian(op.c);
        self.reset_etas();
        self.statistics.number_of_derivative_jacobian_refreshes += 1;
    }

    /// Record the step size the step ended with, once any rescale at its end is done.
    fn record_h_end(&mut self) {
        let h = self.state.h;
        if let Some(record) = self.state.last_step.as_mut() {
            record.h_end = h;
        }
    }

    fn _jacobian_updates(&mut self, c: Eqn::T, state: SolverState) {
        let did_update = if self.jacobian_update.check_rhs_jacobian_update(c, &state) {
            let did_reset = if let Some(op) = self.op.as_mut() {
                op.set_jacobian_is_stale();
                self.nonlinear_solver
                    .reset_jacobian(op, &self.state.y, self.state.t);
                true
            } else if let Some(s_op) = self.s_op.as_mut() {
                s_op.set_jacobian_is_stale();
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.state.s, self.state.t);
                true
            } else {
                false
            };
            self.jacobian_update.update_rhs_jacobian(c);
            self.jacobian_update.update_jacobian(c);
            self.reset_etas();
            did_reset
        } else if self.jacobian_update.check_jacobian_update(c, &state) {
            let did_reset = if let Some(op) = self.op.as_mut() {
                self.nonlinear_solver
                    .reset_jacobian(op, &self.state.y, self.state.t);
                true
            } else if let Some(s_op) = self.s_op.as_mut() {
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.state.s, self.state.t);
                true
            } else {
                false
            };
            self.jacobian_update.update_jacobian(c);
            self.reset_etas();
            did_reset
        } else {
            false
        };

        if did_update {
            self.statistics.record_linear_solver_setup(state);
        }
    }

    fn _update_step_size(&mut self, factor: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        //If step size h is changed then also need to update the terms in
        //the first equation of page 9 of [1]:
        //
        //- constant c = h / (1-kappa) gamma_k term
        //- lu factorisation of (M - c * J) used in newton iteration (same equation)

        let new_h = factor * self.state.h;
        self.n_equal_steps = 0;

        // update D using equations in section 3.2 of [1]
        let order = self.state.order;
        let ru = Self::_compute_r(order, factor).mat_mul(&self.u);
        {
            if self.op.is_some() {
                Self::_update_diff_for_step_size(&ru, &mut self.state.diff, order);
                if self.ode_problem.integrate_out {
                    Self::_update_diff_for_step_size(&ru, &mut self.state.gdiff, order);
                }
            }
            if self.s_op.is_some() {
                Self::_update_diff_for_step_size(&ru, &mut self.state.sdiff, order);
                // no output sensitivities to rescale when the augmented equations have no output
                if self.state.sgdiff.nrows() > 0 {
                    Self::_update_diff_for_step_size(&ru, &mut self.state.sgdiff, order);
                }
            }
        }

        if let Some(op) = self.op.as_mut() {
            op.set_c(new_h, self.alpha[order]);
        }
        if let Some(s_op) = self.s_op.as_mut() {
            s_op.set_c(new_h, self.alpha[order]);
        }

        self.state.h = new_h;
        self.convergence.reset_eta_timestep_change();
        self.sens_convergence.reset_eta_timestep_change();

        // if step size too small, then fail
        if self.state.h.abs() < self.config.minimum_timestep {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.state.t.to_f64().unwrap(),
            }));
        }

        Ok(new_h)
    }

    // D[0:order+1] = D[0:order+1] * R * U
    fn _update_diff_for_step_size(ru: &BdfMat<Eqn::T>, diff: &mut M, order: usize) {
        diff.mul_cols_by(order + 1, ru.as_slice());
    }

    fn calculate_output_delta(&mut self) {
        // integrate output function
        let state = &mut self.state;
        let out = self.ode_problem.eqn.out().unwrap();
        out.call_inplace(&self.y_new, self.t_predict, &mut state.dg);
        self.op.as_ref().unwrap().integrate_out(
            &state.dg,
            &state.gdiff,
            &self.gamma.as_slice()[1..],
            self.alpha.as_slice(),
            state.order,
            &mut self.g_delta,
        );
    }

    fn calculate_sens_output_delta(&mut self) {
        let state = &mut self.state;
        let s_op = self.s_op.as_ref().unwrap();

        // integrate sensitivity output equations, all channels at once
        let out = s_op.eqn().out().unwrap();
        out.call_inplace(&state.s, self.t_predict, &mut state.dsg);

        s_op.integrate_out(
            &state.dsg,
            &state.sgdiff,
            &self.gamma.as_slice()[1..],
            self.alpha.as_slice(),
            state.order,
            &mut self.sg_deltas,
        );
    }

    fn update_differences_and_integrate_out(&mut self) {
        let order = self.state.order;
        let state = &mut self.state;

        // update differences
        Self::_update_diff(order, &self.y_delta, &mut state.diff);

        // integrate output function
        if self.ode_problem.integrate_out {
            Self::_predict_using_diff(&mut state.g, &state.gdiff, order);
            state.g.axpy(Eqn::T::one(), &self.g_delta, Eqn::T::one());

            // update output difference
            Self::_update_diff(order, &self.g_delta, &mut state.gdiff);
        }

        // do the same for sensitivities, every channel in one operation
        if let Some(s_op) = self.s_op.as_ref() {
            // update sensitivity differences
            Self::_update_diff(order, &self.s_deltas, &mut state.sdiff);

            // integrate sensitivity output equations
            if s_op.eqn().out().is_some() {
                Self::_predict_using_diff(&mut state.sg, &state.sgdiff, order);
                state.sg.axpy(Eqn::T::one(), &self.sg_deltas, Eqn::T::one());

                // update sensitivity output difference
                Self::_update_diff(order, &self.sg_deltas, &mut state.sgdiff);
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
        //(see DenseMatrix::update_backward_diff)
        diff.update_backward_diff(order, d);
    }

    // predict forward to new step (eq 2 in [1])
    //
    // This is a gemv over the columns [0..order] of diff
    fn _predict_using_diff(y_predict: &mut Eqn::V, diff: &M, order: usize) {
        let ones = [Eqn::T::one(); COEFFS_LEN];
        diff.gemv_cols(
            0,
            order + 1,
            Eqn::T::one(),
            &ones,
            Eqn::T::zero(),
            y_predict,
        );
    }

    fn _predict_forward(&mut self) {
        let state = &self.state;
        Self::_predict_using_diff(&mut self.y_predict, &state.diff, state.order);

        // update psi and c (h, D, y0 has changed)
        if let Some(op) = self.op.as_mut() {
            op.set_psi_and_y0(
                &state.diff,
                &self.gamma.as_slice()[1..],
                self.alpha.as_slice(),
                state.order,
                &self.y_predict,
            );
        }

        // update time
        let t_new = state.t + state.h;
        self.t_predict = t_new;
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        // check if the we are at tstop
        let state = &self.state;
        let troundoff = <Eqn::T as FromPrimitive>::from_f64(100.0).unwrap()
            * Eqn::T::EPSILON
            * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if (state.h > M::T::zero() && tstop < state.t - troundoff)
            || (state.h < M::T::zero() && tstop > state.t + troundoff)
        {
            let error = OdeSolverError::StopTimeBeforeCurrentTime {
                stop_time: self.tstop.unwrap().to_f64().unwrap(),
                state_time: state.t.to_f64().unwrap(),
            };
            self.tstop = None;

            return Err(DiffsolError::from(error));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if (state.h > M::T::zero() && state.t + state.h > tstop + troundoff)
            || (state.h < M::T::zero() && state.t + state.h < tstop - troundoff)
        {
            debug!(
                "Adjusting step size to hit tstop at t = {}",
                tstop.to_f64().unwrap()
            );
            let factor = (tstop - state.t) / state.h;
            // update step size ignoring the possible "step size too small" error
            let _ = self._update_step_size(factor);
        }
        Ok(None)
    }

    fn initialise_to_first_order(&mut self) {
        self.n_equal_steps = 0;

        // NOTE: tried this but did not give an improvement in perf.
        // keeping it for now just in case it is useful in the future.
        //
        // scale step size h to account for reduction in order at order n error is O(h^{n+2}))
        // error_b = C * h_old^{n+2}
        // error_a = C * h_new^{3}
        // set error_a = error_b
        // => h_new = h_old^{(n+2)/3}
        //let sign_h = if self.state.h >= Eqn::T::zero() {
        //    Eqn::T::one()
        //} else {
        //    -Eqn::T::one()
        //};
        //self.state.h = sign_h * self.state.h.abs().powf(M::T::from_f64((self.state.order + 2) as f64 / 3.0).unwrap());

        self.state.initialise_diff_to_first_order();

        if self.ode_problem.integrate_out {
            self.state.initialise_gdiff_to_first_order();
        }
        if let Some(s_op) = self.s_op.as_ref() {
            self.state.initialise_sdiff_to_first_order();
            if s_op.eqn().out().is_some() {
                self.state.initialise_sgdiff_to_first_order();
            }
        }

        self.u = Self::_compute_r(1, Eqn::T::one());

        self.is_state_modified = false;
    }

    //interpolate solution at time values t* where t-h < t* < t
    //definition of the interpolating polynomial can be found on page 7 of [1]
    fn interpolate_from_diff(
        t: Eqn::T,
        diff: &M,
        t1: Eqn::T,
        h: Eqn::T,
        order: usize,
        y: &mut Eqn::V,
    ) {
        // one weight per column `0..=order`, so the length says what `[..=order]` used to
        let mut weights = BdfVec::zeros(order + 1);
        let mut time_factor = Eqn::T::one();
        weights[0] = Eqn::T::one();
        for (i, weight) in weights.as_mut_slice().iter_mut().enumerate().skip(1) {
            let i_t = <Eqn::T as FromPrimitive>::from_f64((i - 1) as f64).unwrap();
            time_factor *= (t - (t1 - h * i_t)) / (h * (Eqn::T::one() + i_t));
            *weight = time_factor;
        }
        diff.gemv_cols(
            0,
            order + 1,
            Eqn::T::one(),
            weights.as_slice(),
            Eqn::T::zero(),
            y,
        );
    }

    // Interpolate the time derivative dy/dt of the BDF polynomial at time t.
    // Uses the product-rule differentiation of the interpolating polynomial from page 7 of [1].
    // For each term pi_i(t) = prod_{j=0}^{i-1} (t - (t1 - j*h)) / ((j+1)*h),
    // we maintain pi (the product) and d_pi (its derivative) iteratively.
    fn interpolate_derivative_from_diff(
        t: Eqn::T,
        diff: &M,
        t1: Eqn::T,
        h: Eqn::T,
        order: usize,
        dy: &mut Eqn::V,
    ) {
        // weights[i] is the weight of column i + 1; column 0 is constant in t so it drops out
        let mut weights = BdfVec::zeros(order);
        let mut pi = Eqn::T::one();
        let mut d_pi = Eqn::T::zero();
        for (i, weight) in weights.as_mut_slice().iter_mut().enumerate() {
            let i_t = <Eqn::T as FromPrimitive>::from_f64(i as f64).unwrap();
            let denom = h * (Eqn::T::one() + i_t);
            let w = (t - (t1 - h * i_t)) / denom;
            let dw = Eqn::T::one() / denom;
            // d(pi_{i+1})/dt = d(pi_i)/dt * w + pi_i * dw
            let new_d_pi = d_pi * w + pi * dw;
            pi *= w;
            d_pi = new_d_pi;
            *weight = d_pi;
        }
        // an order of zero leaves an empty range, which zeroes dy
        diff.gemv_cols(
            1,
            order + 1,
            Eqn::T::one(),
            weights.as_slice(),
            Eqn::T::zero(),
            dy,
        );
    }

    /// The squared weighted norm of the local error estimate of the step just solved, combining
    /// eqs. 3, 4 and 6 of [1]: the error is `C_k h^{k+1} y^{(k+1)}`, and the correction
    /// `d = y_{n+1} - y_pred = D^{k+1} y_{n+1}` approximates `h^{k+1} y^{(k+1)}`, so the error is
    /// `C_k d`, with `error_const2[k] = C_k^2`.
    fn error_control(&self) -> Eqn::T {
        let state = &self.state;
        let order = state.order;
        let output_in_error_control = self.ode_problem.output_in_error_control();
        let integrate_sens = self.s_op.is_some();
        let sens_in_error_control =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().include_in_error_control();
        let integrate_sens_out =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().out().is_some();
        let sens_output_in_error_control = integrate_sens_out
            && self
                .s_op
                .as_ref()
                .unwrap()
                .eqn()
                .include_out_in_error_control();

        let mut error_norm = M::T::zero();
        if self.op.is_some() {
            let atol = &self.ode_problem.atol;
            let rtol = self.ode_problem.rtol;
            let err = self.y_delta.squared_norm(&state.y, atol, rtol) * self.error_const2[order];
            error_norm = error_norm.max(err);
            if output_in_error_control {
                let rtol = self.ode_problem.out_rtol.unwrap();
                let atol = self.ode_problem.out_atol.as_ref().unwrap();
                let err =
                    self.g_delta.squared_norm(&state.g, atol, rtol) * self.error_const2[order];
                error_norm = error_norm.max(err);
            }
        }
        // the batched norm already reduces by taking the maximum over the channel lanes
        if sens_in_error_control {
            let aug_eqn = self.s_op.as_ref().unwrap().eqn();
            let sens_rtol = aug_eqn.rtol().unwrap();
            let sens_atol = aug_eqn
                .atol()
                .expect("aug eqn in error control should always have an atol");
            let err = self.s_deltas.squared_norm(&state.s, sens_atol, sens_rtol)
                * self.error_const2[order];
            error_norm = error_norm.max(err);
        }
        if sens_output_in_error_control {
            let rtol = self.s_op.as_ref().unwrap().eqn().out_rtol().unwrap();
            let atol = self.s_op.as_ref().unwrap().eqn().out_atol().unwrap();
            let err = self.sg_deltas.squared_norm(&state.sg, atol, rtol) * self.error_const2[order];
            error_norm = error_norm.max(err);
        }
        error_norm
    }

    fn predict_error_control(&self, order: usize) -> Eqn::T {
        let state = &self.state;
        let output_in_error_control = self.ode_problem.output_in_error_control();
        let integrate_sens = self.s_op.is_some();
        let sens_in_error_control =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().include_in_error_control();
        let integrate_sens_out =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().out().is_some();
        let sens_output_in_error_control = integrate_sens_out
            && self
                .s_op
                .as_ref()
                .unwrap()
                .eqn()
                .include_out_in_error_control();

        let atol = &self.ode_problem.atol;
        let rtol = self.ode_problem.rtol;
        let mut error_norm = M::T::zero();
        if self.op.is_some() {
            let err = state
                .diff
                .column(order + 1)
                .squared_norm(&state.y, atol, rtol)
                * self.error_const2[order];
            error_norm = error_norm.max(err);
            if output_in_error_control {
                let rtol = self.ode_problem.out_rtol.unwrap();
                let atol = self.ode_problem.out_atol.as_ref().unwrap();
                let err = state
                    .gdiff
                    .column(order + 1)
                    .squared_norm(&state.g, atol, rtol)
                    * self.error_const2[order];
                error_norm = error_norm.max(err);
            }
        }
        if sens_in_error_control {
            let aug_eqn = self.s_op.as_ref().unwrap().eqn();
            let sens_rtol = aug_eqn.rtol().unwrap();
            let sens_atol = aug_eqn.atol().unwrap();
            let err = state
                .sdiff
                .column(order + 1)
                .squared_norm(&state.s, sens_atol, sens_rtol)
                * self.error_const2[order];
            error_norm = error_norm.max(err);
        }
        if sens_output_in_error_control {
            let rtol = self.s_op.as_ref().unwrap().eqn().out_rtol().unwrap();
            let atol = self.s_op.as_ref().unwrap().eqn().out_atol().unwrap();
            let err = state
                .sgdiff
                .column(order + 1)
                .squared_norm(&state.sg, atol, rtol)
                * self.error_const2[order];
            error_norm = error_norm.max(err);
        }
        error_norm
    }

    fn sensitivity_solve(&mut self, t_new: Eqn::T) -> Result<(), DiffsolError> {
        let order = self.state.order;

        // update for new state
        if let Some(op) = self.op.as_ref() {
            let s_op = self.s_op.as_mut().unwrap();
            let dy_new = op.tmp();
            s_op.eqn_mut()
                .update_rhs_out_state(&self.y_new, &dy_new, t_new);
        }

        // solve for the sensitivity equations discretised using BDF, every channel at once
        let s_op = self.s_op.as_mut().unwrap();
        {
            let state = &self.state;
            // predict forward to new step
            Self::_predict_using_diff(&mut self.s_predict, &state.sdiff, order);

            // setup op
            s_op.set_psi_and_y0(
                &state.sdiff,
                &self.gamma.as_slice()[1..],
                self.alpha.as_slice(),
                order,
                &self.s_predict,
            );
        }

        // with the main equations, the sensitivity equations are a linear solve with the Jacobian
        // at the corrected solution, the derivative of the step. Converge it to the derivative
        // tolerance and, if the Newton matrix is too poor a preconditioner for that, refresh it
        // and retry once rather than reject the step. Without them (an adjoint solve backwards)
        // it is the step's own solve
        let derivative = self.op.is_some();
        if derivative {
            self.sens_convergence
                .set_tolerance(self.config.derivative_solve_tolerance);
            self.sens_convergence
                .set_max_iter(self.config.derivative_solve_max_iter);
        }
        let mut result = self.sensitivity_corrector_solve(t_new);
        if derivative && result.is_err() {
            let y_new = self.y_new.clone();
            self.refresh_jacobian_for_derivative(&y_new, t_new);
            result = self.sensitivity_corrector_solve(t_new);
        }
        result?;

        if self.s_op.as_ref().unwrap().eqn().out().is_some() {
            self.calculate_sens_output_delta();
        }
        Ok(())
    }

    /// The corrector solve of the sensitivity equations, from the prediction set up by
    /// [Self::sensitivity_solve].
    fn sensitivity_corrector_solve(&mut self, t_new: Eqn::T) -> Result<(), DiffsolError> {
        let s_op = self.s_op.as_ref().unwrap();
        // solve every channel at once: the Jacobian is the one already factorised for the main
        // equations, shared by all channels, and the solution holds one channel per batch lane
        let s_new = &mut self.state.s;
        s_new.copy_from(&self.s_predict);
        trace!("Solving sensitivity equations");
        let convergence = if self.op.is_some() {
            &mut self.sens_convergence
        } else {
            &mut self.convergence
        };
        self.nonlinear_solver
            .solve_in_place(s_op, s_new, t_new, &self.s_predict, convergence)?;
        self.statistics.number_of_nonlinear_solver_iterations += convergence.niter();
        let s_new = &*s_new;
        Eqn::V::for_each_elem_mut(
            [&mut self.s_deltas],
            [s_new, &self.s_predict],
            |[d], [s_new, s_predict], _lane, i| *d = s_new[i] - s_predict[i],
        );
        Ok(())
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> OdeSolverMethod<'a, Eqn> for Bdf<'a, Eqn, LS, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicit,
    AugmentedEqn: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    Eqn::V: DefaultDenseMatrix,
    LS: LinearSolver<Eqn::M>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    type State = BdfState<Eqn::V, M>;
    type Config = BdfConfig<Eqn::T>;

    fn config(&self) -> &BdfConfig<Eqn::T> {
        &self.config
    }

    fn get_statistics(&self) -> &OdeSolverStatistics {
        &self.statistics
    }

    fn config_mut(&mut self) -> &mut BdfConfig<Eqn::T> {
        &mut self.config
    }

    fn order(&self) -> usize {
        self.state.order
    }

    fn apply_reset(&mut self) -> Result<(), DiffsolError> {
        let problem = self.problem();
        self.state.as_mut().apply_reset_with_mass::<LS, _>(problem)
    }

    fn apply_reset_with_sens(&mut self, root_idx: usize) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let problem = self.problem();
        self.state
            .as_mut()
            .apply_reset_with_sens_mass::<LS, _>(problem, root_idx)
    }

    fn jacobian(&self) -> Option<Ref<'_, <Eqn>::M>> {
        let t = self.state.t;
        if let Some(op) = self.op.as_ref() {
            let x = &self.state.y;
            Some(op.rhs_jac(x, t))
        } else {
            let x = &self.state.s;
            self.s_op.as_ref().map(|s_op| s_op.rhs_jac(x, t))
        }
    }

    fn mass(&self) -> Option<Ref<'_, <Eqn>::M>> {
        let t = self.state.t;
        if let Some(op) = self.op.as_ref() {
            Some(op.mass(t))
        } else {
            self.s_op.as_ref().map(|s_op| s_op.mass(t))
        }
    }

    fn set_state(&mut self, state: Self::State) {
        debug!(
            "Manually setting BDF state, step size = {}, order = {}, time = {}",
            state.h, state.order, state.t
        );
        let old_order = self.state.order;
        self.state = state;

        if let Some(op) = self.op.as_mut() {
            op.set_c(self.state.h, self.alpha[self.state.order]);
        }
        if let Some(s_op) = self.s_op.as_mut() {
            s_op.set_c(self.state.h, self.alpha[self.state.order]);
        }

        // order might have changed
        if self.state.order != old_order {
            self.u = Self::_compute_r(self.state.order, Eqn::T::one());
        }

        // reinitialise jacobian updates as if a checkpoint was taken
        self._jacobian_updates(
            self.state.h * self.alpha[self.state.order],
            SolverState::Checkpoint,
        );
    }

    fn interpolate_inplace(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        let state = &self.state;
        check_interpolate_shape(y, &state.y)?;
        if !check_interpolate_time(
            t,
            y,
            &state.y,
            state.t,
            state.h,
            None,
            self.is_state_modified,
        )? {
            return Ok(());
        }
        Self::interpolate_from_diff(t, &state.diff, state.t, state.h, state.order, y);
        Ok(())
    }

    fn interpolate_dy_inplace(&self, t: Eqn::T, dy: &mut Eqn::V) -> Result<(), DiffsolError> {
        let state = &self.state;
        check_interpolate_shape(dy, &state.y)?;
        if !check_interpolate_time(
            t,
            dy,
            &state.dy,
            state.t,
            state.h,
            None,
            self.is_state_modified,
        )? {
            return Ok(());
        }
        Self::interpolate_derivative_from_diff(t, &state.diff, state.t, state.h, state.order, dy);
        Ok(())
    }

    fn interpolate_out_inplace(&self, t: Eqn::T, g: &mut Eqn::V) -> Result<(), DiffsolError> {
        let state = &self.state;
        check_interpolate_shape(g, &state.g)?;
        if !check_interpolate_time(
            t,
            g,
            &state.g,
            state.t,
            state.h,
            None,
            self.is_state_modified,
        )? {
            return Ok(());
        }
        Self::interpolate_from_diff(t, &state.gdiff, state.t, state.h, state.order, g);
        Ok(())
    }

    fn interpolate_sens_inplace(
        &self,
        t: <Eqn as Op>::T,
        sens: &mut Eqn::V,
    ) -> Result<(), DiffsolError> {
        let state = &self.state;
        check_interpolate_shape(sens, &state.s)?;

        // no sensitivities to interpolate
        if state.s.len() == 0 {
            return Ok(());
        }

        if !check_interpolate_time(
            t,
            sens,
            &state.s,
            state.t,
            state.h,
            None,
            self.is_state_modified,
        )? {
            return Ok(());
        }
        Self::interpolate_from_diff(t, &state.sdiff, state.t, state.h, state.order, sens);
        Ok(())
    }

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.ode_problem
    }

    fn state(&self) -> StateRef<'_, Eqn::V> {
        self.state.as_ref()
    }

    fn into_state(mut self) -> BdfState<Eqn::V, M> {
        let ctx = self.problem().eqn.context().clone();
        std::mem::replace(&mut self.state, BdfState::new_empty(ctx))
    }

    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        self.is_state_modified = true;
        self.state.as_mut()
    }

    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        let state = &mut self.state;
        if self.is_state_modified {
            if t != state.t {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
            // already at the requested time, nothing to do
            return Ok(());
        }
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }
        let current_t = state.t;
        let current_h = state.h;
        let order = state.order;
        Self::interpolate_from_diff(t, &state.diff, current_t, current_h, order, &mut state.y);
        Self::interpolate_derivative_from_diff(
            t,
            &state.diff,
            current_t,
            current_h,
            order,
            &mut state.dy,
        );
        if self.ode_problem.integrate_out {
            Self::interpolate_from_diff(t, &state.gdiff, current_t, current_h, order, &mut state.g);
        }
        if self.s_op.is_some() {
            Self::interpolate_from_diff(t, &state.sdiff, current_t, current_h, order, &mut state.s);
            Self::interpolate_derivative_from_diff(
                t,
                &state.sdiff,
                current_t,
                current_h,
                order,
                &mut state.ds,
            );
        }
        state.t = t;
        // the state is no longer the end of the recorded step
        state.last_step = None;
        self.is_state_modified = true;
        Ok(())
    }

    fn checkpoint(&mut self) -> Self::State {
        debug!("Taking checkpoint");
        self._jacobian_updates(
            self.state.h * self.alpha[self.state.order],
            SolverState::Checkpoint,
        );
        self.state.clone()
    }

    fn state_clone(&self) -> Self::State {
        self.state.clone()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        debug!(
            "Taking BDF step at time {} with step size {} and order {}",
            self.state.t, self.state.h, self.state.order
        );
        let mut safety: Eqn::T;
        let mut error_norm: Eqn::T;
        let problem = self.ode_problem;
        let integrate_out = problem.integrate_out;
        let integrate_sens = self.s_op.is_some();
        let old_num_error_test_failures = self.statistics.number_of_error_test_failures;

        let mut convergence_fail = false;

        if self.is_state_modified {
            // reinitalise root finder if needed
            if let (Some(root_fn), Some(root_finder)) =
                (problem.eqn.root(), self.root_finder.as_ref())
            {
                let state = &self.state;
                root_finder.init(&root_fn, &state.y, state.t);
            }
            // reinitialise diff matrix
            self.initialise_to_first_order();
            let c = self.state.h * self.alpha[self.state.order];
            if let Some(op) = self.op.as_mut() {
                op.set_c(self.state.h, self.alpha[self.state.order]);
            }
            if let Some(s_op) = self.s_op.as_mut() {
                s_op.set_c(self.state.h, self.alpha[self.state.order]);
            }
            self._jacobian_updates(c, SolverState::StepSuccess);
            self.prev_error_norm = None;

            // reinitialise tstop if needed
            if let Some(t_stop) = self.tstop {
                self.set_stop_time(t_stop)?;
            }
            debug!(
                "State was modified, reinitialised to first order with h = {}",
                self.state.h
            );
        }

        self._predict_forward();

        // loop until step is accepted
        loop {
            let order = self.state.order;
            self.y_new.copy_from(&self.y_predict);

            // solve BDF equation using y0 as starting point
            let mut solve_result = Ok(());
            if let Some(op) = self.op.as_ref() {
                solve_result = self.nonlinear_solver.solve_in_place(
                    op,
                    &mut self.y_new,
                    self.t_predict,
                    &self.y_predict,
                    &mut self.convergence,
                );
                // update statistics
                self.statistics.number_of_nonlinear_solver_iterations += self.convergence.niter();

                if solve_result.is_ok() {
                    // the correction d = y_{n+1} - y_pred, used for the error estimate and to
                    // update the differences
                    Eqn::V::for_each_elem_mut(
                        [&mut self.y_delta],
                        [&self.y_new, &self.y_predict],
                        |[d], [y_new, y_predict], _lane, i| *d = y_new[i] - y_predict[i],
                    );

                    // deal with output equations
                    if integrate_out {
                        self.calculate_output_delta();
                    }
                }
            }

            // only calculate sensitivities if solve was successful
            if solve_result.is_ok()
                && integrate_sens
                && self.sensitivity_solve(self.t_predict).is_err()
            {
                solve_result = Err(ode_solver_error!(SensitivitySolveFailed));
                // with the main equations, the sensitivity solve has already refreshed the Newton
                // matrix at this c and retried, so another refresh would not help: cut the step
                if self.op.is_some() {
                    convergence_fail = true;
                }
            }

            // handle case where either nonlinear solve failed
            if solve_result.is_err() {
                self.statistics.number_of_nonlinear_solver_fails += 1;
                if self.statistics.number_of_nonlinear_solver_fails
                    > self.config.maximum_newton_fails
                {
                    return Err(DiffsolError::from(
                        OdeSolverError::TooManyNonlinearSolverFailures {
                            time: self.state.t.to_f64().unwrap(),
                            num_failures: self.statistics.number_of_nonlinear_solver_fails,
                        },
                    ));
                }
                if convergence_fail {
                    // newton iteration did not converge, but jacobian has already been
                    // evaluated so reduce step size by 0.3 (as per [1]) and try again
                    self.prev_error_norm = None;
                    let new_h =
                        self._update_step_size(<Eqn::T as FromPrimitive>::from_f64(0.3).unwrap())?;
                    debug!(
                        "Second convergence failure, reducing step size to {:.3e} and trying again",
                        new_h.to_f64().unwrap()
                    );
                    self._jacobian_updates(
                        new_h * self.alpha[order],
                        SolverState::SecondConvergenceFail,
                    );

                    // new prediction
                    self._predict_forward();

                    // update statistics
                } else {
                    // newton iteration did not converge, so update jacobian and try again
                    debug!("First convergence failure, updating Jacobian and trying again",);
                    self.prev_error_norm = None;
                    self._jacobian_updates(
                        self.state.h * self.alpha[order],
                        SolverState::FirstConvergenceFail,
                    );
                    convergence_fail = true;
                    // same prediction as last time
                }
                continue;
            }

            error_norm = self.error_control();

            // need to caulate safety even if step is accepted
            let maxiter = self.convergence.max_iter() as f64;
            let niter = self.convergence.niter() as f64;
            safety = <Eqn::T as FromPrimitive>::from_f64(
                0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter),
            )
            .unwrap();

            // do the error test
            if error_norm <= Eqn::T::one() {
                // step is accepted
                debug!(
                    "Step accepted with error norm {:.3e}",
                    error_norm.to_f64().unwrap()
                );
                break;
            } else {
                // step is rejected
                // calculate optimal step size factor using PI controller
                // and reduce step size and try again
                let mut factor = safety
                    * pi_controller_raw(
                        error_norm,
                        self.prev_error_norm,
                        self.ode_problem.ode_options.pi_control_integral,
                        self.ode_problem.ode_options.pi_control_proportional,
                        order + 1,
                    );
                self.prev_error_norm = None;
                if factor < self.config.minimum_timestep_shrink {
                    factor = self.config.minimum_timestep_shrink;
                }
                debug!(
                    "Step rejected with error norm {:.3e}, reducing step size by factor {:.3e}",
                    error_norm.to_f64().unwrap(),
                    factor.to_f64().unwrap()
                );
                let new_h = self._update_step_size(factor)?;
                self._jacobian_updates(new_h * self.alpha[order], SolverState::ErrorTestFail);

                // new prediction
                self._predict_forward();

                // update statistics
                self.statistics.number_of_error_test_failures += 1;
                if self.statistics.number_of_error_test_failures - old_num_error_test_failures
                    >= self.config.maximum_error_test_failures
                {
                    return Err(DiffsolError::from(
                        OdeSolverError::TooManyErrorTestFailures {
                            time: self.state.t.to_f64().unwrap(),
                            num_failures: self.statistics.number_of_error_test_failures,
                        },
                    ));
                }
            }
        }

        // record the accepted attempt and the history it started from, so that `step_vjp` can
        // replay it: h_init and the previous record still describe that history. Without either (a state built some other way) assume no rescale.
        let h_ref = match (self.state.h_init, self.state.last_step) {
            (Some(h_init), _) => h_init,
            (None, Some(previous)) => previous.h_end,
            (None, None) => self.state.h,
        };
        self.state.last_step = Some(BdfStepRecord {
            h_ref,
            h: self.state.h,
            order: self.state.order,
            from_init: self.state.h_init.is_some(),
            h_end: self.state.h,
        });
        self.state.h_init = None;

        // take the accepted step
        self.update_differences_and_integrate_out();

        {
            let state = &mut self.state;
            // without main equations there is no corrector solve, so y stays at the prediction
            if self.op.is_some() {
                state.y.copy_from(&self.y_new);
            } else {
                state.y.copy_from(&self.y_predict);
            }
            state.t = self.t_predict;
            // dy = diff[:, 1] / h, in one pass rather than a copy and a scale
            let inv_h = Eqn::T::one() / state.h;
            state
                .dy
                .axpy_v(inv_h, &state.diff.column(1), Eqn::T::zero());
            // and likewise ds from the sensitivities' history; s is already the corrector's
            if self.s_op.is_some() {
                state
                    .ds
                    .axpy_v(inv_h, &state.sdiff.column(1), Eqn::T::zero());
            }
        }

        // update statistics
        self.statistics.number_of_steps += 1;
        self.jacobian_update.step();
        self.prev_error_norm = Some(error_norm);

        // a change in order is only done after running at order k for k + 1 steps
        // (see page 83 of [2])
        self.n_equal_steps += 1;

        if self.n_equal_steps > self.state.order {
            let order = self.state.order;
            // similar to the optimal step size factor we calculated above for the current
            // order k, we need to calculate the optimal step size factors for orders
            // k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
            let error_m_norm = if order > 1 {
                self.predict_error_control(order - 1)
            } else {
                Eqn::T::INFINITY
            };
            let error_p_norm = if order < MAX_ORDER {
                self.predict_error_control(order + 1)
            } else {
                Eqn::T::INFINITY
            };

            let prev_error = self.prev_error_norm;
            let pi_i = self.ode_problem.ode_options.pi_control_integral;
            let pi_p = self.ode_problem.ode_options.pi_control_proportional;
            let factors: [Eqn::T; 3] = [
                pi_controller_raw(error_m_norm, prev_error, pi_i, pi_p, order),
                pi_controller_raw(error_norm, prev_error, pi_i, pi_p, order + 1),
                pi_controller_raw(error_p_norm, prev_error, pi_i, pi_p, order + 2),
            ];

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
                let old_order = self.state.order;
                let new_order = match max_index {
                    0 => old_order - 1,
                    1 => old_order,
                    2 => old_order + 1,
                    _ => unreachable!(),
                };
                self.state.order = new_order;
                if max_index != 1 {
                    self.u = Self::_compute_r(new_order, Eqn::T::one());
                }
                new_order
            };

            let mut factor = safety * factors[max_index];
            if factor > self.config.maximum_timestep_growth {
                factor = self.config.maximum_timestep_growth;
            }
            if factor < self.config.minimum_timestep_shrink {
                factor = self.config.minimum_timestep_shrink;
            }
            if factor >= self.config.minimum_timestep_growth
                || factor <= self.config.maximum_timestep_shrink
                || max_index == 0
                || max_index == 2
            {
                // the rescale is done even if the new step size is too small, so the step record
                // must still end with it
                let new_h = self
                    ._update_step_size(factor)
                    .inspect_err(|_| self.record_h_end())?;
                debug!(
                    "Changing step size to {:.6} and order to {}",
                    new_h.to_f64().unwrap(),
                    order
                );
                self._jacobian_updates(new_h * self.alpha[order], SolverState::StepSuccess);
            }
        }

        self.record_h_end();

        // check for root within accepted step
        if let (Some(root_fn), Some(root_finder)) =
            (self.ode_problem.eqn.root(), self.root_finder.as_ref())
        {
            let ret = root_finder.check_root(
                &|t: <Eqn as Op>::T, y: &mut <Eqn as Op>::V| self.interpolate_inplace(t, y),
                &root_fn,
                self.state.as_ref().y,
                self.state.as_ref().t,
            );
            if let Some((root, root_idx)) = ret {
                debug!("Root found at time {}", root);
                return Ok(OdeSolverStopReason::RootFound(root, root_idx));
            }
        }

        if let Some(tstop) = self.tstop {
            if let Some(reason) = self.handle_tstop(tstop).unwrap() {
                return Ok(reason);
            }
            // the step may have been shortened to hit tstop next time
            self.record_h_end();
        }

        // just a normal step, no roots or tstop reached
        Ok(OdeSolverStopReason::InternalTimestep)
    }

    fn set_stop_time(&mut self, tstop: <Eqn as Op>::T) -> Result<(), DiffsolError> {
        self.tstop = Some(tstop);
        if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
            let error = OdeSolverError::StopTimeAtCurrentTime;
            self.tstop = None;
            return Err(DiffsolError::from(error));
        }
        Ok(())
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> Bdf<'a, Eqn, LS, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    AugmentedEqn: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    Eqn::V: DefaultDenseMatrix,
    LS: LinearSolver<Eqn::M>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    /// Rescale the first `order + 1` columns of `diff` from step size `from` to `to`, as
    /// `_update_step_size` does, or apply the transpose of that rescale.
    fn ad_rescale(diff: &mut M, order: usize, from: Eqn::T, to: Eqn::T, transpose: bool) {
        if from == to {
            return;
        }
        let ru =
            Self::_compute_r(order, to / from).mat_mul(&Self::_compute_r(order, Eqn::T::one()));
        if transpose {
            diff.mul_cols_by_transpose(order + 1, ru.as_slice());
        } else {
            diff.mul_cols_by(order + 1, ru.as_slice());
        }
    }

    /// Run a solve with the operator's Jacobian at `end`, the state at the end of the step,
    /// using the current Newton matrix as a preconditioner, with `x` holding the initial guess
    /// and returning the solution. If that is too poor a preconditioner for the solve to
    /// converge, refresh it at `end` and retry once.
    fn vjp_solve(
        &mut self,
        end: &BdfState<Eqn::V, M>,
        x: &mut Eqn::V,
        solve: impl Fn(
            &NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
            &BdfCallable<&'a Eqn>,
            &mut Eqn::V,
            &mut Convergence<'a, Eqn::V>,
        ) -> Result<(), DiffsolError>,
    ) -> Result<(), DiffsolError> {
        let x0 = x.clone();
        // a matrix factorised here is already fresh, so a failed solve with it is not retried
        let mut refreshed = !self.nonlinear_solver.is_jacobian_set();
        if refreshed {
            self.refresh_jacobian_for_derivative(&end.y, end.t);
        }
        loop {
            self.vjp_convergence
                .set_tolerance(self.config.derivative_solve_tolerance);
            self.vjp_convergence
                .set_max_iter(self.config.derivative_solve_max_iter);
            let op = self.op.as_ref().unwrap();
            match solve(&self.nonlinear_solver, op, x, &mut self.vjp_convergence) {
                Ok(()) => return Ok(()),
                Err(_) if !refreshed => {
                    self.refresh_jacobian_for_derivative(&end.y, end.t);
                    x.copy_from(&x0);
                    refreshed = true;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Check the inputs of [StepVjp::step_vjp], and return the record of the step that ended at
    /// `end`: the solver integrates the main equations, `end` is the end of a step, the
    /// cotangent has the sizes of `end` and `p_bar` has `nparams` elements, the cotangent's
    /// lanes are a whole number of directions for each lane of `end`, and `p_bar` has exactly
    /// as many.
    fn vjp_check_inputs(
        &self,
        end: &BdfState<Eqn::V, M>,
        cotangent: &BdfState<Eqn::V, M>,
        p_bar: &Eqn::V,
    ) -> Result<BdfStepRecord<Eqn::T>, DiffsolError> {
        if self.op.is_none() {
            return Err(ode_solver_error!(StepDifferentiationNotSupported));
        }
        let step = end
            .last_step
            .ok_or(ode_solver_error!(NoStepToDifferentiate))?;
        let nparams = self.ode_problem.eqn.rhs().nparams();
        // sizes as (rows, columns of a matrix), formatted only on a mismatch
        let shape = |m: &M| (m.nrows(), Some(m.ncols()));
        let vector = |v: &Eqn::V| (v.len(), None);
        for (what, found, expected) in [
            ("y", vector(&cotangent.y), vector(&end.y)),
            ("dy", vector(&cotangent.dy), vector(&end.dy)),
            ("diff", shape(&cotangent.diff), shape(&end.diff)),
            ("the parameter cotangent", vector(p_bar), (nparams, None)),
        ] {
            if found != expected {
                let size = |(rows, cols): (usize, Option<usize>)| match cols {
                    Some(cols) => format!("{rows} x {cols}"),
                    None => rows.to_string(),
                };
                return Err(DiffsolError::from(
                    OdeSolverError::StepDifferentiationSizeMismatch {
                        what,
                        found: size(found),
                        expected: size(expected),
                    },
                ));
            }
        }
        let state_lanes = end.y.context().nbatch();
        let lanes = cotangent.y.context().nbatch();
        let mismatch = |what, found, expected: String| {
            Err(DiffsolError::from(
                OdeSolverError::StepDifferentiationBatchMismatch {
                    what,
                    found,
                    expected,
                    state_lanes,
                },
            ))
        };
        if !lanes.is_multiple_of(state_lanes) {
            return mismatch("y", lanes, format!("a multiple of {state_lanes}"));
        }
        for (what, found) in [
            ("dy", cotangent.dy.context().nbatch()),
            ("diff", cotangent.diff.context().nbatch()),
        ] {
            if found != lanes {
                return mismatch(what, found, lanes.to_string());
            }
        }
        let p_lanes = p_bar.context().nbatch();
        if p_lanes != lanes {
            return mismatch("the parameter cotangent", p_lanes, lanes.to_string());
        }
        Ok(step)
    }

    fn ad_vjp(
        &mut self,
        step: BdfStepRecord<Eqn::T>,
        end: &BdfState<Eqn::V, M>,
        c: &mut BdfState<Eqn::V, M>,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError> {
        let BdfStepRecord {
            h_ref,
            h,
            order,
            from_init,
            h_end,
        } = step;
        // differentiate with the step's own c, which step_vjp restores afterwards
        let alpha = self.alpha[order];
        self.op.as_mut().unwrap().set_c(h, alpha);
        let one = Eqn::T::one();
        let zero = Eqn::T::zero();

        // fold in the readouts: y = diff[:, 0] at the end, and dy = diff[:, 1] / h before the
        // rescale at the end of the step
        c.diff.column_mut(0).axpy(one, &c.y, one);
        Self::ad_rescale(&mut c.diff, end.order, h, h_end, true);
        c.diff.column_mut(1).axpy(one / h, &c.dy, one);
        c.y.fill(zero);
        c.dy.fill(zero);

        // update of the history, then the corrector: (M - c J)ᵀ w = d_bar
        let mut d_bar = c.y.clone();
        c.diff.update_backward_diff_transpose(order, &mut d_bar);
        // start from, and measure errors relative to, d_bar: for small c, w is close to M^-T d_bar
        // TODO: this solve takes ~2 more iterations than the JVP's (e.g. 2.7 vs 1.1 on average).
        let mut w = d_bar.clone();
        self.vjp_solve(end, &mut w, |nl, op, x, convergence| {
            nl.solve_jacobian_transpose_in_place(op, &end.y, end.t, &d_bar, x, &d_bar, convergence)
        })?;

        // the operator's inputs: the history (through psi), the prediction and the parameters;
        // the prediction also enters through d = y - y_pred
        let mut y_pred_bar = d_bar;
        y_pred_bar.mul_assign(Scale(-one));
        self.op.as_ref().unwrap().adjoint_inputs(
            &end.y,
            end.t,
            &w,
            &self.gamma.as_slice()[1..],
            self.alpha.as_slice(),
            order,
            &mut c.diff,
            &mut y_pred_bar,
            p_bar,
        );
        for j in 0..=order {
            c.diff.column_mut(j).axpy(one, &y_pred_bar, one);
        }

        // the rescale before the accepted attempt, then the initialisation of the history
        Self::ad_rescale(&mut c.diff, order, h_ref, h, true);
        if from_init {
            c.y.copy_from_view(&c.diff.column(0));
            c.dy.copy_from_view(&c.diff.column(1));
            c.dy.mul_assign(Scale(h_ref));
            c.diff = M::zeros(c.diff.nrows(), c.diff.ncols(), c.diff.context().clone());
        }

        // the cotangent now sits at the start of the step
        c.t = end.t - h;
        c.h = h_ref;
        c.order = order;
        c.last_step = None;
        c.h_init = from_init.then_some(h_ref);
        c.diff_initialised = !from_init;
        Ok(())
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> StepVjp<'a, Eqn> for Bdf<'a, Eqn, LS, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicitAdjoint + 'a,
    AugmentedEqn: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    Eqn::V: DefaultDenseMatrix,
    LS: LinearSolver<Eqn::M>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    /// The cotangent is a [BdfState] whose `diff`, `y` and `dy` are all read, as cotangents of
    /// the history and of its `y` and `dy` readouts. On return, if the step started from a
    /// first-order initialisation of the history, the cotangent is carried by `y` and `dy`
    /// alone (`diff` is zero); otherwise by `diff` alone (`y` and `dy` are zero).
    fn step_vjp(
        &mut self,
        end: &Self::State,
        cotangent: &mut Self::State,
        p_bar: &mut Eqn::V,
    ) -> Result<(), DiffsolError> {
        let step = self.vjp_check_inputs(end, cotangent, p_bar)?;
        let result = self.ad_vjp(step, end, cotangent, p_bar);
        // the derivative used the replayed step's c: put back the one of the solver's own state,
        // which the forward solve and its Jacobian schedule expect
        let alpha = self.alpha[self.state.order];
        self.op.as_mut().unwrap().set_c(self.state.h, alpha);
        result
    }
}

#[cfg(test)]
mod test {
    use crate::ode_equations::test_models::exponential_decay::{
        exponential_decay_problem_batched_sens, exponential_decay_problem_batched_sens_with_reset,
    };
    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    use crate::ode_equations::test_models::{
        exponential_decay::{
            exponential_decay_problem_batched, exponential_decay_problem_batched_adjoint,
            exponential_decay_problem_batched_adjoint_with_reset,
            exponential_decay_problem_batched_with_reset,
        },
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_batched,
    };
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            dydt_y2::dydt_y2_problem,
            exponential_decay::{
                exponential_decay_problem, exponential_decay_problem_adjoint,
                exponential_decay_problem_sens, exponential_decay_problem_with_root,
                exponential_decay_with_single_reset_root_problem_adjoint,
                negative_exponential_decay_problem,
            },
            exponential_decay_with_algebraic::{
                exponential_decay_with_algebraic_adjoint_problem,
                exponential_decay_with_algebraic_problem,
                exponential_decay_with_algebraic_problem_sens,
            },
            foodweb::{foodweb_elem_problem, foodweb_problem},
            gaussian_decay::gaussian_decay_problem,
            heat2d::{head2d_problem, heat2d_elem_problem},
            logistic::logistic_problem_adjoint_no_out,
            robertson::{robertson, robertson_sens},
            robertson_ode::{robertson_ode, robertson_ode_elem_problem},
            robertson_ode_with_sens::robertson_ode_with_sens,
        },
        ode_solver::tests::{
            setup_test_adjoint, setup_test_adjoint_sum_squares,
            setup_test_adjoint_sum_squares_with_single_reset_root,
            single_reset_root_discrete_times, test_adjoint, test_adjoint_sum_squares,
            test_checkpointing, test_config, test_interpolate, test_interpolate_dy,
            test_ode_solver, test_problem, test_solve_adjoint_sum_squares_with_single_reset_root,
            test_solve_adjoint_with_single_reset_root,
            test_solve_soln_adjoint_sum_squares_with_single_reset_root,
            test_solve_soln_adjoint_with_single_reset_root, test_state_mut,
            test_state_mut_on_problem,
        },
        scale, ConstantOp, Context, DenseMatrix, FaerLU, FaerMat, FaerSparseLU, FaerSparseMat,
        MatrixCommon, NalgebraLU, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverStopReason,
        Op, Vector, VectorView,
    };

    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    /// Rescaling by `a` then by `b` is rescaling by `a * b`: each `R(f) U` resamples the same
    /// interpolating polynomial. The step derivatives rely on this to replay all the rescales of
    /// a step's attempts as one.
    #[test]
    fn bdf_rescales_compose() {
        type B<'a> = super::Bdf<'a, crate::ode_solver::tests::TestEqn<M>, LS>;
        for order in 1..=5 {
            let ru = |f: f64| B::_compute_r(order, f).mat_mul(&B::_compute_r(order, 1.0));
            for (a, b) in [(0.5, 0.3), (2.0, 0.7), (1.3, 1.9)] {
                let lhs = ru(a).mat_mul(&ru(b));
                let rhs = ru(a * b);
                for (x, y) in lhs.as_slice().iter().zip(rhs.as_slice()) {
                    assert!(
                        (x - y).abs() < 1e-12 * (1.0 + y.abs()),
                        "order {order}: {x} != {y}"
                    );
                }
            }
            // U is its own inverse, so a factor of one changes nothing
            for (i, x) in ru(1.0).as_slice().iter().enumerate() {
                let expected = if i % (order + 2) == 0 { 1.0 } else { 0.0 };
                assert!(
                    (x - expected).abs() < 1e-12,
                    "order {order}: RU(1) is not I"
                );
            }
        }
    }

    #[test]
    fn bdf_state_mut() {
        test_state_mut(test_problem::<M>(false).bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_config() {
        test_config(robertson_ode::<M>(false, 1).0.bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_interpolate() {
        test_interpolate(test_problem::<M>(false).bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_interpolate_out() {
        test_interpolate(test_problem::<M>(true).bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_interpolate_sens() {
        test_interpolate(test_problem::<M>(false).bdf_sens::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_interpolate_dy() {
        test_interpolate_dy(test_problem::<M>(false).bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = p.bdf_solver::<LS>(p.bdf_state::<LS>().unwrap()).unwrap();
        test_state_mut_on_problem(s, soln);
    }

    #[test]
    fn bdf_restarts_cleanly_after_repeated_state_jumps() {
        let problem = OdeBuilder::<M>::new()
            .rtol(1e-4)
            .atol([1e-4])
            .rhs_implicit(
                |y, _p, _t, dy| dy[0] = -1000.0 * y[0],
                |_y, _p, _t, v, jv| jv[0] = -1000.0 * v[0],
            )
            .init(|_p, _t, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.bdf::<LS>().unwrap();

        {
            let state = solver.state_mut();
            state.y[0] = 1.0;
            state.dy[0] = -1000.0;
        }

        for event_index in 1..=60 {
            solver.set_stop_time(event_index as f64 * 0.1).unwrap();
            loop {
                if matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {
                    break;
                }
            }
            let state = solver.state_mut();
            state.y[0] += 1.0;
            state.dy[0] = -1000.0 * state.y[0];
        }

        assert_eq!(solver.get_statistics().number_of_nonlinear_solver_fails, 0);
    }

    #[test]
    fn bdf_test_nalgebra_negative_exponential_decay() {
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 12
        number_of_steps: 43
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 46
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 11
        number_of_derivative_jacobian_refreshes: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 48
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<FaerSparseMat<f64>>(false);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn bdf_test_cuda_exponential_decay() {
        use crate::{CudaLU, CudaMat};
        let (problem, soln) = exponential_decay_problem::<CudaMat<f64>>(false);
        let mut s = problem.bdf::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn bdf_test_cuda_oxide_exponential_decay() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = exponential_decay_problem::<OxideMat<f64>>(false);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn bdf_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let solver1 = problem.bdf::<LS>().unwrap();
        let solver2 = problem.bdf::<LS>().unwrap();
        test_checkpointing(soln, solver1, solver2);
    }

    #[test]
    fn bdf_test_faer_exponential_decay() {
        type M = FaerMat<f64>;
        type LS = FaerLU<f64>;
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 12
        number_of_steps: 43
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 46
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 11
        number_of_derivative_jacobian_refreshes: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 48
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 12
        number_of_steps: 54
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 113
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 11
        number_of_derivative_jacobian_refreshes: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.statistics(), @r###"
        number_of_calls: 57
        number_of_jac_muls: 63
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<FaerSparseMat<f64>>(false);
        let mut s = problem.bdf_sens::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_diffsl_sens() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem_diffsl;
        let (problem, soln) = exponential_decay_problem_diffsl::<M, diffsl::LlvmModule>(false);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 12
        number_of_steps: 54
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 113
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 11
        number_of_derivative_jacobian_refreshes: 0
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_faer_sparse_exponential_decay_diffsl_sens() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem_diffsl;
        type M = FaerSparseMat<f64>;
        type LS = FaerSparseLU<f64>;
        let (problem, soln) = exponential_decay_problem_diffsl::<M, diffsl::LlvmModule>(false);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true, true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(true, true);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 160
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 120
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay_adjoint() {
        let (mut problem, soln) =
            exponential_decay_problem_adjoint::<FaerSparseMat<f64>>(true, true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<FaerSparseLU<f64>, _>(&mut problem, soln);
        let (problem, _soln) = exponential_decay_problem_adjoint::<FaerSparseMat<f64>>(true, true);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<FaerSparseLU<f64>, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint_no_error_control() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(false, true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(false, true);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 100.0);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true, false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(true, false);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, soln, _stop_reason) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 522
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 634
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_logistic_without_out() {
        let (problem, soln) = logistic_problem_adjoint_no_out::<M>();
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn bdf_test_nalgebra_logistic_without_out_adjoint_sum_squares() {
        let (mut problem, soln) = logistic_problem_adjoint_no_out::<M>();
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = logistic_problem_adjoint_no_out::<M>();
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, soln, _stop_reason) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint_diffsl() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem_diffsl;
        let (mut problem, soln) = exponential_decay_problem_diffsl::<M, diffsl::LlvmModule>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_faer_sparse_exponential_decay_adjoint_diffsl() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem_diffsl;
        type M = FaerSparseMat<f64>;
        type LS = FaerSparseLU<f64>;
        let (mut problem, soln) = exponential_decay_problem_diffsl::<M, diffsl::LlvmModule>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_algebraic_adjoint() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let (problem, _soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 154
        number_of_jac_muls: 12
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 106
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay_algebraic_adjoint() {
        use crate::FaerSparseLU;
        use crate::FaerSparseMat;
        let (mut problem, soln) =
            exponential_decay_with_algebraic_adjoint_problem::<FaerSparseMat<f64>>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<FaerSparseLU<f64>, _>(&mut problem, soln);
        let (problem, _soln) =
            exponential_decay_with_algebraic_adjoint_problem::<FaerSparseMat<f64>>(true);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<FaerSparseLU<f64>, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_algebraic_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, soln, _stop_reason) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 232
        number_of_jac_muls: 12
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 288
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_with_algebraic_adjoint_diffsl() {
        use crate::ode_equations::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_diffsl;
        let (mut problem, soln) =
            exponential_decay_with_algebraic_problem_diffsl::<M, diffsl::LlvmModule>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), None)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let (problem, soln) = exponential_decay_with_algebraic_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 15
        number_of_steps: 32
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 40
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 3
        number_of_linear_solver_setups_from_step_success: 11
        number_of_derivative_jacobian_refreshes: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 44
        number_of_jac_muls: 6
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay_algebraic() {
        let (problem, soln) = exponential_decay_with_algebraic_problem::<FaerSparseMat<f64>>(false);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic_sens() {
        let (problem, soln) = exponential_decay_with_algebraic_problem_sens::<M>();
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 24
        number_of_steps: 46
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 126
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 8
        number_of_linear_solver_setups_from_step_success: 15
        number_of_derivative_jacobian_refreshes: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 67
        number_of_jac_muls: 74
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_faer_sparse_exponential_decay_algebraic_sens() {
        let (problem, soln) = exponential_decay_with_algebraic_problem_sens::<FaerSparseMat<f64>>();
        let mut s = problem.bdf_sens::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_algebraic_diffsl_sens() {
        use crate::ode_equations::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_diffsl;
        let (_problem, mut soln) = exponential_decay_with_algebraic_problem_sens::<M>();
        let (problem, _soln) =
            exponential_decay_with_algebraic_problem_diffsl::<M, diffsl::LlvmModule>(false);
        soln.atol = problem.atol.clone();
        soln.rtol = problem.rtol;
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 24
        number_of_steps: 46
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 126
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 8
        number_of_linear_solver_setups_from_step_success: 15
        number_of_derivative_jacobian_refreshes: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 78
        number_of_steps: 304
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 634
        number_of_nonlinear_solver_fails: 11
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 11
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 7
        number_of_linear_solver_setups_from_step_success: 59
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 637
        number_of_jac_muls: 42
        number_of_matrix_evals: 14
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_robertson() {
        let (problem, soln) = robertson::<FaerSparseMat<f64>>(false);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "suitesparse")]
    #[test]
    fn bdf_test_faer_sparse_ku_robertson() {
        let (problem, soln) = robertson::<FaerSparseMat<f64>>(false);
        let mut s = problem.bdf::<crate::KLU<_>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_diffsl_robertson() {
        use diffsl::LlvmModule;

        use crate::ode_equations::test_models::robertson;
        let (problem, soln) = robertson::robertson_diffsl_problem::<M, LlvmModule>();
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_diffsl_robertson_ode_adjoint() {
        use crate::ode_equations::test_models::robertson_ode;
        use diffsl::LlvmModule;
        let (mut problem, soln) = robertson_ode::robertson_ode_diffsl_problem::<M, LlvmModule>();
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (mut problem, _soln) = robertson_ode::robertson_ode_diffsl_problem::<M, LlvmModule>();
        problem.ode_options.max_nonlinear_solver_failures = 1000;
        problem.ode_options.max_error_test_failures = 200;
        problem.ode_options.pi_control_integral = 0.5;
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, soln, _stop_reason) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(s), Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
    }

    #[test]
    fn test_bdf_nalgebra_robertson_sens() {
        let (mut problem, soln) = robertson_sens::<M>();
        problem.ode_options.max_nonlinear_solver_failures = 70;
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 72
        number_of_steps: 311
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 1178
        number_of_nonlinear_solver_fails: 9
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 8
        number_of_linear_solver_setups_from_second_convergence_fail: 1
        number_of_linear_solver_setups_from_error_test_fail: 2
        number_of_linear_solver_setups_from_step_success: 60
        number_of_derivative_jacobian_refreshes: 9
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 578
        number_of_jac_muls: 689
        number_of_matrix_evals: 20
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let (problem, soln) = robertson::<M>(true);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 78
        number_of_steps: 304
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 634
        number_of_nonlinear_solver_fails: 11
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 11
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 7
        number_of_linear_solver_setups_from_step_success: 59
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 637
        number_of_jac_muls: 43
        number_of_matrix_evals: 14
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let (problem, soln) = robertson_ode::<M>(false, 3);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 97
        number_of_steps: 427
        number_of_error_test_failures: 15
        number_of_nonlinear_solver_iterations: 948
        number_of_nonlinear_solver_fails: 4
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 4
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 15
        number_of_linear_solver_setups_from_step_success: 77
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 950
        number_of_jac_muls: 90
        number_of_matrix_evals: 10
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode_sens() {
        let (problem, soln) = robertson_ode_with_sens::<M>(false);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 204
        number_of_steps: 725
        number_of_error_test_failures: 103
        number_of_nonlinear_solver_iterations: 2349
        number_of_nonlinear_solver_fails: 9
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 8
        number_of_linear_solver_setups_from_second_convergence_fail: 1
        number_of_linear_solver_setups_from_error_test_fail: 103
        number_of_linear_solver_setups_from_step_success: 91
        number_of_derivative_jacobian_refreshes: 4
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1116
        number_of_jac_muls: 1305
        number_of_matrix_evals: 18
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let (problem, soln) = dydt_y2_problem::<M>(false, 10);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 26
        number_of_steps: 157
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 282
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 1
        number_of_linear_solver_setups_from_step_success: 24
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 284
        number_of_jac_muls: 40
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let (problem, soln) = dydt_y2_problem::<M>(true, 10);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 26
        number_of_steps: 157
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 282
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 1
        number_of_linear_solver_setups_from_step_success: 24
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 284
        number_of_jac_muls: 5
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let (problem, soln) = gaussian_decay_problem::<M>(false, 10);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 22
        number_of_steps: 64
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 102
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 7
        number_of_linear_solver_setups_from_step_success: 14
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 104
        number_of_jac_muls: 20
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_faer_sparse_heat2d() {
        let (problem, soln) = head2d_problem::<FaerSparseMat<f64>, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 23
        number_of_steps: 157
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 171
        number_of_nonlinear_solver_fails: 0
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 0
        number_of_linear_solver_setups_from_second_convergence_fail: 0
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 22
        number_of_derivative_jacobian_refreshes: 0
        ");
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 174
        number_of_jac_muls: 24
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 0
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_bdf_faer_sparse_heat2d_diffsl() {
        use diffsl::LlvmModule;

        use crate::ode_equations::test_models::heat2d;
        let (problem, soln) = heat2d::heat2d_diffsl_problem::<FaerSparseMat<f64>, LlvmModule, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_faer_sparse_foodweb() {
        let (problem, soln) = foodweb_problem::<FaerSparseMat<f64>, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @"
        number_of_linear_solver_setups: 41
        number_of_steps: 161
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 347
        number_of_nonlinear_solver_fails: 14
        number_of_linear_solver_setups_from_checkpoint: 1
        number_of_linear_solver_setups_from_first_convergence_fail: 13
        number_of_linear_solver_setups_from_second_convergence_fail: 1
        number_of_linear_solver_setups_from_error_test_fail: 0
        number_of_linear_solver_setups_from_step_success: 26
        number_of_derivative_jacobian_refreshes: 0
        ");
    }

    #[test]
    fn test_bdf_faer_sparse_heat2d_elem() {
        let (problem, soln) = heat2d_elem_problem::<FaerSparseMat<f64>, 10>(1);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_nalgebra_heat2d_elem_batched() {
        let (problem, soln) = heat2d_elem_problem::<NalgebraMat<f64>, 10>(2);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_faer_sparse_foodweb_elem() {
        let (problem, soln) = foodweb_elem_problem::<FaerSparseMat<f64>, 10>(1);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_faer_sparse_robertson_ode_elem() {
        let (problem, soln) = robertson_ode_elem_problem::<FaerSparseMat<f64>>(1);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_heat2d_elem() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = heat2d_elem_problem::<OxideMat<f64>, 10>(1);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_heat2d_elem_batched() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = heat2d_elem_problem::<OxideMat<f64>, 10>(2);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_foodweb_elem() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = foodweb_elem_problem::<OxideMat<f64>, 10>(1);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_foodweb_elem_batched() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = foodweb_elem_problem::<OxideMat<f64>, 10>(2);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_robertson_ode_elem() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = robertson_ode_elem_problem::<OxideMat<f64>>(1);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_robertson_ode_elem_batched() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = robertson_ode_elem_problem::<OxideMat<f64>>(4);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_bdf_faer_sparse_foodweb_diffsl() {
        use crate::ode_equations::test_models::foodweb;
        use diffsl::LlvmModule;
        let (problem, soln) =
            foodweb::foodweb_diffsl_problem::<FaerSparseMat<f64>, LlvmModule, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tstop_bdf() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched() {
        use crate::{CudaLU, CudaMat};
        let (problem, soln) = exponential_decay_problem_batched::<CudaMat<f64>>(2);
        let mut s = problem.bdf::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_exponential_decay_batched() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem_batched;
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = exponential_decay_problem_batched::<OxideMat<f64>>(2);
        let mut s = problem.bdf::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched_with_reset() {
        use crate::{CudaLU, CudaMat, OdeSolverStopReason};
        let nbatch = 2;
        let (problem, p_f64) = exponential_decay_problem_batched_with_reset::<CudaMat<f64>>(nbatch);
        let final_time = 10.0;
        let t_event = 5.0;
        let mut solver = problem.bdf::<CudaLU<f64>>().unwrap();
        let (ys, ts, stop_reason) = solver.solve(final_time).unwrap();
        assert_eq!(stop_reason, OdeSolverStopReason::TstopReached);
        let t_last = *ts.last().unwrap();
        assert!(
            (t_last - final_time).abs() < 1e-2,
            "expected solve to reach final_time ≈ {final_time}, got {t_last}",
        );
        let reset_val = 0.4;
        let reset_tol = 0.1;
        let reset_col = (0..ts.len())
            .find(|&i| {
                let col = ys.column(i).into_owned();
                (ts[i] - t_event).abs() < 0.5
                    && (col.get_batch(0).get_index(0) - reset_val).abs() < reset_tol
            })
            .expect("expected reset state near t_event");
        let actual_reset_time = ts[reset_col];
        for b in 0..nbatch {
            let k = p_f64[b * 2];
            let expected_final = reset_val * (-k * (final_time - actual_reset_time)).exp();
            let final_col = ys.column(ts.len() - 1).into_owned();
            let actual_final = final_col.get_batch(b).get_index(0);
            let err = (actual_final - expected_final).abs();
            assert!(
                err < 0.05,
                "batch {b}: expected final ≈ {expected_final}, got {actual_final} (err={err})",
            );
        }
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_batched_sens() {
        // nbatch = 2 and nparams = 2, so the augmented state has four lanes: this is what pins
        // the `batch * nparams + param` lane layout down (a transposed layout would pair each
        // batch with the other batch's parameters)
        let (problem, soln) = exponential_decay_problem_batched_sens::<M>(2);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_batched_sens_with_reset() {
        let (problem, soln) = exponential_decay_problem_batched_sens_with_reset::<M>(2);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched_sens() {
        use crate::{CudaLU, CudaMat};
        let (problem, soln) = exponential_decay_problem_batched_sens::<CudaMat<f64>>(2);
        let mut s = problem.bdf_sens::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[cfg(feature = "cuda-oxide")]
    #[test]
    fn test_bdf_cuda_oxide_exponential_decay_batched_sens() {
        use crate::{OxideLU, OxideMat};
        let (problem, soln) = exponential_decay_problem_batched_sens::<OxideMat<f64>>(2);
        let mut s = problem.bdf_sens::<OxideLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched_sens_with_reset() {
        use crate::{CudaLU, CudaMat};
        let nbatch = 2;
        let (problem, soln) =
            exponential_decay_problem_batched_sens_with_reset::<CudaMat<f64>>(nbatch);
        let mut s = problem.bdf_sens::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched_adjoint() {
        use crate::{CudaLU, CudaMat};
        let nbatch = 2;
        let (mut problem, soln) =
            exponential_decay_problem_batched_adjoint::<CudaMat<f64>>(nbatch, true, true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<CudaLU<f64>, _>(&mut problem, soln);
        let (problem, _soln) =
            exponential_decay_problem_batched_adjoint::<CudaMat<f64>>(nbatch, true, true);
        let mut s = problem.bdf::<CudaLU<f64>>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<CudaLU<f64>, _>(checkpointer, Some(s), Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu, 40.0);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched_adjoint_sum_squares() {
        use crate::{CudaLU, CudaMat};
        let nbatch = 2;
        let (mut problem, soln) =
            exponential_decay_problem_batched_adjoint::<CudaMat<f64>>(nbatch, true, false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) =
            setup_test_adjoint_sum_squares::<CudaLU<f64>, _>(&mut problem, times.as_slice());
        let (problem, _soln) =
            exponential_decay_problem_batched_adjoint::<CudaMat<f64>>(nbatch, true, false);
        let mut s = problem.bdf::<CudaLU<f64>>().unwrap();
        let (checkpointer, soln, _stop_reason) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<CudaLU<f64>, _>(checkpointer, Some(s), Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_batched_adjoint_with_reset() {
        use crate::{CudaLU, CudaMat};
        let nbatch = 2;
        let (mut problem, soln) =
            exponential_decay_problem_batched_adjoint_with_reset::<CudaMat<f64>>(nbatch, true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdp_check = setup_test_adjoint::<CudaLU<f64>, _>(&mut problem, soln);
        let mut s = problem.bdf::<CudaLU<f64>>().unwrap();
        let (checkpointer, _y, _t, _stop_reason) =
            s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<CudaLU<f64>, _>(checkpointer, Some(s), Some(dgdp_check.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdp_check, 40.0);
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_bdf_cuda_exponential_decay_with_algebraic_batched() {
        use crate::{CudaLU, CudaMat};
        let (problem, soln) = exponential_decay_with_algebraic_problem_batched::<CudaMat<f64>>(2);
        let mut s = problem.bdf::<CudaLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_root_finder_bdf() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false, true);
        let mut s = problem.bdf::<LS>().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);

        let y0 = problem.eqn.init().call(0.0);
        let mut p = problem.context().vector_zeros(2);
        problem.eqn.get_params(&mut p);
        let k = p.get_index(0);
        let target = 0.6_f64;
        let t_root = -((target / y0.get_index(0)).ln()) / k;
        let expected = y0.clone() * scale(f64::exp(-k * t_root));
        y.assert_eq_norm(&expected, &problem.atol, problem.rtol, 15.0);

        let g = s.interpolate_out(t_root).unwrap();
        let integral = (1.0 - f64::exp(-k * t_root)) / k;
        let expected_g = y0 * scale(integral);
        g.assert_eq_norm(&expected_g, &problem.atol, problem.rtol, 15.0);
    }

    #[test]
    fn test_param_sweep_bdf() {
        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        let ctx = problem.eqn.context();
        let mut ps = Vec::new();
        for y0 in (1..10).map(f64::from) {
            ps.push(ctx.vector_from_vec(vec![0.1, y0]));
        }

        let mut old_soln: Option<<M as MatrixCommon>::V> = None;
        for p in ps {
            problem.eqn_mut().set_params(&p);
            let mut s = problem.bdf::<LS>().unwrap();
            let (ys, _ts, _stop_reason) = s.solve(10.0).unwrap();
            // check that the new solution is different from the old one
            if let Some(old_soln) = &mut old_soln {
                let new_soln = ys.column(ys.ncols() - 1).into_owned();
                let error = new_soln - &*old_soln;
                let diff = error
                    .squared_norm(old_soln, &problem.atol, problem.rtol)
                    .sqrt();
                assert!(diff > 1.0e-6, "diff: {diff}");
            }
            old_soln = Some(ys.column(ys.ncols() - 1).into_owned());
        }
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn test_ball_bounce_bdf() {
        use crate::ode_solver::tests::test_ball_bounce_problem;
        type M = crate::NalgebraMat<f64>;
        type LS = crate::NalgebraLU<f64>;
        let (x, v, t) = crate::ode_solver::tests::test_ball_bounce(
            test_ball_bounce_problem::<M>().bdf::<LS>().unwrap(),
        );

        // compare with the exact solution after the bounce at t_b = sqrt(2 h / g), where the
        // velocity reverses to e g t_b, at whatever steps the solver took
        let (g, h, e) = (9.81_f64, 10.0_f64, 0.8_f64);
        let t_bounce = (2.0 * h / g).sqrt();
        let v_bounce = e * g * t_bounce;
        assert_eq!(t.len(), 3);
        for ((x, v), t) in x.iter().zip(v.iter()).zip(t.iter()) {
            let dt = t - t_bounce;
            assert!(dt > 0.0, "t: {t} is not after the bounce at {t_bounce}");
            let (x_exact, v_exact) = (v_bounce * dt - 0.5 * g * dt * dt, v_bounce - g * dt);
            assert!((x - x_exact).abs() < 1e-4, "x: {x} expected: {x_exact}");
            assert!((v - v_exact).abs() < 1e-4, "v: {v} expected: {v_exact}");
        }
    }

    /// Test that `step()` includes the root index in `RootFound(t, index)`.
    ///
    /// Uses a two-output root function (no Reset); root 0 fires at t ≈ 5.108
    /// (exponential decay y[0] = 0.6, k=0.1) and the test verifies the returned index is 0.
    #[test]
    fn test_root_found_index_bdf() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_two_roots_problem;
        use crate::ode_solver::tests::test_root_found_index;
        let (problem, soln) = exponential_decay_with_two_roots_problem::<M>();
        let solver = problem.bdf::<LS>().unwrap();
        test_root_found_index(solver, &soln, 0, 1e-4);
    }

    /// Test that `solve()` applies resets and continues to the final time.
    #[test]
    fn test_solve_with_reset_bdf() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_reset_problem;
        use crate::ode_solver::tests::test_solve_with_reset;
        let (problem, soln) = exponential_decay_with_reset_problem::<M>();
        let solver = problem.bdf::<LS>().unwrap();
        test_solve_with_reset(solver, &soln, 100.0);
    }

    /// Test that `solve_dense()` applies resets and continues to the final evaluation time.
    #[test]
    fn test_solve_dense_with_reset_bdf() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_reset_problem;
        use crate::ode_solver::tests::test_solve_dense_with_reset;
        let (problem, soln) = exponential_decay_with_reset_problem::<M>();
        let solver = problem.bdf::<LS>().unwrap();
        test_solve_dense_with_reset(solver, &soln);
    }

    /// Test that `solve_dense_sensitivities()` applies resets and continues for BDF.
    #[test]
    fn test_solve_dense_sensitivities_with_reset_bdf() {
        use crate::ode_equations::test_models::exponential_decay::exponential_decay_with_reset_problem_sens;
        use crate::ode_solver::tests::test_solve_dense_sensitivities_with_reset;
        let (problem, soln) = exponential_decay_with_reset_problem_sens::<M>();
        let solver = problem.bdf_sens::<LS>().unwrap();
        test_solve_dense_sensitivities_with_reset(solver, &soln);
    }

    #[test]
    fn test_solve_adjoint_with_single_reset_root_bdf() {
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(true);
        test_solve_adjoint_with_single_reset_root(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.bdf_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn),
            true,
        );
    }

    #[test]
    fn test_solve_adjoint_with_single_reset_root_bdf_without_replay_solver() {
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(true);
        test_solve_adjoint_with_single_reset_root(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.bdf_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn),
            false,
        );
    }

    #[test]
    fn test_solve_soln_adjoint_with_single_reset_root_bdf() {
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(true);
        test_solve_soln_adjoint_with_single_reset_root(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.bdf_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn),
            true,
        );
    }

    #[test]
    fn test_solve_adjoint_sum_squares_with_single_reset_root_bdf() {
        let (mut problem, soln) =
            exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        let times = single_reset_root_discrete_times(soln.solution_points[0].t);
        let (dgdp, data) = setup_test_adjoint_sum_squares_with_single_reset_root::<LS, _>(
            &mut problem,
            times.as_slice(),
        );
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        test_solve_adjoint_sum_squares_with_single_reset_root(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.bdf_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn),
            true,
            dgdp,
            data,
            times.as_slice(),
        );
    }

    #[test]
    fn test_solve_soln_adjoint_sum_squares_with_single_reset_root_bdf() {
        let (mut problem, soln) =
            exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        let times = single_reset_root_discrete_times(soln.solution_points[0].t);
        let (dgdp, data) = setup_test_adjoint_sum_squares_with_single_reset_root::<LS, _>(
            &mut problem,
            times.as_slice(),
        );
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        test_solve_soln_adjoint_sum_squares_with_single_reset_root(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.bdf_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn),
            true,
            dgdp,
            data,
            times.as_slice(),
        );
    }

    #[test]
    fn test_solve_adjoint_sum_squares_with_single_reset_root_bdf_without_replay_solver() {
        let (mut problem, soln) =
            exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        let times = single_reset_root_discrete_times(soln.solution_points[0].t);
        let (dgdp, data) = setup_test_adjoint_sum_squares_with_single_reset_root::<LS, _>(
            &mut problem,
            times.as_slice(),
        );
        let (problem, soln) = exponential_decay_with_single_reset_root_problem_adjoint::<M>(false);
        test_solve_adjoint_sum_squares_with_single_reset_root(
            |state| match state {
                Some(state) => problem.bdf_solver(state),
                None => problem.bdf::<LS>(),
            },
            &soln,
            |adjoint_eqn| problem.bdf_state_adjoint::<LS, _>(adjoint_eqn),
            |state, adjoint_eqn| problem.bdf_solver_adjoint_from_state::<LS, _>(state, adjoint_eqn),
            false,
            dgdp,
            data,
            times.as_slice(),
        );
    }

    #[test]
    fn bdf_clone_resets_statistics_and_preserves_operator_presence() {
        let (problem, _soln) = exponential_decay_problem::<M>(false);
        let mut solver = problem.bdf::<LS>().unwrap();
        solver.step().unwrap();
        assert!(solver.get_statistics().number_of_steps > 0);
        assert!(solver.op.is_some());
        assert!(solver.s_op.is_none());

        let cloned = solver.clone();
        assert!(cloned.op.is_some());
        assert!(cloned.s_op.is_none());
        assert_eq!(cloned.get_statistics().number_of_steps, 0);
        assert_eq!(cloned.state().t, solver.state().t);
    }

    #[test]
    fn bdf_clone_preserves_sensitivity_operator_and_resets_statistics() {
        let (problem, _soln) = exponential_decay_problem_sens::<M>(false);
        let mut solver = problem.bdf_sens::<LS>().unwrap();
        solver.step().unwrap();
        assert!(solver.get_statistics().number_of_steps > 0);
        assert!(solver.op.is_some());
        assert!(solver.s_op.is_some());

        let cloned = solver.clone();
        assert!(cloned.op.is_some());
        assert!(cloned.s_op.is_some());
        assert_eq!(cloned.get_statistics().number_of_steps, 0);
        assert_eq!(cloned.state().s.len(), solver.state().s.len());
    }
}
