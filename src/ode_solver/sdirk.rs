use num_traits::abs;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::ops::MulAssign;
use std::rc::Rc;

use crate::error::DiffsolError;
use crate::error::OdeSolverError;
use crate::matrix::MatrixRef;
use crate::nonlinear_solver::newton::newton_iteration;
use crate::ode_solver_error;
use crate::vector::VectorRef;
use crate::LinearSolver;
use crate::NewtonNonlinearSolver;
use crate::OdeSolverStopReason;
use crate::RootFinder;
use crate::SdirkState;
use crate::SensEquations;
use crate::Tableau;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    DenseMatrix, JacobianUpdate, NonLinearOp, OdeEquations, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, Op, Scalar, Vector, VectorViewMut,
};

use super::bdf::BdfStatistics;
use super::jacobian_update::SolverState;

/// A singly diagonally implicit Runge-Kutta method. Can optionally have an explicit first stage for ESDIRK methods.
///
/// The particular method is defined by the [Tableau] used to create the solver.
/// If the `beta` matrix of the [Tableau] is present this is used for interpolation, otherwise hermite interpolation is used.
///
/// Restrictions:
/// - The upper triangular part of the `a` matrix must be zero (i.e. not fully implicit).
/// - The diagonal of the `a` matrix must be the same non-zero value for all rows (i.e. an SDIRK method), except for the first row which can be zero for ESDIRK methods.
/// - The last row of the `a` matrix must be the same as the `b` vector, and the last element of the `c` vector must be 1 (i.e. a stiffly accurate method)
pub struct Sdirk<M, Eqn, LS>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    LS: LinearSolver<SdirkCallable<Eqn>>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    tableau: Tableau<M>,
    problem: Option<OdeSolverProblem<Eqn>>,
    nonlinear_solver: NewtonNonlinearSolver<SdirkCallable<Eqn>, LS>,
    state: Option<SdirkState<Eqn::V>>,
    diff: M,
    sdiff: Vec<M>,
    gamma: Eqn::T,
    is_sdirk: bool,
    s_op: Option<SdirkCallable<SensEquations<Eqn>>>,
    old_t: Eqn::T,
    old_y: Eqn::V,
    old_y_sens: Vec<Eqn::V>,
    old_f: Eqn::V,
    old_f_sens: Vec<Eqn::V>,
    a_rows: Vec<Eqn::V>,
    statistics: BdfStatistics,
    root_finder: Option<RootFinder<Eqn::V>>,
    tstop: Option<Eqn::T>,
    is_state_mutated: bool,
    jacobian_update: JacobianUpdate<Eqn::T>,
}

impl<M, Eqn, LS> Sdirk<M, Eqn, LS>
where
    LS: LinearSolver<SdirkCallable<Eqn>>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    const NEWTON_MAXITER: usize = 10;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-13;

    pub fn new(tableau: Tableau<M>, linear_solver: LS) -> Self {
        let nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);

        // check that the upper triangular part of a is zero
        let s = tableau.s();
        for i in 0..s {
            for j in (i + 1)..s {
                assert_eq!(
                    tableau.a()[(i, j)],
                    Eqn::T::zero(),
                    "Invalid tableau, expected a(i, j) = 0 for i > j"
                );
            }
        }
        let gamma = tableau.a()[(1, 1)];
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            assert_eq!(
                tableau.a()[(i, i)],
                gamma,
                "Invalid tableau, expected a(i, i) = gamma = {} for i = 1..s-1",
                gamma
            );
        }
        // if a(0, 0) = gamma, then we're a SDIRK method
        // if a(0, 0) = 0, then we're a ESDIRK method
        // otherwise, error
        let zero = Eqn::T::zero();
        if tableau.a()[(0, 0)] != zero && tableau.a()[(0, 0)] != gamma {
            panic!("Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma");
        }
        let is_sdirk = tableau.a()[(0, 0)] == gamma;

        let mut a_rows = Vec::with_capacity(s);
        for i in 0..s {
            let mut row = Vec::with_capacity(i);
            for j in 0..i {
                row.push(tableau.a()[(i, j)]);
            }
            a_rows.push(Eqn::V::from_vec(row));
        }

        // check last row of a is the same as b
        for i in 0..s {
            assert_eq!(
                tableau.a()[(s - 1, i)],
                tableau.b()[i],
                "Invalid tableau, expected a(s-1, i) = b(i)"
            );
        }

        // check that last c is 1
        assert_eq!(
            tableau.c()[s - 1],
            Eqn::T::one(),
            "Invalid tableau, expected c(s-1) = 1"
        );

        // check that the first c is 0 for esdirk methods
        if !is_sdirk {
            assert_eq!(
                tableau.c()[0],
                Eqn::T::zero(),
                "Invalid tableau, expected c(0) = 0 for esdirk methods"
            );
        }

        let n = 1;
        let old_t = Eqn::T::zero();
        let old_y = <Eqn::V as Vector>::zeros(n);
        let old_f = <Eqn::V as Vector>::zeros(n);
        let statistics = BdfStatistics::default();
        let old_f_sens = Vec::new();
        let old_y_sens = Vec::new();
        let diff = M::zeros(n, s);
        let sdiff = Vec::new();
        Self {
            old_y_sens,
            old_f_sens,
            diff,
            sdiff,
            tableau,
            nonlinear_solver,
            state: None,
            problem: None,
            s_op: None,
            gamma,
            is_sdirk,
            old_t,
            old_y,
            a_rows,
            old_f,
            statistics,
            root_finder: None,
            tstop: None,
            is_state_mutated: false,
            jacobian_update: JacobianUpdate::default(),
        }
    }

    pub fn get_statistics(&self) -> &BdfStatistics {
        &self.statistics
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        let state = self.state.as_mut().unwrap();

        // check if the we are at tstop
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if tstop < state.t - troundoff {
            return Err(DiffsolError::from(
                OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time: tstop.into(),
                    state_time: state.t.into(),
                },
            ));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if state.t + state.h > tstop + troundoff {
            let factor = (tstop - state.t) / state.h;
            state.h *= factor;
            self.nonlinear_solver.problem().f.set_h(state.h);
        }
        Ok(None)
    }

    fn predict_stage(i: usize, diff: &M, dy: &mut Eqn::V, tableau: &Tableau<M>) {
        if i == 0 {
            dy.fill(Eqn::T::zero());
        } else if i == 1 {
            dy.copy_from_view(&diff.column(i - 1));
        } else {
            let c =
                (tableau.c()[i] - tableau.c()[i - 2]) / (tableau.c()[i - 1] - tableau.c()[i - 2]);
            // dy = c1  + c * (c1 - c2)
            dy.copy_from_view(&diff.column(i - 1));
            dy.axpy_v(-c, &diff.column(i - 2), Eqn::T::one() + c);
        }
    }

    fn solve_for_sensitivities(&mut self, i: usize, t: Eqn::T) -> Result<(), DiffsolError> {
        // update for new state
        {
            self.problem()
                .as_ref()
                .unwrap()
                .eqn_sens
                .as_ref()
                .unwrap()
                .rhs()
                .update_state(&self.old_y, &self.old_f, t);
        }

        // reuse linear solver from nonlinear solver
        let ls = |x: &mut Eqn::V| -> Result<(), DiffsolError> {
            self.nonlinear_solver.solve_linearised_in_place(x)
        };

        // construct bdf discretisation of sensitivity equations
        let op = self.s_op.as_ref().unwrap();
        op.set_h(self.state.as_ref().unwrap().h);

        // solve for sensitivities equations discretised using sdirk equation
        let fun = |x: &Eqn::V, y: &mut Eqn::V| op.call_inplace(x, t, y);
        let mut convergence = self.nonlinear_solver.convergence().clone();
        let nparams = self.problem().as_ref().unwrap().eqn.rhs().nparams();
        for j in 0..nparams {
            let s0 = &self.state.as_ref().unwrap().s[j];
            op.set_phi(&self.sdiff[j].columns(0, i), s0, &self.a_rows[i]);
            op.eqn().as_ref().rhs().set_param_index(j);
            let ds = &mut self.old_f_sens[j];
            Self::predict_stage(i, &self.sdiff[j], ds, &self.tableau);

            // solve
            {
                newton_iteration(ds, &mut self.old_y_sens[j], s0, fun, ls, &mut convergence)?;
                self.old_y_sens[j].copy_from(&op.get_last_f_eval());
                self.statistics.number_of_nonlinear_solver_iterations += convergence.niter();
            }
        }
        Ok(())
    }

    fn interpolate_from_diff(y0: &Eqn::V, beta_f: &Eqn::V, diff: &M) -> Eqn::V {
        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        let mut ret = y0.clone();
        diff.gemv(Eqn::T::one(), beta_f, Eqn::T::one(), &mut ret);
        ret
    }

    fn interpolate_beta_function(theta: Eqn::T, beta: &M) -> Eqn::V {
        let poly_order = beta.ncols();
        let s_star = beta.nrows();
        let mut thetav = Vec::with_capacity(poly_order);
        thetav.push(theta);
        for i in 1..poly_order {
            thetav.push(theta * thetav[i - 1]);
        }
        // beta_poly = beta * thetav
        let thetav = Eqn::V::from_vec(thetav);
        let mut beta_f = <Eqn::V as Vector>::zeros(s_star);
        beta.gemv(Eqn::T::one(), &thetav, Eqn::T::zero(), &mut beta_f);
        beta_f
    }

    fn interpolate_hermite(theta: Eqn::T, u0: &Eqn::V, u1: &Eqn::V, diff: &M) -> Eqn::V {
        let hf0 = diff.column(0);
        let hf1 = diff.column(diff.ncols() - 1);
        u0 * scale(Eqn::T::from(1.0) - theta)
            + u1 * scale(theta)
            + ((u1 - u0) * scale(Eqn::T::from(1.0) - Eqn::T::from(2.0) * theta)
                + hf0 * scale(theta - Eqn::T::from(1.0))
                + hf1 * scale(theta))
                * scale(theta * (theta - Eqn::T::from(1.0)))
    }

    fn _jacobian_updates(&mut self, h: Eqn::T, state: SolverState) {
        if self.jacobian_update.check_rhs_jacobian_update(h, &state) {
            self.nonlinear_solver.problem().f.set_jacobian_is_stale();
            self.nonlinear_solver
                .reset_jacobian(&self.old_f, self.state.as_ref().unwrap().t);
            self.jacobian_update.update_rhs_jacobian();
            self.jacobian_update.update_jacobian(h);
        } else if self.jacobian_update.check_jacobian_update(h, &state) {
            self.nonlinear_solver
                .reset_jacobian(&self.old_f, self.state.as_ref().unwrap().t);
            self.jacobian_update.update_jacobian(h);
        }
    }

    fn _update_step_size(&mut self, factor: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        let new_h = self.state.as_ref().unwrap().h * factor;

        // if step size too small, then fail
        if abs(new_h) < Eqn::T::from(Self::MIN_TIMESTEP) {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.state.as_ref().unwrap().t.into(),
            }));
        }

        // update h for new step size
        self.nonlinear_solver.problem().f.set_h(new_h);

        // update state
        self.state.as_mut().unwrap().h = new_h;

        Ok(new_h)
    }
}

impl<M, Eqn, LS> OdeSolverMethod<Eqn> for Sdirk<M, Eqn, LS>
where
    LS: LinearSolver<SdirkCallable<Eqn>>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    type State = SdirkState<Eqn::V>;

    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.problem.as_ref()
    }

    fn order(&self) -> usize {
        self.tableau.order()
    }

    fn take_state(&mut self) -> Option<SdirkState<Eqn::V>> {
        Option::take(&mut self.state)
    }
    
    fn checkpoint(&mut self) -> Result<Self::State, DiffsolError> {
        if self.state.is_none() {
            return Err(ode_solver_error!(StateNotSet));
        }
        Ok(self.state.as_ref().unwrap().clone())
    }

    fn set_problem(&mut self, mut state: SdirkState<Eqn::V>, problem: &OdeSolverProblem<Eqn>) -> Result<(), DiffsolError> {
        // setup linear solver for first step
        let callable = Rc::new(SdirkCallable::new(problem, self.gamma));
        callable.set_h(state.h);
        self.jacobian_update.update_jacobian(state.h);
        self.jacobian_update.update_rhs_jacobian();
        let nonlinear_problem = SolverProblem::new_from_ode_problem(callable, problem);
        self.nonlinear_solver.set_problem(&nonlinear_problem);

        // set max iterations for nonlinear solver
        self.nonlinear_solver
            .convergence_mut()
            .set_max_iter(Self::NEWTON_MAXITER);

        // update statistics
        self.statistics = BdfStatistics::default();


        state.check_consistent_with_problem(problem)?;
        
        let nstates = state.y.len();
        let order = self.tableau.s();
        if self.diff.nrows() != nstates || self.diff.ncols() != order {
            self.diff = M::zeros(nstates, order);
        }
        if problem.eqn_sens.is_some() {
            
            state.check_sens_consistent_with_problem(problem)?;
            let nparams = problem.eqn.rhs().nparams();
            if 
                self.sdiff.len() != nparams ||
                self.sdiff[0].nrows() != nstates ||
                self.sdiff[0].ncols() != order
            {
                self.sdiff = vec![M::zeros(nstates, order); nparams];
                self.old_f_sens = vec![<Eqn::V as Vector>::zeros(nstates); nparams];
                self.old_y_sens = vec![<Eqn::V as Vector>::zeros(nstates); nparams];
                self.s_op = Some(SdirkCallable::from_eqn(
                    problem.eqn_sens.as_ref().unwrap().clone(),
                    self.gamma,
                ));
            }
        }

        self.old_f = state.dy.clone();
        self.old_t = state.t;
        self.old_y = state.y.clone();

        state.set_problem(problem)?;
        self.state = Some(state);
        self.problem = Some(problem.clone());
        if let Some(root_fn) = problem.eqn.root() {
            let state = self.state.as_ref().unwrap();
            self.root_finder = Some(RootFinder::new(root_fn.nout()));
            self.root_finder
                .as_ref()
                .unwrap()
                .init(root_fn.as_ref(), &state.y, state.t);
        }
        Ok(())
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        if self.state.is_none() {
            return Err(ode_solver_error!(StateNotSet));
        }
        let n = self.state.as_ref().unwrap().y.len();

        // optionally do the first step
        let start = if self.is_sdirk { 0 } else { 1 };
        let mut updated_jacobian = false;

        // dont' reset jacobian for the first attempt at the step
        let mut error = <Eqn::V as Vector>::zeros(n);

        let mut factor: Eqn::T;

        // loop until step is accepted
        'step: loop {
            let t0 = self.state.as_ref().unwrap().t;
            let h = self.state.as_ref().unwrap().h;
            // if start == 1, then we need to compute the first stage
            // from the last stage of the previous step
            if start == 1 {
                {
                    let state = self.state.as_ref().unwrap();
                    let mut hf = self.diff.column_mut(0);
                    hf.copy_from(&state.dy);
                    hf *= scale(h);
                }

                // sensitivities too
                if self.problem().as_ref().unwrap().eqn_sens.is_some() {
                    for (diff, dy) in self
                        .sdiff
                        .iter_mut()
                        .zip(self.state.as_ref().unwrap().ds.iter())
                    {
                        let mut hf = diff.column_mut(0);
                        hf.copy_from(dy);
                        hf *= scale(h);
                    }
                }
            }

            for i in start..self.tableau.s() {
                let t = t0 + self.tableau.c()[i] * h;
                self.nonlinear_solver.problem().f.set_phi(
                    &self.diff.columns(0, i),
                    &self.state.as_ref().unwrap().y,
                    &self.a_rows[i],
                );

                Self::predict_stage(i, &self.diff, &mut self.old_f, &self.tableau);

                let mut solve_result = self.nonlinear_solver.solve_in_place(
                    &mut self.old_f,
                    t,
                    &self.state.as_ref().unwrap().y,
                );
                self.statistics.number_of_nonlinear_solver_iterations +=
                    self.nonlinear_solver.convergence().niter();

                // only calculate sensitivities if the solve succeeded
                if solve_result.is_ok() {
                    // old_y now has the new y soln and old_f has the new dy soln
                    self.old_y
                        .copy_from(&self.nonlinear_solver.problem().f.get_last_f_eval());
                    if self.problem().as_ref().unwrap().eqn_sens.is_some() {
                        solve_result = self.solve_for_sensitivities(i, t);
                    }
                }

                // handle solve failure
                if solve_result.is_err() {
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    if !updated_jacobian {
                        // newton iteration did not converge, so update jacobian and try again
                        updated_jacobian = true;
                        self._jacobian_updates(h, SolverState::FirstConvergenceFail);
                    } else {
                        // newton iteration did not converge and jacobian has been updated, so we reduce step size and try again
                        let new_h = self._update_step_size(Eqn::T::from(0.3))?;
                        self._jacobian_updates(new_h, SolverState::SecondConvergenceFail);
                    }
                    // try again....
                    continue 'step;
                };

                // update diff with solved dy
                self.diff.column_mut(i).copy_from(&self.old_f);

                if self.problem().as_ref().unwrap().eqn_sens.is_some() {
                    for (diff, old_f_sens) in self.sdiff.iter_mut().zip(self.old_f_sens.iter()) {
                        diff.column_mut(i).copy_from(old_f_sens);
                    }
                }
            }
            // successfully solved for all stages, now compute error
            self.diff
                .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);

            // compute error norm
            let atol = self.problem().as_ref().unwrap().atol.as_ref();
            let rtol = self.problem().as_ref().unwrap().rtol;
            let mut error_norm = error.squared_norm(&self.old_y, atol, rtol);

            // sensitivity errors
            if self.problem().as_ref().unwrap().eqn_sens.is_some()
                && self.problem().as_ref().unwrap().sens_error_control
            {
                for i in 0..self.sdiff.len() {
                    self.sdiff[i].gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);
                    let sens_error_norm = error.squared_norm(&self.old_y_sens[i], atol, rtol);
                    error_norm += sens_error_norm;
                }
                error_norm /= Eqn::T::from((self.sdiff.len() + 1) as f64);
            }

            // adjust step size based on error
            let maxiter = self.nonlinear_solver.convergence().max_iter() as f64;
            let niter = self.nonlinear_solver.convergence().niter() as f64;
            let safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));
            let order = self.tableau.order() as f64;
            factor = safety * error_norm.pow(Eqn::T::from(-0.5 / (order + 1.0)));
            if factor < Eqn::T::from(Self::MIN_FACTOR) {
                factor = Eqn::T::from(Self::MIN_FACTOR);
            }
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break 'step;
            }
            // step is rejected, factor reduces step size, so we try again with the smaller step size
            self.statistics.number_of_error_test_failures += 1;
            let new_h = self._update_step_size(factor)?;
            self._jacobian_updates(new_h, SolverState::ErrorTestFail);
        }

        // take the step
        {
            let state = self.state.as_mut().unwrap();
            self.old_t = state.t;
            state.t += state.h;

            // last stage is the solution and is the same as old_f
            // todo: can we get rid of old_f and just use diff?
            self.old_f.mul_assign(scale(Eqn::T::one() / state.h));
            std::mem::swap(&mut self.old_f, &mut state.dy);

            // old_y already has the new y soln
            std::mem::swap(&mut self.old_y, &mut state.y);

            for i in 0..self.sdiff.len() {
                self.old_f_sens[i].mul_assign(scale(Eqn::T::one() / state.h));
                std::mem::swap(&mut self.old_f_sens[i], &mut state.ds[i]);
                std::mem::swap(&mut self.old_y_sens[i], &mut state.s[i]);
            }
        }

        // update step size for next step
        let new_h = self._update_step_size(factor)?;
        self._jacobian_updates(new_h, SolverState::StepSuccess);

        self.is_state_mutated = false;

        // update statistics
        self.statistics.number_of_linear_solver_setups =
            self.nonlinear_solver.problem().f.number_of_jac_evals();
        self.statistics.number_of_steps += 1;
        self.jacobian_update.step();

        // check for root within accepted step
        if let Some(root_fn) = self.problem.as_ref().unwrap().eqn.root() {
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

        // check if the we are at tstop
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

    fn interpolate_sens(
        &self,
        t: <Eqn as OdeEquations>::T,
    ) -> Result<Vec<<Eqn as OdeEquations>::V>, DiffsolError> {
        if self.state.is_none() {
            return Err(ode_solver_error!(StateNotSet));
        }
        let state = self.state.as_ref().unwrap();

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.s.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_t))
            || (!is_forward && (t < state.t || t > self.old_t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }
        let dt = state.t - self.old_t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = self
                .old_y_sens
                .iter()
                .zip(self.sdiff.iter())
                .map(|(y, diff)| Self::interpolate_from_diff(y, &beta_f, diff))
                .collect();
            Ok(ret)
        } else {
            let ret = self
                .old_y_sens
                .iter()
                .zip(state.s.iter())
                .zip(self.sdiff.iter())
                .map(|((s0, s1), diff)| Self::interpolate_hermite(theta, s0, s1, diff))
                .collect();
            Ok(ret)
        }
    }

    fn interpolate(&self, t: <Eqn>::T) -> Result<<Eqn>::V, DiffsolError> {
        if self.state.is_none() {
            return Err(ode_solver_error!(StateNotSet));
        }
        let state = self.state.as_ref().unwrap();

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_t))
            || (!is_forward && (t < state.t || t > self.old_t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = state.t - self.old_t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = Self::interpolate_from_diff(&self.old_y, &beta_f, &self.diff);
            Ok(ret)
        } else {
            let ret = Self::interpolate_hermite(theta, &self.old_y, &state.y, &self.diff);
            Ok(ret)
        }
    }

    fn state(&self) -> Option<&SdirkState<Eqn::V>> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> Option<&mut SdirkState<Eqn::V>> {
        self.is_state_mutated = true;
        self.state.as_mut()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::{
            test_models::{
                exponential_decay::{
                    exponential_decay_problem, exponential_decay_problem_sens,
                    exponential_decay_problem_with_root, negative_exponential_decay_problem,
                },
                heat2d::head2d_problem,
                robertson::robertson,
                robertson_ode::robertson_ode,
                robertson_sens::robertson_sens,
            },
            tests::{
                test_interpolate, test_no_set_problem, test_ode_solver, test_state_mut,
                test_state_mut_on_problem,
            },
        },
        FaerSparseLU, NalgebraLU, OdeEquations, Op, Sdirk, SparseColMat, Tableau,
    };

    use faer::Mat;
    use num_traits::abs;

    type M = nalgebra::DMatrix<f64>;
    #[test]
    fn sdirk_no_set_problem() {
        let tableau = Tableau::<M>::tr_bdf2();
        test_no_set_problem::<M, _>(Sdirk::<M, _, _>::new(tableau, NalgebraLU::default()));
    }
    #[test]
    fn sdirk_state_mut() {
        let tableau = Tableau::<M>::tr_bdf2();
        test_state_mut::<M, _>(Sdirk::<M, _, _>::new(tableau, NalgebraLU::default()));
    }
    #[test]
    fn sdirk_test_interpolate() {
        let tableau = Tableau::<M>::tr_bdf2();
        test_interpolate::<M, _>(Sdirk::<M, _, _>::new(tableau, NalgebraLU::default()));
    }

    #[test]
    fn sdirk_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let tableau = Tableau::<M>::tr_bdf2();
        let s = Sdirk::<M, _, _>::new(tableau, NalgebraLU::default());
        test_state_mut_on_problem(s, p, soln);
    }

    #[test]
    fn sdirk_test_nalgebra_negative_exponential_decay() {
        let tableau = Tableau::<M>::esdirk34();
        let mut s = Sdirk::<M, _, _>::new(tableau, NalgebraLU::default());
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 4
        number_of_steps: 29
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 116
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 118
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay_sens() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 7
        number_of_steps: 55
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 550
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 222
        number_of_jac_muls: 336
        number_of_matrix_evals: 2
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay() {
        let tableau = Tableau::<M>::esdirk34();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 3
        number_of_steps: 13
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 84
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 86
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay_sens() {
        let tableau = Tableau::<M>::esdirk34();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 5
        number_of_steps: 21
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 333
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 128
        number_of_jac_muls: 211
        number_of_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = robertson::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 97
        number_of_steps: 234
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 1965
        number_of_nonlinear_solver_fails: 13
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1968
        number_of_jac_muls: 36
        number_of_matrix_evals: 12
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_sens() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = robertson_sens::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 115
        number_of_steps: 245
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 4990
        number_of_nonlinear_solver_fails: 37
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1581
        number_of_jac_muls: 3569
        number_of_matrix_evals: 28
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson() {
        let tableau = Tableau::<M>::esdirk34();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = robertson::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 87
        number_of_steps: 137
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 1773
        number_of_nonlinear_solver_fails: 15
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1776
        number_of_jac_muls: 45
        number_of_matrix_evals: 15
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson_sens() {
        let tableau = Tableau::<M>::esdirk34();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = robertson_sens::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 123
        number_of_steps: 156
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 5194
        number_of_nonlinear_solver_fails: 49
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 1695
        number_of_jac_muls: 3719
        number_of_matrix_evals: 38
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_ode() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = robertson_ode::<M>(false, 1);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 113
        number_of_steps: 304
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 2601
        number_of_nonlinear_solver_fails: 15
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 2603
        number_of_jac_muls: 39
        number_of_matrix_evals: 13
        "###);
    }

    #[test]
    fn test_tr_bdf2_faer_sparse_heat2d() {
        let tableau = Tableau::<Mat<f64>>::tr_bdf2();
        let mut s = Sdirk::new(tableau, FaerSparseLU::default());
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        test_ode_solver(&mut s, &problem, soln, None, false);
    }

    #[test]
    fn test_tstop_tr_bdf2() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, true);
    }

    #[test]
    fn test_root_finder_tr_bdf2() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let y = test_ode_solver(&mut s, &problem, soln, None, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }
}
