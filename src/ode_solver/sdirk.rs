use anyhow::anyhow;
use anyhow::Result;
use num_traits::abs;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::ops::MulAssign;
use std::rc::Rc;

use crate::matrix::MatrixRef;
use crate::nonlinear_solver::convergence::Convergence;
use crate::nonlinear_solver::newton::newton_iteration;
use crate::vector::VectorRef;
use crate::LinearSolver;
use crate::NewtonNonlinearSolver;
use crate::OdeSolverStopReason;
use crate::RootFinder;
use crate::Tableau;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    DenseMatrix, NonLinearOp, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op,
    Scalar, Vector, VectorViewMut,
};

use super::bdf::BdfStatistics;

/// A singly diagonally implicit Runge-Kutta method. Can optionally have an explicit first stage for ESDIRK methods.
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
    state: Option<OdeSolverState<Eqn::V>>,
    diff: M,
    sdiff: Vec<M>,
    gamma: Eqn::T,
    is_sdirk: bool,
    old_t: Eqn::T,
    old_y: Eqn::V,
    old_y_sens: Vec<Eqn::V>,
    old_f: Eqn::V,
    old_f_sens: Vec<Eqn::V>,
    a_rows: Vec<Eqn::V>,
    statistics: BdfStatistics<Eqn::T>,
    root_finder: Option<RootFinder<Eqn::V>>,
    tstop: Option<Eqn::T>,
    is_state_mutated: bool,
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
        let mut nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        // set max iterations for nonlinear solver
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);

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
        let s = tableau.s();
        let diff = M::zeros(n, s);
        let old_t = Eqn::T::zero();
        let old_y = <Eqn::V as Vector>::zeros(n);
        let old_f = <Eqn::V as Vector>::zeros(n);
        let statistics = BdfStatistics::default();
        let old_f_sens = Vec::new();
        let sdiff = Vec::new();
        let old_y_sens = Vec::new();
        Self {
            old_y_sens,
            old_f_sens,
            sdiff,
            tableau,
            nonlinear_solver,
            state: None,
            diff,
            problem: None,
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
        }
    }

    pub fn get_statistics(&self) -> &BdfStatistics<Eqn::T> {
        &self.statistics
    }

    fn handle_tstop(&mut self, tstop: Eqn::T) -> Result<Option<OdeSolverStopReason<Eqn::T>>> {
        let state = self.state.as_mut().unwrap();

        // check if the we are at tstop
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if tstop < state.t - troundoff {
            return Err(anyhow::anyhow!(
                "tstop = {} is less than current time t = {}",
                tstop,
                state.t
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

    fn solve_for_sensitivities(&mut self, t: Eqn::T) -> Result<()> {
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
        let ls =
            |x: &mut Eqn::V| -> Result<()> { self.nonlinear_solver.solve_linearised_in_place(x) };

        // construct bdf discretisation of sensitivity equations
        let op = SdirkCallable::from_eqn(
            self.problem
                .as_ref()
                .unwrap()
                .eqn_sens
                .as_ref()
                .unwrap()
                .clone(),
            self.gamma,
        );
        op.set_h(self.state.as_ref().unwrap().h);

        // solve for sensitivities equations discretised using sdirk equation
        let fun = |x: &Eqn::V, y: &mut Eqn::V| op.call_inplace(x, t, y);
        let rtol = self.problem().as_ref().unwrap().rtol;
        let atol = self.problem().as_ref().unwrap().atol.clone();
        let maxiter = self.nonlinear_solver.max_iter();
        let mut convergence = Convergence::new(rtol, atol, maxiter);
        let nparams = self.problem().as_ref().unwrap().eqn.rhs().nparams();
        for i in 0..nparams {
            let s0 = &self.state.as_ref().unwrap().s[i];
            op.set_phi(&self.sdiff[i].columns(0, i), s0, &self.a_rows[i]);
            op.eqn().as_ref().rhs().set_param_index(i);
            let ds = &mut self.old_f_sens[i];
            Self::predict_stage(i, &self.sdiff[i], ds, &self.tableau);

            // solve
            {
                let niter = newton_iteration(ds, fun, ls, &mut convergence)?;
                self.old_y_sens[i].copy_from(&op.get_last_f_eval());
                self.statistics.number_of_nonlinear_solver_iterations += niter;
            }
        }
        Ok(())
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
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.problem.as_ref()
    }

    fn order(&self) -> usize {
        self.tableau.order()
    }

    fn take_state(&mut self) -> Option<OdeSolverState<Eqn::V>> {
        Option::take(&mut self.state)
    }

    fn set_problem(&mut self, state: OdeSolverState<<Eqn>::V>, problem: &OdeSolverProblem<Eqn>) {
        // setup linear solver for first step
        let callable = Rc::new(SdirkCallable::new(problem, self.gamma));
        callable.set_h(state.h);
        let nonlinear_problem = SolverProblem::new_from_ode_problem(callable, problem);
        self.nonlinear_solver.set_problem(&nonlinear_problem);

        // update statistics
        self.statistics = BdfStatistics::default();
        self.statistics.initial_step_size = state.h;

        let nstates = state.y.len();
        let nparams = problem.eqn.rhs().nparams();
        if problem.eqn_sens.is_some() {
            self.sdiff = vec![M::zeros(nstates, self.tableau.s()); nparams];
            self.old_f_sens = vec![<Eqn::V as Vector>::zeros(nstates); nparams];
            self.old_y_sens = vec![<Eqn::V as Vector>::zeros(nstates); nparams];
        }

        self.diff = M::zeros(nstates, self.tableau.s());
        self.old_f = state.dy.clone();
        self.old_t = state.t;
        self.old_y = state.y.clone();
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
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>> {
        // optionally do the first step
        if self.state.is_none() {
            return Err(anyhow!("State not set"));
        }
        let n = self.state.as_ref().unwrap().y.len();

        let start = if self.is_sdirk { 0 } else { 1 };
        let mut updated_jacobian = false;
        let mut error = <Eqn::V as Vector>::zeros(n);

        let mut t1: Eqn::T;

        // loop until step is accepted
        'step: loop {
            let t0 = self.state.as_ref().unwrap().t;
            let h = self.state.as_ref().unwrap().h;
            // if start == 1, then we need to compute the first stage
            if start == 1 {
                let mut hf = self.diff.column_mut(0);
                hf.copy_from(&self.state.as_ref().unwrap().dy);
                hf *= scale(h);
            }
            for i in start..self.tableau.s() {
                let t = t0 + self.tableau.c()[i] * h;
                self.nonlinear_solver.problem().f.set_phi(
                    &self.diff.columns(0, i),
                    &self.state.as_ref().unwrap().y,
                    &self.a_rows[i],
                );

                Self::predict_stage(i, &self.diff, &mut self.old_f, &self.tableau);

                if i == start {
                    // always reset jacobian because we change the step size after every step
                    self.nonlinear_solver.reset_jacobian(&self.old_f, t);
                }
                let mut solve_result = self.nonlinear_solver.solve_in_place(&mut self.old_f, t);
                self.statistics.number_of_nonlinear_solver_iterations +=
                    self.nonlinear_solver.niter();

                // only calculate sensitivities if the solve succeeded
                if solve_result.is_ok() {
                    // old_y now has the new y soln and old_f has the new dy soln
                    self.old_y
                        .copy_from(&self.nonlinear_solver.problem().f.get_last_f_eval());
                    if self.problem().as_ref().unwrap().eqn_sens.is_some() {
                        solve_result = self.solve_for_sensitivities(t);
                    }
                }

                // if we didn't update the jacobian and the solve failed, then we update the jacobian and try again
                let solve_result = if solve_result.is_err() && !updated_jacobian {
                    // newton iteration did not converge, so update jacobian and try again
                    self.nonlinear_solver.problem().f.set_jacobian_is_stale();
                    updated_jacobian = true;

                    Self::predict_stage(i, &self.diff, &mut self.old_f, &self.tableau);

                    self.nonlinear_solver.reset_jacobian(&self.old_f, t);
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    self.nonlinear_solver.solve_in_place(&mut self.old_f, t)
                } else {
                    solve_result
                };

                if solve_result.is_err() {
                    let state = self.state.as_mut().unwrap();
                    // newton iteration did not converge, so we reduce step size and try again
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    state.h *= Eqn::T::from(0.3);

                    // if step size too small, then fail
                    if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                        return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
                    }

                    // update h for new step size
                    self.nonlinear_solver.problem().f.set_h(state.h);

                    // reset nonlinear's linear solver problem as lu factorisation has changed
                    continue 'step;
                };

                // update diff with solved dy
                self.diff.column_mut(i).copy_from(&self.old_f);

                if self.problem().as_ref().unwrap().eqn_sens.is_some() {
                    for diff in self.sdiff.iter_mut() {
                        diff.column_mut(i).copy_from(&self.old_f_sens[i]);
                    }
                }
            }
            // successfully solved for all stages, now compute error
            self.diff
                .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);

            // solve for  (M - h * c * J) * error = error_est as by Hosea, M. E., & Shampine, L. F. (1996). Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20(1-2), 21-37.
            self.nonlinear_solver
                .solve_linearised_in_place(&mut error)?;

            // compute error norm
            let atol = self.problem().as_ref().unwrap().atol.as_ref();
            let rtol = self.problem().as_ref().unwrap().rtol;
            let mut error_norm = error.squared_norm(&self.old_y, atol, rtol);

            // sensitivity errors
            if self.problem().as_ref().unwrap().eqn_sens.is_some() {
                for i in 0..self.sdiff.len() {
                    self.sdiff[i].gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);
                    self.nonlinear_solver
                        .solve_linearised_in_place(&mut error)?;
                    let sens_error_norm = error.squared_norm(&self.old_y_sens[i], atol, rtol);
                    error_norm += sens_error_norm;
                }
                error_norm /= Eqn::T::from((self.sdiff.len() + 1) as f64);
            }

            // adjust step size based on error
            // TODO: if factor close to 1 we shouldn't do this, think there is an alg in the textbook...
            let maxiter = self.nonlinear_solver.max_iter() as f64;
            let niter = self.nonlinear_solver.niter() as f64;
            let safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));
            let order = self.tableau.order() as f64;
            let mut factor = safety * error_norm.pow(Eqn::T::from(-0.5 / (order + 1.0)));
            if factor < Eqn::T::from(Self::MIN_FACTOR) {
                factor = Eqn::T::from(Self::MIN_FACTOR);
            }
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }

            // adjust step size for next step
            let state = self.state.as_mut().unwrap();
            t1 = state.t + state.h;
            state.h *= factor;

            // if step size too small, then fail
            if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
            }

            // update c for new step size
            self.nonlinear_solver.problem().f.set_h(state.h);

            // reset nonlinear's linear solver problem as lu factorisation has changed

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break 'step;
            }
            // step is rejected, factor reduces step size, so we try again with the smaller step size
            self.statistics.number_of_error_test_failures += 1;
        }

        // take the step
        let state = self.state.as_mut().unwrap();
        let dt = t1 - state.t;
        self.old_t = state.t;
        state.t = t1;

        // last stage is the solution and is the same as old_f
        // todo: can we get rid of old_f and just use diff?
        self.old_f.mul_assign(scale(Eqn::T::one() / dt));
        std::mem::swap(&mut self.old_f, &mut state.dy);

        // old_y already has the new y soln
        std::mem::swap(&mut self.old_y, &mut state.y);

        for i in 0..self.sdiff.len() {
            self.old_f_sens[i].mul_assign(scale(Eqn::T::one() / dt));
            std::mem::swap(&mut self.old_f_sens[i], &mut state.ds[i]);
            std::mem::swap(&mut self.old_y_sens[i], &mut state.s[i]);
        }

        self.is_state_mutated = false;

        // update statistics
        self.statistics.number_of_linear_solver_setups =
            self.nonlinear_solver.problem().f.number_of_jac_evals();
        self.statistics.number_of_steps += 1;
        self.statistics.final_step_size = self.state.as_ref().unwrap().h;

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

    fn set_stop_time(&mut self, tstop: <Eqn as OdeEquations>::T) -> Result<()> {
        self.tstop = Some(tstop);
        if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
            self.tstop = None;
            return Err(anyhow::anyhow!(
                "Stop time is at or before current time t = {}",
                self.state.as_ref().unwrap().t
            ));
        }
        Ok(())
    }

    fn interpolate_sens(
        &self,
        _t: <Eqn as OdeEquations>::T,
    ) -> Result<Vec<<Eqn as OdeEquations>::V>> {
        Ok(vec![])
    }

    fn interpolate(&self, t: <Eqn>::T) -> anyhow::Result<<Eqn>::V> {
        if self.state.is_none() {
            return Err(anyhow!("State not set"));
        }
        let state = self.state.as_ref().unwrap();

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(anyhow::anyhow!("Interpolation time is not within the current step. Step size is zero after calling state_mut()"));
            }
        }

        // check that t is within the current step
        if t > state.t || t < self.old_t {
            return Err(anyhow::anyhow!(
                "Interpolation time is not within the current step"
            ));
        }
        let dt = state.t - self.old_t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
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

            // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
            let mut ret = self.old_y.clone();
            self.diff
                .gemv(Eqn::T::one(), &beta_f, Eqn::T::one(), &mut ret);
            Ok(ret)
        } else {
            let hf0 = self.diff.column(0);
            let hf1 = self.diff.column(self.diff.ncols() - 1);
            let u0 = &self.old_y;
            let u1 = &state.y;
            let ret = u0 * scale(Eqn::T::from(1.0) - theta)
                + u1 * scale(theta)
                + ((u1 - u0) * scale(Eqn::T::from(1.0) - Eqn::T::from(2.0) * theta)
                    + hf0 * scale(theta - Eqn::T::from(1.0))
                    + hf1 * scale(theta))
                    * scale(theta * (theta - Eqn::T::from(1.0)));
            Ok(ret)
        }
    }

    fn state(&self) -> Option<&OdeSolverState<Eqn::V>> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> Option<&mut OdeSolverState<Eqn::V>> {
        self.is_state_mutated = true;
        self.state.as_mut()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::{
            test_models::{
                exponential_decay::exponential_decay_problem,
                exponential_decay::exponential_decay_problem_with_root, robertson::robertson,
                robertson_ode::robertson_ode,
            },
            tests::{
                test_interpolate, test_no_set_problem, test_ode_solver, test_state_mut,
                test_state_mut_on_problem,
            },
        },
        NalgebraLU, OdeEquations, Op, Sdirk, Tableau,
    };

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
    fn test_tr_bdf2_nalgebra_exponential_decay() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = exponential_decay_problem::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 28
        number_of_steps: 28
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.011224620483093733
        final_step_size: 0.37808462088748845
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 114
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
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
        number_of_linear_solver_setups: 12
        number_of_steps: 12
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.034484882412482154
        final_step_size: 0.9398383410208245
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 74
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        "###);
    }

    #[cfg(feature = "sundials")]
    #[test]
    fn test_sundials_exponential_decay() {
        let mut s = crate::SundialsIda::default();
        let (problem, soln) = exponential_decay_problem::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 18
        number_of_steps: 43
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 63
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.001
        final_step_size: 0.7770043351266953
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 65
        number_of_jac_muls: 36
        number_of_matrix_evals: 18
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
        number_of_linear_solver_setups: 427
        number_of_steps: 412
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 12
        initial_step_size: 0.0000030885218897033307
        final_step_size: 35655827121.9909
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 3064
        number_of_jac_muls: 42
        number_of_matrix_evals: 14
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
        number_of_linear_solver_setups: 212
        number_of_steps: 193
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 18
        initial_step_size: 0.00007367379016174295
        final_step_size: 44328923924.83207
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 2479
        number_of_jac_muls: 60
        number_of_matrix_evals: 20
        "###);
    }

    #[cfg(feature = "sundials")]
    #[test]
    fn test_sundials_robertson() {
        let mut s = crate::SundialsIda::default();
        let (problem, soln) = robertson::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 59
        number_of_steps: 355
        number_of_error_test_failures: 15
        number_of_nonlinear_solver_iterations: 506
        number_of_nonlinear_solver_fails: 5
        initial_step_size: 0.001
        final_step_size: 11535117835.253025
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 510
        number_of_jac_muls: 180
        number_of_matrix_evals: 60
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_ode() {
        let tableau = Tableau::<M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, NalgebraLU::default());
        let (problem, soln) = robertson_ode::<M>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 248
        number_of_steps: 233
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 15
        initial_step_size: 0.0000027515601924872376
        final_step_size: 31858152718.061752
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 2398
        number_of_jac_muls: 42
        number_of_matrix_evals: 14
        "###);
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
