use crate::error::DiffsolError;
use crate::error::OdeSolverError;
use crate::op::sdirk::SdirkCallable;
use crate::scale;
use crate::AugmentedOdeEquationsImplicit;
use crate::OdeEquationsImplicit;
use crate::OdeSolverStopReason;
use crate::RkState;
use crate::RootFinder;
use crate::Tableau;
use crate::{
    ode_solver_error, AugmentedOdeEquations, Convergence, DefaultDenseMatrix, DenseMatrix,
    NonLinearOp, NonLinearSolver, OdeEquations, OdeSolverProblem, OdeSolverState, Op, Scalar,
    Vector, VectorViewMut,
};
use crate::{TableauMat, TableauVec};
use log::info;
use log::trace;
use num_traits::{abs, FromPrimitive, One, ToPrimitive, Zero};

use super::jacobian_update::SolverState;
use super::pi_controller::pi_controller_raw;
use super::OdeSolverStatistics;
use std::ops::{MulAssign, SubAssign};

/// A Runge-Kutta method.
///
/// The particular method is defined by the [Tableau] used to create the solver.
/// If the `beta` matrix of the [Tableau] is present this is used for interpolation, otherwise hermite interpolation is used.
///
/// Restrictions:
/// - The upper triangular and diagonal parts of the `a` matrix must be zero (i.e. explicit).
/// - The last row of the `a` matrix must be the same as the `b` vector, and the last element of the `c` vector must be 1 (i.e. a stiffly accurate method)
pub struct Rk<'a, Eqn, M = <<Eqn as Op>::V as DefaultDenseMatrix>::M>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    problem: &'a OdeSolverProblem<Eqn>,
    tableau: Tableau<Eqn::T>,
    state: Box<RkState<Eqn::V>>,
    statistics: OdeSolverStatistics,
    root_finder: Option<RootFinder<Eqn::V>>,
    tstop: Option<Eqn::T>,
    diff: M,
    sdiff: Vec<M>,
    sgdiff: Vec<M>,
    gdiff: M,
    // boxed so that swapping state/old_state each accepted step (see `step_accepted`)
    // is a pointer swap instead of a memcpy of the whole (large) RkState struct
    old_state: Box<RkState<Eqn::V>>,
    is_state_mutated: bool,

    error: Option<Eqn::V>,
    out_error: Option<Eqn::V>,
    sens_error: Option<Eqn::V>,
    sens_out_error: Option<Eqn::V>,

    prev_error_norm: Option<Eqn::T>,
}

impl<'a, Eqn, M> Drop for Rk<'a, Eqn, M>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    fn drop(&mut self) {
        info!("Runge-Kutta Solver Statistics: {}", self.statistics);
    }
}

impl<Eqn, M> Clone for Rk<'_, Eqn, M>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    fn clone(&self) -> Self {
        Self {
            old_state: self.old_state.clone(),
            tableau: self.tableau,
            problem: self.problem,
            state: self.state.clone(),
            statistics: self.statistics.clone(),
            root_finder: self.root_finder.clone(),
            tstop: self.tstop,
            is_state_mutated: self.is_state_mutated,
            diff: self.diff.clone(),
            sdiff: self.sdiff.clone(),
            sgdiff: self.sgdiff.clone(),
            gdiff: self.gdiff.clone(),
            error: self.error.clone(),
            out_error: self.out_error.clone(),
            sens_error: self.sens_error.clone(),
            sens_out_error: self.sens_out_error.clone(),
            prev_error_norm: self.prev_error_norm,
        }
    }
}

impl<'a, Eqn, M> Rk<'a, Eqn, M>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
{
    pub(crate) fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<Eqn::T>,
    ) -> Result<Self, DiffsolError> {
        Self::_new(problem, state, tableau, true)
    }

    fn _new(
        problem: &'a OdeSolverProblem<Eqn>,
        mut state: RkState<Eqn::V>,
        tableau: Tableau<Eqn::T>,
        integrate_main_eqn: bool,
    ) -> Result<Self, DiffsolError> {
        // update statistics
        let statistics = OdeSolverStatistics::default();

        state.check_consistent_with_problem(problem)?;

        let nstates = state.y.len();
        let order = tableau.s();

        let ctx = problem.context();

        state.set_problem(problem)?;
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

        let diff = M::zeros(nstates, order, ctx.clone());
        let gdiff_rows = if problem.integrate_out {
            problem.eqn.out().unwrap().nout()
        } else {
            0
        };
        let gdiff = M::zeros(gdiff_rows, order, ctx.clone());

        let old_state = state.clone();

        let error = Some(<Eqn::V as Vector>::zeros(nstates, ctx.clone()));
        let out_error_control = problem.output_in_error_control();
        let out_error = if out_error_control {
            Some(<Eqn::V as Vector>::zeros(
                problem.eqn.out().unwrap().nout(),
                ctx.clone(),
            ))
        } else {
            None
        };

        Ok(Self {
            tableau,
            state: Box::new(state),
            old_state: Box::new(old_state),
            problem,
            statistics,
            root_finder,
            tstop: None,
            is_state_mutated: false,
            diff,
            gdiff,
            sdiff: vec![],
            sgdiff: vec![],
            error,
            out_error,
            sens_error: None,
            sens_out_error: None,
            prev_error_norm: None,
        })
    }

    pub(crate) fn new_augmented<AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<Eqn::T>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        state.check_sens_consistent_with_problem(problem, augmented_eqn)?;
        let integrate_main_eqn = augmented_eqn.integrate_main_eqn();
        let mut ret = Self::_new(problem, state, tableau, integrate_main_eqn)?;
        let naug = augmented_eqn.max_index();
        let nstates = augmented_eqn.rhs().nstates();
        let order = ret.tableau.s();
        let ctx = problem.eqn.context();
        ret.sdiff = vec![M::zeros(nstates, order, ctx.clone()); naug];
        if let Some(out) = augmented_eqn.out() {
            ret.sgdiff = vec![M::zeros(out.nout(), order, ctx.clone()); naug];
        }
        if augmented_eqn.include_in_error_control() {
            ret.sens_error = Some(<Eqn::V as Vector>::zeros(
                augmented_eqn.rhs().nstates(),
                ctx.clone(),
            ))
        };
        if augmented_eqn.include_out_in_error_control() {
            ret.sens_out_error = Some(<Eqn::V as Vector>::zeros(
                augmented_eqn.out().unwrap().nout(),
                ctx.clone(),
            ));
        };
        if !integrate_main_eqn {
            ret.error = None;
            ret.out_error = None;
        }
        Ok(ret)
    }

    pub(crate) fn check_explicit_rk(
        problem: &'a OdeSolverProblem<Eqn>,
        tableau: &Tableau<Eqn::T>,
    ) -> Result<(), DiffsolError> {
        // check that there isn't any mass matrix
        if problem.eqn.mass().is_some() {
            return Err(DiffsolError::from(OdeSolverError::MassMatrixNotSupported));
        }
        // check that the upper triangular and diagonal parts of a are zero
        let s = tableau.s();
        for i in 0..s {
            for j in i..s {
                if tableau.a(i, j) != Eqn::T::zero() {
                    return Err(ode_solver_error!(
                        InvalidTableau,
                        format!(
                            "Invalid tableau, expected a(i, j) = 0 for i >= j, but found a({}, {}) = {}",
                            i,
                            j,
                            tableau.a(i, j)
                        )
                    ));
                }
            }
        }

        // check last row of a is the same as b
        for i in 0..s {
            if tableau.a(s - 1, i) != tableau.b()[i] {
                return Err(ode_solver_error!(
                    InvalidTableau,
                    "Invalid tableau, expected a(s-1, i) = b(i)"
                ));
            }
        }

        // check that last c is 1
        if tableau.c()[s - 1] != Eqn::T::one() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(s-1) = 1"
            ));
        }

        // check that first c is 0
        if tableau.c()[0] != Eqn::T::zero() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(0) = 0"
            ));
        }
        Ok(())
    }

    pub(crate) fn skip_first_stage(&self) -> bool {
        self.tableau.a(0, 0) == Eqn::T::zero()
    }

    pub(crate) fn check_sdirk_rk(tableau: &Tableau<Eqn::T>) -> Result<(), DiffsolError> {
        // check that the upper triangular part of a is zero
        let s = tableau.s();
        for i in 0..s {
            for j in (i + 1)..s {
                if tableau.a(i, j) != Eqn::T::zero() {
                    return Err(ode_solver_error!(
                        InvalidTableau,
                        "Invalid tableau, expected a(i, j) = 0 for i > j"
                    ));
                }
            }
        }
        let gamma = tableau.a(1, 1);
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            if tableau.a(i, i) != gamma {
                return Err(ode_solver_error!(
                    InvalidTableau,
                    format!("Invalid tableau, expected a(i, i) = gamma = {gamma} for i = 1..s-1")
                ));
            }
        }
        // if a(0, 0) = gamma, then we're a SDIRK method
        // if a(0, 0) = 0, then we're a ESDIRK method
        // otherwise, error
        let zero = Eqn::T::zero();
        if tableau.a(0, 0) != zero && tableau.a(0, 0) != gamma {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"
            ));
        }
        let is_sdirk = tableau.a(0, 0) == gamma;

        // check last row of a is the same as b
        for i in 0..s {
            if tableau.a(s - 1, i) != tableau.b()[i] {
                return Err(ode_solver_error!(
                    InvalidTableau,
                    "Invalid tableau, expected a(s-1, i) = b(i)"
                ));
            }
        }

        // check that last c is 1
        if tableau.c()[s - 1] != Eqn::T::one() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(s-1) = 1"
            ));
        }

        // check that the first c is 0 for esdirk methods
        if !is_sdirk && tableau.c()[0] != Eqn::T::zero() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(0) = 0 for esdirk methods"
            ));
        }
        Ok(())
    }

    pub(crate) fn tableau(&self) -> &Tableau<Eqn::T> {
        &self.tableau
    }

    pub(crate) fn get_statistics(&self) -> &OdeSolverStatistics {
        &self.statistics
    }

    pub(crate) fn statistics_mut(&mut self) -> &mut OdeSolverStatistics {
        &mut self.statistics
    }

    pub(crate) fn set_state(&mut self, state: RkState<Eqn::V>) {
        self.is_state_mutated = true;
        *self.state = state;
    }

    pub(crate) fn into_state(mut self) -> RkState<Eqn::V> {
        let ctx = self.problem().eqn.context().clone();
        *std::mem::replace(&mut self.state, Box::new(RkState::new_empty(ctx)))
    }

    pub(crate) fn checkpoint(&mut self) -> RkState<Eqn::V> {
        (*self.state).clone()
    }

    pub(crate) fn order(&self) -> usize {
        self.tableau.order()
    }

    pub(crate) fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.problem
    }

    pub(crate) fn state(&self) -> &RkState<Eqn::V> {
        &self.state
    }

    pub(crate) fn state_mut(&mut self) -> &mut RkState<Eqn::V> {
        self.is_state_mutated = true;
        &mut self.state
    }

    pub(crate) fn state_mut_back(
        &mut self,
        t: M::T,
        integrate_out: bool,
    ) -> Result<(), DiffsolError> {
        let nstates = self.state.y.len();
        let ctx = self.state.y.context().clone();
        let mut y = Eqn::V::zeros(nstates, ctx.clone());
        self.interpolate_inplace(t, &mut y)?;
        let mut dy = Eqn::V::zeros(nstates, ctx.clone());
        self.interpolate_dy_inplace(t, &mut dy)?;
        let g = if integrate_out {
            let nout = self.state.g.len();
            let mut g = Eqn::V::zeros(nout, ctx.clone());
            self.interpolate_out_inplace(t, &mut g)?;
            Some(g)
        } else {
            None
        };
        let nparams = self.state.s.len();
        let s_interp: Vec<Eqn::V> = if nparams > 0 {
            let mut s = vec![Eqn::V::zeros(nstates, ctx); nparams];
            self.interpolate_sens_inplace(t, &mut s)?;
            s
        } else {
            vec![]
        };
        let state = self.state_mut();
        state.y.copy_from(&y);
        state.dy.copy_from(&dy);
        state.t = t;
        if let Some(g) = g.as_ref() {
            state.g.copy_from(g);
        }
        for (j, s_j) in s_interp.iter().enumerate() {
            state.s[j].copy_from(s_j);
        }
        Ok(())
    }

    pub(crate) fn set_stop_time(&mut self, tstop: <Eqn as Op>::T) -> Result<(), DiffsolError> {
        self.tstop = Some(tstop);
        if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
            let error = OdeSolverError::StopTimeAtCurrentTime;
            self.tstop = None;
            return Err(DiffsolError::from(error));
        }
        Ok(())
    }

    pub(crate) fn start_step(&mut self) -> Result<Eqn::T, DiffsolError> {
        if self.is_state_mutated {
            // reinitalise root finder if needed
            if let (Some(root_fn), Some(root_finder)) =
                (self.problem.eqn.root(), self.root_finder.as_ref())
            {
                let state = &self.state;
                root_finder.init(&root_fn, &state.y, state.t);
            }
            // reinitialise tstop if needed
            if let Some(t_stop) = self.tstop {
                self.set_stop_time(t_stop)?;
            }

            self.is_state_mutated = false;
        }

        Ok(self.state.h)
    }

    pub(crate) fn factor(
        &self,
        error_norm: Eqn::T,
        safety_factor: f64,
        min_reduce_factor: Eqn::T,
        max_reduce_factor: Eqn::T,
        min_increase_factor: Eqn::T,
        max_increase_factor: Eqn::T,
    ) -> Eqn::T {
        let safety = Eqn::T::from_f64(0.9 * safety_factor).unwrap();
        let raw = pi_controller_raw(
            error_norm,
            self.prev_error_norm,
            self.problem().ode_options.pi_control_integral,
            self.problem().ode_options.pi_control_proportional,
            self.order() + 1,
        );

        let mut factor = safety * raw;
        if factor > max_reduce_factor && factor < min_increase_factor {
            factor = Eqn::T::one();
        }
        if factor < min_reduce_factor {
            factor = min_reduce_factor;
        }
        if factor > max_increase_factor {
            factor = max_increase_factor;
        }
        factor
    }

    pub(crate) fn set_prev_error(&mut self, error: Eqn::T) {
        self.prev_error_norm = Some(error);
    }

    pub(crate) fn reset_prev_error(&mut self) {
        self.prev_error_norm = None;
    }

    pub(crate) fn start_step_attempt(
        &mut self,
        h: Eqn::T,
        augmented_eqn: Option<&mut impl AugmentedOdeEquations<Eqn>>,
    ) {
        // if start == 1, then we need to compute the first stage
        // from the last stage of the previous step
        if self.skip_first_stage() {
            trace!("Skipping first stage, setting to h * dy from previous step");
            self.diff
                .column_mut(0)
                .axpy(h, &self.state.dy, Eqn::T::zero());

            // sensitivities too
            if augmented_eqn.is_some() {
                for (sdiff, ds) in self.sdiff.iter_mut().zip(self.state.ds.iter()) {
                    sdiff.column_mut(0).axpy(h, ds, Eqn::T::zero());
                }
                for (sgdiff, sdg) in self.sgdiff.iter_mut().zip(self.state.dsg.iter()) {
                    sgdiff.column_mut(0).axpy(h, sdg, Eqn::T::zero());
                }
            }

            // output function
            if self.problem.integrate_out {
                self.gdiff
                    .column_mut(0)
                    .axpy(h, &self.state.dg, Eqn::T::zero());
            }
        }
    }

    pub(crate) fn do_stage(
        &mut self,
        i: usize,
        h: Eqn::T,
        augmented_eqn: Option<&mut impl AugmentedOdeEquations<Eqn>>,
    ) {
        let t = self.state.t + self.tableau.c()[i] * h;

        // main equation
        let integrate_main_eqn = augmented_eqn
            .as_ref()
            .map(|eqn| eqn.integrate_main_eqn())
            .unwrap_or(true);
        if integrate_main_eqn {
            self.old_state.y.copy_from(&self.state.y);
            self.diff.gemv_cols(
                0,
                i,
                Eqn::T::one(),
                self.tableau.stage_coeffs(i),
                Eqn::T::one(),
                &mut self.old_state.y,
            );

            // update diff with solved dy
            self.problem
                .eqn
                .rhs()
                .call_inplace(&self.old_state.y, t, &mut self.old_state.dy);
            self.diff
                .column_mut(i)
                .axpy(h, &self.old_state.dy, Eqn::T::zero());

            // calculate dg and store in gdiff
            if self.problem.integrate_out {
                let out = self.problem.eqn.out().unwrap();
                out.call_inplace(&self.old_state.y, t, &mut self.old_state.dg);
                self.gdiff
                    .column_mut(i)
                    .axpy(h, &self.old_state.dg, Eqn::T::zero());
            }
        }

        // calculate sensitivities
        if let Some(aug_eqn) = augmented_eqn {
            (*aug_eqn).update_rhs_out_state(&self.old_state.y, &self.old_state.dy, t);
            for j in 0..self.sdiff.len() {
                aug_eqn.set_index(j);
                self.old_state.s[j].copy_from(&self.state.s[j]);
                self.sdiff[j].gemv_cols(
                    0,
                    i,
                    Eqn::T::one(),
                    self.tableau.stage_coeffs(i),
                    Eqn::T::one(),
                    &mut self.old_state.s[j],
                );

                aug_eqn
                    .rhs()
                    .call_inplace(&self.old_state.s[j], t, &mut self.old_state.ds[j]);

                self.sdiff[j]
                    .column_mut(i)
                    .axpy(h, &self.old_state.ds[j], Eqn::T::zero());

                // calculate sdg and store in sgdiff
                if let Some(out) = aug_eqn.out() {
                    out.call_inplace(&self.old_state.s[j], t, &mut self.old_state.dsg[j]);
                    self.sgdiff[j]
                        .column_mut(i)
                        .axpy(h, &self.old_state.dsg[j], Eqn::T::zero());
                }
            }
        }
    }

    fn predict_stage_sdirk(
        i: usize,
        h: Eqn::T,
        dy0: &Eqn::V,
        diff: &M,
        hdy: &mut Eqn::V,
        tableau: &Tableau<Eqn::T>,
    ) {
        if i == 0 {
            hdy.axpy(h, dy0, Eqn::T::zero());
        } else if i == 1 {
            hdy.copy_from_view(&diff.column(i - 1));
        } else {
            let c =
                (tableau.c()[i] - tableau.c()[i - 2]) / (tableau.c()[i - 1] - tableau.c()[i - 2]);
            // dy = c1  + c * (c1 - c2)
            hdy.copy_from_view(&diff.column(i - 1));
            hdy.axpy_v(-c, &diff.column(i - 2), Eqn::T::one() + c);
        }
    }

    pub(crate) fn do_stage_sdirk<AugEqn>(
        &mut self,
        i: usize,
        h: Eqn::T,
        op: Option<&SdirkCallable<&Eqn>>,
        mut s_op: Option<&mut SdirkCallable<AugEqn>>,
        nonlinear_solver: &mut impl NonLinearSolver<Eqn::M>,
        convergence: &mut Convergence<'a, Eqn::V>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
        AugEqn: AugmentedOdeEquationsImplicit<Eqn>,
    {
        let t = self.state.t + self.tableau.c()[i] * h;
        // main equation
        if let Some(op) = op {
            op.set_phi(
                Eqn::T::one(),
                &self.diff,
                i,
                &self.state.y,
                self.tableau.stage_coeffs(i),
            );
            Self::predict_stage_sdirk(
                i,
                h,
                &self.state.dy,
                &self.diff,
                &mut self.old_state.dy,
                &self.tableau,
            );
            if !nonlinear_solver.is_jacobian_set() {
                nonlinear_solver.reset_jacobian(op, &self.state.y, t);
                self.statistics
                    .record_linear_solver_setup(SolverState::Checkpoint);
            }
            let solve_result = nonlinear_solver.solve_in_place(
                op,
                &mut self.old_state.dy,
                t,
                &self.state.y,
                //&self.diff.column(0).into_owned(),
                convergence,
            );
            self.statistics.number_of_nonlinear_solver_iterations += convergence.niter();
            solve_result?;
            op.get_f_eval(&self.old_state.dy, &mut self.old_state.y);

            // update diff with solved dy
            self.diff.column_mut(i).copy_from(&self.old_state.dy);

            // calculate dg and store in gdiff
            if self.problem.integrate_out {
                let out = self.problem.eqn.out().unwrap();
                out.call_inplace(&self.old_state.y, t, &mut self.old_state.dg);
                self.gdiff
                    .column_mut(i)
                    .axpy(h, &self.old_state.dg, Eqn::T::zero());
            }
        }

        // calculate sensitivities
        if let Some(ref mut op) = s_op {
            // update for new state
            op.eqn_mut()
                .update_rhs_out_state(&self.old_state.y, &self.old_state.dy, t);

            // solve for sensitivities equations discretised using sdirk equation
            for j in 0..self.sdiff.len() {
                op.set_phi(
                    Eqn::T::one(),
                    &self.sdiff[j],
                    i,
                    &self.state.s[j],
                    self.tableau.stage_coeffs(i),
                );
                op.eqn_mut().set_index(j);
                Self::predict_stage_sdirk(
                    i,
                    h,
                    &self.state.ds[j],
                    &self.sdiff[j],
                    &mut self.old_state.ds[j],
                    &self.tableau,
                );

                if !nonlinear_solver.is_jacobian_set() {
                    nonlinear_solver.reset_jacobian::<SdirkCallable<AugEqn>>(
                        op,
                        &self.old_state.s[j],
                        t,
                    );
                    self.statistics
                        .record_linear_solver_setup(SolverState::Checkpoint);
                }

                // solve
                let solver_result = nonlinear_solver.solve_in_place(
                    *op,
                    &mut self.old_state.ds[j],
                    t,
                    &self.state.s[j],
                    //&self.sdiff[j].column(0).into_owned(),
                    convergence,
                );
                self.statistics.number_of_nonlinear_solver_iterations += convergence.niter();
                solver_result?;

                op.get_f_eval(&self.old_state.ds[j], &mut self.old_state.s[j]);
                self.sdiff[j].column_mut(i).copy_from(&self.old_state.ds[j]);

                // calculate sdg and store in sgdiff
                if let Some(out) = op.eqn().out() {
                    out.call_inplace(&self.old_state.s[j], t, &mut self.old_state.dsg[j]);
                    self.sgdiff[j]
                        .column_mut(i)
                        .axpy(h, &self.old_state.dsg[j], Eqn::T::zero());
                }
            }
        }
        Ok(())
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        let state = &mut self.state;
        // check if the we are at tstop
        let troundoff =
            Eqn::T::from_f64(100.0).unwrap() * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if (state.h > Eqn::T::zero() && tstop < state.t - troundoff)
            || (state.h < Eqn::T::zero() && tstop > state.t + troundoff)
        {
            return Err(DiffsolError::from(
                OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time: tstop.to_f64().unwrap(),
                    state_time: (state.t).to_f64().unwrap(),
                },
            ));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if (state.h > Eqn::T::zero() && state.t + state.h > tstop + troundoff)
            || (state.h < Eqn::T::zero() && state.t + state.h < tstop - troundoff)
        {
            let factor = (tstop - state.t) / state.h;
            state.h.mul_assign(factor);
        }
        Ok(None)
    }

    pub(crate) fn error_norm(
        &mut self,
        _h: Eqn::T,
        augmented_eqn: Option<&mut impl AugmentedOdeEquations<Eqn>>,
        linear_solver: impl FnOnce(&mut Eqn::V) -> Result<(), DiffsolError>,
    ) -> Result<Eqn::T, DiffsolError> {
        let s = self.tableau.s();
        let mut error_norm = Eqn::T::zero();
        if let Some(error) = self.error.as_mut() {
            self.diff.gemv_cols(
                0,
                s,
                Eqn::T::one(),
                self.tableau.d().as_slice(),
                Eqn::T::zero(),
                error,
            );
            linear_solver(error)?;

            // compute error norm
            let atol = &self.problem.atol;
            let rtol = self.problem.rtol;
            let err = error.squared_norm(&self.state.y, atol, rtol);
            error_norm = error_norm.max(err);
        }

        if let Some(out_error) = self.out_error.as_mut() {
            // output errors
            self.gdiff.gemv_cols(
                0,
                s,
                Eqn::T::one(),
                self.tableau.d().as_slice(),
                Eqn::T::zero(),
                out_error,
            );
            let atol = self.problem.out_atol.as_ref().unwrap();
            let rtol = self.problem.out_rtol.unwrap();
            let out_error_norm = out_error.squared_norm(&self.state.g, atol, rtol);
            error_norm = error_norm.max(out_error_norm);
        }

        // sensitivity errors
        if let Some(sens_error) = self.sens_error.as_mut() {
            let aug_eqn = augmented_eqn.as_ref().unwrap();
            let rtol = aug_eqn.rtol().unwrap();
            for i in 0..self.sdiff.len() {
                self.sdiff[i].gemv_cols(
                    0,
                    s,
                    Eqn::T::one(),
                    self.tableau.d().as_slice(),
                    Eqn::T::zero(),
                    sens_error,
                );
                let atol = aug_eqn.atol(i).unwrap();
                let err = sens_error.squared_norm(&self.state.s[i], atol, rtol);
                error_norm = error_norm.max(err);
            }
        }

        // sensitivity output errors
        if let Some(sens_out_error) = self.sens_out_error.as_mut() {
            let aug_eqn = augmented_eqn.as_ref().unwrap();
            let atol = aug_eqn.out_atol().unwrap();
            let rtol = aug_eqn.out_rtol().unwrap();
            for i in 0..self.sgdiff.len() {
                self.sgdiff[i].gemv_cols(
                    0,
                    s,
                    Eqn::T::one(),
                    self.tableau.d().as_slice(),
                    Eqn::T::zero(),
                    sens_out_error,
                );
                let err = sens_out_error.squared_norm(&self.state.sg[i], atol, rtol);
                error_norm = error_norm.max(err);
            }
        }
        Ok(error_norm)
    }

    pub(crate) fn error_test_fail(
        &mut self,
        h: Eqn::T,
        nattempts: usize,
        max_error_test_fails: usize,
        min_timestep: Eqn::T,
    ) -> Result<(), DiffsolError> {
        self.statistics.number_of_error_test_failures += 1;
        // if too many error test failures, then fail
        if nattempts >= max_error_test_fails {
            return Err(DiffsolError::from(
                OdeSolverError::TooManyErrorTestFailures {
                    time: self.state.t.to_f64().unwrap(),
                    num_failures: nattempts,
                },
            ));
        }
        // if step size too small, then fail
        if abs(h) < min_timestep {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.state.t.to_f64().unwrap(),
            }));
        }
        Ok(())
    }

    pub(crate) fn solve_fail(
        &mut self,
        h: Eqn::T,
        min_timestep: Eqn::T,
        max_nonlinear_solver_fails: usize,
    ) -> Result<(), DiffsolError> {
        self.statistics.number_of_nonlinear_solver_fails += 1;
        // if too many nonlinear solver failures, then fail
        if self.statistics.number_of_nonlinear_solver_fails > max_nonlinear_solver_fails {
            return Err(DiffsolError::from(
                OdeSolverError::TooManyNonlinearSolverFailures {
                    time: self.state.t.to_f64().unwrap(),
                    num_failures: max_nonlinear_solver_fails,
                },
            ));
        }
        // if step size too small, then fail
        if abs(h) < min_timestep {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.state.t.to_f64().unwrap(),
            }));
        }
        Ok(())
    }

    pub(crate) fn step_accepted(
        &mut self,
        h: Eqn::T,
        new_h: Eqn::T,
        rescale_dy: bool,
    ) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let s = self.tableau.s();
        // step accepted, so integrate output functions
        if self.problem.integrate_out {
            self.old_state.g.copy_from(&self.state.g);
            self.gdiff.gemv_cols(
                0,
                s,
                Eqn::T::one(),
                self.tableau.b().as_slice(),
                Eqn::T::one(),
                &mut self.old_state.g,
            );
        }

        for i in 0..self.sgdiff.len() {
            self.old_state.sg[i].copy_from(&self.state.sg[i]);
            self.sgdiff[i].gemv_cols(
                0,
                s,
                Eqn::T::one(),
                self.tableau.b().as_slice(),
                Eqn::T::one(),
                &mut self.old_state.sg[i],
            );
        }

        // take the step
        self.old_state.t = self.state.t + h;
        self.old_state.h = new_h;
        if rescale_dy {
            self.old_state.dy *= scale(Eqn::T::one() / h);
            for ds in self.old_state.ds.iter_mut() {
                ds.mul_assign(scale(Eqn::T::one() / h));
            }
        }
        std::mem::swap(&mut self.old_state, &mut self.state);

        // update statistics
        self.statistics.number_of_steps += 1;

        // check for root within accepted step
        if let (Some(root_fn), Some(root_finder)) =
            (self.problem.eqn.root(), self.root_finder.as_ref())
        {
            let ret = root_finder.check_root(
                &|t, y| self.interpolate_inplace(t, y),
                &root_fn,
                &self.state.y,
                self.state.t,
            );
            if let Some((root, root_idx)) = ret {
                return Ok(OdeSolverStopReason::RootFound(root, root_idx));
            }
        }

        // check if the we are at tstop
        if let Some(tstop) = self.tstop {
            if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
                self.tstop = None; // reset tstop
                return Ok(OdeSolverStopReason::TstopReached);
            }
        }

        // just a normal step, no roots or tstop reached
        Ok(OdeSolverStopReason::InternalTimestep)
    }

    fn interpolate_from_diff(y0: &M::V, beta_f: &[M::T], diff: &M, ret: &mut M::V) {
        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        ret.copy_from(y0);
        diff.gemv_cols(0, beta_f.len(), M::T::one(), beta_f, M::T::one(), ret);
    }

    /// `beta * [theta, theta^2, ..., theta^p]`, scaled by `scale`: one weight per stage.
    ///
    /// `beta_t` is the tableau's beta matrix transposed, so stage `i`'s polynomial coefficients
    /// are the contiguous column `i`. The returned length is the stage count, which is what
    /// callers hand to `gemv_cols` as the column count.
    fn interpolate_beta_weights(
        theta: M::T,
        beta_t: &TableauMat<M::T>,
        scale: M::T,
    ) -> TableauVec<M::T> {
        let mut out = TableauVec::zeros(beta_t.ncols());
        for (i, o) in out.as_mut_slice().iter_mut().enumerate() {
            let row = beta_t.as_col_slice(i);
            // theta^(k+1), accumulated as k advances
            let mut theta_pow = theta;
            let mut acc = M::T::zero();
            for r in row {
                acc += *r * theta_pow;
                theta_pow *= theta;
            }
            *o = scale * acc;
        }
        out
    }

    /// Derivative of [`Self::interpolate_beta_weights`] w.r.t. `theta`:
    /// `d/dtheta [theta, theta^2, ..., theta^p] = [1, 2*theta, ..., p*theta^{p-1}]`.
    fn interpolate_beta_weights_deriv(
        theta: M::T,
        beta_t: &TableauMat<M::T>,
        scale: M::T,
    ) -> TableauVec<M::T> {
        let mut out = TableauVec::zeros(beta_t.ncols());
        for (i, o) in out.as_mut_slice().iter_mut().enumerate() {
            let row = beta_t.as_col_slice(i);
            let mut theta_pow = M::T::one();
            let mut acc = M::T::zero();
            for (k, r) in row.iter().enumerate() {
                let coeff = M::T::from_f64(k as f64 + 1.0).unwrap();
                acc += *r * coeff * theta_pow;
                theta_pow *= theta;
            }
            *o = scale * acc;
        }
        out
    }

    fn interpolate_hermite(
        scale_diff: M::T,
        theta: M::T,
        u0: &M::V,
        u1: &M::V,
        diff: &M,
        y: &mut M::V,
    ) {
        let f0 = diff.column(0);
        let f1 = diff.column(diff.ncols() - 1);

        y.copy_from(u1);
        y.sub_assign(u0);
        y.axpy_v(
            scale_diff * (theta - M::T::one()),
            &f0,
            M::T::one() - M::T::from_f64(2.0).unwrap() * theta,
        );
        y.axpy_v(scale_diff * theta, &f1, M::T::one());
        y.axpy(M::T::one() - theta, u0, theta * (theta - M::T::one()));
        y.axpy(theta, u1, M::T::one());
    }

    // Derivative of the Hermite interpolant w.r.t. t.
    //
    // The Hermite polynomial is p(theta) = theta*(theta-1)*Q(theta) + (1-theta)*u0 + theta*u1
    // where Q(theta) = (1-2*theta)*(u1-u0) + scale_diff*(theta-1)*f0 + scale_diff*theta*f1
    //
    // Its derivative w.r.t. theta is:
    //   p'(theta) = (2*theta-1)*Q(theta) + theta*(theta-1)*Q'(theta) + (u1-u0)
    // where Q'(theta) = -2*(u1-u0) + scale_diff*(f0 + f1)
    //
    // And dy/dt = p'(theta) / dt.
    fn interpolate_hermite_deriv(
        scale_diff: M::T,
        theta: M::T,
        dt: M::T,
        u0: &M::V,
        u1: &M::V,
        diff: &M,
        dy: &mut M::V,
    ) {
        let f0 = diff.column(0);
        let f1 = diff.column(diff.ncols() - 1);
        let nstates = dy.len();

        // Build Q(theta) into a temporary
        let mut q = <M::V as Vector>::zeros(nstates, diff.context().clone());
        q.copy_from(u1);
        q.sub_assign(u0); // q = u1 - u0
        q.axpy_v(
            scale_diff * (theta - M::T::one()),
            &f0,
            M::T::one() - M::T::from_f64(2.0).unwrap() * theta,
        ); // q = (1-2*theta)*(u1-u0) + scale_diff*(theta-1)*f0
        q.axpy_v(scale_diff * theta, &f1, M::T::one()); // q = Q(theta)

        // dy = (u1-u0)/dt + (2*theta-1)/dt * Q(theta)
        dy.copy_from(u1);
        dy.sub_assign(u0); // dy = u1 - u0
        dy.axpy(
            (M::T::from_f64(2.0).unwrap() * theta - M::T::one()) / dt,
            &q,
            M::T::one() / dt,
        ); // dy = (1/dt)*(u1-u0) + (2*theta-1)/dt * Q(theta)

        // Reuse q for Q'(theta) = 2*(u0-u1) + scale_diff*(f0 + f1)
        q.copy_from(u0);
        q.sub_assign(u1); // q = u0 - u1
        q.axpy_v(scale_diff, &f0, M::T::from_f64(2.0).unwrap()); // q = 2*(u0-u1) + scale_diff*f0
        q.axpy_v(scale_diff, &f1, M::T::one()); // q = Q'(theta)

        // dy += theta*(theta-1)/dt * Q'(theta)
        dy.axpy(theta * (theta - M::T::one()) / dt, &q, M::T::one());
    }

    pub(crate) fn interpolate_inplace(&self, t: M::T, ret: &mut M::V) -> Result<(), DiffsolError> {
        if ret.len() != self.state.y.len() {
            return Err(DiffsolError::from(
                OdeSolverError::InterpolationVectorWrongSize {
                    expected: self.state.y.len(),
                    found: ret.len(),
                },
            ));
        }
        if self.is_state_mutated {
            if t == self.state.t {
                ret.copy_from(&self.state.y);
                return Ok(());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = self.state.h > M::T::zero();
        if (is_forward && (t > self.state.t || t < self.old_state.t))
            || (!is_forward && (t < self.state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = self.state.t - self.old_state.t;
        let theta = if dt == M::T::zero() {
            M::T::one()
        } else {
            (t - self.old_state.t) / dt
        };
        let scale_diff = Eqn::T::one();
        if let Some(beta_t) = self.tableau.beta_t() {
            let beta_f = Self::interpolate_beta_weights(theta, beta_t, scale_diff);
            Self::interpolate_from_diff(&self.old_state.y, beta_f.as_slice(), &self.diff, ret);
        } else {
            Self::interpolate_hermite(
                scale_diff,
                theta,
                &self.old_state.y,
                &self.state.y,
                &self.diff,
                ret,
            );
        }
        Ok(())
    }

    pub(crate) fn interpolate_dy_inplace(
        &self,
        t: M::T,
        dy: &mut M::V,
    ) -> Result<(), DiffsolError> {
        if dy.len() != self.state.y.len() {
            return Err(DiffsolError::from(
                OdeSolverError::InterpolationVectorWrongSize {
                    expected: self.state.y.len(),
                    found: dy.len(),
                },
            ));
        }
        if self.is_state_mutated {
            if t == self.state.t {
                dy.copy_from(&self.state.dy);
                return Ok(());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = self.state.h > M::T::zero();
        if (is_forward && (t > self.state.t || t < self.old_state.t))
            || (!is_forward && (t < self.state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = self.state.t - self.old_state.t;
        if dt == M::T::zero() {
            dy.copy_from(&self.state.dy);
            return Ok(());
        }
        let theta = (t - self.old_state.t) / dt;
        let scale_diff = Eqn::T::one();
        if let Some(beta_t) = self.tableau.beta_t() {
            // dy/dt = (scale_diff / dt) * diff * d_beta_f, with the scalar folded into the
            // weights so no extra pass is needed
            let d_beta_f = Self::interpolate_beta_weights_deriv(theta, beta_t, scale_diff / dt);
            self.diff.gemv_cols(
                0,
                d_beta_f.len(),
                M::T::one(),
                d_beta_f.as_slice(),
                M::T::zero(),
                dy,
            );
        } else {
            Self::interpolate_hermite_deriv(
                scale_diff,
                theta,
                dt,
                &self.old_state.y,
                &self.state.y,
                &self.diff,
                dy,
            );
        }
        Ok(())
    }

    pub(crate) fn interpolate_out_inplace(
        &self,
        t: M::T,
        g: &mut M::V,
    ) -> Result<(), DiffsolError> {
        if g.len() != self.state.g.len() {
            return Err(DiffsolError::from(
                OdeSolverError::InterpolationVectorWrongSize {
                    expected: self.state.g.len(),
                    found: g.len(),
                },
            ));
        }
        if self.is_state_mutated {
            if t == self.state.t {
                g.copy_from(&self.state.g);
                return Ok(());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = self.state.h > M::T::zero();
        if (is_forward && (t > self.state.t || t < self.old_state.t))
            || (!is_forward && (t < self.state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = self.state.t - self.old_state.t;
        let theta = if dt == M::T::zero() {
            M::T::one()
        } else {
            (t - self.old_state.t) / dt
        };
        let scale_diff = Eqn::T::one();
        if let Some(beta_t) = self.tableau.beta_t() {
            let beta_f = Self::interpolate_beta_weights(theta, beta_t, scale_diff);
            Self::interpolate_from_diff(&self.old_state.g, beta_f.as_slice(), &self.gdiff, g);
        } else {
            Self::interpolate_hermite(
                scale_diff,
                theta,
                &self.old_state.g,
                &self.state.g,
                &self.gdiff,
                g,
            );
        }
        Ok(())
    }

    pub(crate) fn interpolate_sens_inplace(
        &self,
        t: Eqn::T,
        ret: &mut [M::V],
    ) -> Result<(), DiffsolError> {
        if ret.len() != self.state.s.len() {
            return Err(DiffsolError::from(
                OdeSolverError::SensitivityCountMismatch {
                    expected: self.state.s.len(),
                    found: ret.len(),
                },
            ));
        }
        for s in ret.iter() {
            if s.len() != self.state.s[0].len() {
                return Err(DiffsolError::from(
                    OdeSolverError::InterpolationVectorWrongSize {
                        expected: self.state.s[0].len(),
                        found: s.len(),
                    },
                ));
            }
        }
        if self.is_state_mutated {
            if t == self.state.t {
                for (r, s) in ret.iter_mut().zip(self.state.s.iter()) {
                    r.copy_from(s);
                }
                return Ok(());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = self.state.h > M::T::zero();
        if (is_forward && (t > self.state.t || t < self.old_state.t))
            || (!is_forward && (t < self.state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = self.state.t - self.old_state.t;
        let theta = if dt == M::T::zero() {
            M::T::one()
        } else {
            (t - self.old_state.t) / dt
        };
        let scale_diff = Eqn::T::one();
        if let Some(beta_t) = self.tableau.beta_t() {
            let beta_f = Self::interpolate_beta_weights(theta, beta_t, scale_diff);
            for ((y, diff), r) in self
                .old_state
                .s
                .iter()
                .zip(self.sdiff.iter())
                .zip(ret.iter_mut())
            {
                Self::interpolate_from_diff(y, beta_f.as_slice(), diff, r);
            }
        } else {
            for ((s0, s1), (diff, r)) in self
                .old_state
                .s
                .iter()
                .zip(self.state.s.iter())
                .zip(self.sdiff.iter().zip(ret.iter_mut()))
            {
                Self::interpolate_hermite(scale_diff, theta, s0, s1, diff, r);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext,
        error::{DiffsolError, OdeSolverError},
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            exponential_decay::exponential_decay_problem,
            exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem,
        },
        DefaultDenseMatrix, OdeEquations, OdeSolverProblem, Tableau,
    };

    use super::Rk;
    use crate::TableauMat;

    type M = NalgebraMat<f64>;

    fn check_sdirk_for_problem<Eqn>(
        _problem: &OdeSolverProblem<Eqn>,
        tableau: &Tableau<Eqn::T>,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<T = f64, V = crate::NalgebraVec<f64>, C = NalgebraContext>,
        Eqn::V: DefaultDenseMatrix<T = f64, C = NalgebraContext, M = M>,
    {
        Rk::<Eqn, M>::check_sdirk_rk(tableau)
    }

    /// `base` with `a[0, 0]` overwritten, rebuilt through the public natural-orientation API.
    fn tableau_with_a00(base: Tableau<f64>, a00: f64) -> Tableau<f64> {
        let s = base.s();
        let mut a = TableauMat::zeros(s, s);
        for i in 0..s {
            for j in 0..s {
                a[(i, j)] = base.a(i, j);
            }
        }
        a[(0, 0)] = a00;
        Tableau::new(
            a,
            *base.b(),
            *base.c(),
            *base.d(),
            base.order(),
            base.beta_t().map(|beta_t| beta_t.transposed()),
        )
    }

    fn make_invalid_explicit_tableau() -> Tableau<f64> {
        tableau_with_a00(Tableau::tsit45(), 1.0)
    }

    fn make_invalid_sdirk_tableau() -> Tableau<f64> {
        tableau_with_a00(Tableau::tr_bdf2(), 0.25)
    }

    fn expect_invalid_tableau(err: DiffsolError) {
        assert!(err.to_string().contains("Invalid tableau"));
    }

    #[test]
    fn explicit_rk_rejects_mass_matrices() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<M>(false);
        let err = Rk::<_, M>::check_explicit_rk(&problem, &Tableau::tsit45()).unwrap_err();
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::MassMatrixNotSupported)
        ));
    }

    #[test]
    fn explicit_rk_rejects_invalid_a_diagonal() {
        let (problem, _soln) = exponential_decay_problem::<M>(false);
        let err =
            Rk::<_, M>::check_explicit_rk(&problem, &make_invalid_explicit_tableau()).unwrap_err();
        expect_invalid_tableau(err);
    }

    #[test]
    fn sdirk_rk_rejects_invalid_first_diagonal_entry() {
        let (problem, _soln) = exponential_decay_problem::<M>(false);
        let err = check_sdirk_for_problem(&problem, &make_invalid_sdirk_tableau()).unwrap_err();
        expect_invalid_tableau(err);
    }
}
