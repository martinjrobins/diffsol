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
    MatrixView, NonLinearOp, NonLinearSolver, OdeEquations, OdeSolverProblem, OdeSolverState, Op,
    Scalar, Vector, VectorViewMut,
};
use num_traits::{abs, FromPrimitive, One, Pow, ToPrimitive, Zero};

use super::bdf::BdfStatistics;
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
    tableau: Tableau<M>,
    state: RkState<Eqn::V>,
    a_rows: Vec<Eqn::V>,
    statistics: BdfStatistics,
    root_finder: Option<RootFinder<Eqn::V>>,
    tstop: Option<Eqn::T>,
    diff: M,
    sdiff: Vec<M>,
    sgdiff: Vec<M>,
    gdiff: M,
    old_state: RkState<Eqn::V>,
    is_state_mutated: bool,

    error: Option<Eqn::V>,
    out_error: Option<Eqn::V>,
    sens_error: Option<Eqn::V>,
    sens_out_error: Option<Eqn::V>,
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
            tableau: self.tableau.clone(),
            problem: self.problem,
            state: self.state.clone(),
            a_rows: self.a_rows.clone(),
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
        tableau: Tableau<M>,
    ) -> Result<Self, DiffsolError> {
        Self::_new(problem, state, tableau)
    }

    fn _new(
        problem: &'a OdeSolverProblem<Eqn>,
        mut state: RkState<Eqn::V>,
        tableau: Tableau<M>,
    ) -> Result<Self, DiffsolError> {
        // update statistics
        let statistics = BdfStatistics::default();

        state.check_consistent_with_problem(problem)?;

        let nstates = state.y.len();
        let order = tableau.s();

        let s = tableau.s();
        let mut a_rows = Vec::with_capacity(s);
        let ctx = problem.context();
        for i in 0..s {
            let mut row = Vec::with_capacity(i);
            for j in 0..i {
                row.push(tableau.a().get_index(i, j));
            }
            a_rows.push(Eqn::V::from_vec(row, ctx.clone()));
        }

        state.set_problem(problem)?;
        let root_finder = if let Some(root_fn) = problem.eqn.root() {
            let root_finder = RootFinder::new(root_fn.nout(), problem.eqn.nstates(), ctx.clone());
            root_finder.init(&root_fn, &state.y, state.t);
            Some(root_finder)
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
            state,
            old_state,
            problem,
            a_rows,
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
        })
    }

    pub(crate) fn new_augmented<AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<M>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        state.check_sens_consistent_with_problem(problem, augmented_eqn)?;
        let mut ret = Self::_new(problem, state, tableau)?;
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
        if !augmented_eqn.integrate_main_eqn() {
            ret.error = None;
            ret.out_error = None;
        }
        Ok(ret)
    }

    pub(crate) fn check_explicit_rk(
        problem: &'a OdeSolverProblem<Eqn>,
        tableau: &Tableau<M>,
    ) -> Result<(), DiffsolError> {
        // check that there isn't any mass matrix
        if problem.eqn.mass().is_some() {
            return Err(DiffsolError::from(OdeSolverError::MassMatrixNotSupported));
        }
        // check that the upper triangular and diagonal parts of a are zero
        let s = tableau.s();
        for i in 0..s {
            for j in i..s {
                if tableau.a().get_index(i, j) != Eqn::T::zero() {
                    return Err(ode_solver_error!(
                        InvalidTableau,
                        format!(
                            "Invalid tableau, expected a(i, j) = 0 for i >= j, but found a({}, {}) = {}",
                            i,
                            j,
                            tableau.a().get_index(i, j)
                        )
                    ));
                }
            }
        }

        // check last row of a is the same as b
        for i in 0..s {
            if tableau.a().get_index(s - 1, i) != tableau.b().get_index(i) {
                return Err(ode_solver_error!(
                    InvalidTableau,
                    "Invalid tableau, expected a(s-1, i) = b(i)"
                ));
            }
        }

        // check that last c is 1
        if tableau.c().get_index(s - 1) != Eqn::T::one() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(s-1) = 1"
            ));
        }

        // check that first c is 0
        if tableau.c().get_index(0) != Eqn::T::zero() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(0) = 0"
            ));
        }
        Ok(())
    }

    pub(crate) fn skip_first_stage(&self) -> bool {
        self.tableau.a().get_index(0, 0) == Eqn::T::zero()
    }

    pub(crate) fn check_sdirk_rk(tableau: &Tableau<M>) -> Result<(), DiffsolError> {
        // check that the upper triangular part of a is zero
        let s = tableau.s();
        for i in 0..s {
            for j in (i + 1)..s {
                if tableau.a().get_index(i, j) != Eqn::T::zero() {
                    return Err(ode_solver_error!(
                        InvalidTableau,
                        "Invalid tableau, expected a(i, j) = 0 for i > j"
                    ));
                }
            }
        }
        let gamma = tableau.a().get_index(1, 1);
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            if tableau.a().get_index(i, i) != gamma {
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
        if tableau.a().get_index(0, 0) != zero && tableau.a().get_index(0, 0) != gamma {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"
            ));
        }
        let is_sdirk = tableau.a().get_index(0, 0) == gamma;

        // check last row of a is the same as b
        for i in 0..s {
            if tableau.a().get_index(s - 1, i) != tableau.b().get_index(i) {
                return Err(ode_solver_error!(
                    InvalidTableau,
                    "Invalid tableau, expected a(s-1, i) = b(i)"
                ));
            }
        }

        // check that last c is 1
        if tableau.c().get_index(s - 1) != Eqn::T::one() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(s-1) = 1"
            ));
        }

        // check that the first c is 0 for esdirk methods
        if !is_sdirk && tableau.c().get_index(0) != Eqn::T::zero() {
            return Err(ode_solver_error!(
                InvalidTableau,
                "Invalid tableau, expected c(0) = 0 for esdirk methods"
            ));
        }
        Ok(())
    }

    pub(crate) fn tableau(&self) -> &Tableau<M> {
        &self.tableau
    }

    pub(crate) fn get_statistics(&self) -> &BdfStatistics {
        &self.statistics
    }

    pub(crate) fn set_state(&mut self, state: RkState<Eqn::V>) {
        self.is_state_mutated = true;
        self.state = state;
    }

    pub(crate) fn into_state(self) -> RkState<Eqn::V> {
        self.state
    }

    pub(crate) fn checkpoint(&mut self) -> RkState<Eqn::V> {
        self.state.clone()
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

    pub(crate) fn old_state(&self) -> &RkState<Eqn::V> {
        &self.old_state
    }

    pub(crate) fn state_mut(&mut self) -> &mut RkState<Eqn::V> {
        self.is_state_mutated = true;
        &mut self.state
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
            if let Some(root_fn) = self.problem.eqn.root() {
                let state = &self.state;
                self.root_finder
                    .as_ref()
                    .unwrap()
                    .init(&root_fn, &state.y, state.t);
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
        min_factor: Eqn::T,
        max_factor: Eqn::T,
    ) -> Eqn::T {
        let safety = Eqn::T::from_f64(0.9 * safety_factor).unwrap();
        let mut factor =
            safety * error_norm.pow(Eqn::T::from_f64(-0.5 / (self.order() as f64 + 1.0)).unwrap());
        if factor < min_factor {
            factor = min_factor;
        }
        if factor > max_factor {
            factor = max_factor;
        }
        factor
    }

    pub(crate) fn start_step_attempt(
        &mut self,
        h: Eqn::T,
        augmented_eqn: Option<&mut impl AugmentedOdeEquations<Eqn>>,
    ) {
        // if start == 1, then we need to compute the first stage
        // from the last stage of the previous step
        if self.skip_first_stage() {
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
        let t = self.state.t + self.tableau.c().get_index(i) * h;

        // main equation
        let integrate_main_eqn = augmented_eqn
            .as_ref()
            .map(|eqn| eqn.integrate_main_eqn())
            .unwrap_or(true);
        if integrate_main_eqn {
            self.old_state.y.copy_from(&self.state.y);
            self.diff.columns(0, i).gemv_o(
                Eqn::T::one(),
                &self.a_rows[i],
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
                self.sdiff[j].columns(0, i).gemv_o(
                    Eqn::T::one(),
                    &self.a_rows[i],
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
        dy: &mut Eqn::V,
        tableau: &Tableau<M>,
    ) {
        if i == 0 {
            dy.axpy(h, dy0, Eqn::T::zero());
        } else if i == 1 {
            dy.copy_from_view(&diff.column(i - 1));
        } else {
            let c = (tableau.c().get_index(i) - tableau.c().get_index(i - 2))
                / (tableau.c().get_index(i - 1) - tableau.c().get_index(i - 2));
            // dy = c1  + c * (c1 - c2)
            dy.copy_from_view(&diff.column(i - 1));
            dy.axpy_v(-c, &diff.column(i - 2), Eqn::T::one() + c);
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
        let t = self.state.t + self.tableau.c().get_index(i) * h;

        // main equation
        if let Some(op) = op {
            op.set_phi(
                Eqn::T::one(),
                &self.diff.columns(0, i),
                &self.state.y,
                &self.a_rows[i],
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
                nonlinear_solver.reset_jacobian(op, &self.old_state.dy, t);
            }
            let solve_result = nonlinear_solver.solve_in_place(
                op,
                &mut self.old_state.dy,
                t,
                &self.state.y,
                convergence,
            );
            self.statistics.number_of_nonlinear_solver_iterations += convergence.niter();
            solve_result?;
            self.old_state.y.copy_from(&op.get_last_f_eval());

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
                    &self.sdiff[j].columns(0, i),
                    &self.state.s[j],
                    &self.a_rows[i],
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
                        &self.old_state.ds[j],
                        t,
                    );
                }

                // solve
                let solver_result = nonlinear_solver.solve_in_place(
                    *op,
                    &mut self.old_state.ds[j],
                    t,
                    &self.state.s[j],
                    convergence,
                );
                self.statistics.number_of_nonlinear_solver_iterations += convergence.niter();
                solver_result?;

                self.old_state.s[j].copy_from(&op.get_last_f_eval());
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
        self.statistics.number_of_linear_solver_setups = op.map_or_else(
            || s_op.as_ref().unwrap().number_of_jac_evals(),
            |op| op.number_of_jac_evals(),
        );
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
    ) -> Eqn::T {
        let mut ncontributions = 0;
        let mut error_norm = Eqn::T::zero();
        if let Some(error) = self.error.as_mut() {
            self.diff
                .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), error);

            // compute error norm
            let atol = &self.problem.atol;
            let rtol = self.problem.rtol;
            error_norm += error.squared_norm(&self.state.y, atol, rtol);
            ncontributions += 1;
        }

        if let Some(out_error) = self.out_error.as_mut() {
            // output errors
            self.gdiff
                .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), out_error);
            let atol = self.problem.out_atol.as_ref().unwrap();
            let rtol = self.problem.out_rtol.unwrap();
            let out_error_norm = out_error.squared_norm(&self.state.g, atol, rtol);
            error_norm += out_error_norm;
            ncontributions += 1;
        }

        // sensitivity errors
        if let Some(sens_error) = self.sens_error.as_mut() {
            let aug_eqn = augmented_eqn.as_ref().unwrap();
            let atol = aug_eqn.atol().unwrap();
            let rtol = aug_eqn.rtol().unwrap();
            for i in 0..self.sdiff.len() {
                self.sdiff[i].gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), sens_error);
                error_norm += sens_error.squared_norm(&self.state.s[i], atol, rtol);
                ncontributions += 1;
            }
        }

        // sensitivity output errors
        if let Some(sens_out_error) = self.sens_out_error.as_mut() {
            let aug_eqn = augmented_eqn.as_ref().unwrap();
            let atol = aug_eqn.out_atol().unwrap();
            let rtol = aug_eqn.out_rtol().unwrap();
            for i in 0..self.sgdiff.len() {
                self.sgdiff[i].gemv(
                    Eqn::T::one(),
                    self.tableau.d(),
                    Eqn::T::zero(),
                    sens_out_error,
                );
                error_norm += sens_out_error.squared_norm(&self.state.sg[i], atol, rtol);
                ncontributions += 1;
            }
        }
        if ncontributions > 1 {
            error_norm /= Eqn::T::from_f64(ncontributions as f64).unwrap();
        }
        error_norm
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
    ) -> Result<(), DiffsolError> {
        self.statistics.number_of_nonlinear_solver_fails += 1;
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
        // step accepted, so integrate output functions
        if self.problem.integrate_out {
            self.old_state.g.copy_from(&self.state.g);
            self.gdiff.gemv(
                Eqn::T::one(),
                self.tableau.b(),
                Eqn::T::one(),
                &mut self.old_state.g,
            );
        }

        for i in 0..self.sgdiff.len() {
            self.old_state.sg[i].copy_from(&self.state.sg[i]);
            self.sgdiff[i].gemv(
                Eqn::T::one(),
                self.tableau.b(),
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
        if let Some(root_fn) = self.problem.eqn.root() {
            let ret = self.root_finder.as_ref().unwrap().check_root(
                &|t, y| self.interpolate_inplace(t, y),
                &root_fn,
                &self.state.y,
                self.state.t,
            );
            if let Some(root) = ret {
                return Ok(OdeSolverStopReason::RootFound(root));
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

    fn interpolate_from_diff(scale_diff: M::T, y0: &M::V, beta_f: &M::V, diff: &M, ret: &mut M::V) {
        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        ret.copy_from(y0);
        diff.gemv(scale_diff, beta_f, M::T::one(), ret);
    }

    fn interpolate_beta_function(theta: M::T, beta: &M) -> M::V {
        let poly_order = beta.ncols();
        let s_star = beta.nrows();
        let mut thetav = Vec::with_capacity(poly_order);
        thetav.push(theta);
        for i in 1..poly_order {
            thetav.push(theta * thetav[i - 1]);
        }
        // beta_poly = beta * thetav
        let thetav = M::V::from_vec(thetav, beta.context().clone());
        let mut beta_f = <M::V as Vector>::zeros(s_star, beta.context().clone());
        beta.gemv(M::T::one(), &thetav, M::T::zero(), &mut beta_f);
        beta_f
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
        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            Self::interpolate_from_diff(scale_diff, &self.old_state.y, &beta_f, &self.diff, ret);
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
        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            Self::interpolate_from_diff(scale_diff, &self.old_state.g, &beta_f, &self.gdiff, g);
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
        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            for ((y, diff), r) in self
                .old_state
                .s
                .iter()
                .zip(self.sdiff.iter())
                .zip(ret.iter_mut())
            {
                Self::interpolate_from_diff(scale_diff, y, &beta_f, diff, r);
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
