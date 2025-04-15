use crate::error::DiffsolError;
use crate::error::OdeSolverError;
use crate::ode_solver_error;
use crate::vector::VectorRef;
use crate::NoAug;
use crate::OdeSolverStopReason;
use crate::RkState;
use crate::RootFinder;
use crate::Tableau;
use crate::{
    scale, AugmentedOdeEquations, DefaultDenseMatrix, DenseMatrix, MatrixView, NonLinearOp,
    OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, Scalar, StateRef,
    StateRefMut, Vector, VectorViewMut,
};
use num_traits::abs;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;

use super::bdf::BdfStatistics;
use super::method::AugmentedOdeSolverMethod;

impl<'a, Eqn, M, AugEqn> AugmentedOdeSolverMethod<'a, Eqn, AugEqn>
    for ExplicitRk<'a, Eqn, M, AugEqn>
where
    Eqn: OdeEquations,
    AugEqn: AugmentedOdeEquations<Eqn>,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugEqn>) {
        (self.state, self.augmented_eqn)
    }
    fn augmented_eqn(&self) -> Option<&AugEqn> {
        self.augmented_eqn.as_ref()
    }
}

/// An explicit Runge-Kutta method.
///
/// The particular method is defined by the [Tableau] used to create the solver.
/// If the `beta` matrix of the [Tableau] is present this is used for interpolation, otherwise hermite interpolation is used.
///
/// Restrictions:
/// - The upper triangular and diagonal parts of the `a` matrix must be zero (i.e. explicit).
/// - The last row of the `a` matrix must be the same as the `b` vector, and the last element of the `c` vector must be 1 (i.e. a stiffly accurate method)
pub struct ExplicitRk<
    'a,
    Eqn,
    M = <<Eqn as Op>::V as DefaultDenseMatrix>::M,
    AugmentedEqn = NoAug<Eqn>,
> where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    tableau: Tableau<M>,
    problem: &'a OdeSolverProblem<Eqn>,
    augmented_eqn: Option<AugmentedEqn>,
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
}

impl<Eqn, M, AugmentedEqn> Clone for ExplicitRk<'_, Eqn, M, AugmentedEqn>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
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
            augmented_eqn: self.augmented_eqn.clone(),
        }
    }
}

impl<'a, Eqn, M, AugmentedEqn> ExplicitRk<'a, Eqn, M, AugmentedEqn>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
{
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-13;
    const MAX_ERROR_TEST_FAILS: usize = 40;

    pub fn new(
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
        // check that there isn't any mass matrix
        if problem.eqn.mass().is_some() {
            return Err(DiffsolError::from(OdeSolverError::MassMatrixNotSupported));
        }
        // check that the upper triangular and diagonal parts of a are zero
        let s = tableau.s();
        for i in 0..s {
            for j in i..s {
                assert_eq!(
                    tableau.a().get_index(i, j),
                    Eqn::T::zero(),
                    "Invalid tableau, expected a(i, j) = 0 for i >= j"
                );
            }
        }

        let mut a_rows = Vec::with_capacity(s);
        let ctx = problem.context();
        for i in 0..s {
            let mut row = Vec::with_capacity(i);
            for j in 0..i {
                row.push(tableau.a().get_index(i, j));
            }
            a_rows.push(Eqn::V::from_vec(row, ctx.clone()));
        }

        // check last row of a is the same as b
        for i in 0..s {
            assert_eq!(
                tableau.a().get_index(s - 1, i),
                tableau.b().get_index(i),
                "Invalid tableau, expected a(s-1, i) = b(i)"
            );
        }

        // check that last c is 1
        assert_eq!(
            tableau.c().get_index(s - 1),
            Eqn::T::one(),
            "Invalid tableau, expected c(s-1) = 1"
        );

        // check that first c is 0
        assert_eq!(
            tableau.c().get_index(0),
            Eqn::T::zero(),
            "Invalid tableau, expected c(0) = 0"
        );

        // update statistics
        let statistics = BdfStatistics::default();

        state.check_consistent_with_problem(problem)?;

        let nstates = state.y.len();
        let order = tableau.s();

        state.set_problem(problem)?;
        let root_finder = if let Some(root_fn) = problem.eqn.root() {
            let root_finder = RootFinder::new(root_fn.nout(), ctx.clone());
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
            augmented_eqn: None,
        })
    }

    pub fn new_augmented(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        tableau: Tableau<M>,
        augmented_eqn: AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        state.check_sens_consistent_with_problem(problem, &augmented_eqn)?;
        let mut ret = Self::_new(problem, state, tableau)?;
        let naug = augmented_eqn.max_index();
        let nstates = augmented_eqn.rhs().nstates();
        let order = ret.tableau.s();
        let ctx = problem.eqn.context();
        ret.sdiff = vec![M::zeros(nstates, order, ctx.clone()); naug];
        if let Some(out) = augmented_eqn.out() {
            ret.sgdiff = vec![M::zeros(out.nout(), order, ctx.clone()); naug];
        }
        ret.augmented_eqn = Some(augmented_eqn);
        Ok(ret)
    }

    pub fn get_statistics(&self) -> &BdfStatistics {
        &self.statistics
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        let state = &mut self.state;

        // check if the we are at tstop
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if (state.h > M::T::zero() && tstop < state.t - troundoff)
            || (state.h < M::T::zero() && tstop > state.t + troundoff)
        {
            return Err(DiffsolError::from(
                OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time: tstop.into(),
                    state_time: state.t.into(),
                },
            ));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if (state.h > M::T::zero() && state.t + state.h > tstop + troundoff)
            || (state.h < M::T::zero() && state.t + state.h < tstop - troundoff)
        {
            let factor = (tstop - state.t) / state.h;
            state.h *= factor;
        }
        Ok(None)
    }

    fn interpolate_from_diff(h: Eqn::T, y0: &Eqn::V, beta_f: &Eqn::V, diff: &M) -> Eqn::V {
        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        let mut ret = y0.clone();
        diff.gemv(h, beta_f, Eqn::T::one(), &mut ret);
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
        let thetav = Eqn::V::from_vec(thetav, beta.context().clone());
        let mut beta_f = <Eqn::V as Vector>::zeros(s_star, beta.context().clone());
        beta.gemv(M::T::one(), &thetav, Eqn::T::zero(), &mut beta_f);
        beta_f
    }

    fn interpolate_hermite(h: Eqn::T, theta: Eqn::T, u0: &Eqn::V, u1: &Eqn::V, diff: &M) -> Eqn::V {
        let f0 = diff.column(0);
        let f1 = diff.column(diff.ncols() - 1);
        u0 * scale(Eqn::T::from(1.0) - theta)
            + u1 * scale(theta)
            + ((u1 - u0) * scale(Eqn::T::from(1.0) - Eqn::T::from(2.0) * theta)
                + f0 * scale(h * (theta - Eqn::T::from(1.0)))
                + f1 * scale(h * theta))
                * scale(theta * (theta - Eqn::T::from(1.0)))
    }
}

impl<'a, Eqn, M, AugmentedEqn> OdeSolverMethod<'a, Eqn> for ExplicitRk<'a, Eqn, M, AugmentedEqn>
where
    Eqn: OdeEquations,
    M: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
{
    type State = RkState<Eqn::V>;

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.problem
    }

    fn jacobian(&self) -> Option<std::cell::Ref<<Eqn>::M>> {
        None
    }

    fn mass(&self) -> Option<std::cell::Ref<<Eqn>::M>> {
        None
    }

    fn order(&self) -> usize {
        self.tableau.order()
    }

    fn set_state(&mut self, state: Self::State) {
        self.state = state;
    }

    fn into_state(self) -> RkState<Eqn::V> {
        self.state
    }

    fn checkpoint(&mut self) -> Self::State {
        self.state.clone()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let n = self.state.y.len();
        let old_num_error_test_fails = self.statistics.number_of_error_test_failures;

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

        let ctx = self.problem.eqn.context();
        // todo: remove this allocation?
        let mut error = <Eqn::V as Vector>::zeros(n, ctx.clone());
        let out_error_control = self.problem().output_in_error_control();
        let mut out_error = if out_error_control {
            <Eqn::V as Vector>::zeros(self.problem().eqn.out().unwrap().nout(), ctx.clone())
        } else {
            <Eqn::V as Vector>::zeros(0, ctx.clone())
        };
        let sens_error_control = self.augmented_eqn.is_some()
            && self
                .augmented_eqn
                .as_ref()
                .unwrap()
                .include_in_error_control();
        let mut sens_error = if sens_error_control {
            <Eqn::V as Vector>::zeros(
                self.augmented_eqn.as_ref().unwrap().rhs().nstates(),
                ctx.clone(),
            )
        } else {
            <Eqn::V as Vector>::zeros(0, ctx.clone())
        };
        let sens_out_error_control = self.augmented_eqn.is_some()
            && self
                .augmented_eqn
                .as_ref()
                .unwrap()
                .include_out_in_error_control();
        let mut sens_out_error = if sens_out_error_control {
            <Eqn::V as Vector>::zeros(
                self.augmented_eqn.as_ref().unwrap().out().unwrap().nout(),
                ctx.clone(),
            )
        } else {
            <Eqn::V as Vector>::zeros(0, ctx.clone())
        };
        let integrate_main_eqn = if let Some(aug_eqn) = self.augmented_eqn.as_ref() {
            aug_eqn.integrate_main_eqn()
        } else {
            true
        };

        // loop until step is accepted
        let t0 = self.state.t;
        let mut h = self.state.h;
        let mut factor: Eqn::T;
        'step: loop {
            // since start == 1, then we need to compute the first stage
            // from the last stage of the previous step
            self.diff.column_mut(0).copy_from(&self.state.dy);

            // sensitivities too
            if self.augmented_eqn.is_some() {
                for (diff, dy) in self.sdiff.iter_mut().zip(self.state.ds.iter()) {
                    diff.column_mut(0).copy_from(dy);
                }
                for (diff, dg) in self.sgdiff.iter_mut().zip(self.state.dsg.iter()) {
                    diff.column_mut(0).copy_from(dg);
                }
            }

            // output function
            if self.problem.integrate_out {
                self.gdiff.column_mut(0).copy_from(&self.state.dg);
            }

            for i in 1..self.tableau.s() {
                let t = t0 + self.tableau.c().get_index(i) * h;

                // main equation
                if integrate_main_eqn {
                    self.old_state.y.copy_from(&self.state.y);
                    self.diff.columns(0, i).gemv_o(
                        h,
                        &self.a_rows[i],
                        Eqn::T::one(),
                        &mut self.old_state.y,
                    );

                    // update diff with solved dy
                    self.problem.eqn.rhs().call_inplace(
                        &self.old_state.y,
                        t,
                        &mut self.old_state.dy,
                    );
                    self.diff.column_mut(i).copy_from(&self.old_state.dy);

                    // calculate dg and store in gdiff
                    if self.problem.integrate_out {
                        let out = self.problem.eqn.out().unwrap();
                        out.call_inplace(&self.old_state.y, t, &mut self.old_state.dg);
                        self.gdiff.column_mut(i).copy_from(&self.old_state.dg);
                    }
                }

                // calculate sensitivities
                if let Some(aug_eqn) = self.augmented_eqn.as_mut() {
                    aug_eqn.update_rhs_out_state(&self.old_state.y, &self.old_state.dy, t);
                    for j in 0..self.sdiff.len() {
                        aug_eqn.set_index(j);
                        self.old_state.s[j].copy_from(&self.state.s[j]);
                        self.sdiff[j].columns(0, i).gemv_o(
                            h,
                            &self.a_rows[i],
                            Eqn::T::one(),
                            &mut self.old_state.s[j],
                        );

                        aug_eqn.rhs().call_inplace(
                            &self.old_state.s[j],
                            t,
                            &mut self.old_state.ds[j],
                        );

                        self.sdiff[j].column_mut(i).copy_from(&self.old_state.ds[j]);

                        // calculate sdg and store in sgdiff
                        if let Some(out) = aug_eqn.out() {
                            out.call_inplace(&self.old_state.s[j], t, &mut self.old_state.dsg[j]);
                            self.sgdiff[j]
                                .column_mut(i)
                                .copy_from(&self.old_state.dsg[j]);
                        }
                    }
                }
            }
            let mut ncontributions = 0;
            let mut error_norm = Eqn::T::zero();
            // successfully solved for all stages, now compute error
            if integrate_main_eqn {
                self.diff
                    .gemv(h, self.tableau.d(), Eqn::T::zero(), &mut error);

                // compute error norm
                let atol = &self.problem().atol;
                let rtol = self.problem().rtol;
                error_norm += error.squared_norm(&self.old_state.y, atol, rtol);
                ncontributions += 1;

                // output errors
                if out_error_control {
                    self.gdiff
                        .gemv(h, self.tableau.d(), Eqn::T::zero(), &mut out_error);
                    let atol = self.problem().out_atol.as_ref().unwrap();
                    let rtol = self.problem().out_rtol.unwrap();
                    let out_error_norm = out_error.squared_norm(&self.state.g, atol, rtol);
                    error_norm += out_error_norm;
                    ncontributions += 1;
                }
            }

            // sensitivity errors
            if sens_error_control {
                let aug_eqn = self.augmented_eqn.as_ref().unwrap();
                let atol = aug_eqn.atol().unwrap();
                let rtol = aug_eqn.rtol().unwrap();
                for i in 0..self.sdiff.len() {
                    self.sdiff[i].gemv(h, self.tableau.d(), Eqn::T::zero(), &mut sens_error);
                    let sens_error_norm = sens_error.squared_norm(&self.old_state.s[i], atol, rtol);
                    error_norm += sens_error_norm;
                    ncontributions += 1;
                }
            }

            // sensitivity output errors
            if sens_out_error_control {
                let aug_eqn = self.augmented_eqn.as_ref().unwrap();
                let atol = aug_eqn.atol().unwrap();
                let rtol = aug_eqn.rtol().unwrap();
                for i in 0..self.sgdiff.len() {
                    self.sgdiff[i].gemv(h, self.tableau.d(), Eqn::T::zero(), &mut sens_out_error);
                    let sens_error_norm =
                        sens_out_error.squared_norm(&self.state.sg[i], atol, rtol);
                    error_norm += sens_error_norm;
                    ncontributions += 1;
                }
            }
            if ncontributions > 1 {
                error_norm /= Eqn::T::from(ncontributions as f64);
            }

            // adjust step size based on error
            let safety = Eqn::T::from(0.9);
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
            if self.statistics.number_of_error_test_failures - old_num_error_test_fails
                >= Self::MAX_ERROR_TEST_FAILS
            {
                return Err(DiffsolError::from(
                    OdeSolverError::TooManyErrorTestFailures {
                        time: self.state.t.into(),
                    },
                ));
            }
            h *= factor;

            // if step size too small, then fail
            if abs(h) < Eqn::T::from(Self::MIN_TIMESTEP) {
                return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                    time: self.state.t.into(),
                }));
            }
        }

        // step accepted, so integrate output functions
        if self.problem.integrate_out {
            self.old_state.g.copy_from(&self.state.g);
            self.gdiff
                .gemv(h, self.tableau.b(), Eqn::T::one(), &mut self.old_state.g);
        }

        for i in 0..self.sgdiff.len() {
            self.old_state.sg[i].copy_from(&self.state.sg[i]);
            self.sgdiff[i].gemv(
                h,
                self.tableau.b(),
                Eqn::T::one(),
                &mut self.old_state.sg[i],
            );
        }

        // take the step
        self.old_state.t = t0 + h;
        // update step size for next step
        self.old_state.h = factor * h;
        std::mem::swap(&mut self.old_state, &mut self.state);

        // update statistics
        self.statistics.number_of_steps += 1;

        // check for root within accepted step
        if let Some(root_fn) = self.problem.eqn.root() {
            let ret = self.root_finder.as_ref().unwrap().check_root(
                &|t| self.interpolate(t),
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
            if let Some(reason) = self.handle_tstop(tstop).unwrap() {
                return Ok(reason);
            }
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

    fn interpolate_sens(&self, t: <Eqn as Op>::T) -> Result<Vec<<Eqn as Op>::V>, DiffsolError> {
        let state = &self.state;

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.s.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_state.t))
            || (!is_forward && (t < state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }
        let dt = state.t - self.old_state.t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_state.t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = self
                .old_state
                .s
                .iter()
                .zip(self.sdiff.iter())
                .map(|(y, diff)| Self::interpolate_from_diff(dt, y, &beta_f, diff))
                .collect();
            Ok(ret)
        } else {
            let ret = self
                .old_state
                .s
                .iter()
                .zip(state.s.iter())
                .zip(self.sdiff.iter())
                .map(|((s0, s1), diff)| Self::interpolate_hermite(dt, theta, s0, s1, diff))
                .collect();
            Ok(ret)
        }
    }

    fn interpolate(&self, t: <Eqn>::T) -> Result<<Eqn>::V, DiffsolError> {
        let state = &self.state;

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_state.t))
            || (!is_forward && (t < state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = state.t - self.old_state.t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_state.t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = Self::interpolate_from_diff(dt, &self.old_state.y, &beta_f, &self.diff);
            Ok(ret)
        } else {
            let ret = Self::interpolate_hermite(dt, theta, &self.old_state.y, &state.y, &self.diff);
            Ok(ret)
        }
    }

    fn interpolate_out(&self, t: <Eqn>::T) -> Result<<Eqn>::V, DiffsolError> {
        let state = &self.state;

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.g.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_state.t))
            || (!is_forward && (t < state.t || t > self.old_state.t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = state.t - self.old_state.t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_state.t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = Self::interpolate_from_diff(dt, &self.old_state.g, &beta_f, &self.gdiff);
            Ok(ret)
        } else {
            let ret =
                Self::interpolate_hermite(dt, theta, &self.old_state.g, &state.g, &self.gdiff);
            Ok(ret)
        }
    }

    fn state(&self) -> StateRef<Eqn::V> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> StateRefMut<Eqn::V> {
        self.is_state_mutated = true;
        self.state.as_mut()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_solver::{
            test_models::exponential_decay::{
                exponential_decay_problem, exponential_decay_problem_adjoint,
                exponential_decay_problem_sens, exponential_decay_problem_with_root,
                negative_exponential_decay_problem,
            },
            tests::{
                setup_test_adjoint, setup_test_adjoint_sum_squares, test_adjoint,
                test_adjoint_sum_squares, test_checkpointing, test_interpolate, test_ode_solver,
                test_problem, test_state_mut, test_state_mut_on_problem,
            },
        },
        Context, DenseMatrix, MatrixCommon, NalgebraLU, NalgebraVec, OdeEquations, OdeSolverMethod,
        Op, Vector, VectorView,
    };

    use num_traits::abs;

    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn explicit_rk_state_mut() {
        test_state_mut(test_problem::<M>().tsit45().unwrap());
    }
    #[test]
    fn explicit_rk_test_interpolate() {
        test_interpolate(test_problem::<M>().tsit45().unwrap());
    }

    #[test]
    fn explicit_rk_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let s1 = problem.tsit45().unwrap();
        let s2 = problem.tsit45().unwrap();
        test_checkpointing(soln, s1, s2);
    }

    #[test]
    fn explicit_rk_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = p.tsit45().unwrap();
        test_state_mut_on_problem(s, soln);
    }

    #[test]
    fn explicit_rk_test_nalgebra_negative_exponential_decay() {
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tsit45_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 0
        number_of_steps: 5
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 32
        number_of_jac_muls: 0
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tsit45_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.tsit45_sens().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 0
        number_of_steps: 7
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 0
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 44
        number_of_jac_muls: 86
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn explicit_rk_test_tsit45_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.tsit45().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem.tsit45_solver_adjoint(checkpointer, None).unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 434
        number_of_jac_muls: 8
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 123
        "###);
    }

    #[test]
    fn explicit_rk_test_nalgebra_exponential_decay_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(false);
        let mut s = problem.tsit45().unwrap();
        let (checkpointer, soln) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .tsit45_solver_adjoint(checkpointer, Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 747
        number_of_jac_muls: 0
        number_of_matrix_evals: 0
        number_of_jac_adj_muls: 1707
        "###);
    }

    #[test]
    fn test_tstop_tsit45() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tsit45().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[test]
    fn test_root_finder_tsit45() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let mut s = problem.tsit45().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }

    #[test]
    fn test_param_sweep_tsit45() {
        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        let mut ps = Vec::new();
        for y0 in (1..10).map(f64::from) {
            ps.push(problem.context().vector_from_vec(vec![0.1, y0]));
        }

        let mut old_soln: Option<NalgebraVec<f64>> = None;
        for p in ps {
            problem.eqn_mut().set_params(&p);
            let mut s = problem.tsit45().unwrap();
            let (ys, _ts) = s.solve(10.0).unwrap();
            // check that the new solution is different from the old one
            if let Some(old_soln) = &mut old_soln {
                let new_soln = ys.column(ys.ncols() - 1).into_owned();
                let error = new_soln - &*old_soln;
                let diff = error
                    .squared_norm(old_soln, &problem.atol, problem.rtol)
                    .sqrt();
                assert!(diff > 1.0e-6, "diff: {}", diff);
            }
            old_soln = Some(ys.column(ys.ncols() - 1).into_owned());
        }
    }

    #[cfg(feature = "diffsl")]
    #[test]
    fn test_ball_bounce_tsit45() {
        type M = crate::NalgebraMat<f64>;
        let (x, v, t) = crate::ode_solver::tests::test_ball_bounce(
            crate::ode_solver::tests::test_ball_bounce_problem::<M>()
                .tsit45()
                .unwrap(),
        );
        let expected_x = [6.375884661615263];
        let expected_v = [0.6878538646461059];
        let expected_t = [2.5];
        for (i, ((x, v), t)) in x.iter().zip(v.iter()).zip(t.iter()).enumerate() {
            assert!((x - expected_x[i]).abs() < 1e-4);
            assert!((v - expected_v[i]).abs() < 1e-4);
            assert!((t - expected_t[i]).abs() < 1e-4);
        }
    }
}
