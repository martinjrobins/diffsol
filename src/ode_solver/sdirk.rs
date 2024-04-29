use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::ops::MulAssign;
use std::rc::Rc;

use crate::{LinearSolver, NonLinearOp, LinearOp};
use crate::matrix::MatrixRef;
use crate::op::linearise::LinearisedOp;
use crate::vector::VectorRef;
use crate::NewtonNonlinearSolver;
use crate::Tableau;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    DenseMatrix, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Vector,
    VectorView, VectorViewMut,
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
pub struct Sdirk<M, Eqn>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    tableau: Tableau<M>,
    problem: Option<OdeSolverProblem<Eqn>>,
    nonlinear_solver: NewtonNonlinearSolver<SdirkCallable<Eqn>>,
    state: Option<OdeSolverState<Eqn::V>>,
    diff: M,
    gamma: Eqn::T,
    is_sdirk: bool,
    old_t: Eqn::T,
    old_y: Eqn::V,
    old_f: Eqn::V,
    f: Eqn::V,
    a_rows: Vec<Eqn::V>,
    statistics: BdfStatistics<Eqn::T>,
}

impl<M, Eqn> Sdirk<M, Eqn>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    const NEWTON_MAXITER: usize = 10;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-13;

    pub fn new(
        tableau: Tableau<M>,
        linear_solver: impl LinearSolver<SdirkCallable<Eqn>> + 'static,
    ) -> Self {
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
        let f = <Eqn::V as Vector>::zeros(n);
        let statistics = BdfStatistics::default();
        Self {
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
            f,
            statistics,
        }
    }

    pub fn get_statistics(&self) -> &BdfStatistics<Eqn::T> {
        &self.statistics
    }
}

impl<M, Eqn> OdeSolverMethod<Eqn> for Sdirk<M, Eqn>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.problem.as_ref()
    }

    fn set_problem(
        &mut self,
        mut state: OdeSolverState<<Eqn>::V>,
        problem: &OdeSolverProblem<Eqn>,
    ) {
        // update initial step size based on function
        let mut scale_factor = state.y.abs();
        scale_factor *= scale(problem.rtol);
        scale_factor += problem.atol.as_ref();

        // compute first step based on alg in Hairer, Norsett, Wanner
        // Solving Ordinary Differential Equations I, Nonstiff Problems
        // Section II.4.2
        let f0 = problem.eqn.rhs().call(&state.y, state.t);
        let hf0 = &f0 * scale(state.h);

        let mut tmp = f0.clone();
        tmp.component_div_assign(&scale_factor);
        let d0 = tmp.norm();

        tmp = state.y.clone();
        tmp.component_div_assign(&scale_factor);
        let d1 = f0.norm();

        let h0 = if d0 < Eqn::T::from(1e-5) || d1 < Eqn::T::from(1e-5) {
            Eqn::T::from(1e-6)
        } else {
            Eqn::T::from(0.01) * (d0 / d1)
        };

        let y1 = &state.y + hf0;
        let t1 = state.t + h0;
        let f1 = problem.eqn.rhs().call(&y1, t1);

        let mut df = f1 - &f0;
        df *= scale(Eqn::T::one() / h0);
        df.component_div_assign(&scale_factor);
        let d2 = df.norm();

        let mut max_d = d2;
        if max_d < d1 {
            max_d = d1;
        }
        let h1 = if max_d < Eqn::T::from(1e-15) {
            let h1 = h0 * Eqn::T::from(1e-3);
            if h1 < Eqn::T::from(1e-6) {
                Eqn::T::from(1e-6)
            } else {
                h1
            }
        } else {
            (Eqn::T::from(0.01) / max_d)
                .pow(Eqn::T::one() / Eqn::T::from(1.0 + self.tableau.order() as f64))
        };

        state.h = Eqn::T::from(100.0) * h0;
        if state.h > h1 {
            state.h = h1;
        }

        // setup linear solver for first step
        let callable = Rc::new(SdirkCallable::new(problem, self.gamma));
        callable.set_h(state.h);
        let nonlinear_problem = SolverProblem::new_from_ode_problem(callable, problem);
        self.nonlinear_solver.set_problem(&nonlinear_problem);

        // update statistics
        self.statistics = BdfStatistics::default();
        self.statistics.initial_step_size = state.h;

        self.diff = M::zeros(state.y.len(), self.tableau.s());
        self.old_f = f0.clone();
        self.f = f0;
        self.old_t = state.t;
        self.old_y = state.y.clone();
        self.state = Some(state);
        self.problem = Some(problem.clone());
    }

    fn step(&mut self) -> anyhow::Result<()> {
        // optionally do the first step
        let state = self.state.as_mut().unwrap();
        let n = state.y.len();
        let y0 = &state.y;
        let start = if self.is_sdirk { 0 } else { 1 };
        let mut updated_jacobian = false;
        let mut error = <Eqn::V as Vector>::zeros(n);

        let mut t1: Eqn::T;

        // loop until step is accepted
        'step: loop {
            // if start == 1, then we need to compute the first stage
            if start == 1 {
                let mut hf = self.diff.column_mut(0);
                hf.copy_from(&self.f);
                hf *= scale(state.h);
            }
            for i in start..self.tableau.s() {
                let t = state.t + self.tableau.c()[i] * state.h;
                self.nonlinear_solver.set_time(t).unwrap();
                {
                    let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                    callable.set_phi(&self.diff.columns(0, i), y0, &self.a_rows[i]);
                }

                let mut dy = if i == 0 {
                    self.diff.column(self.diff.ncols() - 1).into_owned()
                } else if i == 1 {
                    self.diff.column(i - 1).into_owned()
                } else {
                    let df = self.diff.column(i - 1) - self.diff.column(i - 2);
                    let c = (self.tableau.c()[i] - self.tableau.c()[i - 2])
                        / (self.tableau.c()[i - 1] - self.tableau.c()[i - 2]);
                    self.diff.column(i - 1) + df * scale(c)
                };
                let solve_result = self.nonlinear_solver.solve_in_place(&mut dy);

                // if we didn't update the jacobian and the solve failed, then we update the jacobian and try again
                let solve_result = if solve_result.is_err() && !updated_jacobian {
                    // newton iteration did not converge, so update jacobian and try again
                    {
                        let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                        callable.set_jacobian_is_stale();
                    }
                    self.nonlinear_solver.reset_jacobian();
                    updated_jacobian = true;

                    let mut dy = if i == 0 {
                        self.diff.column(self.diff.ncols() - 1).into_owned()
                    } else if i == 1 {
                        self.diff.column(i - 1).into_owned()
                    } else {
                        let df = self.diff.column(i - 1) - self.diff.column(i - 2);
                        let c = (self.tableau.c()[i] - self.tableau.c()[i - 2])
                            / (self.tableau.c()[i - 1] - self.tableau.c()[i - 2]);
                        self.diff.column(i - 1) + df * scale(c)
                    };
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    self.nonlinear_solver.solve_in_place(&mut dy)
                } else {
                    solve_result
                };

                if solve_result.is_err() {
                    // newton iteration did not converge, so we reduce step size and try again
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    state.h *= Eqn::T::from(0.3);

                    // if step size too small, then fail
                    if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                        return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
                    }

                    // update h for new step size
                    let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                    callable.set_h(state.h);

                    // reset nonlinear's linear solver problem as lu factorisation has changed
                    self.nonlinear_solver.reset_jacobian();
                    continue 'step;
                };

                // update diff with solved dy
                self.diff.column_mut(i).copy_from(&dy);
            }
            // successfully solved for all stages, now compute error
            self.diff
                .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);

            // solve for  (M - h * c * J) * error = error_est as by Hosea, M. E., & Shampine, L. F. (1996). Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20(1-2), 21-37.
            self.nonlinear_solver
                .linear_solver()
                .solve_in_place(&mut error)?;

            // do not include algebraic variables in error calculation
            //let algebraic = self.problem.as_ref().unwrap().eqn.algebraic_indices();
            //for i in 0..algebraic.len() {
            //    error[algebraic[i]] = Eqn::T::zero();
            //}

            // scale error and compute norm
            let scale_y = {
                let y1_ref = self
                    .nonlinear_solver
                    .problem()
                    .unwrap()
                    .f
                    .as_ref()
                    .get_last_f_eval();
                let ode_problem = self.problem.as_ref().unwrap();
                let mut scale_y = y1_ref.abs() * scale(ode_problem.rtol);
                scale_y += ode_problem.atol.as_ref();
                scale_y
            };
            error.component_div_assign(&scale_y);
            let error_norm = error.norm() / M::T::from((n as f64).sqrt());

            // adjust step size based on error
            let maxiter = self.nonlinear_solver.max_iter() as f64;
            let niter = self.nonlinear_solver.niter() as f64;
            let safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));
            let order = self.tableau.order() as f64;
            let mut factor = safety * error_norm.pow(Eqn::T::from(-1.0 / (order + 1.0)));
            if factor < Eqn::T::from(Self::MIN_FACTOR) {
                factor = Eqn::T::from(Self::MIN_FACTOR);
            }
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }

            // adjust step size for next step
            t1 = state.t + state.h;
            state.h *= factor;

            // if step size too small, then fail
            if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
            }

            // update c for new step size
            let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
            callable.set_h(state.h);

            // reset nonlinear's linear solver problem as lu factorisation has changed
            self.nonlinear_solver.reset_jacobian();

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break 'step;
            }
            // step is rejected, factor reduces step size, so we try again with the smaller step size
            self.statistics.number_of_error_test_failures += 1;
        }

        // take the step
        let dt = t1 - state.t;
        self.old_t = state.t;
        state.t = t1;

        self.old_f
            .copy_from_view(&self.diff.column(self.diff.ncols() - 1));
        self.old_f.mul_assign(scale(Eqn::T::one() / dt));
        std::mem::swap(&mut self.old_f, &mut self.f);

        let y1 = self
            .nonlinear_solver
            .problem()
            .unwrap()
            .f
            .as_ref()
            .get_last_f_eval();
        self.old_y.copy_from(&y1);
        std::mem::swap(&mut self.old_y, &mut state.y);

        // update statistics
        self.statistics.number_of_linear_solver_setups = self
            .nonlinear_solver
            .problem()
            .unwrap()
            .f
            .number_of_jac_evals();
        self.statistics.number_of_steps += 1;
        self.statistics.final_step_size = self.state.as_ref().unwrap().h;

        Ok(())
    }

    fn interpolate(&self, t: <Eqn>::T) -> anyhow::Result<<Eqn>::V> {
        let state = self.state.as_ref().expect("State not set");

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

    fn state(&self) -> Option<&OdeSolverState<<Eqn>::V>> {
        self.state.as_ref()
    }

    fn take_state(&mut self) -> Option<OdeSolverState<<Eqn>::V>> {
        Option::take(&mut self.state)
    }
}
