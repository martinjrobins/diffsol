use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::rc::Rc;

use crate::linear_solver;
use crate::linear_solver::LinearSolver;
use crate::matrix::MatrixRef;
use crate::nonlinear_solver;
use crate::op::linearise::LinearisedOp;
use crate::vector::VectorRef;
use crate::NewtonNonlinearSolver;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    DenseMatrix, MatrixView, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
    Vector, VectorIndex, VectorView, VectorViewMut,
};

pub struct Tableau<M: DenseMatrix> {
    at: M, // A transpose
    b: M::V,
    c: M::V,
    d: M::V,
    beta: M,
    order: usize,
}

impl<M: DenseMatrix> Tableau<M> {
    /// TR-BDF2 method
    /// from R.E. Bank, W.M. Coughran Jr, W. Fichtner, E.H. Grosse, D.J. Rose and R.K. Smith, Transient simulation of silicon devices and circuits, IEEE Trans. Comput.-Aided Design 4 (1985) 436-451.
    /// analysed in M.E. Hosea and L.F. Shampine. Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20:21–37, 1996.
    pub fn tr_bdf2() -> Self {
        let gamma = M::T::from(2.0 - 2.0_f64.sqrt());
        let d = gamma / M::T::from(2.0);
        let w = M::T::from(2.0_f64.sqrt() / 4.0);

        let mut a = M::zeros(3, 3);
        a[(1, 0)] = d;
        a[(1, 1)] = d;

        a[(2, 0)] = w;
        a[(2, 1)] = w;
        a[(2, 2)] = d;

        let mut at = M::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                at[(i, j)] = a[(j, i)];
            }
        }

        let b = M::V::from_vec(vec![w, w, d]);
        let b_hat = M::V::from_vec(vec![
            (M::T::from(1.0) - w) / M::T::from(3.0),
            (M::T::from(3.0) * w + M::T::from(1.0)) / M::T::from(3.0),
            d / M::T::from(3.0),
        ]);
        let mut d = M::V::zeros(3);
        for i in 0..3 {
            d[i] = b[i] - b_hat[i];
        }

        let c = M::V::from_vec(vec![M::T::zero(), gamma, M::T::one()]);

        let beta = M::zeros(1, 1);

        let order = 2;

        Self {
            at,
            b,
            c,
            d,
            beta,
            order,
        }
    }
    /// L-stable SDIRK method of order 4, from
    /// Hairer, Norsett, Wanner, Solving Ordinary Differential Equations II, Stiff and Differential-Algebraic Problems, 2nd Edition
    /// Section IV.6, page 107
    pub fn sdirk4() -> Self {
        let mut a = M::zeros(5, 5);
        a[(0, 0)] = M::T::from(1.0 / 4.0);

        a[(1, 0)] = M::T::from(1.0 / 2.0);
        a[(1, 1)] = M::T::from(1.0 / 4.0);

        a[(2, 0)] = M::T::from(17.0 / 50.0);
        a[(2, 1)] = M::T::from(-1.0 / 25.0);
        a[(2, 2)] = M::T::from(1.0 / 4.0);

        a[(3, 0)] = M::T::from(371.0 / 1360.0);
        a[(3, 1)] = M::T::from(-137.0 / 2720.0);
        a[(3, 2)] = M::T::from(15.0 / 544.0);
        a[(3, 3)] = M::T::from(1.0 / 4.0);

        a[(4, 0)] = M::T::from(25.0 / 24.0);
        a[(4, 1)] = M::T::from(-49.0 / 48.0);
        a[(4, 2)] = M::T::from(125.0 / 16.0);
        a[(4, 3)] = M::T::from(-85.0 / 12.0);
        a[(4, 4)] = M::T::from(1.0 / 4.0);

        let mut at = M::zeros(5, 5);
        for i in 0..5 {
            for j in 0..5 {
                at[(i, j)] = a[(j, i)];
            }
        }

        let b = M::V::from_vec(vec![
            M::T::from(25.0 / 24.0),
            M::T::from(-49.0 / 48.0),
            M::T::from(125.0 / 16.0),
            M::T::from(-85.0 / 12.0),
            M::T::from(1.0 / 4.0),
        ]);

        let c = M::V::from_vec(vec![
            M::T::from(1.0 / 4.0),
            M::T::from(3.0 / 4.0),
            M::T::from(11.0 / 20.0),
            M::T::from(1.0 / 2.0),
            M::T::from(1.0),
        ]);

        let d = M::V::from_vec(vec![
            M::T::from(-3.0 / 16.0),
            M::T::from(-27.0 / 32.0),
            M::T::from(25.0 / 32.0),
            M::T::from(0.0),
            M::T::from(1.0 / 4.0),
        ]);

        let mut beta = M::zeros(5, 4);
        beta[(0, 0)] = M::T::from(11.0 / 3.0);
        beta[(0, 1)] = M::T::from(-463.0 / 72.0);
        beta[(0, 2)] = M::T::from(217.0 / 36.0);
        beta[(0, 3)] = M::T::from(-20.0 / 9.0);

        beta[(1, 0)] = M::T::from(11.0 / 2.0);
        beta[(1, 1)] = M::T::from(-385.0 / 16.0);
        beta[(1, 2)] = M::T::from(661.0 / 24.0);
        beta[(1, 3)] = M::T::from(-10.0);

        beta[(2, 0)] = M::T::from(-128.0 / 18.0);
        beta[(2, 1)] = M::T::from(20125.0 / 432.0);
        beta[(2, 2)] = M::T::from(-8875.0 / 216.0);
        beta[(2, 3)] = M::T::from(250.0 / 27.0);

        beta[(3, 0)] = M::T::from(0.0);
        beta[(3, 1)] = M::T::from(-85.0 / 4.0);
        beta[(3, 2)] = M::T::from(85.0 / 6.0);
        beta[(3, 3)] = M::T::from(0.0);

        beta[(4, 0)] = M::T::from(-11.0 / 19.0);
        beta[(4, 1)] = M::T::from(557.0 / 108.0);
        beta[(4, 2)] = M::T::from(-359.0 / 54.0);
        beta[(4, 3)] = M::T::from(80.0 / 27.0);

        let order = 4;
        Self::new(at, b, c, d, beta, order)
    }
    pub fn new(at: M, b: M::V, c: M::V, d: M::V, beta: M, order: usize) -> Self {
        let s = c.len();
        assert_eq!(at.nrows(), s, "Invalid number of rows in a, expected {}", s);
        assert_eq!(
            at.ncols(),
            s,
            "Invalid number of columns in a, expected {}",
            s
        );
        assert_eq!(
            b.len(),
            s,
            "Invalid number of elements in b, expected {}",
            s
        );
        assert_eq!(
            c.len(),
            s,
            "Invalid number of elements in c, expected {}",
            s
        );
        assert_eq!(
            beta.nrows(),
            s,
            "Invalid number of rows in beta, expected {}",
            s
        );
        Self {
            at,
            b,
            c,
            d,
            beta,
            order,
        }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn s(&self) -> usize {
        self.c.len()
    }

    pub fn at(&self) -> &M {
        &self.at
    }

    pub fn b(&self) -> &M::V {
        &self.b
    }

    pub fn c(&self) -> &M::V {
        &self.c
    }

    pub fn d(&self) -> &M::V {
        &self.d
    }

    pub fn beta(&self) -> &M {
        &self.beta
    }
}

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
    state: Option<OdeSolverState<Eqn::M>>,
    diff: M,
    gamma: Eqn::T,
    is_sdirk: bool,
    old_t: Eqn::T,
    old_y: Eqn::V,
    old_f: Eqn::V,
    f: Eqn::V,
    a_rows: Vec<Eqn::V>,
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
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn new(
        tableau: Tableau<M>,
        linear_solver: impl LinearSolver<LinearisedOp<SdirkCallable<Eqn>>> + 'static,
    ) -> Self {
        let mut nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);
        // set max iterations for nonlinear solver
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);

        // check that the upper triangular part of a is zero
        let s = tableau.s();
        for i in 0..s {
            for j in (i + 1)..s {
                assert_eq!(
                    tableau.at()[(j, i)],
                    Eqn::T::zero(),
                    "Invalid tableau, expected a(i, j) = 0 for i > j"
                );
            }
        }
        let gamma = tableau.at()[(1, 1)];
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            assert_eq!(
                tableau.at()[(i, i)],
                gamma,
                "Invalid tableau, expected a(i, i) = gamma = {} for i = 1..s-1",
                gamma
            );
        }
        // if a(0, 0) = gamma, then we're a SDIRK method
        // if a(0, 0) = 0, then we're a ESDIRK method
        // otherwise, error
        let zero = Eqn::T::zero();
        if tableau.at()[(0, 0)] != zero && tableau.at()[(0, 0)] != gamma {
            panic!("Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma");
        }
        let is_sdirk = tableau.at()[(0, 0)] == gamma;

        let mut a_rows = Vec::with_capacity(s);
        for i in 0..s {
            let mut row = Vec::with_capacity(i);
            for j in 0..i {
                row.push(tableau.at()[(j, i)]);
            }
            a_rows.push(Eqn::V::from_vec(row));
        }

        // check last row of a is the same as b
        for i in 0..s {
            assert_eq!(
                tableau.at()[(i, s - 1)],
                tableau.b()[i],
                "Invalid tableau, expected a(s-1, i) = b(i)"
            );
        }

        let n = 1;
        let s = tableau.s();
        let diff = M::zeros(n, s);
        let old_t = Eqn::T::zero();
        let old_y = <Eqn::V as Vector>::zeros(n);
        let old_f = <Eqn::V as Vector>::zeros(n);
        let f = <Eqn::V as Vector>::zeros(n);
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
        }
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
        mut state: OdeSolverState<<Eqn>::M>,
        problem: &OdeSolverProblem<Eqn>,
    ) {
        // update initial step size based on function
        let mut scale_factor = state.y.abs();
        scale_factor *= scale(problem.rtol);
        scale_factor += problem.atol.as_ref();

        // compute first step based on alg in Hairer, Norsett, Wanner
        // Solving Ordinary Differential Equations I, Nonstiff Problems
        // Section II.4.2
        let f0 = problem.eqn.rhs(state.t, &state.y);
        let hf0 = &f0 * state.h;

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
        let f1 = problem.eqn.rhs(t1, &y1);

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
        self.nonlinear_solver.set_problem(nonlinear_problem);

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
        let scale_y = {
            let ode_problem = self.problem.as_ref().unwrap();
            let mut scale_y = y0.abs() * scale(ode_problem.rtol);
            scale_y += ode_problem.atol.as_ref();
            scale_y
        };
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
                let mut phi = y0.clone();
                if i > 0 {
                    let a_row = &self.a_rows[i];
                    self.diff
                        .columns(0, i)
                        .gemv_o(Eqn::T::one(), a_row, Eqn::T::one(), &mut phi);
                }

                self.nonlinear_solver.set_time(t).unwrap();
                {
                    let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                    callable.set_phi(phi);
                }

                let mut dy = if i == 0 {
                    self.diff.column(self.diff.ncols() - 1).into_owned()
                } else {
                    self.diff.column(i - 1).into_owned()
                };
                let solve_result = self.nonlinear_solver.solve_in_place(&mut dy);

                // if we didn't update the jacobian and the solve failed, then we update the jacobian and try again
                let solve_result = if solve_result.is_err() && !updated_jacobian {
                    // newton iteration did not converge, so update jacobian and try again
                    {
                        let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                        callable.set_rhs_jacobian_is_stale();
                    }
                    self.nonlinear_solver.reset();
                    updated_jacobian = true;

                    if i == 0 {
                        dy.copy_from_view(&self.diff.column(self.diff.ncols() - 1));
                    } else {
                        dy.copy_from_view(&self.diff.column(i - 1));
                    };
                    self.nonlinear_solver.solve_in_place(&mut dy)
                } else {
                    solve_result
                };

                if solve_result.is_err() {
                    // newton iteration did not converge, so we reduce step size and try again
                    state.h *= Eqn::T::from(0.3);

                    // if step size too small, then fail
                    if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                        return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
                    }

                    // update h for new step size
                    let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                    callable.set_h(state.h);

                    // reset nonlinear's linear solver problem as lu factorisation has changed
                    self.nonlinear_solver.reset();
                    continue 'step;
                };

                // update diff with solved dy
                self.diff.column_mut(i).copy_from(&dy);
            }

            // successfully solved for all stages, now compute error
            let mut error = <Eqn::V as Vector>::zeros(n);
            self.diff
                .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);

            // solve for  (M - h * c * J) * error = error_est as by Hosea, M. E., & Shampine, L. F. (1996). Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20(1-2), 21-37.
            self.nonlinear_solver
                .linear_solver()
                .solve_in_place(&mut error)?;

            // do not include algebraic variables in error calculation
            let algebraic = self.problem.as_ref().unwrap().eqn.algebraic_indices();
            for i in 0..algebraic.len() {
                error[algebraic[i]] = Eqn::T::zero();
            }

            // scale error and compute norm
            error.component_div_assign(&scale_y);
            let error_norm = error.norm();

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
            self.nonlinear_solver.reset();

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break 'step;
            }
            // step is rejected, factor reduces step size, so we try again with the smaller step size
        }

        // take the step
        let dt = t1 - state.t;
        self.old_t = state.t;
        state.t = t1;

        if self.tableau.c()[self.tableau.s() - 1] == Eqn::T::one() {
            self.old_f
                .copy_from_view(&self.diff.column(self.diff.ncols() - 1));
            self.old_f *= scale(Eqn::T::one() / dt);
            std::mem::swap(&mut self.old_f, &mut self.f);
        } else {
            unimplemented!();
        }

        self.old_y.copy_from(&state.y);
        let y1 = &mut state.y;
        self.diff
            .gemv(Eqn::T::one(), self.tableau.b(), Eqn::T::one(), y1);
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
            Eqn::T::zero()
        } else {
            (t - self.old_t) / dt
        };

        let f0 = &self.old_f;
        let f1 = &self.f;
        let u0 = &self.old_y;
        let u1 = &state.y;
        let ret = u0 * (Eqn::T::from(1.0) - theta)
            + u1 * theta
            + ((u1 - u0) * scale(Eqn::T::from(1.0) - Eqn::T::from(2.0) * theta)
                + f0 * ((theta - Eqn::T::from(1.0)) * dt)
                + f1 * (theta * dt))
                * scale(theta * (theta - Eqn::T::from(1.0)));

        //let poly_order = self.tableau.beta().ncols();
        //let s_star = self.tableau.beta().nrows();
        //let mut thetav = Vec::with_capacity(poly_order);
        //thetav.push(theta);
        //for i in 1..poly_order {
        //    thetav.push(theta * thetav[i - 1]);
        //}
        //// beta_poly = beta * thetav
        //let thetav = Eqn::V::from_vec(thetav);
        //let mut beta = <Eqn::V as Vector>::zeros(s_star);
        //self.tableau
        //    .beta()
        //    .gemv(Eqn::T::one(), &thetav, Eqn::T::zero(), &mut beta);

        //// ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        //let mut ret = self.old_y.clone();
        //self.diff.gemv(dt, &beta, Eqn::T::one(), &mut ret);
        Ok(ret)
    }

    fn state(&self) -> Option<&OdeSolverState<<Eqn>::M>> {
        self.state.as_ref()
    }

    fn take_state(&mut self) -> Option<OdeSolverState<<Eqn>::M>> {
        self.state.take()
    }
}
