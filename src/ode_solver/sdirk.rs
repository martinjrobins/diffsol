use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::rc::Rc;

use crate::matrix::MatrixRef;
use crate::vector::VectorRef;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    DenseMatrix, MatrixView, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
    Vector, VectorView, VectorViewMut,
};

struct Tableau<M: DenseMatrix> {
    at: M, // A transpose
    b: M::V,
    c: M::V,
    d: M::V,
    beta: M,
    order: usize,
}

impl<M: DenseMatrix> Tableau<M> {
    pub fn new(at: M, b: M::V, c: M::V, d: M::V, beta: M, order: usize) -> Self {
        let s = b.len();
        assert_eq!(at.nrows(), s, "Invalid number of rows in a, expected {}", s);
        assert_eq!(
            at.ncols(),
            s,
            "Invalid number of columns in a, expected {}",
            s
        );
        assert_eq!(
            c.len(),
            s,
            "Invalid number of elements in c, expected {}",
            s
        );
        assert_eq!(
            d.len(),
            s,
            "Invalid number of elements in d, expected {}",
            s
        );
        assert_eq!(
            beta.nrows(),
            s,
            "Invalid number of rows in beta, expected {}",
            s
        );
        assert_eq!(
            beta.ncols(),
            s,
            "Invalid number of columns in beta, expected {}",
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
        self.b.len()
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

struct Sdirk<M, Eqn, NS>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
{
    tableau: Tableau<M>,
    problem: Option<OdeSolverProblem<Eqn>>,
    nonlinear_solver: NS,
    state: Option<OdeSolverState<Eqn::M>>,
    diff: M,
    gamma: Eqn::T,
    is_sdirk: bool,
    old_t: Eqn::T,
    old_y: Eqn::V,
}

impl<M, Eqn, NS> Sdirk<M, Eqn, NS>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
{
    const NEWTON_MAXITER: usize = 4;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn new(tableau: Tableau<M>, mut nonlinear_solver: NS) -> Self {
        // set max iterations for nonlinear solver
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);

        // check that the upper triangular part of a is zero
        for i in 0..tableau.s() {
            for j in 0..i {
                assert_eq!(
                    tableau.at()[(j, i)],
                    Eqn::T::zero(),
                    "Invalid tableau, expected a(i, j) = 0 for i > j"
                );
            }
        }
        let gamma = tableau.at()[(0, 0)];
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
        let is_sdirk = match tableau.at()[(0, 0)] {
            gamma => true,
            zero => false,
            _ => panic!("Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"),
        };
        assert!(
            gamma == Eqn::T::zero() || gamma == tableau.at()[(0, 0)],
            "Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"
        );
        let n = 1;
        let s = tableau.s();
        let diff = M::zeros(n, s);
        let old_t = Eqn::T::zero();
        let old_y = <Eqn::V as Vector>::zeros(n);
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
        }
    }

    fn nonlinear_problem_op(&self) -> Option<&Rc<SdirkCallable<Eqn>>> {
        Some(&self.nonlinear_solver.problem()?.f)
    }
}

impl<M, Eqn, NS> OdeSolverMethod<Eqn> for Sdirk<M, Eqn, NS>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
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

        let y1 = &state.y + &f0 * h0;
        let t1 = state.t + h0;
        let f1 = problem.eqn.rhs(t1, &y1);

        let mut df = f1 - f0;
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
        let callable = Rc::new(SdirkCallable::new(problem));
        callable.set_c(state.h, self.gamma);
        let nonlinear_problem = SolverProblem::new_from_ode_problem(callable, problem);
        self.nonlinear_solver.set_problem(nonlinear_problem);

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
        let mut t = state.t;

        // loop until step is accepted
        loop {
            t = state.t;
            for i in start..self.tableau.s() {
                t += self.tableau.c()[i] * state.h;
                let mut phi = y0.clone();
                let at_col = self.tableau.at().column(i);
                self.diff
                    .columns(0, i)
                    .gemv_v(state.h, &at_col, Eqn::T::one(), &mut phi);

                self.nonlinear_solver.set_time(t).unwrap();
                {
                    let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                    callable.set_phi(phi);
                }

                let mut dy = if i == 0 {
                    self.diff.column(0).into_owned()
                } else {
                    self.diff.column(i - 1).into_owned()
                };
                match self.nonlinear_solver.solve_in_place(&mut dy) {
                    Ok(r) => Ok(r),
                    Err(e) => {
                        if !updated_jacobian {
                            // newton iteration did not converge, so update jacobian and try again
                            {
                                let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
                                callable.set_rhs_jacobian_is_stale();
                            }
                            self.nonlinear_solver.reset();
                            updated_jacobian = true;

                            if i == 0 {
                                dy.copy_from_view(&self.diff.column(0));
                            } else {
                                dy.copy_from_view(&self.diff.column(i - 1));
                            };
                            self.nonlinear_solver.solve_in_place(&mut dy)
                        } else {
                            Err(e)
                        }
                    }
                }?;

                // update diff with solved dy
                self.diff.column_mut(i).copy_from(&dy);
            }
            let mut error = <Eqn::V as Vector>::zeros(n);
            self.diff
                .gemv(state.h, self.tableau.d(), Eqn::T::zero(), &mut error);

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
            state.h *= factor;

            // if step size too small, then fail
            if state.h < Eqn::T::from(Self::MIN_TIMESTEP) {
                return Err(anyhow::anyhow!("Step size too small at t = {}", state.t));
            }

            // update c for new step size
            let callable = self.nonlinear_solver.problem().unwrap().f.as_ref();
            callable.set_c(state.h, self.gamma);

            // reset nonlinear's linear solver problem as lu factorisation has changed
            self.nonlinear_solver.reset();

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break;
            }
            // step is rejected, factor reduces step size, so we try again with the smaller step size
        }

        // take the step
        self.old_t = state.t;
        state.t = t;
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
        let theta = (t - self.old_t) / (state.t - self.old_t);
        let poly_order = self.tableau.beta().ncols();
        let s_star = self.tableau.beta().nrows();
        let mut thetav = vec![Eqn::T::from(1.0); poly_order];
        for i in 1..poly_order {
            thetav[i] = theta * thetav[i - 1];
        }
        // beta_poly = beta * thetav
        let thetav = Eqn::V::from_vec(thetav);
        let mut beta = <Eqn::V as Vector>::zeros(s_star);
        self.tableau
            .beta()
            .gemv(state.h, &thetav, Eqn::T::zero(), &mut beta);

        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        let mut ret = thetav;
        ret.copy_from(&self.old_y);
        self.diff
            .gemv(Eqn::T::one(), &beta, Eqn::T::one(), &mut ret);
        Ok(ret)
    }

    fn state(&self) -> Option<&OdeSolverState<<Eqn>::M>> {
        self.state.as_ref()
    }

    fn take_state(&mut self) -> Option<OdeSolverState<<Eqn>::M>> {
        self.state.take()
    }
}
