use num_traits::Zero;
use std::cmp::{max, min};
use std::{f32::consts::E, rc::Rc};

use crate::matrix::DenseMatrix;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    Matrix, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Scalar, Vector,
};

struct Tableau<M: DenseMatrix> {
    a: M,
    b: M::V,
    c: M::V,
    d: M::V,
    beta: M,
    order: usize,
}

impl<M: DenseMatrix> Tableau<M> {
    pub fn new(a: M, b: M::V, c: M::V, d: M::V, beta: M, order: usize) -> Self {
        let s = b.len();
        assert_eq!(a.rows(), s, "Invalid number of rows in a, expected {}", s);
        assert_eq!(
            a.cols(),
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
            beta.rows(),
            s,
            "Invalid number of rows in beta, expected {}",
            s
        );
        assert_eq!(
            beta.cols(),
            s,
            "Invalid number of columns in beta, expected {}",
            s
        );
        Self {
            a,
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

    pub fn a(&self, i: usize, j: usize) -> T {
        self.a[(i, j)]
    }

    pub fn b(&self, i: usize) -> T {
        self.b[i]
    }

    pub fn c(&self, i: usize) -> T {
        self.c[i]
    }

    pub fn d(&self, i: usize) -> T {
        self.d[i]
    }
}

struct Sdirk<Eqn, NS>
where
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
{
    tableau: Tableau<Eqn::T>,
    problem: Option<OdeSolverProblem<Eqn>>,
    nonlinear_solver: NS,
    state: Option<OdeSolverState<Eqn::M>>,
    diff: Eqn::M,
    gamma: Eqn::T,
    is_sdirk: bool,
    old_t: Eqn::T,
    old_y: Eqn::V,
}

impl<Eqn, NS> Sdirk<Eqn, NS>
where
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
{
    const NEWTON_MAXITER: usize = 4;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn new(tableau: Tableau<Eqn::T>, mut nonlinear_solver: NS) -> Self {
        // set max iterations for nonlinear solver
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);

        // check that the upper triangular part of a is zero
        for i in 0..tableau.s() {
            for j in 0..i {
                assert_eq!(
                    tableau.a(i, j),
                    Eqn::T::zero(),
                    "Invalid tableau, expected a(i, j) = 0 for i > j"
                );
            }
        }
        let gamma = tableau.a(0, 0);
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            assert_eq!(
                tableau.a(i, i),
                gamma,
                "Invalid tableau, expected a(i, i) = gamma = {} for i = 1..s-1",
                gamma
            );
        }
        // if a(0, 0) = gamma, then we're a SDIRK method
        // if a(0, 0) = 0, then we're a ESDIRK method
        // otherwise, error
        let is_sdirk = match tableau.a(0, 0) {
            gamma => true,
            Eqn::T::zero() => false,
            _ => panic!("Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"),
        };
        assert!(
            gamma == Eqn::T::zero() || gamma == tableau.a(0, 0),
            "Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"
        );
        let n = 1;
        let s = tableau.s();
        let diff = Eqn::M::zeros(n, s);
        Self {
            tableau,
            nonlinear_solver,
            state: None,
            diff,
            problem: None,
            gamma,
            is_sdirk,
        }
    }

    fn nonlinear_problem_op(&self) -> Option<&Rc<SdirkCallable<Eqn>>> {
        Some(&self.nonlinear_solver.problem()?.f)
    }
}

impl<Eqn, NS> OdeSolverMethod<Eqn> for Sdirk<Eqn, NS>
where
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
{
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.problem.as_ref()
    }

    fn set_problem(&mut self, state: OdeSolverState<<Eqn>::M>, problem: &OdeSolverProblem<Eqn>) {
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

        let y1 = &state.y + scale(h0) * f0;
        let t1 = state.t + h0;
        let f1 = problem.eqn.rhs(t1, &y1);

        let mut df = f1 - f0;
        df *= scale(Eqn::T::one() / h0);
        df.component_div_assign(&scale_factor);
        let d2 = df.norm();

        let max_d = max(d1, d2);
        let h1 = if max_d < Eqn::T::from(1e-15) {
            max(Eqn::T::from(1e-6), h0 * Eqn::T::from(1e-3))
        } else {
            (Eqn::T::from(0.01) / max_d)
                .pow(Eqn::T::one() / Eqn::T::from(1.0 + self.tableau.order() as f64))
        };

        state.h = min(100 * h0, h1);

        // setup linear solver for first step
        let callable = Rc::new(SdirkCallable::new(problem));
        callable.set_c(state.h, self.gamma);
        let nonlinear_problem = SolverProblem::new_from_ode_problem(callable, problem);
        self.nonlinear_solver
            .as_mut()
            .set_problem(nonlinear_problem);

        self.state = Some(state);
        self.problem = Some(problem.clone());
        self.old_t = state.t;
        self.old_y = state.y.clone();
    }

    fn step(&mut self) -> anyhow::Result<()> {
        // optionally do the first step
        let state = self.state.as_mut().unwrap();
        let n = state.y.len();
        let y0 = &state.y;
        let start = if self.is_sdirk { 0 } else { 1 };
        let mut updated_jacobian = false;
        let mut phi = Eqn::V::zeros(n);
        let scale_y = {
            let ode_problem = self.ode_problem.as_ref().unwrap();
            let mut scale_y = y0.clone();
            scale_y = y0.abs() * scale(ode_problem.rtol);
            scale_y += ode_problem.atol.as_ref();
        };
        let mut t = state.t;

        // loop until step is accepted
        loop {
            t = state.t;
            for i in start..self.tableau.s() {
                t += self.tableau.c(i) * state.h;
                phi.copy_from(y0);
                for j in 0..i {
                    phi += self.tableau.a(i, j) * self.diff.column(j);
                }
                phi *= state.h;
                self.nonlinear_solver.as_mut().set_time(t).unwrap();
                {
                    let callable = self.nonlinear_problem_op().unwrap();
                    callable.set_phi(phi);
                }

                let dy = if i == 0 {
                    self.diff.column(0).to_owned()
                } else {
                    self.diff.column(i - 1).to_owned()
                };
                match self.nonlinear_solver.solve_in_place(dy) {
                    Ok(result) => Ok(()),
                    Err(e) => {
                        if !updated_jacobian {
                            // newton iteration did not converge, so update jacobian and try again
                            {
                                let callable = self.nonlinear_problem_op().unwrap();
                                callable.set_rhs_jacobian_is_stale();
                            }
                            self.nonlinear_solver.as_mut().reset();
                            updated_jacobian = true;
                            self.nonlinear_solver.solve_in_place(dy)
                        } else {
                            Err(e)
                        }
                    }
                }?;
            }
            let mut error = Eqn::V::zeros(n);
            for i in 0..self.tableau.s() {
                error += self.tableau.d(i) * self.diff.column(i);
            }
            error *= state.h;

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
            let callable = self.nonlinear_problem_op().unwrap();
            callable.set_c(state.h, self.gamma);

            // reset nonlinear's linear solver problem as lu factorisation has changed
            self.nonlinear_solver.as_mut().reset();

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break;
            }
            // step is rejected
            // reduce step size and try again
        }

        // take the step
        self.old_t = state.t;
        state.t = t;
        self.old_y.copy_from(state.y);
        let y1 = &mut state.y;
        for i in 0..self.tableau.s() {
            y1 += self.tableau.b(i) * self.diff.column(i);
        }
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
        let poly_order = self.tableau.interpolation_order();
        let s_star = self.tableau.interpolation_stages();
        let mut thetav = vec![Eqn::T::from(1.0); poly_order];
        for i in 1..poly_order {
            thetav[i] = theta * thetav[i - 1];
        }
        // beta_poly = beta * thetav
        let mut beta = Eqn::V::from_vec(thetav);
        self.tableau
            .beta()
            .gemv(Eqn::T::one(), &beta, Eqn::T::zero(), &mut beta);
        beta *= state.h;

        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        let mut ret = self.old_y.clone();
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
