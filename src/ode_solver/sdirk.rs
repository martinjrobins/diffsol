use anyhow::Result;
use nalgebra::ComplexField;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::ops::MulAssign;
use std::rc::Rc;

use crate::linear_solver::LinearSolver;
use crate::matrix::MatrixRef;
use crate::op::linearise::LinearisedOp;
use crate::op::matrix::MatrixOp;
use crate::vector::VectorRef;
use crate::NewtonNonlinearSolver;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem,
    DenseMatrix, MatrixView, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
    Vector, VectorView, VectorViewMut,
};

use super::bdf::BdfStatistics;

pub struct Tableau<M: DenseMatrix> {
    a: M,
    b: M::V,
    c: M::V,
    d: M::V,
    order: usize,
}

impl<M: DenseMatrix> Tableau<M> {
    /// TR-BDF2 method
    /// from R.E. Bank, W.M. Coughran Jr, W. Fichtner, E.H. Grosse, D.J. Rose and R.K. Smith, Transient simulation of silicon devices and circuits, IEEE Trans. Comput.-Aided Design 4 (1985) 436-451.
    /// analysed in M.E. Hosea and L.F. Shampine. Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20:21–37, 1996.
    pub fn tr_bdf2(linear_solver: impl LinearSolver<MatrixOp<M>>) -> Result<Self> {
        let gamma = M::T::from(2.0 - 2.0_f64.sqrt());
        let d = gamma / M::T::from(2.0);
        let w = M::T::from(2.0_f64.sqrt() / 4.0);

        let mut a = M::zeros(3, 3);
        a[(1, 0)] = d;
        a[(1, 1)] = d;

        a[(2, 0)] = w;
        a[(2, 1)] = w;
        a[(2, 2)] = d;

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

        let order = 2;

        Ok(Self::new(a, b, c, d, order, beta))
    }
    /// L-stable SDIRK method of order 4, from
    /// Hairer, Norsett, Wanner, Solving Ordinary Differential Equations II, Stiff and Differential-Algebraic Problems, 2nd Edition
    /// Section IV.6, page 107
    pub fn sdirk4() -> Result<Self> {
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

        let order = 4;
        let beta = Self::compute_beta(&a, &b, &c, order, linear_solver)?;
        //beta[(0, 0)] = M::T::from(11.0 / 3.0);
        //beta[(0, 1)] = M::T::from(-463.0 / 72.0);
        //beta[(0, 2)] = M::T::from(217.0 / 36.0);
        //beta[(0, 3)] = M::T::from(-20.0 / 9.0);

        //beta[(1, 0)] = M::T::from(11.0 / 2.0);
        //beta[(1, 1)] = M::T::from(-385.0 / 16.0);
        //beta[(1, 2)] = M::T::from(661.0 / 24.0);
        //beta[(1, 3)] = M::T::from(-10.0);

        //beta[(2, 0)] = M::T::from(-128.0 / 18.0);
        //beta[(2, 1)] = M::T::from(20125.0 / 432.0);
        //beta[(2, 2)] = M::T::from(-8875.0 / 216.0);
        //beta[(2, 3)] = M::T::from(250.0 / 27.0);

        //beta[(3, 0)] = M::T::from(0.0);
        //beta[(3, 1)] = M::T::from(-85.0 / 4.0);
        //beta[(3, 2)] = M::T::from(85.0 / 6.0);
        //beta[(3, 3)] = M::T::from(0.0);

        //beta[(4, 0)] = M::T::from(-11.0 / 19.0);
        //beta[(4, 1)] = M::T::from(557.0 / 108.0);
        //beta[(4, 2)] = M::T::from(-359.0 / 54.0);
        //beta[(4, 3)] = M::T::from(80.0 / 27.0);

        Ok(Self::new(a, b, c, d, order, beta))
    }
    fn compute_beta(
        a: &M,
        b: &M::V,
        c: &M::V,
        order: usize,
        mut linear_solver: impl LinearSolver<MatrixOp<M>>,
    ) -> Result<M> {
        if order > 4 {
            panic!("Invalid order, expected order <= 4");
        }
        let s = c.len();
        let q = order;
        let e = M::V::from_element(s, M::T::one());
        let mat_c = M::from_diagonal(c);
        let o = 2_usize.pow(u32::try_from(order - 1).unwrap());
        println!("o: {}", o);
        println!("order: {}", order);
        println!("s: {}", s);
        println!("q: {}", q);

        let neqn = o * q + s;
        let nunknown = s * q;

        let add_extra_condition = if neqn == nunknown {
            false
        } else if neqn + s == nunknown {
            true
        } else {
            panic!("Error, expected neqn = nunknown or neqn + s = nunknown, got neqn = {}, nunknown = {}, s = {}", neqn, nunknown, s);
        };

        // construct psi
        println!("o: {}", o);
        println!("order: {}", order);
        let mut psi = M::zeros(o, s);
        for i in 0..s {
            psi[(0, i)] = e[i];
        }
        if order >= 2 {
            let mut ce = M::V::zeros(s);
            mat_c.gemv(M::T::one(), &e, M::T::zero(), &mut ce);
            for i in 0..s {
                psi[(1, i)] = ce[i];
            }
            if order >= 3 {
                let mut c2e = M::V::zeros(s);
                mat_c.gemv(M::T::one(), &ce, M::T::zero(), &mut c2e);
                let mut ace = M::V::zeros(s);
                a.gemv(M::T::one(), &ce, M::T::zero(), &mut ace);
                for i in 0..s {
                    psi[(2, i)] = c2e[i];
                    psi[(3, i)] = ace[i];
                }
                if order >= 4 {
                    let mut c3e = M::V::zeros(s);
                    mat_c.gemv(M::T::one(), &c2e, M::T::zero(), &mut c3e);
                    let mut cace = M::V::zeros(s);
                    mat_c.gemv(M::T::one(), &ace, M::T::zero(), &mut cace);
                    let mut ac2e = M::V::zeros(s);
                    a.gemv(M::T::one(), &c2e, M::T::zero(), &mut ac2e);
                    let mut a2ce = M::V::zeros(s);
                    a.gemv(M::T::one(), &ace, M::T::zero(), &mut a2ce);
                    for i in 0..s {
                        psi[(4, i)] = c3e[i];
                        psi[(5, i)] = cace[i];
                        psi[(6, i)] = ac2e[i];
                        psi[(7, i)] = a2ce[i];
                    }
                }
            }
        }

        println!("psi: {:?}", psi);

        let nrows = if add_extra_condition { q * o + s + s } else { q * o + s };
        let mut order_c = M::zeros(nrows, q * s);
        // orderC = | I_q kron psi | (q * o, q * s)
        //          | e kron I_s |   (s, q * s)
        //          | c^k kron I_s | (s, q * s) (if add_extra_condition)
        for i in 0..q {
            for j in 0..o {
                for k in 0..s {
                    order_c[(i * o + j, i * s + k)] = psi[(j, k)];
                }
            }
            for j in 0..s {
                order_c[(q * o + j, i * s + j)] = M::T::one();
            }
            if add_extra_condition {
                for j in 0..s {
                    order_c[(q * o + s + j, i * s + j)] = c[s - 2].pow(i as i32);
                }
            }
        }
        println!("orderC: {:?}", order_c);
        let mut gamma = M::zeros(o, q);
        gamma[(0, 0)] = M::T::one();

        if order >= 2 {
            gamma[(1, 1)] = M::T::from(1.0 / 2.0);
        }

        if order >= 3 {
            gamma[(2, 2)] = M::T::from(1.0 / 3.0);
            gamma[(3, 2)] = M::T::from(1.0 / 6.0);
        }

        if order >= 4 {
            gamma[(4, 3)] = M::T::from(1.0 / 4.0);
            gamma[(5, 3)] = M::T::from(1.0 / 8.0);
            gamma[(6, 3)] = M::T::from(1.0 / 12.0);
            gamma[(7, 3)] = M::T::from(1.0 / 24.0);
        }

        println!("gamma: {:?}", gamma);

        let mut vec_gamma = M::V::zeros(nrows);
        for j in 0..q {
            for i in 0..o {
                vec_gamma[j * o + i] = gamma[(i, j)];
            }
        }
        for i in 0..s {
            vec_gamma[q * o + i] = b[i];
        }
        if add_extra_condition {
            for i in 0..s {
                vec_gamma[q * o + s + i] = a[(s - 2, i)];
            }
        }

        // solve orderC * vec_b = vec_gamma
        let op = MatrixOp::new(order_c);
        let atol = M::V::from_element(q * o, M::T::from(1e-8));
        let rtol = M::T::from(1e-8);
        let problem = SolverProblem::new(Rc::new(op), M::T::zero(), Rc::new(atol), rtol);
        linear_solver.set_problem(problem);
        let vec_b = linear_solver.solve(&vec_gamma)?;

        // construct beta
        let mut beta = M::zeros(s, q);
        for i in 0..s {
            for j in 0..q {
                beta[(i, j)] = vec_b[j * s + i];
            }
        }

        // check that sum of 1st column of beta is 1
        let mut sum = M::T::zero();
        for i in 0..s {
            sum += beta[(i, 0)];
        }
        if (sum - M::T::one()).abs() > M::T::from(1e-8).abs() {
            panic!(
                "Invalid beta, expected sum of 1st column to be 1 but is {}",
                sum
            );
        }
        // check that sum of rows of beta equals b
        for i in 0..s {
            sum = M::T::zero();
            for j in 0..q {
                sum += beta[(i, j)];
            }
            if (sum - b[i]).abs() > M::T::from(1e-8).abs() {
                panic!(
                    "Invalid beta, expected sum of rows to equal b but is {}",
                    sum
                );
            }
        }
        Ok(beta)
    }
    pub fn new(a: M, b: M::V, c: M::V, d: M::V, order: usize, beta: M) -> Self {
        let s = c.len();
        assert_eq!(a.ncols(), s, "Invalid number of rows in a, expected {}", s);
        assert_eq!(
            a.nrows(),
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
        self.c.len()
    }

    pub fn a(&self) -> &M {
        &self.a
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
                } else if i == 1 {
                    self.diff.column(i - 1).into_owned()
                } else {
                    let df = self.diff.column(i - 1) - self.diff.column(i - 2);
                    let c = (self.tableau.c[i] - self.tableau.c[i - 2])
                        / (self.tableau.c[i - 1] - self.tableau.c[i - 2]);
                    self.diff.column(i - 1) + df * scale(c)
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

                    let mut dy = if i == 0 {
                        self.diff.column(self.diff.ncols() - 1).into_owned()
                    } else if i == 1 {
                        self.diff.column(i - 1).into_owned()
                    } else {
                        let df = self.diff.column(i - 1) - self.diff.column(i - 2);
                        let c = (self.tableau.c[i] - self.tableau.c[i - 2])
                            / (self.tableau.c[i - 1] - self.tableau.c[i - 2]);
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
            //let algebraic = self.problem.as_ref().unwrap().eqn.algebraic_indices();
            //for i in 0..algebraic.len() {
            //    error[algebraic[i]] = Eqn::T::zero();
            //}

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
            self.statistics.number_of_error_test_failures += 1;
        }

        // take the step
        let dt = t1 - state.t;
        self.old_t = state.t;
        state.t = t1;

        if self.tableau.c()[self.tableau.s() - 1] == Eqn::T::one() {
            self.old_f
                .copy_from_view(&self.diff.column(self.diff.ncols() - 1));
            self.old_f.mul_assign(scale(Eqn::T::one() / dt));
            std::mem::swap(&mut self.old_f, &mut self.f);
        } else {
            unimplemented!();
        }

        self.old_y.copy_from(&state.y);
        let y1 = &mut state.y;
        self.diff
            .gemv(Eqn::T::one(), self.tableau.b(), Eqn::T::one(), y1);

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
            Eqn::T::zero()
        } else {
            (t - self.old_t) / dt
        };

        let f0 = &self.old_f;
        let f1 = &self.f;
        let u0 = &self.old_y;
        let u1 = &state.y;
        let ret_h = u0 * (Eqn::T::from(1.0) - theta)
            + u1 * theta
            + ((u1 - u0) * scale(Eqn::T::from(1.0) - Eqn::T::from(2.0) * theta)
                + f0 * ((theta - Eqn::T::from(1.0)) * dt)
                + f1 * (theta * dt))
                * scale(theta * (theta - Eqn::T::from(1.0)));

        let poly_order = self.tableau.beta().ncols();
        let s_star = self.tableau.beta().nrows();
        let mut thetav = Vec::with_capacity(poly_order);
        thetav.push(theta);
        for i in 1..poly_order {
            thetav.push(theta * thetav[i - 1]);
        }
        // beta_poly = beta * thetav
        let thetav = Eqn::V::from_vec(thetav);
        let mut beta = <Eqn::V as Vector>::zeros(s_star);
        self.tableau
            .beta()
            .gemv(Eqn::T::one(), &thetav, Eqn::T::zero(), &mut beta);

        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        let mut ret = self.old_y.clone();
        self.diff.gemv(dt, &beta, Eqn::T::one(), &mut ret);
        println!("ret: {:?}", ret);
        println!("ret_h: {:?}", ret_h);
        Ok(ret)
    }

    fn state(&self) -> Option<&OdeSolverState<<Eqn>::M>> {
        self.state.as_ref()
    }

    fn take_state(&mut self) -> Option<OdeSolverState<<Eqn>::M>> {
        self.state.take()
    }
}
