use std::rc::Rc;
use num_traits::Zero;

use crate::{nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, solver::SolverProblem, Matrix, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Scalar, Vector};



struct Tableau<T: Scalar> {
    a: Vec<T>,
    b: Vec<T>,
    c: Vec<T>,
    c: Vec<T>,
    d: Vec<T>,
}

impl<T: Scalar> Tableau<T> {
    pub fn new(a: Vec<T>, b: Vec<T>, c: Vec<T>, d: Vec<T>) -> Self {
        let s = b.len();
        assert_eq!(a.len(), s * s, "Invalid number of elements in a, expected {}", s * s);
        assert_eq!(c.len(), s, "Invalid number of elements in c, expected {}", s);
        assert_eq!(d.len(), s, "Invalid number of elements in d, expected {}", s);
        Self { a, b, c, d }
    }

    pub fn s(&self) -> usize {
        self.b.len()
    }

    pub fn a(&self, i: usize, j: usize) -> T {
        self.a[i * self.b.len() + j]
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
}

impl<Eqn, NS> Sdirk<Eqn, NS> 
where
    Eqn: OdeEquations,
    NS: NonLinearSolver<SdirkCallable<Eqn>>,
{
    pub fn new(tableau: Tableau<Eqn::T>, nonlinear_solver: NS) -> Self {
        // check that the upper triangular part of a is zero
        for i in 0..tableau.s() {
            for j in 0..i {
                assert_eq!(tableau.a(i, j), Eqn::T::zero(), "Invalid tableau, expected a(i, j) = 0 for i > j");
            }
        }
        let gamma = tableau.a(0, 0);
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            assert_eq!(tableau.a(i, i), gamma, "Invalid tableau, expected a(i, i) = gamma = {} for i = 1..s-1", gamma);
        }
        // if a(0, 0) = gamma, then we're a SDIRK method
        // if a(0, 0) = 0, then we're a ESDIRK method
        // otherwise, error
        let is_sdirk = match tableau.a(0, 0) {
            gamma => true,
            Eqn::T::zero() => false,
            _ => panic!("Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma"),
        };
        assert!(gamma == Eqn::T::zero() || gamma == tableau.a(0, 0), "Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma");
        let n = 1;
        let s = tableau.s();
        let diff = Eqn::M::zeros(n, s);
        Self { tableau, nonlinear_solver, state: None, diff, problem: None, gamma, is_sdirk }
    }

    fn nonlinear_problem_op(&self) -> Option<&Rc<SdirkCallable<Eqn>>> {
        Some(&self.nonlinear_solver.problem()?.f)
    }
}

impl<Eqn, NS> OdeSolverMethod<Eqn> for Sdirk<Eqn, NS> 
where
    Eqn: OdeEquations,
    NS: NonLinearSolver<BdfCallable<Eqn>>,
{
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.problem.as_ref()
    }
    
    fn set_problem(&mut self, state: OdeSolverState<<Eqn>::M>, problem: &OdeSolverProblem<Eqn>) {
        // update initial step size based on function
        let mut scale_factor = state.y.abs();
        scale_factor *= scale(problem.rtol);
        scale_factor += problem.atol.as_ref();

        let f0 = problem.eqn.rhs(state.t, &state.y);
        let hf0 = &f0 * state.h;
        let y1 = &state.y + &hf0;
        let t1 = state.t + state.h;
        let f1 = problem.eqn.rhs(t1, &y1);

        let mut df = f1 - f0;
        df.component_div_assign(&scale_factor);
        let d2 = df.norm();

        let mut new_h = state.h * d2.pow(-0.5);
        if new_h > Eqn::T::from(100.0) * state.h {
            new_h = Eqn::T::from(100.0) * state.h;
        }
        state.h = new_h;

        // setup linear solver for first step
        let callable = Rc::new(SdirkCallable::new(problem));
        callable.set_c(state.h, self.gamma);
        let nonlinear_problem = SolverProblem::new_from_ode_problem(callable, problem);
        self.nonlinear_solver
            .as_mut()
            .set_problem(nonlinear_problem);

        self.state = Some(state);
        self.problem = Some(problem.clone());
    }
    
    fn step(&mut self) -> anyhow::Result<()> {
        // optionally do the first step
        let state = self.state.as_mut().unwrap();
        let n = state.y.len();
        let y0 = &state.y;
        let start = if self.is_sdirk { 0 } else { 1 };
        let mut t = state.t;
        let mut updated_jacobian = false;
        loop {
            for i in start..self.tableau.s() {
                t += self.tableau.c(i) * state.h;
                let mut phi = y0.clone();
                for j in 0..i {
                    phi += self.tableau.a(i, j) * self.diff.column(j);
                }
                phi *= state.h;
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
                    Err(e) =>  {
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
            let mut error = Eqn::T::zero();
            for i in 0..self.tableau.s() {
                error += self.tableau.d(i) * self.diff.column(i);
            }
            error *= state.h;

            // test error is within tolerance
            {
                let ode_problem = self.ode_problem.as_ref().unwrap();
                scale_y = y_new.abs() * scale(ode_problem.rtol);
                scale_y += ode_problem.atol.as_ref();
            }
            error.component_div_assign(&scale_y);
            let error_norm = error.norm();
            if error_norm <= Eqn::T::from(1.0) {
                break;
            } else {
                // step is rejected
                // reduce step size and try again
                todo!()
            }
        }


        let y1 = &mut state.y;
        for i in 0..self.tableau.s() {
            y1 += self.tableau.b(i) * self.diff.column(i);
        }
        Ok(())
    }
    
    fn interpolate(&self, t: <Eqn>::T) -> anyhow::Result<<Eqn>::V> {
        todo!()
    }
    
    fn state(&self) -> Option<&OdeSolverState<<Eqn>::M>> {
        self.state.as_ref()
    }
    
    fn take_state(&mut self) -> Option<OdeSolverState<<Eqn>::M>> {
        self.state.take()
    }


}
