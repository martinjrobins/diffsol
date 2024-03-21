use std::rc::Rc;
use anyhow::Context;

use crate::{op::{filter::FilterCallable, ode_rhs::OdeRhs}, Matrix, NonLinearSolver, OdeEquations, SolverProblem, Vector, VectorIndex};


use anyhow::Result;
use num_traits::{One, Zero};

use self::equations::{OdeSolverEquations, OdeSolverEquationsMassI};

#[cfg(feature = "diffsl")]
use self::diffsl::DiffSl;

#[cfg(feature = "diffsl")]
use crate::op::Op;


pub mod bdf;
pub mod test_models;
pub mod equations;


#[cfg(feature = "diffsl")]
pub mod diffsl;

pub trait OdeSolverMethod<Eqn: OdeEquations> {
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<Eqn::M>, problem: &OdeSolverProblem<Eqn>);
    fn step(&mut self, state: &mut OdeSolverState<Eqn::M>) -> Result<()>;
    fn interpolate(&self, state: &OdeSolverState<Eqn::M>, t: Eqn::T) -> Eqn::V;
    fn solve(&mut self, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Result<Eqn::V> {
        let problem = problem.clone();
        let mut state = OdeSolverState::new(&problem);
        self.set_problem(&mut state, &problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
    fn make_consistent_and_solve<RS: NonLinearSolver<FilterCallable<OdeRhs<Eqn>>>>(&mut self, problem: &OdeSolverProblem<Eqn>, t: Eqn::T, root_solver: &mut RS) -> Result<Eqn::V> {
        let problem = problem.clone();
        let mut state = OdeSolverState::new_consistent(&problem, root_solver)?;
        self.set_problem(&mut state, &problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
}




pub struct OdeSolverState<M: Matrix> {
    pub y: M::V,
    pub t: M::T,
    pub h: M::T,
    _phantom: std::marker::PhantomData<M>,
}

impl <M: Matrix> OdeSolverState<M> {
    pub fn new<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {

        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init(t);
        Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        }
        
    }
    fn new_consistent<Eqn, S>(ode_problem: &OdeSolverProblem<Eqn>, root_solver: &mut S) -> Result<Self>
    where
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
        S: NonLinearSolver<FilterCallable<OdeRhs<Eqn>>> + ?Sized,
    {

        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let indices = ode_problem.eqn.algebraic_indices();
        let mut y = ode_problem.eqn.init(t);
        if indices.len() == 0 {
            return Ok(Self {
                y,
                t,
                h,
                _phantom: std::marker::PhantomData,
            })
        }
        let mut y_filtered = y.filter(&indices);
        let atol = Rc::new(ode_problem.atol.as_ref().filter(&indices));
        let rhs = Rc::new(OdeRhs::new(ode_problem.eqn.clone()));
        let f = Rc::new(FilterCallable::new(rhs, &y, indices));
        let rtol = ode_problem.rtol;
        let init_problem = SolverProblem::new(f, t, atol, rtol);
        root_solver.set_problem(init_problem);
        root_solver.solve_in_place(&mut y_filtered)?;
        let init_problem = root_solver.problem().unwrap();
        let indices = init_problem.f.indices();
        y.scatter_from(&y_filtered, indices);
        Ok(Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        })
    }
}




pub struct OdeSolverProblem<Eqn: OdeEquations> {
    pub eqn: Rc<Eqn>,
    pub rtol: Eqn::T,
    pub atol: Rc<Eqn::V>,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
}

// impl clone
impl <Eqn: OdeEquations> Clone for OdeSolverProblem<Eqn> {
    fn clone(&self) -> Self {
        Self {
            eqn: self.eqn.clone(),
            rtol: self.rtol,
            atol: self.atol.clone(),
            t0: self.t0,
            h0: self.h0,
        }
    }
}

impl <Eqn: OdeEquations> OdeSolverProblem<Eqn> {
    pub fn default_rtol() -> Eqn::T {
        Eqn::T::from(1e-6)
    }
    pub fn default_atol(nstates: usize) -> Eqn::V {
        Eqn::V::from_element(nstates, Eqn::T::from(1e-6))
    }
    pub fn new(eqn: Eqn) -> Self {
        let t0 = Eqn::T::zero();
        let eqn = Rc::new(eqn);
        let h0 = Eqn::T::one();
        let nstates = eqn.nstates();
        let rtol = Self::default_rtol();
        let atol = Rc::new(Self::default_atol(nstates));
        Self {
            eqn,
            rtol,
            atol,
            t0,
            h0,
        }
    }
    pub fn set_params(&mut self, p: Eqn::V) -> Result<()> {
        let eqn = Rc::get_mut(&mut self.eqn).context("Failed to get mutable reference to equations, is there a solver created with this problem?")?;
        eqn.set_params(p);
        Ok(())
    }
}

impl<M, F, G, H, I> OdeSolverProblem<OdeSolverEquations<M, F, G, H, I>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode_with_mass(rhs: F, rhs_jac: G, mass: H, init: I, p: M::V) -> Self {
        let eqn = OdeSolverEquations::new_ode_with_mass(rhs, rhs_jac, mass, init, p);
        OdeSolverProblem::new(eqn)
    }
} 

impl<M, F, G, I> OdeSolverProblem<OdeSolverEquationsMassI<M, F, G, I>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode(rhs: F, rhs_jac: G, init: I, p: M::V) -> Self {
        let eqn = OdeSolverEquationsMassI::new_ode(rhs, rhs_jac, init, p);
        OdeSolverProblem::new(eqn)
    }
}

#[cfg(feature = "diffsl")]
impl OdeSolverProblem<DiffSl> {
    pub fn new_diffsl(source: &str, p: <DiffSl as Op>::V) -> Result<Self> {
        let eqn = DiffSl::new(source, p)?;
        Ok(OdeSolverProblem::new(eqn))
    }
}



pub struct OdeSolverSolutionPoint<V: Vector> {
    pub state: V,
    pub t: V::T,
}

pub struct OdeSolverSolution<V: Vector> {
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
}

impl <V: Vector> OdeSolverSolution<V> {
    pub fn push(&mut self, state: V, t: V::T) {
        // find the index to insert the new point keeping the times sorted
        let index = self.solution_points.iter().position(|x| x.t > t).unwrap_or(self.solution_points.len());
        // insert the new point at that index
        self.solution_points.insert(index, OdeSolverSolutionPoint{state, t});
    }
}

impl<V: Vector> Default for OdeSolverSolution<V> {
    fn default() -> Self {
        Self {
            solution_points: Vec::new(),
        }
    }
}






#[cfg(test)]
mod tests {
    use crate::nonlinear_solver::newton::NewtonNonlinearSolver;
    use tests::bdf::Bdf;
    use super::*;
    use super::test_models::{
        exponential_decay::exponential_decay_problem, 
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem,
        robertson::robertson,
        robertson_ode::robertson_ode,
    };
    
    
    fn test_ode_solver<M, Eqn>(
        method: &mut impl OdeSolverMethod<Eqn>, 
        mut root_solver: impl NonLinearSolver<FilterCallable<OdeRhs<Eqn>>>, 
        problem: OdeSolverProblem<Eqn>, 
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
    )
    where 
        M: Matrix + 'static,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let mut state = OdeSolverState::new_consistent(&problem, &mut root_solver).unwrap();
        method.set_problem(&mut state, &problem);
        for point in solution.solution_points.iter() {
            while state.t < point.t {
                method.step(&mut state).unwrap();
            }

            let soln = method.interpolate(&state, point.t);

            if let Some(override_tol) = override_tol {
                soln.assert_eq(&point.state, override_tol);
            } else {
                let tol = {
                    let problem = method.problem().unwrap();
                    soln.abs() * problem.rtol + problem.atol.as_ref()
                };
                soln.assert_eq(&point.state, tol[0]);
            }
        }
    }
    
    type Mcpu = nalgebra::DMatrix<f64>;
    
    #[test]
    fn test_bdf_nalgebra_exponential_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = exponential_decay_problem::<Mcpu>();
        test_ode_solver(&mut s, rs, problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_rhs_jac_evals: 1
        number_of_rhs_evals: 54
        number_of_linear_solver_setups: 16
        number_of_jac_mul_evals: 0
        number_of_steps: 19
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 54
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.011892071150027213
        final_step_size: 0.23215911532645564
        "###);
    }
    
    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>();
        test_ode_solver(&mut s, rs, problem, soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_rhs_jac_evals: 2
        number_of_rhs_evals: 58
        number_of_linear_solver_setups: 18
        number_of_jac_mul_evals: 0
        number_of_steps: 21
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 58
        number_of_nonlinear_solver_fails: 2
        initial_step_size: 0.004450050658086208
        final_step_size: 0.20974041151932246
        "###);
    }
    
    
    #[test]
    fn test_bdf_nalgebra_robertson() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = robertson::<Mcpu>();
        test_ode_solver(&mut s, rs, problem, soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_rhs_jac_evals: 18
        number_of_rhs_evals: 1046
        number_of_linear_solver_setups: 129
        number_of_jac_mul_evals: 0
        number_of_steps: 374
        number_of_error_test_failures: 17
        number_of_nonlinear_solver_iterations: 1046
        number_of_nonlinear_solver_fails: 25
        initial_step_size: 0.0000045643545698038086
        final_step_size: 7622676567.923919
        "###);
    }
    
    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = robertson_ode::<Mcpu>();
        test_ode_solver(&mut s, rs, problem, soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_rhs_jac_evals: 17
        number_of_rhs_evals: 941
        number_of_linear_solver_setups: 105
        number_of_jac_mul_evals: 0
        number_of_steps: 346
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 941
        number_of_nonlinear_solver_fails: 18
        initial_step_size: 0.0000038381494276795106
        final_step_size: 7310380599.023874
        "###);
    }
}
