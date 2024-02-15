use std::rc::Rc;

use crate::{callable::{closure::Closure, constant_closure::ConstantClosure, filter::FilterCallable, unit::UnitCallable, ConstantOp, LinearOp, NonLinearOp}, matrix::Matrix, solver::SolverProblem, Solver, Vector, VectorIndex};

use anyhow::Result;
use num_traits::{One, Zero};


pub mod bdf;
pub mod test_models;

pub trait OdeSolverMethod<CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<V = CRhs::V, T = CRhs::T>> {
    fn problem(&self) -> Option<&Rc<OdeSolverProblem<CRhs, CMass, CInit>>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<CRhs::V>, problem: Rc<OdeSolverProblem<CRhs, CMass, CInit>>);
    fn step(&mut self, state: &mut OdeSolverState<CRhs::V>) -> Result<()>;
    fn interpolate(&self, state: &OdeSolverState<CRhs::V>, t: CRhs::T) -> CRhs::V;
    fn get_statistics(&self) -> OdeSolverStatistics;
    fn solve(&mut self, problem: OdeSolverProblem<CRhs, CMass, CInit>, t: CRhs::T) -> Result<CRhs::V> {
        let mut state = OdeSolverState::new(&problem);
        self.set_problem(&mut state, Rc::new(problem));
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
    fn make_consistent_and_solve<RS: Solver<FilterCallable<CRhs>>>(&mut self, problem: OdeSolverProblem<CRhs, CMass, CInit>, t: CRhs::T, root_solver: &mut RS) -> Result<CRhs::V> {
        let mut state = OdeSolverState::new_consistent(&problem, root_solver)?;
        self.set_problem(&mut state, Rc::new(problem));
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
}

#[derive(Clone, Debug, Default)]
pub struct OdeSolverStatistics {
    pub number_of_rhs_jac_evals: usize,
    pub number_of_rhs_evals: usize,
    pub number_of_jacobian_evals: usize,
    pub number_of_jac_mul_evals: usize,
}


pub struct OdeSolverState<V: Vector> {
    pub y: V,
    pub t: V::T,
    pub h: V::T,
}

impl <V: Vector> OdeSolverState<V> {
    fn new<CRhs, CMass, CInit>(ode_problem: &OdeSolverProblem<CRhs, CMass, CInit>) -> Self
    where
        CRhs: NonLinearOp<V = V, T = V::T>, 
        CMass: LinearOp<V = V, T = V::T>, 
        CInit: ConstantOp<V = V, T = V::T>,
    {

        let p = &ode_problem.problem.p;
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let init = ode_problem.init.as_ref();
        let y = init.call(p, t);
        Self {
            y,
            t,
            h,
        }
        
    }
    fn new_consistent<CRhs, CMass, CInit, S>(ode_problem: &OdeSolverProblem<CRhs, CMass, CInit>, root_solver: &mut S) -> Result<Self>
    where
        CRhs: NonLinearOp<V = V, T = V::T>, 
        CMass: LinearOp<V = V, T = V::T>, 
        CInit: ConstantOp<V = V, T = V::T>,
        S: Solver<FilterCallable<CRhs>> + ?Sized,
    {

        let p = &ode_problem.problem.p;
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let diag = ode_problem.mass.jacobian_diagonal(p, t);
        let indices = diag.filter_indices(|x| x == CRhs::T::zero());
        let mut y = ode_problem.init.call(p, t);
        if indices.len() == 0 {
            return Ok(Self {
                y,
                t,
                h,
            })
        }
        let mut y_filtered = y.filter(&indices);
        let f = Rc::new(FilterCallable::new(ode_problem.problem.f.clone(), &y, indices));
        let init_problem = Rc::new(SolverProblem::new(f, p.clone(), t));
        root_solver.set_problem(init_problem.clone());
        root_solver.solve_in_place(&mut y_filtered)?;
        let indices = init_problem.as_ref().f.indices();
        y.scatter_from(&y_filtered, indices);
        Ok(Self {
            y,
            t,
            h,
        })
    }
}


pub struct OdeSolverProblem<CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V>, CInit: ConstantOp<V = CRhs::V>> {
    pub problem: Rc<SolverProblem<CRhs>>,
    pub mass: Rc<CMass>,
    pub init: Rc<CInit>,
    pub t0: CRhs::T,
    pub h0: CRhs::T,
}


impl <CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V>, CInit: ConstantOp<V = CRhs::V>> OdeSolverProblem<CRhs, CMass, CInit> {
    pub fn new(rhs: CRhs, mass: CMass, init: CInit, p: CRhs::V) -> Self {
        let t0 = CRhs::T::zero();
        let problem = Rc::new(SolverProblem::new(Rc::new(rhs), p, t0));
        let mass = Rc::new(mass);
        let init = Rc::new(init);
        let h0 = CRhs::T::one();
        //TODO: check mass does not depend on state
        //TODO: check init does not depend on state
        Self {
            problem,
            mass,
            init,
            t0,
            h0,
        }
    }
}
impl <M, F, G, H> OdeSolverProblem<Closure<M, F, G>, UnitCallable<M>, ConstantClosure<M, H>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, M::T, &mut M::V),
{
    pub fn new_ode(rhs: F, jac: G, init: H, nstates: usize, nparams: usize) -> Self {
        let t0 = M::T::zero();
        let h0 = M::T::one();
        let rhs = Rc::new(Closure::new(rhs, jac, nstates, nstates, nparams));
        let mass = Rc::new(UnitCallable::new(nstates));
        let init = Rc::new(ConstantClosure::new(init, nstates, nstates, nparams));
        let problem = Rc::new(SolverProblem::new(rhs, M::V::zeros(nparams), M::T::zero()));
        Self {
            problem,
            mass,
            init,
            t0,
            h0
        }
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
    use crate::{nonlinear_solver::newton::NewtonNonlinearSolver, Matrix};
    use tests::bdf::Bdf;
    use super::*;
    use super::test_models::{
        exponential_decay::exponential_decay_problem, 
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem
    };
    
    
    fn test_ode_solver<M, CRhs, CMass, CInit>(
        mut method: impl OdeSolverMethod<CRhs, CMass, CInit>, mut root_solver: impl Solver<FilterCallable<CRhs>>, problem: OdeSolverProblem<CRhs, CMass, CInit>, solution: OdeSolverSolution<M::V>
    )
    where 
        M: Matrix + 'static,
        CRhs: NonLinearOp<M = M, V = M::V, T = M::T>,
        CMass: LinearOp<M = M, V = M::V, T = M::T>,
        CInit: ConstantOp<M = M, V = M::V, T = M::T>,
    {
        let problem = Rc::new(problem);
        method.set_problem(&mut OdeSolverState::new(&problem), problem.clone());
        let mut state = OdeSolverState::new_consistent(&problem, &mut root_solver).unwrap();
        for point in solution.solution_points.iter() {
            while state.t < point.t {
                method.step(&mut state).unwrap();
            }
            let soln = method.interpolate(&state, point.t);
            soln.assert_eq(&point.state, M::T::from(1e-5));
        }
    }
    
    type Mcpu = nalgebra::DMatrix<f64>;
    
    #[test]
    fn test_bdf_nalgebra_exponential_decay() {
        let s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = exponential_decay_problem::<Mcpu>();
        test_ode_solver(s, rs, problem, soln);
    }
    
    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>();
        test_ode_solver(s, rs, problem, soln);
    }
}