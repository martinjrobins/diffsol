use std::rc::Rc;

use crate::{Scalar, Vector, Callable, IndexType, Solver, NewtonNonlinearSolver, LU, Jacobian, callable::filter::FilterCallable, solver::SolverProblem};

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use num_traits::{One, Zero};

use self::bdf::Bdf;

pub mod bdf;

trait OdeSolverMethod<CRhs: Callable, CMass: Callable<V = CRhs::V>> {
    fn problem(&self) -> Option<&Rc<OdeSolverProblem<CRhs, CMass>>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<CRhs::V>, problem: Rc<OdeSolverProblem<CRhs, CMass>>);
    fn step(&mut self, state: &mut OdeSolverState<CRhs::V>) -> Result<()>;
    fn interpolate(&self, state: &OdeSolverState<CRhs::V>, t: CRhs::T) -> CRhs::V;
    fn get_statistics(&self) -> &OdeSolverStatistics;
}

pub struct OdeSolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub struct OdeSolverState<V: Vector> {
    pub y: V,
    pub t: V::T,
    pub h: V::T,
}

pub struct OdeSolverProblem<CRhs: Callable, CMass: Callable<V = CRhs::V>> {
    pub problem: Rc<SolverProblem<CRhs>>,
    pub mass: Rc<CMass>,
}

impl <CRhs: Callable, CMass: Callable<V = CRhs::V>> OdeSolverProblem<CRhs, CMass> {
    pub fn new(rhs: CRhs, mass: CMass, p: CRhs::V) -> Self {
        let problem = Rc::new(SolverProblem::new(Rc::new(rhs), p));
        let mass = Rc::new(mass);
        Self {
            problem,
            mass,
        }
    }
}


impl <V: Vector> OdeSolverState<V> {
    fn new(rhs: & impl Callable<V = V>) -> Self {
        Self {
            y: V::zeros(rhs.nstates()),
            t: V::T::zero(),
            h: V::T::one(),
        }
    }
}


pub struct OdeSolver<CRhs: Callable, CMass: Callable<V = CRhs::V>> {
    state: OdeSolverState<CRhs::V>,
    ode_problem: Rc<OdeSolverProblem<CRhs, CMass>>,
    method: Box<dyn OdeSolverMethod<CRhs, CMass>>,
}

impl <CRhs: Callable, CMass: Callable<V = CRhs::V>> OdeSolver<CRhs, CMass> {
    fn state(&self) -> &OdeSolverState<CRhs::V> {
        &self.state
    }
    
    fn state_mut(&mut self) -> &mut OdeSolverState<CRhs::V> {
        &mut self.state
    }
    
    pub fn solve(&mut self, t: CRhs::T) -> Result<CRhs::V> {
        while self.state.t <= t {
            self.method.step(&mut self.state)?;
        }
       Ok(self.method.interpolate(&self.state, t))
    }
}

impl <T: Scalar, CRhs: Jacobian<V = DVector<T>, M = DMatrix<T>, T = T> + 'static, CMass: Jacobian<V = DVector<T>, M = DMatrix<T>, T = T> + 'static> OdeSolver<CRhs, CMass> {
    fn calculate_consistent_y0(&mut self) -> Result<&mut Self> {
        let rhs = self.ode_problem.problem.f.clone();
        let y = &self.state.y;
        let mass = self.ode_problem.mass.as_ref();
        let p = self.ode_problem.problem.p.clone();
        let diag = mass.jacobian(&y, &p).diagonal();
        let indices = diag.filter_indices(|x| x == T::zero());
        let f = Rc::new(FilterCallable::new(rhs, y, indices));
        let mut newton = NewtonNonlinearSolver::new(LU::<T>::default());
        let problem = Rc::new(SolverProblem::new(f, p));
        newton.set_problem(y, problem);
        newton.solve_in_place(&mut self.state.y)?;
        Ok(self)
    }
    fn new(rhs: CRhs, mass: CMass, params: DVector<T>) -> Self {
        let state = OdeSolverState::new(&rhs);
        let problem = Rc::new(OdeSolverProblem::new(rhs, mass, params));
        let method = Box::new(Bdf::<DMatrix<T>, CRhs, CMass>::new());
        
        Self {
            state,
            method,
            ode_problem: problem,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{callable::closure::Closure, Matrix, Solver, SolverProblem};
    use super::*;
    use num_traits::{One, Zero};
    
    // 0 = J * x * x - 8
    fn square<M: Matrix>(x: &M::V, _p: &M::V, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
        y.component_mul_assign(x);
        y.add_scalar_mut(M::T::from(-8.0));
    }

    // J = 2 * J * x * dx
    fn square_jacobian<M: Matrix>(x: &M::V, _p: &M::V, v: &M::V, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::from(2.0), x, M::T::zero(), y); // y = 2 * J * x
        y.component_mul_assign(v);
    }
    
    pub type SquareClosure<M: Matrix> = Closure<M, fn(&M::V, &M::V, &mut M::V, &M), fn(&M::V, &M::V, &M::V, &mut M::V, &M), M>;
    
    pub fn get_square_problem<M: Matrix>() -> SquareClosure<M> {
        let jac = Matrix::from_diagonal(&M::V::from_vec(vec![2.0.into(), 2.0.into()]));
        Closure::new(
            square,
            square_jacobian,
            jac, 
            2,
        )
    }
    
    pub fn test_ode_solver<M: Matrix, S: OdeSolver<SquareClosure<M>>> (mut solver: S) 
    {
        let op = Rc::new(get_square_problem::<M>());
        let problem = Rc::new(SolverProblem::new(op, <M::V as Vector>::zeros(0)));
        let x0 = M::V::from_vec(vec![2.1.into(), 2.1.into()]);
        solver.set_problem(&x0, problem);
        let x = solver.solve(&x0).unwrap();
        let expect = M::V::from_vec(vec![2.0