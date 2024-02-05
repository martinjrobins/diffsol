use std::rc::Rc;

use crate::{Vector, VectorIndex, Matrix, IndexType, Solver, Jacobian, callable::{filter::FilterCallable, ConstantJacobian, ConstantOp, LinearOp, NonLinearOp}, solver::SolverProblem};

use anyhow::Result;
use num_traits::{One, Zero};


pub mod bdf;

trait OdeSolverMethod<CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V>, CInit: ConstantOp<V = CRhs::V>> {
    fn problem(&self) -> Option<&Rc<OdeSolverProblem<CRhs, CMass, CInit>>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<CRhs::V>, problem: Rc<OdeSolverProblem<CRhs, CMass, CInit>>);
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

pub struct OdeSolverProblem<CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V>, CInit: ConstantOp<V = CRhs::V>> {
    pub problem: Rc<SolverProblem<CRhs>>,
    pub mass: Rc<CMass>,
    pub init: Rc<CInit>,
    pub t0: CRhs::T,
    pub h0: CRhs::T,
}

impl <CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V>, CInit: ConstantOp<V = CRhs::V>> OdeSolverProblem<CRhs, CMass, CInit> {
    pub fn new(rhs: CRhs, mass: CMass, init: CInit, p: CRhs::V) -> Self {
        let problem = Rc::new(SolverProblem::new(Rc::new(rhs), p));
        let mass = Rc::new(mass);
        let init = Rc::new(init);
        let t0 = CRhs::T::zero();
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


impl <V: Vector> OdeSolverState<V> {
    fn new<CRhs, CMass, CInit>(ode_problem: &OdeSolverProblem<CRhs, CMass, CInit>) -> Self
    where
        CRhs: NonLinearOp<V = V, T = V::T> + 'static, 
        CMass: ConstantJacobian<V = V, T = V::T> + 'static, 
        CInit: ConstantOp<V = V, T = V::T>,
    {

        let p = &ode_problem.problem.p;
        let rhs = ode_problem.problem.f.as_ref();
        let mass = ode_problem.mass.as_ref();
        let init = ode_problem.init.as_ref();
        let dummy_x = V::zeros(0);
        let y = init.call(&p);
        return Self {
            y,
            t: ode_problem.t0,
            h: ode_problem.h0,
        }
        
    }
    fn new_consistent<CRhs, CMass, CInit, S>(ode_problem: &OdeSolverProblem<CRhs, CMass, CInit>, root_solver: &mut S) -> Result<Self>
    where
        CRhs: NonLinearOp<V = V, T = V::T> + 'static, 
        CMass: ConstantJacobian<V = V, T = V::T> + 'static, 
        CInit: ConstantOp<V = V, T = V::T>,
        S: Solver<FilterCallable<CRhs>> + 'static + ?Sized,
    {

        let p = &ode_problem.problem.p;
        let dummy_x = V::zeros(0);
        let diag = ode_problem.mass.jacobian(&p).diagonal();
        let indices = diag.filter_indices(|x| x == CRhs::T::zero());
        let mut y = ode_problem.init.call(&p);
        if indices.len() == 0 {
            return Ok(Self {
                y,
                t: ode_problem.t0,
                h: ode_problem.h0,
            })
        }
        let f = Rc::new(FilterCallable::new(ode_problem.problem.f.clone(), &dummy_x, indices));
        let init_problem = Rc::new(SolverProblem::new(f, p.clone()));
        root_solver.set_problem(init_problem);
        root_solver.solve_in_place(&mut y)?;
        Ok(Self {
            y,
            t: ode_problem.t0,
            h: ode_problem.h0,
        })
    }
}


pub struct OdeSolver<CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<V = CRhs::V, T = CRhs::T>> {
    state: Option<OdeSolverState<CRhs::V>>,
    ode_problem: Rc<OdeSolverProblem<CRhs, CMass, CInit>>,
    method: Box<dyn OdeSolverMethod<CRhs, CMass, CInit>>,
    root_solver: Option<Box<dyn Solver<FilterCallable<CRhs>>>>,
}



impl <CRhs: NonLinearOp + 'static, CMass: ConstantJacobian<V = CRhs::V, T = CRhs::T> + 'static, CInit: ConstantOp<V = CRhs::V, T = CRhs::T> + 'static> OdeSolver<CRhs, CMass, CInit> {
    fn new(ode_problem: OdeSolverProblem<CRhs, CMass, CInit>, method: impl OdeSolverMethod<CRhs, CMass, CInit> + 'static) -> Self 
    {
        let ode_problem = Rc::new(ode_problem);
        let state = None;
        let method = Box::new(method);
        let root_solver = None;
        
        Self {
            state,
            method,
            ode_problem,
            root_solver,
        }
    }
    fn initialise(&mut self) -> Result<()> {
        let root_solver = self.root_solver.as_mut();
        match root_solver {
            Some(solver) => {
                let state = OdeSolverState::new_consistent(&self.ode_problem, solver.as_mut())?;
                self.state = Some(state);
            }
            None => {
                let state = OdeSolverState::new(&self.ode_problem);
                self.state = Some(state);
            }
        }
        Ok(())
    }
    fn state(&self) -> Option<&OdeSolverState<CRhs::V>> {
        self.state.as_ref()
    }
    
    fn state_mut(&mut self) -> Option<&mut OdeSolverState<CRhs::V>> {
        self.state.as_mut()
    }
    
    pub fn solve(&mut self, t: CRhs::T) -> Result<CRhs::V> {
        if self.state.is_none() {
            return Err(anyhow::anyhow!("OdeSolver::solve() called before initialise"));
        }
        while self.state.as_ref().unwrap().t <= t {
            self.method.step(self.state.as_mut().unwrap())?;
        }
        Ok(self.method.interpolate(self.state.as_ref().unwrap(), t))
    }
}


#[cfg(test)]
mod tests {
    use std::ops::{IndexMut, MulAssign};

    use crate::{callable::{closure::Closure, constant_closure::ConstantClosure, linear_closure::LinearClosure}, vector::VectorRef, Matrix};
    use nalgebra::ComplexField;
    use super::*;
    
    // exponential decay problem
    // dy/dt = -ay (p = [a])
    fn exponential_decay<M: Matrix>(x: &M::V, p: &M::V, y: &mut M::V, _data: &M) {
        y.copy_from(x);
        y.mul_assign(-p[0]);
    }

    // Jv = -av
    fn exponential_decay_jacobian<M: Matrix>(_x: &M::V, p: &M::V, v: &M::V, y: &mut M::V, _jac: &M) {
        y.copy_from(v);
        y.mul_assign(-p[0]);
    }
    
    fn exponential_decay_mass<M: Matrix>(x: &M::V, _p: &M::V, y: &mut M::V, _data: &M) {
        y.copy_from(x);
    }
    
    fn exponential_decay_mass_jacobian<M: Matrix>(_p: &M::V, v: &M::V, y: &mut M::V, _jac: &M) {
        y.copy_from(v);
    }

    fn exponential_decay_init<M: Matrix>(p: &M::V, y: &mut M::V, _data: &M) {
        let y0 = M::V::from_vec(vec![1.0.into(), 1.0.into()]);
        y.copy_from(&y0);
    }

    fn exponential_decay_problem<M: Matrix + 'static>() -> OdeSolverProblem<Closure<M, M>, LinearClosure<M, M>, ConstantClosure<M, M>> {
        let nstates = 2;
        let data = M::zeros(1, 1);
        let rhs = Closure::<M, M>::new(
            exponential_decay,
            exponential_decay_jacobian,
            data.clone(), 
            nstates,
        );
        let mass = LinearClosure::<M, M>::new(
            exponential_decay_mass,
            exponential_decay_mass_jacobian,
            data.clone(), 
            nstates,
        );
        let init = ConstantClosure::<M, M>::new(
            exponential_decay_init,
            data.clone(), 
            nstates,
        );
        let p = M::V::from_vec(vec![0.1.into()]);
        OdeSolverProblem::new(rhs, mass, init, p)
    }
    
    // exponential decay problem with algebraic constraint
    // dy/dt = -ay
    // 0 = z - y
    fn exponential_decay_with_algebraic<M: Matrix>(x: &M::V, p: &M::V, y: &mut M::V, _data: &M) 
    {
        y.copy_from(x);
        y.mul_assign(-p[0]);
        let nstates = y.len();
    }
    
    // Jv = [-av; 0]
    fn exponential_decay_with_algebraic_jacobian<M: Matrix>(_x: &M::V, p: &M::V, v: &M::V, y: &mut M::V, _jac: &M) {
        y.copy_from(v);
        y.mul_assign(-p[0]);
        let nstates = y.len();
        y[nstates - 1] = M::T::zero();
    }
    
    fn exponential_decay_with_algebraic_mass<M: Matrix>(x: &M::V, _p: &M::V, y: &mut M::V, _data: &M) {
        y.copy_from(x);
        let nstates = y.len();
        y[nstates - 1] = M::T::zero();
    }
    
    fn exponential_decay_with_algebraic_mass_jacobian<M: Matrix>(_p: &M::V, v: &M::V, y: &mut M::V, _jac: &M) {
        y.copy_from(v);
        let nstates = y.len();
        y[nstates - 1] = M::T::zero();
    }

    fn exponential_decay_with_algebraic_init<M: Matrix>(p: &M::V, y: &mut M::V, _data: &M) {
        let y0 = M::V::from_vec(vec![1.0.into(), 1.0.into(), 0.0.into()]);
        y.copy_from(&y0);
    }

    fn exponential_decay_with_algebraic_problem<M: Matrix + 'static>() -> OdeSolverProblem<Closure<M, M>, LinearClosure<M, M>, ConstantClosure<M, M>> {
        let nstates = 3;
        let data = M::zeros(1, 1);
        let rhs = Closure::<M, M>::new(
            exponential_decay_with_algebraic,
            exponential_decay_with_algebraic_jacobian,
            data.clone(), 
            nstates,
        );
        let mass = LinearClosure::<M, M>::new(
            exponential_decay_with_algebraic_mass,
            exponential_decay_with_algebraic_mass_jacobian,
            data.clone(), 
            nstates,
        );
        let init = ConstantClosure::<M, M>::new(
            exponential_decay_with_algebraic_init,
            data.clone(), 
            nstates,
        );
        let p = M::V::from_vec(vec![0.1.into()]);
        OdeSolverProblem::new(rhs, mass, init, p)
    }
    
    pub fn test_ode_solver<M: Matrix + 'static, SM: OdeSolverMethod<Closure<M, M>, LinearClosure<M, M>, ConstantClosure<M, M>> + Clone + 'static> (method: SM) 
    where 
        for <'a> &'a M::V: VectorRef<M::V>,
    {
        let problems = vec![
            exponential_decay_problem::<M>(),
            exponential_decay_with_algebraic_problem::<M>(),
        ];
        let y0s = vec![
            M::V::from_vec(vec![1.0.into(), 1.0.into()]),
            M::V::from_vec(vec![1.0.into(), 1.0.into(), 1.0.into()]),
        ];
        let t1 = M::T::from(1.0);
        let solutions_at_t1 = vec![
            &y0s[0] * M::T::exp(-problems[0].problem.p[0] * t1),
            &y0s[1] * M::T::exp(-problems[1].problem.p[0] * t1),
        ];
        for ((problem, y0), soln) in problems.into_iter().zip(y0s.into_iter()).zip(solutions_at_t1.into_iter()) {
            let mut solver = OdeSolver::new(problem, method);
            solver.initialise().unwrap();
            solver.state().unwrap().y.assert_eq(&y0, M::T::from(1e-6));
            let result = solver.solve(t1);
            result.unwrap().assert_eq(&soln, M::T::from(1e-6));
        }
    }
}