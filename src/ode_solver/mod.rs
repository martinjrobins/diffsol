use std::rc::Rc;

use crate::{callable::{filter::FilterCallable, ConstantOp, LinearOp, NonLinearOp}, solver::SolverProblem, IndexType, Solver, Vector, VectorIndex};

use anyhow::Result;
use num_traits::{One, Zero};


pub mod bdf;

pub trait OdeSolverMethod<CRhs: NonLinearOp, CMass: LinearOp<V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<V = CRhs::V, T = CRhs::T>> {
    fn problem(&self) -> Option<&Rc<OdeSolverProblem<CRhs, CMass, CInit>>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<CRhs::V>, problem: Rc<OdeSolverProblem<CRhs, CMass, CInit>>);
    fn step(&mut self, state: &mut OdeSolverState<CRhs::V>) -> Result<()>;
    fn interpolate(&self, state: &OdeSolverState<CRhs::V>, t: CRhs::T) -> CRhs::V;
    fn get_statistics(&self) -> &OdeSolverStatistics;
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
        CRhs: NonLinearOp<V = V, T = V::T>, 
        CMass: LinearOp<V = V, T = V::T>, 
        CInit: ConstantOp<V = V, T = V::T>,
    {

        let p = &ode_problem.problem.p;
        let init = ode_problem.init.as_ref();
        let y = init.call(p);
        Self {
            y,
            t: ode_problem.t0,
            h: ode_problem.h0,
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
        let dummy_x = V::zeros(0);
        let diag = ode_problem.mass.jacobian_diagonal(p);
        let indices = diag.filter_indices(|x| x == CRhs::T::zero());
        let mut y = ode_problem.init.call(p);
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



#[cfg(test)]
mod tests {
    use std::ops::MulAssign;

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

    fn exponential_decay_init<M: Matrix>(_p: &M::V, y: &mut M::V, _data: &M) {
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
    // remove warning about unused mut
    #[allow(unused_mut)]
    fn exponential_decay_with_algebraic<M: Matrix>(x: &M::V, p: &M::V, mut y: &mut M::V, _data: &M) 
    {
        y.copy_from(x);
        y.mul_assign(-p[0]);
        let nstates = y.len();
        y[nstates - 1] = x[nstates - 1] - x[nstates - 2];
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

    fn exponential_decay_with_algebraic_init<M: Matrix>(_p: &M::V, y: &mut M::V, _data: &M) {
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
    
    pub fn test_ode_solver<M: Matrix + 'static, SM: OdeSolverMethod<Closure<M, M>, LinearClosure<M, M>, ConstantClosure<M, M>> + 'static, RS: Solver<FilterCallable<Closure<M, M>>>>(mut method: SM, mut root_solver: RS) 
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
        for ((problem, _y0), soln) in problems.into_iter().zip(y0s.into_iter()).zip(solutions_at_t1.into_iter()) {
            method.make_consistent_and_solve(problem, t1, &mut root_solver).unwrap().assert_eq(&soln, M::T::from(1e-6));
        }
    }
}