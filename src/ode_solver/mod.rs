use std::rc::Rc;

use crate::{op::{closure::Closure, constant_closure::ConstantClosure, filter::FilterCallable, linear_closure::LinearClosure, unit::UnitCallable, ConstantOp, LinearOp, NonLinearOp}, Matrix, NonLinearSolver, SolverProblem, Vector, VectorIndex};

use anyhow::Result;
use num_traits::{One, Zero};


pub mod bdf;
pub mod test_models;

pub trait OdeSolverMethod<CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> {
    fn problem(&self) -> Option<&OdeSolverProblem<CRhs, CMass, CInit>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<CRhs::M>, problem: OdeSolverProblem<CRhs, CMass, CInit>);
    fn step(&mut self, state: &mut OdeSolverState<CRhs::M>) -> Result<()>;
    fn interpolate(&self, state: &OdeSolverState<CRhs::M>, t: CRhs::T) -> CRhs::V;
    fn solve(&mut self, problem: OdeSolverProblem<CRhs, CMass, CInit>, t: CRhs::T) -> Result<CRhs::V> {
        let mut state = OdeSolverState::new(&problem);
        self.set_problem(&mut state, problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
    fn make_consistent_and_solve<RS: NonLinearSolver<FilterCallable<CRhs>>>(&mut self, problem: OdeSolverProblem<CRhs, CMass, CInit>, t: CRhs::T, root_solver: &mut RS) -> Result<CRhs::V> {
        let mut state = OdeSolverState::new_consistent(&problem, root_solver)?;
        self.set_problem(&mut state, problem);
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
    fn new<CRhs, CMass, CInit>(ode_problem: &OdeSolverProblem<CRhs, CMass, CInit>) -> Self
    where
        CRhs: NonLinearOp<M = M, V = M::V, T = M::T>, 
        CMass: LinearOp<M = M, V = M::V, T = M::T>, 
        CInit: ConstantOp<M = M, V = M::V, T = M::T>,
    {

        let p = &ode_problem.p;
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let init = ode_problem.init.as_ref();
        let y = init.call(p, t);
        Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        }
        
    }
    fn new_consistent<CRhs, CMass, CInit, S>(ode_problem: &OdeSolverProblem<CRhs, CMass, CInit>, root_solver: &mut S) -> Result<Self>
    where
        CRhs: NonLinearOp<M = M, V = M::V, T = M::T>, 
        CMass: LinearOp<M = M, V = M::V, T = M::T>, 
        CInit: ConstantOp<M = M, V = M::V, T = M::T>,
        S: NonLinearSolver<FilterCallable<CRhs>> + ?Sized,
    {

        let p = &ode_problem.p;
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
                _phantom: std::marker::PhantomData,
            })
        }
        let mut y_filtered = y.filter(&indices);
        let atol = Rc::new(ode_problem.atol.as_ref().filter(&indices));
        let f = Rc::new(FilterCallable::new(ode_problem.rhs.clone(), &y, indices));
        let p = p.clone();
        let rtol = ode_problem.rtol;
        let init_problem = SolverProblem::new(f, p, t, atol, rtol);
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


pub struct OdeSolverProblem<CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> {
    pub rhs: Rc<CRhs>,
    pub mass: Rc<CMass>,
    pub init: Rc<CInit>,
    pub p: Rc<CRhs::V>,
    pub rtol: CRhs::T,
    pub atol: Rc<CRhs::V>,
    pub t0: CRhs::T,
    pub h0: CRhs::T,
}

// impl clone
impl <CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> Clone for OdeSolverProblem<CRhs, CMass, CInit> {
    fn clone(&self) -> Self {
        Self {
            rhs: self.rhs.clone(),
            mass: self.mass.clone(),
            init: self.init.clone(),
            p: self.p.clone(),
            rtol: self.rtol,
            atol: self.atol.clone(),
            t0: self.t0,
            h0: self.h0,
        }
    }
}

impl <CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>, CInit: ConstantOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> OdeSolverProblem<CRhs, CMass, CInit> {
    pub fn default_rtol() -> CRhs::T {
        CRhs::T::from(1e-6)
    }
    pub fn default_atol(nstates: usize) -> CRhs::V {
        CRhs::V::from_element(nstates, CRhs::T::from(1e-6))
    }
    pub fn new(rhs: CRhs, mass: CMass, init: CInit, p: CRhs::V) -> Self {
        let t0 = CRhs::T::zero();
        let rhs = Rc::new(rhs);
        let p = Rc::new(p);
        let mass = Rc::new(mass);
        let init = Rc::new(init);
        let h0 = CRhs::T::one();
        let nstates = init.nstates();
        let rtol = Self::default_rtol();
        let atol = Rc::new(Self::default_atol(nstates));
        Self {
            rhs,
            mass,
            init,
            p,
            rtol,
            atol,
            t0,
            h0,
        }
    }

    pub fn init(&self) -> &CInit {
        self.init.as_ref()
    }
}
impl <M, F, G> OdeSolverProblem<Closure<M, F, G>, UnitCallable<M>, ConstantClosure<M>> 
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    pub fn new_ode(rhs: F, jac: G, init: impl Fn(&M::V, M::T) -> M::V + 'static, p: M::V) -> Self {
        let t0 = M::T::zero();
        let h0 = M::T::one();
        let nparams = p.len();
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let rhs = Rc::new(Closure::new(rhs, jac, nstates, nstates, nparams));
        let mass = Rc::new(UnitCallable::new(nstates));
        let init = Rc::new(ConstantClosure::new(init, nstates, nstates, nparams));
        let rtol = Self::default_rtol();
        let atol = Rc::new(Self::default_atol(nstates));
        let p = Rc::new(p);
        Self {
            rhs,
            mass,
            init,
            p,
            rtol,
            atol,
            t0,
            h0
        }
    }
}

impl <M, F, G, H> OdeSolverProblem<Closure<M, F, G>, LinearClosure<M, H>, ConstantClosure<M>> 
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    pub fn new_ode_with_mass(rhs: F, jac: G, mass: H, init: impl Fn(&M::V, M::T) -> M::V + 'static, p: M::V) -> Self {
        let t0 = M::T::zero();
        let h0 = M::T::one();
        let nparams = p.len();
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let rhs = Rc::new(Closure::new(rhs, jac, nstates, nstates, nparams));
        let mass = Rc::new(LinearClosure::new(mass, nstates, nstates, nparams));
        let init = Rc::new(ConstantClosure::new(init, nstates, nstates, nparams));
        let rtol = Self::default_rtol();
        let atol = Rc::new(Self::default_atol(nstates));
        let p = Rc::new(p);
        Self {
            rhs,
            mass,
            init,
            p,
            rtol,
            atol,
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
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem,
        robertson::robertson,
        robertson_ode::robertson_ode,
    };
    
    
    fn test_ode_solver<M, CRhs, CMass, CInit>(
        method: &mut impl OdeSolverMethod<CRhs, CMass, CInit>, 
        mut root_solver: impl NonLinearSolver<FilterCallable<CRhs>>, 
        problem: OdeSolverProblem<CRhs, CMass, CInit>, 
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
    )
    where 
        M: Matrix + 'static,
        CRhs: NonLinearOp<M = M, V = M::V, T = M::T>,
        CMass: LinearOp<M = M, V = M::V, T = M::T>,
        CInit: ConstantOp<M = M, V = M::V, T = M::T>,
    {
        let mut state = OdeSolverState::new_consistent(&problem, &mut root_solver).unwrap();
        method.set_problem(&mut state, problem);
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
        final_step_size: 7622668559.795311
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
