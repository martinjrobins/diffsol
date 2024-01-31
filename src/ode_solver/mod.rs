use crate::{Scalar, Vector, Matrix, Callable, IndexType, Solver, SolverOptions, NewtonNonlinearSolver, LU, Jacobian, callable::filter::FilterCallable, solver::{Options, SolverProblem}, solver::{Problem}};

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use ouroboros::self_referencing;
use num_traits::{One, Zero};

use self::bdf::Bdf;

pub mod bdf;

trait OdeSolverMethod<'a, V: Vector, CRhs: Callable<V>, CMass: Callable<V>> {
    fn options(&self) -> Option<&OdeSolverOptions<V::T>>;
    fn set_options(&mut self, options: OdeSolverOptions<V::T>);
    fn problem(&self) -> Option<&OdeSolverProblem<'a, V, CRhs, CMass>>;
    fn set_problem(&mut self, state: &OdeSolverState<V>, problem: OdeSolverProblem<'a, V, CRhs, CMass>);
    fn step(&mut self, state: OdeSolverState<V>) -> Result<OdeSolverState<V>>;
    fn interpolate(&self, state: &OdeSolverState<V>, t: V::T) -> V;
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

pub struct OdeSolverProblem<'a, V: Vector, CRhs: Callable<V>, CMass: Callable<V>> {
    pub p: &'a V,
    pub atol: &'a V,
    pub mass: &'a CMass,
    pub rhs: &'a CRhs,
}

impl <'a, V: Vector, CRhs: Callable<V>, CMass: Callable<V>> OdeSolverProblem<'a, V, CRhs, CMass> {
    pub fn new(rhs: &'a CRhs, atol: &'a V, mass: &'a CMass, p: &'a V) -> Self {
        Self {
            p,
            atol,
            mass,
            rhs,
        }
    }
}

impl<'a, V: Vector, CRhs: Callable<V>, CMass: Callable<V>> Problem<V> for OdeSolverProblem<'a, V, CRhs, CMass> {
    fn atol(&self) -> Option<&V> {
        Some(self.atol)
    }
    fn nstates(&self) -> IndexType {
        self.rhs.nstates()
    }
}


pub struct OdeSolverOptions<T: Scalar> {
    pub atol: T,
    pub rtol: T,
    pub root_solver_max_iter: IndexType,
    pub nonlinear_max_iter: IndexType,
}

impl <T: Scalar> Options<T> for OdeSolverOptions<T> {
    fn atol(&self) -> T {
        self.atol
    }
}

impl <V: Vector> OdeSolverState<V> {
    fn new(rhs: impl Callable<V>) -> Self {
        Self {
            y: V::zeros(rhs.nstates()),
            t: V::T::zero(),
            h: V::T::one(),
        }
    }
}

impl <T: Scalar> Default for OdeSolverOptions<T> {
    fn default() -> Self {
        let options = SolverOptions::default();
        Self {
            atol: options.atol,
            rtol: options.rtol,
            root_solver_max_iter: 15,
            nonlinear_max_iter: 4,
        }
    }
}

#[self_referencing]
pub struct OdeSolverSelf<V: Vector + 'static, CRhs: Callable<V> + 'static, CMass: Callable<V> + 'static> {
    rhs: CRhs,
    mass: CMass,
    atol: V,
    p: V,
    #[borrows(rhs, mass, atol, p)]
    #[covariant]
    method: Box<dyn OdeSolverMethod<'this, V, CRhs, CMass>>,
}

pub struct OdeSolver<V: Vector + 'static, CRhs: Callable<V> + 'static, CMass: Callable<V> + 'static> {
    state: OdeSolverState<V>,
    options_and_method: OdeSolverSelf<V, CRhs, CMass>,
}

impl <V: Vector, CRhs: Callable<V>, CMass: Callable<V>> OdeSolver<V, CRhs, CMass> {
    

    fn state(&self) -> &OdeSolverState<V> {
        &self.state
    }
    
    fn state_mut(&mut self) -> &mut OdeSolverState<V> {
        &mut self.state
    }
    
    pub fn solve(&mut self, t: V::T) -> Result<V> {
        while self.state.t <= t {
            self.state = self.options_and_method.borrow_method().step(self.state)?;
        }
       Ok(self.options_and_method.borrow_method().interpolate(&self.state, t))
    }
}

impl <T: Scalar, CRhs: Callable<DVector<T>> + Jacobian<DMatrix<T>>, CMass: Callable<DVector<T>> + Jacobian<DMatrix<T>>> OdeSolver<DVector<T>, CRhs, CMass> {
    fn calculate_consistent_y0(&mut self) -> Result<&mut Self> {
        let rhs = self.options_and_method.borrow_rhs();
        let y = &self.state.y;
        let mass = self.options_and_method.borrow_mass();
        let p = self.options_and_method.borrow_p();
        let diag = mass.jacobian(&y, p).diagonal();
        let indices = diag.filter_indices(|x| x == T::zero());
        let f = FilterCallable::<DVector<T>, CRhs>::new(rhs, y, indices);
        let mut newton = NewtonNonlinearSolver::<DVector<T>, FilterCallable::<DVector<T>, CRhs>>::new(LU::<T>::default());
        newton.set_problem(y, SolverProblem::new(&f, p));
        self.state_mut().y = newton.solve(self.state().y)?;
        Ok(self)
    }
    fn new(rhs: CRhs, mass: CMass, params: DVector<T>) -> Self {
        let options = OdeSolverOptions::default();
        let state = OdeSolverState::new(rhs);
        let options_and_method = OdeSolverSelfBuilder {
            rhs,
            mass,
            atol: <DVector<T> as Vector>::from_element(rhs.nstates(), options.atol),
            p: params,
            method_builder: |rhs, mass, atol, p| {
                let method = Box::new(Bdf::<DMatrix<T>, CRhs, CMass>::new());
                let problem = OdeSolverProblem::new(rhs, atol, mass, p);
                method.set_problem(&state, problem);
                method
            },
        }.build();
        Self {
            state,
            options_and_method,
        }
    }
}