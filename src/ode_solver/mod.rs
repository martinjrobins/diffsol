use crate::{Scalar, Vector, Matrix, Callable, IndexType, Solver, SolverOptions, NewtonNonlinearSolver, LU, Jacobian, callable::filter::FilterCallable, solver::{Options, SolverProblem}, solver::{atol::Atol, Problem}};

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use ouroboros::self_referencing;

use self::bdf::Bdf;

pub mod bdf;

trait OdeSolverMethod<'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> {
    fn options(&self) -> Option<&OdeSolverOptions<T>>;
    fn set_options(&mut self, options: OdeSolverOptions<T>);
    fn problem(&self) -> Option<&OdeSolverProblem<'a, T, V, CRhs, CMass>>;
    fn set_problem(&mut self, state: &OdeSolverState<T, V>, problem: OdeSolverProblem<'a, T, V, CRhs, CMass>);
    fn step(&mut self, state: OdeSolverState<T, V>) -> Result<OdeSolverState<T, V>>;
    fn interpolate(&self, state: &OdeSolverState<T, V>, t: T) -> V;
    fn get_statistics(&self) -> &OdeSolverStatistics;
}

pub struct OdeSolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub struct OdeSolverState<T: Scalar, V: Vector<T>> {
    pub y: V,
    pub t: T,
    pub h: T,
}

pub struct OdeSolverProblem<'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> {
    pub p: &'a V,
    pub atol: &'a V,
    pub mass: &'a CMass,
    pub rhs: &'a CRhs,
    _phantom: std::marker::PhantomData<T>,
}

impl <'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> OdeSolverProblem<'a, T, V, CRhs, CMass> {
    pub fn new(rhs: &'a CRhs, atol: &'a V, mass: &'a CMass, p: &'a V) -> Self {
        Self {
            p,
            atol,
            mass,
            rhs,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> Problem<T, V> for OdeSolverProblem<'a, T, V, CRhs, CMass> {
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

impl <T: Scalar, V: Vector<T>> OdeSolverState<T, V> {
    fn new(rhs: impl Callable<T, V>) -> Self {
        Self {
            y: V::zeros(rhs.nstates()),
            t: T::zero(),
            h: T::one(),
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
pub struct OdeSolverSelf<T: Scalar + 'static, V: Vector<T> + 'static, CRhs: Callable<T, V> + 'static, CMass: Callable<T, V> + 'static> {
    rhs: CRhs,
    mass: CMass,
    atol: V,
    p: V,
    #[borrows(rhs, mass, atol, p)]
    #[covariant]
    method: Box<dyn OdeSolverMethod<'this, T, V, CRhs, CMass>>,
}

pub struct OdeSolver<T: Scalar + 'static, V: Vector<T> + 'static, CRhs: Callable<T, V> + 'static, CMass: Callable<T, V> + 'static> {
    state: OdeSolverState<T, V>,
    options_and_method: OdeSolverSelf<T, V, CRhs, CMass>,
}

impl <T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> OdeSolver<T, V, CRhs, CMass> {
    

    fn state(&self) -> &OdeSolverState<T, V> {
        &self.state
    }
    
    fn state_mut(&mut self) -> &mut OdeSolverState<T, V> {
        &mut self.state
    }
    
    pub fn solve(&mut self, t: T) -> Result<V> {
        while self.state.t <= t {
            self.state = self.options_and_method.borrow_method().step(self.state)?;
        }
       Ok(self.options_and_method.borrow_method().interpolate(&self.state, t))
    }
}

impl <T: Scalar, CRhs: Callable<T, DVector<T>> + Jacobian<T, DVector<T>, DMatrix<T>>, CMass: Callable<T, DVector<T>> + Jacobian<T, DVector<T>, DMatrix<T>>> OdeSolver<T, DVector<T>, CRhs, CMass> {
    fn calculate_consistent_y0(&mut self) -> Result<&mut Self> {
        let rhs = self.options_and_method.borrow_rhs();
        let y = &self.state.y;
        let mass = self.options_and_method.borrow_mass();
        let p = self.options_and_method.borrow_p();
        let diag = mass.jacobian(&y, p).diagonal();
        let indices = diag.filter_indices(|x| x == T::zero());
        let f = FilterCallable::<T, DVector<T>, CRhs>::new(rhs, y, indices);
        let mut newton = NewtonNonlinearSolver::<T, DVector<T>, FilterCallable::<T, DVector<T>, CRhs>>::new(LU::<T>::default());
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
            atol: <DVector<T> as Vector<T>>::from_element(rhs.nstates(), options.atol),
            p: params,
            method_builder: |rhs, mass, atol, p| {
                let method = Box::new(Bdf::<T, DVector<T>, DMatrix<T>, CRhs, CMass>::new());
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