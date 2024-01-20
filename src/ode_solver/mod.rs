use crate::{Scalar, Vector, Matrix, Callable, IndexType, Solver, SolverOptions, NewtonNonlinearSolver, LU, Diagonal, GatherCallable};

use anyhow::Result;
use ouroboros::self_referencing;

pub mod bdf;

trait OdeSolverMethod<'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V> + Diagonal<T, V>> {
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

pub struct OdeSolverProblem<'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V> + Diagonal<T, V>> {
    pub p: &'a V,
    pub atol: Option<&'a V>,
    pub mass: &'a CMass,
    pub rhs: &'a CRhs,
    _phantom: std::marker::PhantomData<T>,
}

pub struct OdeSolverOptions<T: Scalar> {
    pub atol: T,
    pub rtol: T,
    pub root_solver_max_iter: IndexType,
    pub nonlinear_max_iter: IndexType,
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
pub struct OdeSolverSelf<T: Scalar + 'static, V: Vector<T> + 'static, CRhs: Callable<T, V> + 'static, CMass: Callable<T, V> + Diagonal<T, V> + 'static> {
    rhs: CRhs,
    atol: Option<V>,
    mass: CMass,
    p: V,
    #[borrows(rhs, mass, p)]
    #[covariant]
    method: Option<Box<dyn OdeSolverMethod<'this, T, V, CRhs, CMass>>>,
}

pub struct OdeSolver<T: Scalar + 'static, V: Vector<T> + 'static, CRhs: Callable<T, V> + 'static, CMass: Callable<T, V> + Diagonal<T, V> + 'static> {
    state: OdeSolverState<T, V>,
    options_and_method: OdeSolverSelf<T, V, CRhs, CMass>,
}

impl <T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V> + Diagonal<T, V>> OdeSolver<T, V, CRhs, CMass> {
    fn new(rhs: CRhs, mass: CMass) -> Self {
        Self {
            state: OdeSolverState::new(rhs),
            options_and_method: OdeSolverSelfBuilder {
                rhs,
                mass,
                atol: None,
                p: V::zeros(rhs.nparams()),
                method_builder: |_, _, _| None,
            }.build()
        }
    }

    fn state(&self) -> &OdeSolverState<T, V> {
        &self.state
    }
    
    fn state_mut(&mut self) -> &mut OdeSolverState<T, V> {
        &mut self.state
    }
    
    fn calculate_consistent_y0(&mut self) -> Result<&mut Self> {
        let rhs = self.options_and_method.borrow_rhs();
        let mass = self.options_and_method.borrow_mass();
        let x = V::zeros(0);
        let p = self.options_and_method.borrow_p();
        let diag = mass.diagonal(&x, p);
        let indices = diag.filter_indices(|&x| x == T::zero());
        let f = GatherCallable::new(rhs, indices);
        let newton = NewtonNonlinearSolver::new(LU::default());
        self.state_mut().y = newton.solve(&self.state().y)?;
        Ok(self)
    }
    
    pub fn solve(&mut self, t: T) -> V {
        if self.method().is_none() {
            self.with_method_mut(|method| {
                method = Some(bdf::Bdf::new());
            });
        }
        while self.t <= t {
            self.t = self.method.step(t);
        }
        self.method.interpolate(t)
    }
}