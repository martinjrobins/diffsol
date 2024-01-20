use std::borrow::Borrow;

use crate::{Scalar, Vector, Matrix, callable::Callable, IndexType, solver::Solver, nonlinear_solver::newton::NewtonNonlinearSolver, linear_solver::lu::LU};

use anyhow::Result;
use ouroboros::self_referencing;

pub mod bdf;

trait OdeSolverMethod<'a, T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> {
    fn reset(&mut self, state: &OdeSolverState<T, V>, options: &'a OdeSolverOptions<T, V, CRhs, CMass>);
    fn step(&mut self, state: &mut OdeSolverState<T, V>, t: T) -> T;
    fn interpolate(&self, state: &OdeSolverState<T, V>, t: T) -> V;
}

pub struct OdeSolverState<T: Scalar, V: Vector<T>> {
    pub y: V,
    pub t: T,
    pub h: T,
}

pub struct OdeSolverOptions<T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> {
    pub p: V,
    pub atol: V,
    pub rtol: T,
    pub mass: CMass,
    pub rhs: CRhs,
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

impl <T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> OdeSolverOptions<T, V, CRhs, CMass> {
    fn new(rhs: CRhs, mass: CMass) -> Self {
        Self {
            p: V::zeros(rhs.nparams()),
            atol: V::zeros(rhs.nstates()),
            rtol: T::zero(),
            mass,
            rhs,
            root_solver_max_iter: 15,
            nonlinear_max_iter: 4,
        }
    }
}

#[self_referencing]
pub struct OdeSolverSelf<T: Scalar + 'static, V: Vector<T> + 'static, CRhs: Callable<T, V> + 'static, CMass: Callable<T, V> + 'static> {
    options: OdeSolverOptions<T, V, CRhs, CMass>,
    #[borrows(options)]
    #[covariant]
    method: Option<Box<dyn OdeSolverMethod<'this, T, V, CRhs, CMass>>>,
}

pub struct OdeSolver<T: Scalar + 'static, V: Vector<T> + 'static, CRhs: Callable<T, V> + 'static, CMass: Callable<T, V> + 'static> {
    state: OdeSolverState<T, V>,
    options_and_method: OdeSolverSelf<T, V, CRhs, CMass>,
}

impl <T: Scalar, V: Vector<T>, CRhs: Callable<T, V>, CMass: Callable<T, V>> OdeSolver<T, V, CRhs, CMass> {
    fn new(rhs: CRhs, mass: CMass) -> Self {
        Self {
            state: OdeSolverState::new(rhs),
            options_and_method: OdeSolverSelfBuilder {
                options: OdeSolverOptions::new(rhs, mass),
                method_builder: |_| None,
            }.build()
        }
    }

    fn state(&self) -> &OdeSolverState<T, V> {
        &self.state
    }
    
    fn state_mut(&mut self) -> &mut OdeSolverState<T, V> {
        &mut self.state
    }
    
    fn options(&self) -> &OdeSolverOptions<T, V, CRhs, CMass> {
        self.options_and_method.borrow_options()
    }
    
    fn options_mut(&mut self) -> &mut OdeSolverOptions<T, V, CRhs, CMass> {
        self.options_and_method.borrow_options_mut()
    }
    
    fn set_suggested_step_size(&mut self, h: T) -> &mut Self {
        self.options_mut().h = h;
        self
    }
    
    fn set_time(&mut self, t: T) -> &mut Self {
        self.state_mut().t = t;
        self
    }
    
    fn set_rtol(&mut self, rtol: T) -> &mut Self {
        self.options_mut().rtol = rtol;
        self
    }
    
    fn set_atol(&mut self, atol: V) -> &mut Self {
        self.options_mut().atol = atol;
        self
    }

    fn set_parameters(&mut self, p: V) -> &mut Self {
        self.options_mut().p = p;
        self
    }
    
    fn set_state(&mut self, y: V) -> &mut Self {
        self.state_mut().y = y;
        self
    }
    
    fn set_mass(&mut self, mass: Box<dyn Callable<T, V>>) -> &mut Self {
        self.options_mut().mass = mass;
        self
    }
    
    fn method_mut(&mut self) -> &mut Option<Box<dyn OdeSolverMethod<T, V, CRhs, CMass>>> {
        self.options_and_method.borrow_method_mut()
    }
    
    fn method(&self) -> &Option<Box<dyn OdeSolverMethod<T, V, CRhs, CMass>>> {
        self.options_and_method.borrow_method()
    }
    
    fn calculate_consistent_y0(&mut self) -> Result<&mut Self> {
        let mut mask = self.get_state().mass.diagonal();
        mask.map(|x| if x == T::zero() { T::one() } else { T::zero() });
        let newton = NewtonNonlinearSolver::new(self.options().rtol, &self.options().atol, self.options().root_solver_max_iter, LU::default(), mask);
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