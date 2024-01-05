use crate::{Scalar, Vector, Matrix, NonLinearSolver, callable::{Callable, ode::BdfCallable}, nonlinear_solver::{self, newton::NewtonNonlinearSolver}, linear_solver::LinearSolver};

use anyhow::{Result, anyhow};

pub mod bdf;

trait OdeSolverMethod<T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> {
    fn step(&mut self, t: T) -> T;
    fn interpolate(&self, t: T) -> V;
}

enum MethodName {
    Bdf,
}

struct OdeSolverState<T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> {
    y: V,
    t: T,
    h: T,
    p: V,
    atol: V,
    rtol: T,
    mass: M,
    method: MethodName,
    rhs: C,
}

pub struct OdeSolver<T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> {
    method: Option<Box<dyn OdeSolverMethod<T, V, M, C>>>,
    state: OdeSolverState<T, V, M, C>,
}

impl <T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> OdeSolver<T, V, M, C> {
    fn new(rhs: C) -> Self {
        Self {
            method: None,
            state: Self::default_state(&rhs),
        }
    }
    
    fn default_state(rhs: &C) -> OdeSolverState<T, V, M, C> {
        let nstates = rhs.nstates();
        let nparams = rhs.nparams();
        let y0 = V::zeros(nstates);
        let t0 = T::zero();
        let h0 = T::one();
        let p = V::zeros(nparams);
        let mass = M::eye(nstates);
        OdeSolverState::new(y0, t0, h0, p, mass, rhs)
    }
    
    fn get_state(&mut self) -> &OdeSolverState<T, V, M, C> {
        if self.method.is_some() {
            panic!("Cannot change state after method is initialized")
        }
        &self.state
    }
    
    fn set_suggested_step_size(&mut self, h: T) -> &mut Self {
        self.get_state().h = h;
        self
    }
    
    fn set_time(&mut self, t: T) -> &mut Self {
        self.get_state().t = t;
        self
    }
    
    fn set_rtol(&mut self, rtol: T) -> &mut Self {
        self.get_state().rtol = rtol;
        self
    }
    
    fn set_atol(&mut self, atol: V) -> &mut Self {
        self.get_state().atol = atol;
        self
    }

    fn set_parameters(&mut self, p: V) -> &mut Self {
        self.get_state().p = p;
        self
    }
    
    fn set_state(&mut self, y: V) -> &mut Self {
        self.get_state().y = y;
        self
    }
    
    fn set_mass(&mut self, mass: M) -> &mut Self {
        self.get_state().mass = mass;
        self
    }
    
    fn calculate_consistent_y0<NLS: NonLinearSolver<T, V, C>>(&mut self) -> Result<&mut Self> {
        let mut mask = self.get_state().mass.diagonal();
        mask.map(|x| if x == T::zero() { T::one() } else { T::zero() });
        let newton = NLS::new(self.rhs, Some(mask));
        self.y = newton.solve(&self.y0)?;
        Ok(self)
    }
    
    pub fn solve(&mut self, t: T) -> V {
        if self.method.is_none() {
            self.method = match self.get_state().method {
                Method::Bdf(method) => self.method = Some(Box::new(method)),
            }
        }
        while self.t <= t {
            self.t = self.method.step(t);
        }
        self.method.interpolate(t)
    }
}