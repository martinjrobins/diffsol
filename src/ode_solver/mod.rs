use crate::{Scalar, Vector, Matrix, callable::Callable, IndexType, solver::Solver};

use anyhow::Result;
use ouroboros::self_referencing;

pub mod bdf;

trait OdeSolverMethod<'a, T: Scalar, V: Vector<T>> {
    fn set_state(&mut self, state: &'a OdeSolverState<T, V>);
    fn is_state_set(&self) -> bool;
    fn clear_state(&mut self);
    fn step(&mut self, t: T) -> T;
    fn interpolate(&self, t: T) -> V;
}

struct OdeSolverState<T: Scalar, V: Vector<T>> {
    y: V,
    t: T,
    h: T,
    p: V,
    atol: V,
    rtol: T,
    mass: Box::<dyn Callable<T, V>>,
    rhs: Box::<dyn Callable<T, V>>,
    root_solver_max_iter: IndexType,
    nonlinear_max_iter: IndexType,
}

impl <T: Scalar, V: Vector<T>, M: Matrix<T, V>> OdeSolverState<T, V> {
    fn new(rhs: impl Callable<T, V>) -> Self {
        Self {
            y: V::zeros(rhs.nstates()),
            t: T::zero(),
            h: T::one(),
            p: V::zeros(rhs.nparams()),
            atol: V::zeros(rhs.nstates()),
            rtol: T::zero(),
            mass: Callable::<T, V>::eye(rhs.nstates()),
            rhs,
            root_solver_max_iter: 15,
            nonlinear_max_iter: 4,
        }
    }
}

#[self_referencing]
struct MyStruct {
    int_data: i32,
    float_data: f32,
    #[borrows(int_data)]
    // the 'this lifetime is created by the #[self_referencing] macro
    // and should be used on all references marked by the #[borrows] macro
    int_reference: &'this i32,
    #[borrows(mut float_data)]
    float_reference: &'this mut f32,
}

#[self_referencing]
pub struct OdeSolver<T: Scalar, V: Vector<T>> {
    state: OdeSolverState<T, V>,
    #[borrows(mut state)]
    method: Box<dyn OdeSolverMethod<'this, T, V>>,
}

impl <T: Scalar, V: Vector<T>> OdeSolver<T, V> {
    fn new<C: Callable<T, V>>(rhs: C) -> Self {
        Self {
            method: None,
            state: Self::default_state(&rhs),
        }
    }
    
    fn get_state(&mut self) -> &OdeSolverState<T, V> {
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
    
    fn set_mass(&mut self, mass: Box<dyn Callable<T, V>>) -> &mut Self {
        self.get_state().mass = mass;
        self
    }
    
    fn calculate_consistent_y0<S: Solver<T, V>>(&mut self) -> Result<&mut Self> {
        let mut mask = self.get_state().mass.diagonal();
        mask.map(|x| if x == T::zero() { T::one() } else { T::zero() });
        let newton = S::new(self.rhs, Some(mask));
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