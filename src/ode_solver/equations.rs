use core::panic;
use std::rc::Rc;
use num_traits::{Zero, One};
use std::ops::AddAssign;

use crate::{op::{closure::Closure, linear_closure::{self, LinearClosure}, ode_rhs::OdeRhs, Op}, vector::VectorRef, ConstantOp, LinearOp, Matrix, NonLinearOp, Vector, VectorIndex};



pub trait OdeEquations: Op {
    /// This must be called first
    fn set_params(&mut self, p: Self::V);

    /// calculates $F(y)$ where $y$ is given in `y` and stores the result in `rhs_y`
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V);

    /// calculates $y = J(x)v$
    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V);

    /// initializes `y` with the initial condition
    fn init(&self, t: Self::T) -> Self::V;
    
    fn rhs(&self, t: Self::T, y: &Self::V) -> Self::V {
        let mut rhs_y = Self::V::zeros(self.nstates());
        self.rhs_inplace(t, y, &mut rhs_y);
        rhs_y
    }

    fn rhs_jac(&self, t: Self::T, x: &Self::V, v: &Self::V) -> Self::V {
        let mut rhs_jac_y = Self::V::zeros(self.nstates());
        self.rhs_jac_inplace(t, x, v, &mut rhs_jac_y);
        rhs_jac_y
    }

    /// rhs jacobian matrix J(x), re-use jacobian calculation from NonLinearOp
    fn rhs_jacobian(&self, x: &Self::V, t: Self::T) -> Self::M {
        let rhs_inplace = |x: &Self::V, p: &Self::V, t: Self::T, y_rhs: &mut Self::V| {
            self.rhs_inplace(t, x, y_rhs);
        };
        let rhs_jac_inplace = |x: &Self::V, p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V| {
            self.rhs_jac_inplace(t, x, v, y);
        };
        let dummy_p = Rc::new(Self::V::zeros(0));
        let closure = Closure::new(rhs_inplace, rhs_jac_inplace, self.nstates(), self.nstates(), dummy_p);
        closure.jacobian(x, t)
    }

    /// mass matrix action: y = M(t)
    fn mass_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // assume identity mass matrix
        y.copy_from(v);
    }

    /// returns the indices of the algebraic state variables
    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        let mass = |y: &Self::V, p: &Self::V, t: Self::T, res: &mut Self::V| {
            self.mass_inplace(t, y, res);
        };
        let dummy_p = Rc::new(Self::V::zeros(0));
        let mass_linear_op = LinearClosure::<Self::M, _>::new(mass, self.nstates(), self.nstates(), dummy_p);
        let t = Self::T::zero();
        let diag: Self::V = mass_linear_op.jacobian_diagonal(t);
        diag.filter_indices(|x| x == Self::T::zero())
    }
    
    /// mass matrix, re-use jacobian calculation from LinearOp
    fn mass_matrix(&self, t: Self::T) -> Self::M {
        let mass = |y: &Self::V, p: &Self::V, t: Self::T, res: &mut Self::V| {
            self.rhs_inplace(t, y, res);
        };
        let dummy_p = Rc::new(Self::V::zeros(0));
        let linear_closure = LinearClosure::new(mass, self.nstates(), self.nstates(), dummy_p);
        LinearOp::jacobian(&linear_closure, t)
    }
}

pub struct OdeSolverEquations<M, F, G, H, I> 
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: F,
    rhs_jac: G,
    mass: H,
    init: I,
    p: Rc<M::V>,
    nstates: usize,
}

impl<M, F, G, H, I> OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode_with_mass(rhs: F, rhs_jac: G, mass: H, init: I, p: M::V) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);
        Self {
            rhs,
            rhs_jac,
            mass,
            init,
            p,
            nstates,
        }
    }
}

// impl Op 
impl <M, F, G, H, I> Op for OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    type M = M;
    type V = M::V;
    type T = M::T;

   fn nout(&self) -> usize {
       self.nstates
   } 
   fn nparams(&self) -> usize {
       self.p.len()
   }
   fn nstates(&self) -> usize {
       self.nstates
   }
}

impl <M, F, G, H, I> OdeEquations for OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{

    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs)(y, p, t, rhs_y);
    }
    
    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs_jac)(x, p, t, v, y);
    }

    fn mass_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.mass)(v, p, t, y);
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }
}

pub struct OdeSolverEquationsMassI<M, F, G, I> 
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: F,
    rhs_jac: G,
    init: I,
    p: Rc<M::V>,
    nstates: usize,
}

impl<M, F, G, I> OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode(rhs: F, rhs_jac: G, init: I, p: M::V) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);
        Self {
            rhs,
            rhs_jac,
            init,
            p,
            nstates,
        }
    }
}

// impl Op 
impl <M, F, G, I> Op for OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    type M = M;
    type V = M::V;
    type T = M::T;

   fn nout(&self) -> usize {
       self.nstates
   } 
   fn nparams(&self) -> usize {
       self.p.len()
   }
   fn nstates(&self) -> usize {
       self.nstates
   }
}

impl <M, F, G, I> OdeEquations for OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs)(y, p, t, rhs_y);
    }
    
    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs_jac)(x, p, t, v, y);
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }
    
    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        <Self::V as Vector>::Index::zeros(0)
    }
}



