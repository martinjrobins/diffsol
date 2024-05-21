use std::{cell::RefCell, rc::Rc};
use num_traits::{One, Zero};

use crate::{NonLinearOp, OdeEquations, OdeSolverState, Op, Matrix, Vector, MatrixSparsity, ConstantOp, LinearOp};

pub struct SensInit<Eqn> 
where 
  Eqn: OdeEquations
{
    eqn: Rc<Eqn>,
    init_sens: RefCell<Eqn::M>,
    index: RefCell<usize>,
}

impl<Eqn> SensInit<Eqn> 
where 
  Eqn: OdeEquations
{
    pub fn new(eqn: &Rc<Eqn>) -> Self {
        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let init_sens = Eqn::M::new_from_sparsity(nstates, nparams, eqn.init().sparsity_sens());
        let init_sens = RefCell::new(init_sens);
        let index = RefCell::new(0);
        Self {
            eqn: eqn.clone(),
            init_sens,
            index,
        }
    }
    pub fn update_state(&self, state: &OdeSolverState<Eqn::V>) {
        let mut init_sens = self.init_sens.borrow_mut();
        self.eqn.init().sens_inplace(state.t, &mut init_sens);
    }
    pub fn set_param_index(&self, index: usize) {
        self.index.replace(index);
    }
}

impl<Eqn> Op for SensInit<Eqn> 
where 
  Eqn: OdeEquations
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> ConstantOp for SensInit<Eqn> 
where 
  Eqn: OdeEquations
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.fill(Eqn::T::zero());
        let init_sens = self.init_sens.borrow();
        let index = *self.index.borrow();
        init_sens.add_column_to_vector(index, y);
    }
}

pub struct SensRhs<Eqn> 
where 
  Eqn: OdeEquations
{
    eqn: Rc<Eqn>,
    sens: RefCell<Eqn::M>,
    rhs_sens: RefCell<Eqn::M>,
    mass_sens: RefCell<Eqn::M>,
    y: RefCell<Eqn::V>,
    index: RefCell<usize>,
}

impl<Eqn> SensRhs<Eqn> 
where 
  Eqn: OdeEquations
{
    pub fn new(eqn: &Rc<Eqn>) -> Self {
        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let rhs_sens = Eqn::M::new_from_sparsity(nstates, nparams, eqn.rhs().sparsity_sens());
        let mass_sens = if let Some(mass) = eqn.mass() {
            Eqn::M::new_from_sparsity(nstates, nparams, mass.sparsity_sens())
        } else {
            let ones = Eqn::V::from_element(nstates, Eqn::T::one());
            Eqn::M::from_diagonal(&ones)
        };
        let sens = if rhs_sens.sparsity().is_some() && mass_sens.sparsity().is_some() {
            let sparsity = mass_sens.sparsity().unwrap().union(rhs_sens.sparsity().unwrap()).unwrap();
            Eqn::M::new_from_sparsity(nstates, nparams, Some(&sparsity))
        } else {
            Eqn::M::new_from_sparsity(nstates, nparams, None)
        };
        let sens = RefCell::new(sens);
        let rhs_sens = RefCell::new(rhs_sens);
        let mass_sens = RefCell::new(mass_sens);
        let y = RefCell::new(Eqn::V::zeros(nstates));
        let index = RefCell::new(0);
        Self {
            eqn: eqn.clone(),
            sens,
            rhs_sens,
            mass_sens,
            y,
            index,
        }
    }
    pub fn update_state(&self, state: &OdeSolverState<Eqn::V>) {
        let mut rhs_sens = self.rhs_sens.borrow_mut();
        let mut mass_sens = self.mass_sens.borrow_mut();
        let mut sens = self.sens.borrow_mut();
        let mut y = self.y.borrow_mut();
        self.eqn.rhs().sens_inplace(&state.y, state.t, &mut rhs_sens);
        if let Some(mass) = self.eqn.mass() {
            mass.sens_inplace(&state.dy, state.t, &mut mass_sens);
        }
        sens.scale_add_and_assign( &rhs_sens, -Eqn::T::one(), &mass_sens);
        y.copy_from(&state.y);
    }
    pub fn set_param_index(&self, index: usize) {
        self.index.replace(index);
    }
}

impl<Eqn> Op for SensRhs<Eqn> 
where 
  Eqn: OdeEquations
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> NonLinearOp for SensRhs<Eqn> 
where 
  Eqn: OdeEquations
{
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let sy = self.y.borrow();
        let sens = self.sens.borrow();
        let index = *self.index.borrow();
        self.eqn.rhs().jac_mul_inplace(&sy, t, x, y);
        sens.add_column_to_vector(index, y);
    }
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let sy = self.y.borrow();
        self.eqn.rhs().jac_mul_inplace(&sy, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        let sy = self.y.borrow();
        self.eqn.rhs().jacobian_inplace(&sy, t, y);
    }
}


pub struct SensEquations<Eqn> 
where 
  Eqn: OdeEquations
{
    eqn: Rc<Eqn>,
    rhs: Rc<SensRhs<Eqn>>,
    init: Rc<SensInit<Eqn>>,
}

impl<Eqn> SensEquations<Eqn> 
where 
  Eqn: OdeEquations
{
    pub fn new(eqn: &Rc<Eqn>) -> Self {
        let rhs = Rc::new(SensRhs::new(eqn));
        let init = Rc::new(SensInit::new(eqn));
        Self {
            rhs,
            init,
            eqn: eqn.clone(),
        }
    }
}

impl<Eqn> Op for SensEquations<Eqn> 
where 
  Eqn: OdeEquations
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nout()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> OdeEquations for SensEquations<Eqn> 
where 
  Eqn: OdeEquations
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type Rhs = SensRhs<Eqn>;
    type Mass = Eqn::Mass;
    type Root = Eqn::Root;
    type Init = SensInit<Eqn>;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn mass(&self) -> Option<&Rc<Self::Mass>> {
        self.eqn.mass()
    }
    fn root(&self) -> Option<&Rc<Self::Root>> {
        self.eqn.root()
    }
    fn init(&self) -> &Rc<Self::Init> {
        &self.init
    }
    fn set_params(&mut self, _p: Self::V) {
        panic!("Not implemented for SensEquations");
    }
}

