use crate::{matrix::{DenseMatrix, MatrixRef}, ode_solver::{equations::OdeEquations, OdeSolverProblem}, IndexType, Matrix, Vector, VectorRef};
use num_traits::{One, Zero};
use std::{cell::RefCell, ops::{Deref, SubAssign}, rc::Rc};

use super::{NonLinearOp, Op};

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0 
pub struct BdfCallable<Eqn: OdeEquations> 
{
    eqn: Rc<Eqn>,
    psi_neg_y0: RefCell<Eqn::V>,
    c: RefCell<Eqn::T>,
    jac: RefCell<Eqn::M>,
    rhs_jac: RefCell<Eqn::M>,
    mass_jac: RefCell<Eqn::M>,
    rhs_jacobian_is_stale: RefCell<bool>,
    jacobian_is_stale: RefCell<bool>,
    mass_jacobian_is_stale: RefCell<bool>,
    number_of_rhs_jac_evals: RefCell<usize>,
    number_of_rhs_evals: RefCell<usize>,
    number_of_jac_evals: RefCell<usize>,
    number_of_jac_mul_evals: RefCell<usize>,
}



impl<Eqn: OdeEquations> BdfCallable<Eqn> 
{
    pub fn new(ode_problem: &OdeSolverProblem<Eqn>) -> Self {
        let eqn = ode_problem.eqn.clone();
        let n = ode_problem.eqn.nstates();
        let c = RefCell::new(Eqn::T::zero());
        let psi_neg_y0 = RefCell::new(<Eqn::V as Vector>::zeros(n));
        let rhs_jac = RefCell::new(Eqn::M::zeros(n, n));
        let jac = RefCell::new(Eqn::M::zeros(n, n));
        let mass_jac = RefCell::new(Eqn::M::zeros(n, n));
        let rhs_jacobian_is_stale = RefCell::new(true);
        let jacobian_is_stale = RefCell::new(true);
        let mass_jacobian_is_stale = RefCell::new(true);
        let number_of_rhs_jac_evals = RefCell::new(0);
        let number_of_rhs_evals = RefCell::new(0);
        let number_of_jac_evals = RefCell::new(0);
        let number_of_jac_mul_evals = RefCell::new(0);

        Self { eqn, psi_neg_y0, c, jac, rhs_jac, mass_jac, rhs_jacobian_is_stale, jacobian_is_stale, mass_jacobian_is_stale, number_of_rhs_jac_evals, number_of_rhs_evals, number_of_jac_evals, number_of_jac_mul_evals }
    }

    pub fn number_of_rhs_jac_evals(&self) -> usize {
        *self.number_of_rhs_jac_evals.borrow()
    }
    pub fn number_of_rhs_evals(&self) -> usize {
        *self.number_of_rhs_evals.borrow()
    }
    pub fn number_of_jac_evals(&self) -> usize {
        *self.number_of_jac_evals.borrow()
    }
    pub fn number_of_jac_mul_evals(&self) -> usize {
        *self.number_of_jac_mul_evals.borrow()
    }
    pub fn set_c(&self, h: Eqn::T, alpha: &[Eqn::T], order: IndexType) 
    where 
        for <'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        self.c.replace(h * alpha[order]);
        if !*self.rhs_jacobian_is_stale.borrow() && !*self.mass_jacobian_is_stale.borrow() {
            let rhs_jac_ref = self.rhs_jac.borrow();
            let rhs_jac = rhs_jac_ref.deref();
            let mass_jac_ref = self.mass_jac.borrow();
            let mass_jac = mass_jac_ref.deref();
            let c = *self.c.borrow().deref();
            self.jac.replace(mass_jac - rhs_jac * c); 
        } else {
            self.jacobian_is_stale.replace(true);
        }
    }
    pub fn set_psi_and_y0<M: DenseMatrix<T=Eqn::T, V=Eqn::V>>(&self, diff: &M, gamma: &[Eqn::T], alpha: &[Eqn::T], order: usize, y0: &Eqn::V) {
        // update psi term as defined in second equation on page 9 of [1]
        let mut new_psi_neg_y0 = diff.column(1) * gamma[1];
        for (i, &gamma_i) in gamma.iter().enumerate().take(order + 1).skip(2) {
            new_psi_neg_y0 += diff.column(i) * gamma_i
        }
        new_psi_neg_y0 *= alpha[order];

        // now negate y0
        new_psi_neg_y0.sub_assign(y0);
        self.psi_neg_y0.replace(new_psi_neg_y0);
    }
    pub fn set_rhs_jacobian_is_stale(&self) {
        self.rhs_jacobian_is_stale.replace(true);
        self.jacobian_is_stale.replace(true);
    }
}


impl<Eqn: OdeEquations> Op for  BdfCallable<Eqn> 
{
    type V = Eqn::V;
    type T = Eqn::T;
    type M = Eqn::M;
    fn nstates(&self) -> usize {
        self.eqn.nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.nparams()
    }
}

// callable to solve for F(y) = M (y' + psi) - f(y) = 0 
impl<Eqn: OdeEquations> NonLinearOp for  BdfCallable<Eqn> 
where 
    for <'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for <'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        let psi_neg_y0_ref = self.psi_neg_y0.borrow();
        let psi_neg_y0 = psi_neg_y0_ref.deref();
        let mut tmp = x + psi_neg_y0;
        self.eqn.mass_inplace(t, &tmp, y);
        self.eqn.rhs_inplace(t, x, &mut tmp);
        // y = - c * tmp  + y``
        let c = *self.c.borrow().deref();
        y.axpy(-c, &tmp, Eqn::T::one());

        let number_of_rhs_evals = *self.number_of_rhs_evals.borrow() + 1;
        self.number_of_rhs_evals.replace(number_of_rhs_evals);
    }
    // (M - c * f'(y)) v
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.eqn.mass_inplace(t, v, y);
        let tmp = self.eqn.rhs_jac(t, x, v);
        // y = - c * tmp  + y
        let c = *self.c.borrow().deref();
        y.axpy(-c, &tmp, Eqn::T::one());

        let number_of_jac_mul_evals = *self.number_of_jac_mul_evals.borrow() + 1;
        self.number_of_jac_mul_evals.replace(number_of_jac_mul_evals);
    }

    fn jacobian(&self, x: &Eqn::V, t: Eqn::T) -> Eqn::M {
        if *self.mass_jacobian_is_stale.borrow() {
            self.mass_jac.replace(self.eqn.mass_matrix(t));
            self.mass_jacobian_is_stale.replace(false);
            self.jacobian_is_stale.replace(true);
        }
        if *self.rhs_jacobian_is_stale.borrow() {
            self.rhs_jac.replace(self.eqn.rhs_jacobian(x, t));
            let number_of_rhs_jac_evals = *self.number_of_rhs_jac_evals.borrow() + 1;
            self.number_of_rhs_jac_evals.replace(number_of_rhs_jac_evals);
            self.rhs_jacobian_is_stale.replace(false);
            self.jacobian_is_stale.replace(true);
        }
        if *self.jacobian_is_stale.borrow() {
            let rhs_jac_ref = self.rhs_jac.borrow();
            let rhs_jac = rhs_jac_ref.deref();
            let mass_jac_ref = self.mass_jac.borrow();
            let mass_jac = mass_jac_ref.deref();
            let c = *self.c.borrow().deref();
            self.jac.replace(mass_jac - rhs_jac * c); 
            self.jacobian_is_stale.replace(false);
        }
        let number_of_jac_evals = *self.number_of_jac_evals.borrow() + 1;
        self.number_of_jac_evals.replace(number_of_jac_evals);
        self.jac.borrow().clone()
    }
}


