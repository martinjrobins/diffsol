use crate::{matrix::MatrixRef, ode_solver::OdeSolverProblem, IndexType, Matrix, Vector, VectorRef};
use num_traits::{One, Zero};
use std::{cell::RefCell, ops::{Deref, SubAssign}, rc::Rc};

use super::{ConstantOp, LinearOp, NonLinearOp, Op};

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0 
pub struct BdfCallable<CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> 
{
    rhs: Rc<CRhs>,
    mass: Rc<CMass>,
    psi_neg_y0: RefCell<CRhs::V>,
    c: RefCell<CRhs::T>,
    rhs_jac: RefCell<CRhs::M>,
    jac: RefCell<CRhs::M>,
    mass_jac: RefCell<CRhs::M>,
    rhs_jacobian_is_stale: RefCell<bool>,
    jacobian_is_stale: RefCell<bool>,
    mass_jacobian_is_stale: RefCell<bool>,
    number_of_rhs_jac_evals: RefCell<usize>,
    number_of_rhs_evals: RefCell<usize>,
    number_of_jac_evals: RefCell<usize>,
    number_of_jac_mul_evals: RefCell<usize>,
}

impl<CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> BdfCallable<CRhs, CMass> 
{
    pub fn new<CInit: ConstantOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>>(ode_problem: Rc<OdeSolverProblem<CRhs, CMass, CInit>>) -> Self {
        let n = ode_problem.problem.f.nstates();
        let c = RefCell::new(CRhs::T::zero());
        let psi_neg_y0 = RefCell::new(<CRhs::V as Vector>::zeros(n));
        let rhs_jac = RefCell::new(CRhs::M::zeros(n, n));
        let jac = RefCell::new(CRhs::M::zeros(n, n));
        let mass_jac = RefCell::new(CRhs::M::zeros(n, n));
        let rhs_jacobian_is_stale = RefCell::new(true);
        let jacobian_is_stale = RefCell::new(true);
        let mass_jacobian_is_stale = RefCell::new(true);
        let rhs = ode_problem.problem.f.clone();
        let mass = ode_problem.mass.clone();
        let number_of_rhs_jac_evals = RefCell::new(0);
        let number_of_rhs_evals = RefCell::new(0);
        let number_of_jac_evals = RefCell::new(0);
        let number_of_jac_mul_evals = RefCell::new(0);

        Self { rhs, mass, psi_neg_y0, c, rhs_jac, jac, mass_jac, rhs_jacobian_is_stale, jacobian_is_stale, mass_jacobian_is_stale, number_of_rhs_jac_evals, number_of_rhs_evals, number_of_jac_evals, number_of_jac_mul_evals }
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
    pub fn set_c(&self, h: CRhs::T, alpha: &[CRhs::T], order: IndexType) 
    where 
        for <'b> &'b CRhs::M: MatrixRef<CRhs::M>,
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
    pub fn set_psi_and_y0(&self, diff: &CRhs::M, gamma: &[CRhs::T], alpha: &[CRhs::T], order: usize, y0: &CRhs::V) {
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


impl<CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> Op for  BdfCallable<CRhs, CMass> 
{
    type V = CRhs::V;
    type T = CRhs::T;
    type M = CRhs::M;
    fn nstates(&self) -> usize {
        self.rhs.nstates()
    }
    fn nout(&self) -> usize {
        self.rhs.nout()
    }
    fn nparams(&self) -> usize {
        self.rhs.nparams()
    }
}

// callable to solve for F(y) = M (y' + psi) - f(y) = 0 
impl<CRhs: NonLinearOp, CMass: LinearOp<M = CRhs::M, V = CRhs::V, T = CRhs::T>> NonLinearOp for  BdfCallable<CRhs, CMass> 
where 
    for <'b> &'b CRhs::V: VectorRef<CRhs::V>,
    for <'b> &'b CRhs::M: MatrixRef<CRhs::M>,
{
    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call_inplace(&self, x: &CRhs::V, p: &CRhs::V, t: CRhs::T, y: &mut CRhs::V) {
        self.rhs.call_inplace(x, p, t, y);
        let psi_neg_y0_ref = self.psi_neg_y0.borrow();
        let psi_neg_y0 = psi_neg_y0_ref.deref();
        let c = *self.c.borrow().deref();
        let tmp = x + psi_neg_y0;
        self.mass.gemv(&tmp, p, t, CRhs::T::one(), -c, y);
        let number_of_rhs_evals = *self.number_of_rhs_evals.borrow() + 1;
        self.number_of_rhs_evals.replace(number_of_rhs_evals);
}
    fn jac_mul_inplace(&self, x: &CRhs::V, p: &CRhs::V, t: CRhs::T, v: &CRhs::V, y: &mut CRhs::V) {
        let c = *self.c.borrow().deref();
        self.rhs.jac_mul_inplace(x, p, t, v, y);
        self.mass.gemv(v, p, t,  CRhs::T::one(), -c, y);
        let number_of_jac_mul_evals = *self.number_of_jac_mul_evals.borrow() + 1;
        self.number_of_jac_mul_evals.replace(number_of_jac_mul_evals);
    }

    fn jacobian(&self, x: &CRhs::V, p: &CRhs::V, t: CRhs::T) -> CRhs::M {
        if *self.mass_jacobian_is_stale.borrow() {
            self.mass_jac.replace(self.mass.jacobian(p, t));
            self.mass_jacobian_is_stale.replace(false);
            self.jacobian_is_stale.replace(true);
        }
        if *self.rhs_jacobian_is_stale.borrow() {
            self.rhs_jac.replace(self.rhs.jacobian(x, p, t));
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


