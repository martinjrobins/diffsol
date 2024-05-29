use crate::{
    MatrixSparsityRef, MatrixRef, ode_solver::equations::OdeEquations, LinearOp, Matrix, MatrixSparsity,
    OdeSolverProblem, Vector, VectorRef,
};
use num_traits::{One, Zero};
use std::{
    cell::{Ref, RefCell},
    ops::{AddAssign, Deref, SubAssign},
    rc::Rc,
};

use super::{NonLinearOp, Op};

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0
pub struct BdfCallable<Eqn: OdeEquations> {
    eqn: Rc<Eqn>,
    psi_neg_y0: RefCell<Eqn::V>,
    c: RefCell<Eqn::T>,
    tmp: RefCell<Eqn::V>,
    rhs_jac: RefCell<Eqn::M>,
    mass_jac: RefCell<Eqn::M>,
    jacobian_is_stale: RefCell<bool>,
    number_of_jac_evals: RefCell<usize>,
    sparsity: Option<<Eqn::M as Matrix>::Sparsity>,
}

impl<Eqn: OdeEquations> BdfCallable<Eqn> {
    pub fn from_eqn(eqn: &Rc<Eqn>) -> Self {
        let eqn = eqn.clone();
        let n = eqn.rhs().nstates();
        let c = RefCell::new(Eqn::T::zero());
        let psi_neg_y0 = RefCell::new(<Eqn::V as Vector>::zeros(n));
        let jacobian_is_stale = RefCell::new(true);
        let number_of_jac_evals = RefCell::new(0);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(n));
        let rhs_jac = RefCell::new(<Eqn::M as Matrix>::zeros(0, 0));
        let mass_jac = RefCell::new(<Eqn::M as Matrix>::zeros(0, 0));
        let sparsity = None;
        Self {
            eqn,
            psi_neg_y0,
            c,
            rhs_jac,
            mass_jac,
            jacobian_is_stale,
            number_of_jac_evals,
            tmp,
            sparsity,
        }
    }
    pub fn eqn(&self) -> &Rc<Eqn> {
        &self.eqn
    }
    pub fn new(ode_problem: &OdeSolverProblem<Eqn>) -> Self {
        let eqn = ode_problem.eqn.clone();
        let n = ode_problem.eqn.rhs().nstates();
        let c = RefCell::new(Eqn::T::zero());
        let psi_neg_y0 = RefCell::new(<Eqn::V as Vector>::zeros(n));
        let jacobian_is_stale = RefCell::new(true);
        let number_of_jac_evals = RefCell::new(0);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(n));

        // create the mass and rhs jacobians according to the sparsity pattern
        let rhs_jac_sparsity = eqn.rhs().sparsity();
        let rhs_jac = RefCell::new(Eqn::M::new_from_sparsity(n, n, rhs_jac_sparsity.map(|s| s.to_owned())));
        let sparsity = if let Some(rhs_jac_sparsity) = eqn.rhs().sparsity() {
            if let Some(mass) = eqn.mass() {
                // have mass, use the union of the mass and rhs jacobians sparse patterns
                Some(mass.sparsity().unwrap().to_owned().union(rhs_jac_sparsity).unwrap())
            } else {
                // no mass, use the identity
                let mass_sparsity = <Eqn::M as Matrix>::Sparsity::new_diagonal(n);
                Some(mass_sparsity.union(rhs_jac_sparsity).unwrap())
            }
        } else {
            None
        };

        let mass_jac = if eqn.mass().is_none() {
            // no mass matrix, so just use the identity
            Eqn::M::from_diagonal(&Eqn::V::from_element(n, Eqn::T::one()))
        } else {
            // mass is not constant, so just create a matrix with the correct sparsity
            Eqn::M::new_from_sparsity(n, n, eqn.mass().unwrap().sparsity().map(|s| s.to_owned()))
        };

        let mass_jac = RefCell::new(mass_jac);

        Self {
            eqn,
            psi_neg_y0,
            c,
            rhs_jac,
            mass_jac,
            jacobian_is_stale,
            number_of_jac_evals,
            tmp,
            sparsity,
        }
    }

    #[cfg(test)]
    fn set_c_direct(&mut self, c: Eqn::T) {
        self.c.replace(c);
    }

    #[cfg(test)]
    fn set_psi_neg_y0_direct(&mut self, psi_neg_y0: Eqn::V) {
        self.psi_neg_y0.replace(psi_neg_y0);
    }

    pub fn tmp(&self) -> Ref<Eqn::V> {
        self.tmp.borrow()
    }

    pub fn number_of_jac_evals(&self) -> usize {
        *self.number_of_jac_evals.borrow()
    }
    pub fn set_c(&self, h: Eqn::T, alpha: Eqn::T)
    where
        for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        self.c.replace(h * alpha);
    }
    pub fn set_psi_and_y0(&self, psi: Eqn::V, y0: &Eqn::V) {
        let mut new_psi_neg_y0 = psi;

        // now negate y0
        new_psi_neg_y0.sub_assign(y0);
        self.psi_neg_y0.replace(new_psi_neg_y0);
    }
    pub fn set_jacobian_is_stale(&self) {
        self.jacobian_is_stale.replace(true);
    }
}

impl<Eqn: OdeEquations> Op for BdfCallable<Eqn> {
    type V = Eqn::V;
    type T = Eqn::T;
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
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.sparsity.as_ref().map(|s| s.as_ref())
    }
}

// dF(y)/dp = dM/dp (y - y0 + psi) + Ms - c * df(y)/dp - c df(y)/dy s = 0
// jac is M - c * df(y)/dy, same
// callable to solve for F(y) = M (y' + psi) - f(y) = 0
impl<Eqn: OdeEquations> NonLinearOp for BdfCallable<Eqn>
where
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        let psi_neg_y0_ref = self.psi_neg_y0.borrow();
        let psi_neg_y0 = psi_neg_y0_ref.deref();

        self.eqn.rhs().call_inplace(x, t, y);

        let mut tmp = self.tmp.borrow_mut();
        tmp.copy_from(x);
        tmp.add_assign(psi_neg_y0);
        let c = *self.c.borrow().deref();
        // y = M tmp - c * y
        if let Some(mass) = self.eqn.mass() {
            mass.gemv_inplace(&tmp, t, -c, y);
        } else {
            y.axpy(Eqn::T::one(), &tmp, -c);
        }
    }
    // (M - c * f'(y)) v
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.eqn.rhs().jac_mul_inplace(x, t, v, y);
        let c = *self.c.borrow().deref();
        // y = Mv - c y
        if let Some(mass) = self.eqn.mass() {
            mass.gemv_inplace(v, t, -c, y);
        } else {
            y.axpy(Eqn::T::one(), v, -c);
        }
    }

    // M - c * f'(y)
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if *self.jacobian_is_stale.borrow() {
            // calculate the mass and rhs jacobians
            let mut rhs_jac = self.rhs_jac.borrow_mut();
            self.eqn.rhs().jacobian_inplace(x, t, &mut rhs_jac);
            let c = *self.c.borrow().deref();
            if self.eqn.mass().is_none() {
                let mass_jac = self.mass_jac.borrow();
                y.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
            } else {
                let mut mass_jac = self.mass_jac.borrow_mut();
                self.eqn.mass().unwrap().matrix_inplace(t, &mut mass_jac);
                y.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
            }
            self.jacobian_is_stale.replace(false);
        } else {
            // only c has changed, so just do the addition
            let rhs_jac = self.rhs_jac.borrow();
            let mass_jac = self.mass_jac.borrow();
            let c = *self.c.borrow().deref();
            y.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
        }
        let number_of_jac_evals = *self.number_of_jac_evals.borrow() + 1;
        self.number_of_jac_evals.replace(number_of_jac_evals);
    }
}

#[cfg(test)]
mod tests {
    use crate::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use crate::op::NonLinearOp;
    use crate::vector::Vector;

    use super::BdfCallable;
    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn test_bdf_callable() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let mut bdf_callable = BdfCallable::new(&problem);
        let c = 0.1;
        let phi_neg_y0 = Vcpu::from_vec(vec![1.1, 1.2]);
        bdf_callable.set_c_direct(c);
        bdf_callable.set_psi_neg_y0_direct(phi_neg_y0);
        // check that the bdf function is correct
        let y = Vcpu::from_vec(vec![1.0, 1.0]);
        let t = 0.0;
        let mut y_out = Vcpu::from_vec(vec![0.0, 0.0]);

        // F(y) = M (y - y0 + psi) - c * f(y)
        // M = |1 0|
        //     |0 1|
        // y = |1|
        //     |1|
        // f(y) = |-0.1|
        //        |-0.1|
        //  i.e. F(y) = |1 0| |2.1| - 0.1 * |-0.1| =  |2.11|
        //              |0 1| |2.2|         |-0.1|    |2.21|
        bdf_callable.call_inplace(&y, t, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![2.11, 2.21]);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        let v = Vcpu::from_vec(vec![1.0, 1.0]);
        // f'(y)v = |-0.1|
        //          |-0.1|
        // Mv - c * f'(y) v = |1 0| |1| - 0.1 * |-0.1| = |1.01|
        //                    |0 1| |1|         |-0.1|   |1.01|
        bdf_callable.jac_mul_inplace(&y, t, &v, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![1.01, 1.01]);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        // J = M - c * f'(y) = |1 0| - 0.1 * |-0.1 0| = |1.01 0|
        //                     |0 1|         |0 -0.1|   |0 1.01|
        let jac = bdf_callable.jacobian(&y, t);
        assert_eq!(jac[(0, 0)], 1.01);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(1, 1)], 1.01);
    }
}
