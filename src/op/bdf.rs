use crate::{
    matrix::{MatrixCommon, MatrixRef}, ode_solver::equations::OdeEquations, scale, LinearOp, Matrix, MatrixSparsity, OdeSolverProblem, Vector, VectorRef
};
use num_traits::Zero;
use std::{
    cell::RefCell,
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
    jac: RefCell<Eqn::M>,
    rhs_jac: RefCell<Eqn::M>,
    mass_jac: RefCell<Eqn::M>,
    jacobian_is_stale: RefCell<bool>,
    number_of_jac_evals: RefCell<usize>,
    sparsity: <Eqn::M as MatrixCommon>::Sparsity,
}

impl<Eqn: OdeEquations> BdfCallable<Eqn> {
    pub fn new(ode_problem: &OdeSolverProblem<Eqn>) -> Self {
        let eqn = ode_problem.eqn.clone();
        let n = ode_problem.eqn.rhs().nstates();
        let c = RefCell::new(Eqn::T::zero());
        let psi_neg_y0 = RefCell::new(<Eqn::V as Vector>::zeros(n));
        let rhs_jac = RefCell::new(Eqn::M::zeros(n, n));
        let jac = RefCell::new(Eqn::M::zeros(n, n));
        let jacobian_is_stale = RefCell::new(true);
        let number_of_jac_evals = RefCell::new(0);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(n));

        let mass_sparsity = LinearOp::sparsity(eqn.mass());
        let rhs_jac_sparsity = NonLinearOp::sparsity(eqn.rhs());
        let sparsity = mass_sparsity.union(rhs_jac_sparsity).unwrap();

        let mass_jac = if eqn.is_mass_constant() {
            RefCell::new(eqn.mass().matrix(Eqn::T::zero()))
        } else {
            RefCell::new(<Eqn::M as Matrix>::zeros(n, n))
        };

        Self {
            eqn,
            psi_neg_y0,
            c,
            jac,
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

    pub fn number_of_jac_evals(&self) -> usize {
        *self.number_of_jac_evals.borrow()
    }
    pub fn set_c(&self, h: Eqn::T, alpha: Eqn::T)
    where
        for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        self.c.replace(h * alpha);
        if !*self.jacobian_is_stale.borrow() {
            let rhs_jac = self.rhs_jac.borrow();
            let mass_jac = self.mass_jac.borrow();
            let c = *self.c.borrow().deref();
            self.jac
                .replace(mass_jac.deref() - rhs_jac.deref() * scale(c));
        }
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
}

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
        self.eqn.mass().gemv_inplace(&tmp, t, -c, y);
    }
    // (M - c * f'(y)) v
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.eqn.rhs().jac_mul_inplace(x, t, v, y);
        let c = *self.c.borrow().deref();
        // y = Mv - c y
        self.eqn.mass().gemv_inplace(v, t, -c, y);
    }

    fn sparsity(&self) -> &<Self::M as MatrixCommon>::Sparsity {
        &self.sparsity
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if *self.jacobian_is_stale.borrow() {
            let rhs_jac = self.rhs_jac.borrow_mut();
            self.eqn.rhs().jacobian_inplace(x, t, &mut rhs_jac);
            let c = *self.c.borrow().deref();
            if self.eqn.is_mass_constant() {
                let mass_jac = self.mass_jac.borrow();
                let jac = self.jac.borrow_mut();
                jac.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
            } else {
                let mass_jac = self.mass_jac.borrow_mut();
                self.eqn.mass().matrix_inplace(t, &mut mass_jac);
                let jac = self.jac.borrow_mut();
                jac.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
            }
            self.jacobian_is_stale.replace(false);
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
