use crate::{
    matrix::{MatrixRef, MatrixView},
    ode_solver::equations::OdeEquations,
    LinearOp, Matrix, MatrixSparsity, OdeSolverProblem, Vector, VectorRef,
};
use num_traits::{One, Zero};
use std::{
    cell::{Ref, RefCell},
    ops::Deref,
    rc::Rc,
};

use super::{NonLinearOp, Op};

// callable to solve for F(y) = M (y) - h f(phi + a * y) = 0
pub struct SdirkCallable<Eqn: OdeEquations> {
    eqn: Rc<Eqn>,
    c: Eqn::T,
    h: RefCell<Eqn::T>,
    phi: RefCell<Eqn::V>,
    tmp: RefCell<Eqn::V>,
    rhs_jac: RefCell<Eqn::M>,
    mass_jac: RefCell<Eqn::M>,
    jacobian_is_stale: RefCell<bool>,
    number_of_jac_evals: RefCell<usize>,
    sparsity: Option<<Eqn::M as Matrix>::Sparsity>,
}

impl<Eqn: OdeEquations> SdirkCallable<Eqn> {
    pub fn new(ode_problem: &OdeSolverProblem<Eqn>, c: Eqn::T) -> Self {
        let eqn = ode_problem.eqn.clone();
        let n = ode_problem.eqn.rhs().nstates();
        let h = RefCell::new(Eqn::T::zero());
        let phi = RefCell::new(<Eqn::V as Vector>::zeros(n));
        let jacobian_is_stale = RefCell::new(true);
        let number_of_jac_evals = RefCell::new(0);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(n));

        // create the mass and rhs jacobians according to the sparsity pattern
        let mass_sparsity = eqn.mass().sparsity();
        let rhs_jac_sparsity = eqn.rhs().sparsity();
        let rhs_jac = RefCell::new(Eqn::M::new_from_sparsity(n, n, rhs_jac_sparsity));
        let sparsity = if let Some(rhs_jac_sparsity) = rhs_jac_sparsity {
            mass_sparsity.map(|mass_sparsity| mass_sparsity.union(rhs_jac_sparsity).unwrap())
        } else {
            None
        };

        // if mass is constant then pre-compute it
        let mut mass_jac = Eqn::M::new_from_sparsity(n, n, mass_sparsity);
        if eqn.is_mass_constant() {
            eqn.mass().matrix_inplace(Eqn::T::zero(), &mut mass_jac);
        }
        let mass_jac = RefCell::new(mass_jac);

        Self {
            eqn,
            phi,
            c,
            h,
            rhs_jac,
            mass_jac,
            sparsity,
            jacobian_is_stale,
            number_of_jac_evals,
            tmp,
        }
    }

    pub fn number_of_jac_evals(&self) -> usize {
        *self.number_of_jac_evals.borrow()
    }
    pub fn set_h(&self, h: Eqn::T)
    where
        for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    {
        self.h.replace(h);
    }
    pub fn get_last_f_eval(&self) -> Ref<Eqn::V> {
        self.tmp.borrow()
    }
    #[allow(dead_code)]
    fn set_phi_direct(&self, phi: Eqn::V) {
        let mut phi_ref = self.phi.borrow_mut();
        phi_ref.copy_from(&phi);
    }
    pub fn set_phi<'a, M: MatrixView<'a, T = Eqn::T, V = Eqn::V>>(
        &self,
        diff: &M,
        y0: &Eqn::V,
        a: &Eqn::V,
    ) {
        let mut phi = self.phi.borrow_mut();
        phi.copy_from(y0);
        diff.gemv_o(Eqn::T::one(), a, Eqn::T::one(), &mut phi);
    }

    fn set_tmp(&self, x: &Eqn::V) {
        let phi_ref = self.phi.borrow();
        let phi = phi_ref.deref();
        let mut tmp = self.tmp.borrow_mut();
        let c = self.c;

        tmp.copy_from(phi);
        tmp.axpy(c, x, Eqn::T::one());
    }

    pub fn set_jacobian_is_stale(&self) {
        self.jacobian_is_stale.replace(true);
    }
}

impl<Eqn: OdeEquations> Op for SdirkCallable<Eqn> {
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
    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
        self.sparsity.as_ref()
    }
}

impl<Eqn: OdeEquations> NonLinearOp for SdirkCallable<Eqn>
where
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    // F(y) = M (y) - h f(phi + c * y) = 0
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        self.set_tmp(x);
        let tmp = self.tmp.borrow();
        let h = *self.h.borrow().deref();

        self.eqn.rhs().call_inplace(&tmp, t, y);

        // y = Mx - h y
        self.eqn.mass().gemv_inplace(x, t, -h, y);
    }
    // (M - c * h * f'(phi + c * y)) v
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.set_tmp(x);
        let tmp = self.tmp.borrow();
        let h = *self.h.borrow().deref();
        let c = self.c;

        self.eqn.rhs().jac_mul_inplace(&tmp, t, v, y);

        // y = Mv - c h y
        self.eqn.mass().gemv_inplace(v, t, -c * h, y);
    }

    // M - c * h * f'(phi + c * y)
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let c = self.c;
        let h = *self.h.borrow().deref();
        if *self.jacobian_is_stale.borrow() {
            // calculate the mass and rhs jacobians
            let mut rhs_jac = self.rhs_jac.borrow_mut();
            self.set_tmp(x);
            let tmp = self.tmp.borrow();
            self.eqn.rhs().jacobian_inplace(&tmp, t, &mut rhs_jac);

            if self.eqn.is_mass_constant() {
                let mass_jac = self.mass_jac.borrow();
                y.scale_add_and_assign(mass_jac.deref(), -(c * h), rhs_jac.deref());
            } else {
                let mut mass_jac = self.mass_jac.borrow_mut();
                self.eqn.mass().matrix_inplace(t, &mut mass_jac);
                y.scale_add_and_assign(mass_jac.deref(), -(c * h), rhs_jac.deref());
            }
            self.jacobian_is_stale.replace(false);
        } else {
            // only h has changed, so just do the addition
            let rhs_jac = self.rhs_jac.borrow();
            let mass_jac = self.mass_jac.borrow();
            y.scale_add_and_assign(mass_jac.deref(), -(c * h), rhs_jac.deref());
        }
        let number_of_jac_evals = *self.number_of_jac_evals.borrow() + 1;
        self.number_of_jac_evals.replace(number_of_jac_evals);
    }
}

#[cfg(test)]
mod tests {
    use crate::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use crate::ode_solver::test_models::robertson::robertson;
    use crate::op::NonLinearOp;
    use crate::vector::Vector;
    use crate::Matrix;

    use super::SdirkCallable;
    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn test_sdirk_robertson_jacobian() {
        let (problem, _soln) = robertson::<Mcpu>(false);
        let c = 0.1;
        let h = 1.3;
        let phi = Vcpu::from_vec(vec![1.1, 1.2, 1.3]);
        let sdirk_callable = SdirkCallable::new(&problem, c);
        sdirk_callable.set_h(h);
        sdirk_callable.set_phi_direct(phi);
        let t = 0.9;
        let y = Vcpu::from_vec(vec![1.1, 1.2, 1.3]);

        let v = Vcpu::from_vec(vec![2.0, 3.0, 4.0]);
        let jac = sdirk_callable.jacobian(&y, t);
        let jac_mul_v = sdirk_callable.jac_mul(&y, t, &v);
        let mut jac_mul_v2 = Vcpu::from_vec(vec![0.0, 0.0, 0.0]);
        jac.gemv(1.0, &v, 0.0, &mut jac_mul_v2);
        jac_mul_v.assert_eq_st(&jac_mul_v2, 1e-10);
    }

    #[test]
    fn test_sdirk_callable() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let c = 0.1;
        let h = 1.0;
        let sdirk_callable = SdirkCallable::new(&problem, c);
        sdirk_callable.set_h(h);

        let phi = Vcpu::from_vec(vec![1.1, 1.2]);
        sdirk_callable.set_phi_direct(phi);
        // check that the function is correct
        let y = Vcpu::from_vec(vec![1.0, 1.0]);
        let t = 0.0;
        let mut y_out = Vcpu::from_vec(vec![0.0, 0.0]);

        // F(y) = M y -  h f(phi + c * y)
        // M = |1 0|
        //     |0 1|
        // y = |1|
        //     |1|
        // f(y) = |-0.1 * y|
        //        |-0.1 * y|
        // i.e. f(phi + c * y) = |-0.1 * (1.1 + 0.1 * 1)| = |-0.12|
        //                       |-0.1 * (1.2 + 0.1 * 1)| = |-0.13|
        //  i.e. F(y) = |1 0| |1| - |-0.12| =  |1.12|
        //              |0 1| |1|   |-0.13|    |1.13|
        sdirk_callable.call_inplace(&y, t, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![1.12, 1.13]);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        let v = Vcpu::from_vec(vec![1.0, 1.0]);
        // f'(phi + c * y)v = |-0.1| = |-0.1|
        //                    |-0.1| = |-0.1|
        // Mv - c * h * f'(phi + c * y) v = |1 0| |1| - 0.1 * |-0.1| = |1.01|
        //                                  |0 1| |1|         |-0.1|   |1.01|
        sdirk_callable.jac_mul_inplace(&y, t, &v, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![1.01, 1.01]);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        // J = M - c * h * f'(phi + c * y) = |1 0| - 0.1 * |-0.1 0| = |1.01 0|
        //                                   |0 1|         |0 -0.1|   |0 1.01|
        let mut jac = sdirk_callable.jacobian(&y, t);
        assert_eq!(jac[(0, 0)], 1.01);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(1, 1)], 1.01);
        sdirk_callable.jacobian_inplace(&y, t, &mut jac);
        assert_eq!(jac[(0, 0)], 1.01);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(1, 1)], 1.01);
    }
}
