use crate::{
    matrix::MatrixRef, ode_solver::equations::OdeEquations, LinearOp, Matrix, MatrixSparsity,
    OdeSolverProblem, Vector, VectorRef,
};
use num_traits::{One, Zero};
use std::{
    cell::RefCell,
    ops::{AddAssign, Deref, SubAssign},
    rc::Rc,
};

use super::{NonLinearOp, Op};

pub struct InitOp<Eqn: OdeEquations> {
    eqn: Rc<Eqn>,
    jac: Eqn::M,
    pub y0: RefCell<Eqn::V>,
    pub algebraic_indices: <Eqn::V as Vector>::Index,
    neg_mass: Eqn::M,
}

impl<Eqn: OdeEquations> InitOp<Eqn> {
    pub fn new(eqn: &Rc<Eqn>, t0: Eqn::T, y0: &Eqn::V, dy0: &Eqn::V) -> Self {
        let eqn = eqn.clone();
        let n = eqn.rhs().nstates();
        let mass_diagonal = eqn.mass().unwrap().matrix(t0).diagonal();
        let algebraic_indices = mass_diagonal.filter_indices(|x| x == Eqn::T::zero());

        let mut rhs_jac = eqn.rhs().jacobian(&y0, t0);
        let mut mass = eqn.mass().unwrap().matrix(t0);

        // equations are:
        // h(t, u, v, du) = 0
        // g(t, u, v) = 0
        // where u are differential states, v are algebraic states.
        // choose h = -M_u du + f(u, v), where M_u are the differential states of the mass matrix
        // want to solve for du, v, so jacobian is
        // J = (-M_u, df/dv)
        //     (0,    dg/dv)
        // note rhs_jac = (df/du df/dv)
        //                (dg/du dg/dv)
        // according to the algebraic indices.
        let (mut m_u, _, _, _) = mass.split_at_indices(&algebraic_indices)
        m_u *= -Eqn::T::one();
        let (dfdu, dfdv, _, dgdv) = rhs_jac.split_at_indices(&algebraic_indices);
        let zero = Eqn::M::zeros(algebraic_indices.len(), n);
        let jac = Eqn::M::combine_at_indices(&m_u, &dfdv, &zero, &dgdv, &algebraic_indices);
        let neg_mass = Eqn::M::combine_at_indices(&m_u, &zero, &zero, &zero, &algebraic_indices);

        let mut y0 = y0.clone();
        y0.copy_from_indices(&dy0, &algebraic_indices);
        let y0 = RefCell::new(y0);
        Self {
            eqn,
            jac,
            y0,
            neg_mass,
            algebraic_indices,
        }
    }

    pub fn scatter_soln(&self, soln: &Eqn::V, y: &mut Eqn::V, dy: &mut Eqn::V) {
        let tmp = dy.clone();
        dy.copy_from(soln);
        dy.copy_from_indices(&tmp, &self.algebraic_indices);
        y.copy_from_indices(&soln, &self.algebraic_indices);
    }
}

impl<Eqn: OdeEquations> Op for InitOp<Eqn> {
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
        self.jac.sparsity()
    }
}

impl<Eqn: OdeEquations> NonLinearOp for InitOp<Eqn>
{
    // -M_u du + f(u, v)
    // g(t, u, v)
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        // input x = (du, v)
        // self.y0 = (u, v)
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(&x, &self.algebraic_indices);

        // y = (f; g)
        self.eqn.rhs().call_inplace(&y0, t, y);


        // y = -M x + y
        self.neg_mass.gemv(Eqn::T::one(), &x, Eqn::T::one(), y);
    }

    // J v
    fn jac_mul_inplace(&self, _x: &Eqn::V, _t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.jac.gemv(Eqn::T::one(), &v, Eqn::T::one(), y);
    }

    // M - c * f'(y)
    fn jacobian_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::M) {
        y.copy_from(&self.jac);
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
