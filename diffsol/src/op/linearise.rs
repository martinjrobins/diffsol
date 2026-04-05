use num_traits::One;
use std::{cell::RefCell, rc::Rc};

use crate::{LinearOp, Matrix, Op, Vector};

use super::nonlinear_op::NonLinearOpJacobian;

pub struct LinearisedOp<C: NonLinearOpJacobian> {
    callable: Rc<C>,
    x: C::V,
    tmp: RefCell<C::V>,
    x_is_set: bool,
}

impl<C: NonLinearOpJacobian> LinearisedOp<C> {
    pub fn new(callable: Rc<C>) -> Self {
        let x = C::V::zeros(callable.nstates(), callable.context().clone());
        let tmp = RefCell::new(C::V::zeros(callable.nstates(), callable.context().clone()));
        Self {
            callable,
            x,
            tmp,
            x_is_set: false,
        }
    }

    pub fn set_x(&mut self, x: &C::V) {
        self.x.copy_from(x);
        self.x_is_set = true;
    }

    pub fn unset_x(&mut self) {
        self.x_is_set = false;
    }

    pub fn x_is_set(&self) -> bool {
        self.x_is_set
    }
}

impl<C: NonLinearOpJacobian> Op for LinearisedOp<C> {
    type V = C::V;
    type T = C::T;
    type M = C::M;
    type C = C::C;
    fn nstates(&self) -> usize {
        self.callable.nstates()
    }
    fn nout(&self) -> usize {
        self.callable.nout()
    }
    fn nparams(&self) -> usize {
        self.callable.nparams()
    }
    fn context(&self) -> &Self::C {
        self.callable.context()
    }
}

impl<C: NonLinearOpJacobian> LinearOp for LinearisedOp<C> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.callable.jac_mul_inplace(&self.x, t, x, y);
    }
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self.tmp.borrow_mut();
        tmp.copy_from(y);
        self.callable.jac_mul_inplace(&self.x, t, x, y);
        y.axpy(beta, &tmp, Self::T::one());
    }
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.callable.jacobian_inplace(&self.x, t, y);
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.callable.jacobian_sparsity()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{
        context::nalgebra::NalgebraContext, matrix::dense_nalgebra_serial::NalgebraMat,
        matrix::Matrix, DenseMatrix, LinearOp, NonLinearOp, NonLinearOpJacobian, Op, Vector,
    };

    use super::LinearisedOp;

    type M = NalgebraMat<f64>;

    struct FakeJacOp {
        ctx: NalgebraContext,
    }

    impl Op for FakeJacOp {
        type V = crate::NalgebraVec<f64>;
        type T = f64;
        type M = M;
        type C = NalgebraContext;

        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            0
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }

    impl NonLinearOp for FakeJacOp {
        fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
            y.copy_from(x);
        }
    }

    impl NonLinearOpJacobian for FakeJacOp {
        fn jac_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    x.get_index(0) * v.get_index(0) + v.get_index(1),
                    2.0 * v.get_index(0) + x.get_index(1) * v.get_index(1),
                ],
                NalgebraContext,
            ));
        }
    }

    #[test]
    fn linearised_op_tracks_state_and_uses_jacobian_helpers() {
        let callable = Rc::new(FakeJacOp {
            ctx: NalgebraContext,
        });
        let mut op = LinearisedOp::new(callable);
        assert_eq!(op.nstates(), 2);
        assert_eq!(op.nout(), 2);
        assert_eq!(op.nparams(), 0);
        assert!(!op.x_is_set());
        assert!(op.sparsity().is_none());

        let x0 = crate::NalgebraVec::from_vec(vec![3.0, 4.0], NalgebraContext);
        op.set_x(&x0);
        assert!(op.x_is_set());

        let v = crate::NalgebraVec::from_vec(vec![5.0, 6.0], NalgebraContext);
        let mut y = crate::NalgebraVec::zeros(2, NalgebraContext);
        op.call_inplace(&v, 0.0, &mut y);
        y.assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![21.0, 34.0], NalgebraContext),
            1e-12,
        );

        y = crate::NalgebraVec::from_vec(vec![1.0, 2.0], NalgebraContext);
        op.gemv_inplace(&v, 0.0, 0.5, &mut y);
        y.assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![21.5, 35.0], NalgebraContext),
            1e-12,
        );

        let mut matrix = M::zeros(2, 2, NalgebraContext);
        op.matrix_inplace(0.0, &mut matrix);
        assert_eq!(matrix.get_index(0, 0), 3.0);
        assert_eq!(matrix.get_index(1, 0), 2.0);
        assert_eq!(matrix.get_index(0, 1), 1.0);
        assert_eq!(matrix.get_index(1, 1), 4.0);

        op.unset_x();
        assert!(!op.x_is_set());
    }
}
