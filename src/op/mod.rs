use std::rc::Rc;

use crate::{ConstantOp, LinearOp, Matrix, NonLinearOp, Scalar, Vector};

use nonlinear_op::NonLinearOpJacobian;
use serde::Serialize;

pub mod bdf;
pub mod closure;
pub mod closure_no_jac;
pub mod closure_with_adjoint;
pub mod closure_with_sens;
pub mod constant_closure;
pub mod constant_closure_with_adjoint;
pub mod constant_closure_with_sens;
pub mod constant_op;
pub mod init;
pub mod linear_closure;
pub mod linear_closure_with_adjoint;
pub mod linear_op;
pub mod linearise;
pub mod matrix;
pub mod nonlinear_op;
pub mod sdirk;
pub mod unit;

/// A generic operator trait.
///
/// Op is a trait for operators that, given a paramter vector `p`, operates on an input vector `x` to produce an output vector `y`.
/// It defines the number of states (i.e. length of `x`), the number of outputs (i.e. length of `y`), and number of parameters (i.e. length of `p`) of the operator.
/// It also defines the type of the scalar, vector, and matrices used in the operator.
pub trait Op {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    type M: Matrix<T = Self::T, V = Self::V>;

    /// Return the number of input states of the operator.
    fn nstates(&self) -> usize;

    /// Return the number of outputs of the operator.
    fn nout(&self) -> usize;

    /// Return the number of parameters of the operator.
    fn nparams(&self) -> usize {
        0
    }

    /// Set the parameters of the operator to the given value.
    fn set_params(&mut self, p: Rc<Self::V>) {
        assert_eq!(p.len(), self.nparams());
    }

    /// Return statistics about the operator (e.g. how many times it was called, how many times the jacobian was computed, etc.)
    fn statistics(&self) -> OpStatistics {
        OpStatistics::default()
    }
}


#[derive(Default, Clone, Serialize)]
pub struct OpStatistics {
    pub number_of_calls: usize,
    pub number_of_jac_muls: usize,
    pub number_of_matrix_evals: usize,
    pub number_of_jac_adj_muls: usize,
}

impl OpStatistics {
    pub fn new() -> Self {
        Self {
            number_of_jac_muls: 0,
            number_of_calls: 0,
            number_of_matrix_evals: 0,
            number_of_jac_adj_muls: 0,
        }
    }

    pub fn increment_call(&mut self) {
        self.number_of_calls += 1;
    }

    pub fn increment_jac_mul(&mut self) {
        self.number_of_jac_muls += 1;
    }

    pub fn increment_jac_adj_mul(&mut self) {
        self.number_of_jac_adj_muls += 1;
    }

    pub fn increment_matrix(&mut self) {
        self.number_of_matrix_evals += 1;
    }
}

impl<C: Op> Op for &C {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    fn nstates(&self) -> usize {
        C::nstates(*self)
    }
    fn nout(&self) -> usize {
        C::nout(*self)
    }
    fn nparams(&self) -> usize {
        C::nparams(*self)
    }
}

impl<C: NonLinearOp> NonLinearOp for &C {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        C::call_inplace(*self, x, t, y)
    }
}

impl<C: NonLinearOpJacobian> NonLinearOpJacobian for &C {
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::jac_mul_inplace(*self, x, t, v, y)
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        C::jacobian_inplace(*self, x, t, y)
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::jacobian_sparsity(*self)
    }
}

impl<C: LinearOp> LinearOp for &C {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        C::gemv_inplace(*self, x, t, beta, y)
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::sparsity(*self)
    }
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        C::matrix_inplace(*self, t, y)
    }
}

impl<C: ConstantOp> ConstantOp for &C {
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        C::call_inplace(*self, t, y)
    }
}
