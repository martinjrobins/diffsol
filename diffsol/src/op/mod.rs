use crate::{
    ConstantOp, ConstantOpSens, ConstantOpSensAdjoint, Context, LinearOp, LinearOpTranspose,
    Matrix, NonLinearOp, NonLinearOpAdjoint, NonLinearOpSens, NonLinearOpSensAdjoint, Scalar,
    Vector,
};

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
pub mod stoch;
pub mod unit;

/// A generic operator trait.
///
/// Op is a trait for operators that, given a paramter vector `p`, operates on an input vector `x` to produce an output vector `y`.
/// It defines the number of states (i.e. length of `x`), the number of outputs (i.e. length of `y`), and number of parameters (i.e. length of `p`) of the operator.
/// It also defines the type of the scalar, vector, and matrices used in the operator.
pub trait Op {
    type T: Scalar;
    type V: Vector<T = Self::T, C = Self::C>;
    type M: Matrix<T = Self::T, V = Self::V, C = Self::C>;
    type C: Context;

    /// return the context of the operator
    fn context(&self) -> &Self::C;

    /// Return the number of input states of the operator.
    fn nstates(&self) -> usize;

    /// Return the number of outputs of the operator.
    fn nout(&self) -> usize;

    /// Return the number of parameters of the operator.
    fn nparams(&self) -> usize;

    /// Return statistics about the operator (e.g. how many times it was called, how many times the jacobian was computed, etc.)
    fn statistics(&self) -> OpStatistics {
        OpStatistics::default()
    }
}

/// A wrapper for an operator that parameterises it with a parameter vector.
pub struct ParameterisedOp<'a, C: Op> {
    pub op: &'a C,
    pub p: &'a C::V,
}

impl<'a, C: Op> ParameterisedOp<'a, C> {
    pub fn new(op: &'a C, p: &'a C::V) -> Self {
        Self { op, p }
    }
}

pub trait BuilderOp: Op {
    fn set_nstates(&mut self, nstates: usize);
    fn set_nparams(&mut self, nparams: usize);
    fn set_nout(&mut self, nout: usize);
    fn calculate_sparsity(&mut self, y0: &Self::V, t0: Self::T, p: &Self::V);
}

impl<C: Op> Op for ParameterisedOp<'_, C> {
    type V = C::V;
    type T = C::T;
    type M = C::M;
    type C = C::C;
    fn nstates(&self) -> usize {
        self.op.nstates()
    }
    fn nout(&self) -> usize {
        self.op.nout()
    }
    fn nparams(&self) -> usize {
        self.op.nparams()
    }
    fn statistics(&self) -> OpStatistics {
        self.op.statistics()
    }
    fn context(&self) -> &Self::C {
        self.op.context()
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
    type C = C::C;
    fn nstates(&self) -> usize {
        C::nstates(*self)
    }
    fn nout(&self) -> usize {
        C::nout(*self)
    }
    fn nparams(&self) -> usize {
        C::nparams(*self)
    }
    fn statistics(&self) -> OpStatistics {
        C::statistics(*self)
    }
    fn context(&self) -> &Self::C {
        C::context(*self)
    }
}

impl<C: Op> Op for &mut C {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    type C = C::C;
    fn nstates(&self) -> usize {
        C::nstates(*self)
    }
    fn nout(&self) -> usize {
        C::nout(*self)
    }
    fn nparams(&self) -> usize {
        C::nparams(*self)
    }
    fn statistics(&self) -> OpStatistics {
        C::statistics(*self)
    }
    fn context(&self) -> &Self::C {
        C::context(*self)
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

impl<C: NonLinearOpAdjoint> NonLinearOpAdjoint for &C {
    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        C::adjoint_inplace(*self, x, t, y)
    }
    fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::adjoint_sparsity(*self)
    }
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::jac_transpose_mul_inplace(*self, x, t, v, y)
    }
}

impl<C: NonLinearOpSens> NonLinearOpSens for &C {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::sens_mul_inplace(*self, x, t, v, y)
    }
    fn sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        C::sens_inplace(*self, x, t, y)
    }

    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::sens_sparsity(*self)
    }
}

impl<C: NonLinearOpSensAdjoint> NonLinearOpSensAdjoint for &C {
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::sens_transpose_mul_inplace(*self, x, t, v, y)
    }
    fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        C::sens_adjoint_inplace(*self, x, t, y)
    }
    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::sens_adjoint_sparsity(*self)
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

impl<C: LinearOpTranspose> LinearOpTranspose for &C {
    fn gemv_transpose_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        C::gemv_transpose_inplace(*self, x, t, beta, y)
    }
    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        C::transpose_inplace(*self, t, y)
    }
    fn transpose_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::transpose_sparsity(*self)
    }
}

impl<C: ConstantOp> ConstantOp for &C {
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        C::call_inplace(*self, t, y)
    }
}

impl<C: ConstantOpSens> ConstantOpSens for &C {
    fn sens_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::sens_mul_inplace(*self, t, v, y)
    }
    fn sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        C::sens_inplace(*self, t, y)
    }
    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::sens_sparsity(*self)
    }
}

impl<C: ConstantOpSensAdjoint> ConstantOpSensAdjoint for &C {
    fn sens_transpose_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::sens_transpose_mul_inplace(*self, t, v, y)
    }
    fn sens_adjoint_inplace(&self, t: Self::T, y: &mut Self::M) {
        C::sens_adjoint_inplace(*self, t, y)
    }
    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        C::sens_adjoint_sparsity(*self)
    }
}
