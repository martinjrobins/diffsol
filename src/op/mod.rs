use std::rc::Rc;

use crate::{Matrix, MatrixSparsityRef, Scalar, Vector};

use num_traits::{One, Zero};
use serde::Serialize;

pub mod bdf;
pub mod closure;
pub mod closure_with_adjoint;
pub mod closure_no_jac;
pub mod closure_with_sens;
pub mod constant_closure;
pub mod constant_closure_with_sens;
pub mod init;
pub mod linear_closure;
pub mod linear_closure_with_sens;
pub mod linearise;
pub mod matrix;
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

    /// Return sparsity information for the jacobian or matrix (if available)
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        None
    }

    /// Return sparsity information for the jacobian or matrix (if available)
    fn sparsity_adjoint(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        None
    }

    /// Return sparsity information for the sensitivity of the operator wrt a parameter vector p (if available)
    fn sparsity_sens(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        None
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

// NonLinearOp is a trait that defines a nonlinear operator or function `F` that maps an input vector `x` to an output vector `y`, (i.e. `y = F(x, t)`).
// It extends the [Op] trait with methods for computing the operator and its Jacobian.
//
// The operator is defined by the [Self::call_inplace] method, which computes the function `F(x, t)` at a given state and time.
// The Jacobian is defined by the [Self::jac_mul_inplace] method, which computes the product of the Jacobian with a given vector `J(x, t) * v`.
pub trait NonLinearOp: Op {
    /// Compute the operator `F(x, t)` at a given state and time.
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V);

    /// Compute the product of the Jacobian with a given vector `J(x, t) * v`.
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V);

    /// Compute the product of the transpose of the Jacobian with a given vector `-J(x, t)^T * v`.
    /// The default implementation fails with a panic, as this method is not implemented by default
    /// and should be implemented by the user if needed.
    fn jac_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V) {
        panic!("jac_transpose_mul_inplace not implemented");
    }

    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(x, t) * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
    /// Default implementation returns zero and panics if nparams() is not zero.
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        if self.nparams() != 0 {
            panic!("sens_mul_inplace not implemented for non-zero parameters");
        }
        y.fill(Self::T::zero());
    }

    fn has_sens(&self) -> bool {
        false
    }

    /// Compute the operator `F(x, t)` at a given state and time, and return the result.
    /// Use `[Self::call_inplace]` to for a non-allocating version.
    fn call(&self, x: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(x, t, &mut y);
        y
    }

    /// Compute the product of the Jacobian with a given vector `J(x, t) * v`, and return the result.
    /// Use `[Self::jac_mul_inplace]` to for a non-allocating version.
    fn jac_mul(&self, x: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates());
        self.jac_mul_inplace(x, t, v, &mut y);
        y
    }

    /// Compute the product of the partial gradient of F wrt a parameter vector p with a given vector `\parial F/\partial p(x, t) * v`, and return the result.
    /// Use `[Self::sens_mul_inplace]` to for a non-allocating version.
    fn sens_mul(&self, x: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates());
        self.sens_mul_inplace(x, t, v, &mut y);
        y
    }

    /// Compute the Jacobian matrix `J(x, t)` of the operator and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Op::sparsity`].
    /// The default implementation of this method computes the Jacobian using [Self::jac_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_jacobian_inplace(x, t, y);
    }

    /// Default implementation of the Jacobian computation (this is the default for [Self::jacobian_inplace]).
    fn _default_jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jac_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the Jacobian matrix `J(x, t)` of the operator and return it.
    /// See [Self::jacobian_inplace] for a non-allocating version.
    fn jacobian(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y = Self::M::new_from_sparsity(n, n, self.sparsity().map(|s| s.to_owned()));
        self.jacobian_inplace(x, t, &mut y);
        y
    }

    /// Compute the Adjoint matrix `-J^T(x, t)` of the operator and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Op::sparsity`].
    /// The default implementation of this method computes the Jacobian using [Self::jac_transpose_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_adjoint_inplace(x, t, y);
    }

    /// Default implementation of the Adjoint computation (this is the default for [Self::adjoint_inplace]).
    fn _default_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jac_transpose_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the Adjoint matrix `-J^T(x, t)` of the operator and return it.
    /// See [Self::adjoint_inplace] for a non-allocating version.
    fn adjoint(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y = Self::M::new_from_sparsity(n, n, self.sparsity_adjoint().map(|s| s.to_owned()));
        self.adjoint_inplace(x, t, &mut y);
        y
    }

    /// Compute the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Op::sparsity`].
    /// The default implementation of this method computes the gradient using [Self::sens_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_sens_inplace(x, t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_inplace]).
    fn _default_sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nparams());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nparams() {
            v[j] = Self::T::one();
            self.sens_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_inplace] for a non-allocating version.
    fn sens(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let m = self.nparams();
        let mut y = Self::M::new_from_sparsity(n, m, self.sparsity_sens().map(|s| s.to_owned()));
        self.sens_inplace(x, t, &mut y);
        y
    }
}

/// LinearOp is a trait for linear operators (i.e. they only depend linearly on the input `x`), see [NonLinearOp] for a non-linear op.
///
/// An example of a linear operator is a matrix-vector product `y = A(t) * x`, where `A(t)` is a matrix.
/// It extends the [Op] trait with methods for calling the operator via a GEMV-like operation (i.e. `y = t * A * x + beta * y`), and for computing the matrix representation of the operator.
pub trait LinearOp: Op {
    /// Compute the operator `y = A(t) * x` at a given state and time, the default implementation uses [Self::gemv_inplace].
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let beta = Self::T::zero();
        self.gemv_inplace(x, t, beta, y);
    }

    fn has_sens(&self) -> bool {
        false
    }

    /// Compute the operator via a GEMV operation (i.e. `y = A(t) * x + beta * y`)
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V);

    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(t) * x * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
    /// Default implementation returns zero and panics if nparams() is not zero.
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        if self.nparams() != 0 {
            panic!("sens_mul_inplace not implemented for non-zero parameters");
        }
        y.fill(Self::T::zero());
    }

    /// Compute the product of the partial gradient of F wrt a parameter vector p with a given vector `\parial F/\partial p(x, t) * v`, and return the result.
    /// Use `[Self::sens_mul_inplace]` to for a non-allocating version.
    fn sens_mul(&self, x: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates());
        self.sens_mul_inplace(x, t, v, &mut y);
        y
    }

    /// Compute the matrix representation of the operator `A(t)` and return it.
    /// See [Self::matrix_inplace] for a non-allocating version.
    fn matrix(&self, t: Self::T) -> Self::M {
        let mut y = Self::M::new_from_sparsity(
            self.nstates(),
            self.nstates(),
            self.sparsity().map(|s| s.to_owned()),
        );
        self.matrix_inplace(t, &mut y);
        y
    }

    /// Compute the matrix representation of the operator `A(t)` and store it in the matrix `y`.
    /// The default implementation of this method computes the matrix using [Self::gemv_inplace],
    /// but it can be overriden for more efficient implementations.
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_matrix_inplace(t, y);
    }

    /// Default implementation of the matrix computation, see [Self::matrix_inplace].
    fn _default_matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, t, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Op::sparsity`].
    /// The default implementation of this method computes the gradient using [Self::sens_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_sens_inplace(x, t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_inplace]).
    fn _default_sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nparams());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nparams() {
            v[j] = Self::T::one();
            self.sens_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_inplace] for a non-allocating version.
    fn sens(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let m = self.nparams();
        let mut y = Self::M::new_from_sparsity(n, m, self.sparsity_sens().map(|s| s.to_owned()));
        self.sens_inplace(x, t, &mut y);
        y
    }
}

pub trait ConstantOp: Op {
    fn call_inplace(&self, t: Self::T, y: &mut Self::V);
    fn call(&self, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(t, &mut y);
        y
    }

    fn has_sens(&self) -> bool {
        false
    }

    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(x, t) * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
    /// Default implementation returns zero and panics if nparams() is not zero.
    fn sens_mul_inplace(&self, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        if self.nparams() != 0 {
            panic!("sens_mul_inplace not implemented for non-zero parameters");
        }
        y.fill(Self::T::zero());
    }

    /// Compute the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Op::sparsity`].
    /// The default implementation of this method computes the gradient using [Self::sens_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_sens_inplace(t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_inplace]).
    fn _default_sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nparams());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nparams() {
            v[j] = Self::T::one();
            self.sens_mul_inplace(t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_inplace] for a non-allocating version.
    fn sens(&self, t: Self::T) -> Self::M {
        let n = self.nstates();
        let m = self.nparams();
        let mut y = Self::M::new_from_sparsity(n, m, self.sparsity_sens().map(|s| s.to_owned()));
        self.sens_inplace(t, &mut y);
        y
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
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::jac_mul_inplace(*self, x, t, v, y)
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        C::jacobian_inplace(*self, x, t, y)
    }
}

impl<C: LinearOp> LinearOp for &C {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        C::gemv_inplace(*self, x, t, beta, y)
    }
}
