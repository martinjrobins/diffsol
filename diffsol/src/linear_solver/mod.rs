use crate::{error::DiffsolError, Matrix, NonLinearOpJacobian};
use diffsol_la::LinearSolver as LaLinearSolver;
use num_traits::Zero;

pub use diffsol_la::{FaerLU, FaerSparseLU, NalgebraLU};

#[cfg(feature = "suitesparse")]
pub use diffsol_la::KLU;

#[cfg(feature = "cuda")]
pub use diffsol_la::CudaLU;

/// Re-export of the [`diffsol_la`] linear-solver backends' module paths, so that
/// existing `crate::linear_solver::<backend>` paths keep resolving.
pub use diffsol_la::linear_solver::{faer, nalgebra};

#[cfg(feature = "suitesparse")]
pub use diffsol_la::linear_solver::suitesparse;

#[cfg(feature = "cuda")]
pub use diffsol_la::linear_solver::cuda;

/// A borrowing adapter that presents a [NonLinearOpJacobian] (evaluated at a
/// fixed state `x` and time `t`) as a [`diffsol_la::LinearOp`].
///
/// This is the bridge that allows the time-aware, operator-based
/// [LinearSolver] trait in `diffsol` to be implemented on top of the
/// time-unaware [`diffsol_la::LinearSolver`] backends.
pub struct LinearisedRef<'a, C: NonLinearOpJacobian> {
    op: &'a C,
    x: Option<&'a C::V>,
    t: C::T,
}

impl<'a, C: NonLinearOpJacobian> LinearisedRef<'a, C> {
    /// Create an adapter used only to query the sparsity pattern (no state set).
    pub fn sparsity_only(op: &'a C) -> Self {
        Self {
            op,
            x: None,
            t: C::T::zero(),
        }
    }

    /// Create an adapter that evaluates the Jacobian at `(x, t)`.
    pub fn at(op: &'a C, x: &'a C::V, t: C::T) -> Self {
        Self { op, x: Some(x), t }
    }
}

impl<C: NonLinearOpJacobian> diffsol_la::LinearOp for LinearisedRef<'_, C> {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    type C = C::C;

    fn nrows(&self) -> crate::IndexType {
        self.op.nout()
    }

    fn ncols(&self) -> crate::IndexType {
        self.op.nstates()
    }

    fn context(&self) -> &Self::C {
        self.op.context()
    }

    fn matrix_inplace(&self, y: &mut Self::M) {
        let x = self.x.expect("LinearisedRef: state x not set");
        self.op.jacobian_inplace(x, self.t, y);
    }

    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.jacobian_sparsity()
    }
}

/// A solver for the linear problem `Ax = b`, where `A` is a linear operator that is obtained by taking the linearisation of a nonlinear operator `C`.
pub trait LinearSolver<M: Matrix>: Default {
    // sets the point at which the linearisation of the operator is evaluated
    // the operator is assumed to have the same sparsity as that given to [Self::set_problem]
    fn set_linearisation<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
    );

    /// Set the problem to be solved, any previous problem is discarded.
    /// Any internal state of the solver is reset.
    /// This function will normally set the sparsity pattern of the matrix to be solved.
    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C);

    /// Solve the problem `Ax = b` and return the solution `x`.
    /// panics if [Self::set_linearisation] has not been called previously
    fn solve(&self, b: &M::V) -> Result<M::V, DiffsolError> {
        let mut b = b.clone();
        self.solve_in_place(&mut b)?;
        Ok(b)
    }

    fn solve_in_place(&self, b: &mut M::V) -> Result<(), DiffsolError>;
}

/// Any [`diffsol_la::LinearSolver`] backend automatically implements the
/// time-aware, operator-based [LinearSolver] trait via the [LinearisedRef] bridge.
impl<M: Matrix, LS: LaLinearSolver<M>> LinearSolver<M> for LS {
    fn set_linearisation<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
    ) {
        LaLinearSolver::set_linearisation(self, &LinearisedRef::at(op, x, t));
    }

    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C) {
        LaLinearSolver::set_sparsity(self, &LinearisedRef::sparsity_only(op));
    }

    fn solve_in_place(&self, b: &mut M::V) -> Result<(), DiffsolError> {
        LaLinearSolver::solve_in_place(self, b).map_err(Into::into)
    }
}

pub struct LinearSolveSolution<V> {
    pub x: V,
    pub b: V,
}

impl<V> LinearSolveSolution<V> {
    pub fn new(b: V, x: V) -> Self {
        Self { x, b }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        op::{closure::Closure, ParameterisedOp},
        scalar::scale,
        vector::VectorRef,
        Context, FaerMat, FaerVec, LinearSolver, Matrix, NalgebraVec, NonLinearOpJacobian, Op,
        Vector,
    };
    use num_traits::{FromPrimitive, One, Zero};

    use super::LinearSolveSolution;
    use super::{FaerLU, NalgebraLU};

    #[allow(clippy::type_complexity)]
    pub fn linear_problem<M: Matrix + 'static>() -> (
        Closure<
            M,
            impl Fn(&M::V, &M::V, M::T, &mut M::V),
            impl Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        >,
        M::T,
        M::V,
        Vec<LinearSolveSolution<M::V>>,
    ) {
        let diagonal = M::V::from_vec(
            vec![M::T::from_f64(2.0).unwrap(), M::T::from_f64(2.0).unwrap()],
            Default::default(),
        );
        let jac1 = M::from_diagonal(&diagonal);
        let jac2 = M::from_diagonal(&diagonal);
        let ctx = M::C::default();
        let p = M::V::zeros(0, ctx.clone());
        let mut op = Closure::new(
            // f = J * x
            move |x, _p, _t, y| jac1.gemv(M::T::one(), x, M::T::zero(), y),
            move |_x, _p, _t, v, y| jac2.gemv(M::T::one(), v, M::T::zero(), y),
            2,
            2,
            p.len(),
            ctx.clone(),
        );
        op.calculate_sparsity(
            &M::V::from_element(2, M::T::one(), ctx.clone()),
            M::T::zero(),
            &p,
        );
        let rtol = M::T::from_f64(1e-6).unwrap();
        let atol = M::V::from_vec(
            vec![M::T::from_f64(1e-6).unwrap(), M::T::from_f64(1e-6).unwrap()],
            ctx.clone(),
        );
        let solns = vec![LinearSolveSolution::new(
            M::V::from_vec(
                vec![M::T::from_f64(2.0).unwrap(), M::T::from_f64(4.0).unwrap()],
                ctx.clone(),
            ),
            M::V::from_vec(vec![M::T::one(), M::T::from_f64(2.0).unwrap()], ctx.clone()),
        )];
        (op, rtol, atol, solns)
    }

    pub fn test_linear_solver<'a, C>(
        mut solver: impl LinearSolver<C::M>,
        op: C,
        rtol: C::T,
        atol: &'a C::V,
        solns: Vec<LinearSolveSolution<C::V>>,
    ) where
        C: NonLinearOpJacobian,
        for<'b> &'b C::V: VectorRef<C::V>,
    {
        solver.set_problem(&op);
        let x = C::V::zeros(op.nout(), op.context().clone());
        let t = C::T::zero();
        solver.set_linearisation(&op, &x, t);
        for soln in solns {
            let x = solver.solve(&soln.b).unwrap();
            let tol = { &soln.x * scale(rtol) + atol };
            x.assert_eq(&soln.x, &tol);
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn linear_problem_batched<M: Matrix + 'static>(
        ctx: M::C,
    ) -> (
        Closure<
            M,
            impl Fn(&M::V, &M::V, M::T, &mut M::V),
            impl Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        >,
        M::T,
        M::V,
        Vec<LinearSolveSolution<M::V>>,
    ) {
        assert_eq!(ctx.nbatch(), 2);
        let two = M::T::from_f64(2.0).unwrap();
        let three = M::T::from_f64(3.0).unwrap();
        let diag = M::V::from_vec(vec![two, two, three, three], ctx.clone());
        let jac1 = M::from_diagonal(&diag);
        let jac2 = M::from_diagonal(&diag);
        let p = M::V::zeros(0, ctx.clone());
        let mut op = Closure::new(
            move |x, _p, _t, y| jac1.gemv(M::T::one(), x, M::T::zero(), y),
            move |_x, _p, _t, v, y| jac2.gemv(M::T::one(), v, M::T::zero(), y),
            2,
            2,
            p.len(),
            ctx.clone(),
        );
        op.calculate_sparsity(
            &M::V::from_element(2, M::T::one(), ctx.clone()),
            M::T::zero(),
            &p,
        );
        let rtol = M::T::from_f64(1e-6).unwrap();
        let atol_val = M::T::from_f64(1e-6).unwrap();
        let atol = M::V::from_vec(vec![atol_val; 4], ctx.clone());
        let one = M::T::one();
        let four = M::T::from_f64(4.0).unwrap();
        let six = M::T::from_f64(6.0).unwrap();
        let nine = M::T::from_f64(9.0).unwrap();
        let solns = vec![LinearSolveSolution::new(
            M::V::from_vec(vec![two, four, six, nine], ctx.clone()),
            M::V::from_vec(vec![one, two, two, three], ctx),
        )];
        (op, rtol, atol, solns)
    }

    #[test]
    fn test_lu_nalgebra() {
        let (op, rtol, atol, solns) = linear_problem::<NalgebraMat<f64>>();
        let p = NalgebraVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let s = NalgebraLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }

    #[test]
    fn test_lu_faer() {
        let (op, rtol, atol, solns) = linear_problem::<FaerMat<f64>>();
        let p = FaerVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let s = FaerLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }

    #[test]
    fn test_sparse_lu_faer() {
        use crate::FaerSparseMat;
        let (op, rtol, atol, solns) = linear_problem::<FaerSparseMat<f64>>();
        let p = FaerVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let s = super::FaerSparseLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }

    #[cfg(feature = "suitesparse")]
    #[test]
    fn test_klu() {
        use crate::FaerSparseMat;
        let (op, rtol, atol, solns) = linear_problem::<FaerSparseMat<f64>>();
        let p = FaerVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let s = super::KLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lu_cuda() {
        use crate::{CudaLU, CudaMat, CudaVec};
        let (op, rtol, atol, solns) = linear_problem::<CudaMat<f64>>();
        let p = CudaVec::zeros(0, op.context().clone());
        let op = ParameterisedOp::new(&op, &p);
        let s = CudaLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_lu_cuda_batched() {
        use crate::{CudaContext, CudaLU, CudaMat, CudaVec};
        let ctx = CudaContext::default().with_nbatch(2);
        let (op, rtol, atol, solns) = linear_problem_batched::<CudaMat<f64>>(ctx);
        let p = CudaVec::zeros(0, op.context().clone());
        let op = ParameterisedOp::new(&op, &p);
        let s = CudaLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }
}
