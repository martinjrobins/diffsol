use crate::{error::DiffsolError, Matrix, NonLinearOpJacobian};

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;

#[cfg(feature = "suitesparse")]
pub mod suitesparse;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use faer::lu::LU as FaerLU;
pub use nalgebra::lu::LU as NalgebraLU;

/// A solver for the linear problem `Ax = b`, where `A` is a linear operator that is obtained by taking the linearisation of a nonlinear operator `C`
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
        linear_solver::{FaerLU, NalgebraLU},
        matrix::dense_nalgebra_serial::NalgebraMat,
        op::{closure::Closure, ParameterisedOp},
        scalar::scale,
        vector::VectorRef,
        FaerMat, FaerVec, LinearSolver, Matrix, NalgebraVec, NonLinearOpJacobian, Op, Vector,
    };
    use num_traits::{One, Zero};

    use super::LinearSolveSolution;

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
        let diagonal = M::V::from_vec(vec![2.0.into(), 2.0.into()], Default::default());
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
        let rtol = M::T::from(1e-6);
        let atol = M::V::from_vec(vec![1e-6.into(), 1e-6.into()], ctx.clone());
        let solns = vec![LinearSolveSolution::new(
            M::V::from_vec(vec![2.0.into(), 4.0.into()], ctx.clone()),
            M::V::from_vec(vec![1.0.into(), 2.0.into()], ctx.clone()),
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

    #[test]
    fn test_lu_nalgebra() {
        let (op, rtol, atol, solns) = linear_problem::<NalgebraMat<f64>>();
        let p = NalgebraVec::zeros(0, op.context().clone());
        let op = ParameterisedOp::new(&op, &p);
        let s = NalgebraLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }
    #[test]
    fn test_lu_faer() {
        let (op, rtol, atol, solns) = linear_problem::<FaerMat<f64>>();
        let p = FaerVec::zeros(0, op.context().clone());
        let op = ParameterisedOp::new(&op, &p);
        let s = FaerLU::default();
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
}
