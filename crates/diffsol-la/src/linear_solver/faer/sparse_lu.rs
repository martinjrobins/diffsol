use std::cell::RefCell;

use crate::context::broadcast_batch;
use crate::{
    error::LaError, linear_solver::LinearSolver, linear_solver_error, scalar::IndexType, Context,
    FaerContext, FaerScalar, FaerSparseMat, FaerVec, LinearOp, Matrix,
};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuRef, NumericLu, SymbolicLu};
use faer::Conj;

/// Sparse $LU$ factorization state for a (possibly batched) [FaerSparseMat].
///
/// The symbolic factorization, one numeric factorization per batch and both scratch buffers are
/// allocated once, in [`FaerSparseLU::set_sparsity`].  faer sizes a [`NumericLu`]'s own buffers on
/// its first `factorize_numeric_lu` rather than on construction, so the first
/// [`FaerSparseLU::set_linearisation`] after that still allocates inside faer; later ones reuse
/// the capacity.
struct SparseLu<T: FaerScalar> {
    a: FaerSparseMat<T>,
    symbolic: SymbolicLu<IndexType>,
    numeric: Vec<NumericLu<IndexType, T>>,
    factor_scratch: MemBuffer,
    solve_scratch: RefCell<MemBuffer>,
}

impl<T: FaerScalar> SparseLu<T> {
    fn new(a: FaerSparseMat<T>) -> Self {
        let symbolic = factorize_symbolic_lu(a.data[0].symbolic(), Default::default())
            .expect("Failed to create symbolic LU");
        let par = a.context.par;
        // the solve always runs one column at a time, so this size holds however many batches the
        // right-hand side broadcasts into
        let factor_scratch =
            MemBuffer::new(symbolic.factorize_numeric_lu_scratch::<T>(par, Default::default()));
        let solve_scratch =
            RefCell::new(MemBuffer::new(symbolic.solve_in_place_scratch::<T>(1, par)));
        Self {
            numeric: (0..a.data.len()).map(|_| NumericLu::new()).collect(),
            a,
            symbolic,
            factor_scratch,
            solve_scratch,
        }
    }

    fn nbatch(&self) -> usize {
        self.numeric.len()
    }

    fn factor(&mut self) {
        let par = self.a.context.par;
        for (numeric, block) in self.numeric.iter_mut().zip(self.a.data.iter()) {
            self.symbolic
                .factorize_numeric_lu(
                    numeric,
                    block.rb(),
                    par,
                    MemStack::new(&mut self.factor_scratch),
                    Default::default(),
                )
                .expect("Failed to factorise matrix");
        }
    }

    fn solve_block(&self, batch: usize, rhs: faer::MatMut<'_, T>, par: faer::Par) {
        LuRef::new_unchecked(&self.symbolic, &self.numeric[batch]).solve_in_place_with_conj(
            Conj::No,
            rhs,
            par,
            MemStack::new(&mut self.solve_scratch.borrow_mut()),
        );
    }
}

/// A [LinearSolver] that uses the LU decomposition in the [`faer`](https://github.com/sarah-ek/faer-rs) library to solve the linear system.
///
/// The low-level `faer::sparse::linalg::lu` API is used rather than
/// `faer::sparse::linalg::solvers` both so that the parallelism from [`FaerContext::par`] is
/// honoured -- the wrapper types hardcode faer's process-global setting -- and so that the
/// factorizations and scratch space can be allocated once in [`Self::set_sparsity`] and reused.
pub struct FaerSparseLU<T>
where
    T: FaerScalar,
{
    lu: Option<SparseLu<T>>,
}

impl<T> Default for FaerSparseLU<T>
where
    T: FaerScalar,
{
    fn default() -> Self {
        Self { lu: None }
    }
}

impl<T: FaerScalar> LinearSolver<FaerSparseMat<T>> for FaerSparseLU<T> {
    fn set_linearisation<
        C: LinearOp<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let lu = self.lu.as_mut().expect("Matrix not set");
        op.matrix_inplace(&mut lu.a);
        lu.factor();
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), LaError> {
        let Some(lu) = self.lu.as_ref() else {
            return Err(linear_solver_error!(LuNotInitialized));
        };
        x.context
            .assert_broadcastable_into(lu.nbatch(), "sparse_lu_solve");
        let par = x.context.par;
        let nb = x.data.ncols();
        for batch in 0..nb {
            lu.solve_block(
                broadcast_batch(batch, lu.nbatch(), nb),
                x.data.rb_mut().col_mut(batch).as_mat_mut(),
                par,
            );
        }
        Ok(())
    }

    fn set_sparsity<C: LinearOp<T = T, V = FaerVec<T>, M = FaerSparseMat<T>, C = FaerContext>>(
        &mut self,
        op: &C,
    ) {
        let ncols = op.ncols();
        let nrows = op.nrows();
        let a = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), *op.context());
        self.lu = Some(SparseLu::new(a));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linear_solver::tests::{diagonal_op, test_grouped_lu_solve, test_narrow_state_lu_solve},
        Vector,
    };

    #[test]
    fn test_sparse_lu() {
        let mut s = FaerSparseLU::<f64>::default();
        let op = diagonal_op::<FaerSparseMat<f64>>(2.0);
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = FaerVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &FaerVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }

    #[test]
    fn test_grouped_sparse_lu() {
        test_grouped_lu_solve::<FaerSparseMat<f64>, FaerSparseLU<f64>>(FaerContext::with_nbatch(2));
    }

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_narrow_state_sparse_lu() {
        test_narrow_state_lu_solve::<FaerSparseMat<f64>, FaerSparseLU<f64>>(
            FaerContext::with_nbatch(2),
        );
    }
}
