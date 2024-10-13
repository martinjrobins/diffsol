use crate::{Matrix, LinearOp, Op};


pub struct MatrixOp<M: Matrix> {
    m: M,
}

impl<M: Matrix> MatrixOp<M> {
    pub fn new(m: M) -> Self {
        Self { m }
    }
}

impl<M: Matrix> Op for MatrixOp<M> {
    type V = M::V;
    type T = M::T;
    type M = M;
    fn nstates(&self) -> usize {
        self.m.nrows()
    }
    fn nout(&self) -> usize {
        self.m.ncols()
    }
    fn nparams(&self) -> usize {
        0
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.m.sparsity()
    }
}

impl<M: Matrix> LinearOp for MatrixOp<M> {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        self.m.gemv(t, x, beta, y);
    }
}
