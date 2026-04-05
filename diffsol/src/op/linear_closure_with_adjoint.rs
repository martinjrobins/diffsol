use std::cell::RefCell;

use crate::{
    find_matrix_non_zeros, find_transpose_non_zeros, jacobian::JacobianColoring,
    matrix::sparsity::MatrixSparsity, LinearOp, LinearOpTranspose, Matrix, Op,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

pub struct LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    func: F,
    func_adjoint: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    coloring_adjoint: Option<JacobianColoring<M>>,
    sparsity_adjoint: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F, G> LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    pub fn new(
        func: F,
        func_adjoint: G,
        nstates: usize,
        nout: usize,
        nparams: usize,
        ctx: M::C,
    ) -> Self {
        Self {
            func,
            func_adjoint,
            nstates,
            statistics: RefCell::new(OpStatistics::default()),
            nout,
            nparams,
            coloring: None,
            sparsity: None,
            coloring_adjoint: None,
            sparsity_adjoint: None,
            ctx,
        }
    }

    pub fn calculate_sparsity(&mut self, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_matrix_non_zeros(&op, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new(
            self.sparsity.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }
    pub fn calculate_adjoint_sparsity(&mut self, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_transpose_non_zeros(&op, t0);
        self.sparsity_adjoint = Some(
            MatrixSparsity::try_from_indices(self.nstates, self.nout, non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring_adjoint = Some(JacobianColoring::new(
            self.sparsity_adjoint.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }
}

impl<M, F, G> Op for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }

    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, G> BuilderOp for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_sparsity(t0, p);
        self.calculate_adjoint_sparsity(t0, p);
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
    }
}

impl<M, F, G> LinearOp for ParameterisedOp<'_, LinearClosureWithAdjoint<M, F, G>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, beta, y)
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.op.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = &self.op.coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity.clone()
    }
}

impl<M, F, G> LinearOpTranspose for ParameterisedOp<'_, LinearClosureWithAdjoint<M, F, G>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn gemv_transpose_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        (self.op.func_adjoint)(x, self.p, t, beta, y)
    }
    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.op.coloring_adjoint {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_transpose_inplace(t, y);
        }
    }

    fn transpose_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity_adjoint.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, matrix::dense_nalgebra_serial::NalgebraMat,
        matrix::Matrix, DenseMatrix, LinearOp, LinearOpTranspose, Op, Vector,
    };

    use super::{super::BuilderOp, LinearClosureWithAdjoint};

    type M = NalgebraMat<f64>;
    type V = crate::NalgebraVec<f64>;

    fn forward(x: &V, p: &V, _t: f64, beta: f64, y: &mut V) {
        let out = V::from_vec(
            vec![
                p.get_index(0) * x.get_index(0),
                x.get_index(0) + p.get_index(1) * x.get_index(1),
            ],
            NalgebraContext,
        );
        y.axpy(1.0, &out, beta);
    }

    fn adjoint(x: &V, p: &V, _t: f64, beta: f64, y: &mut V) {
        let out = V::from_vec(
            vec![
                p.get_index(0) * x.get_index(0) + x.get_index(1),
                p.get_index(1) * x.get_index(1),
            ],
            NalgebraContext,
        );
        y.axpy(1.0, &out, beta);
    }

    fn make_op() -> LinearClosureWithAdjoint<M, fn(&V, &V, f64, f64, &mut V), fn(&V, &V, f64, f64, &mut V)> {
        LinearClosureWithAdjoint::new(forward, adjoint, 2, 2, 2, NalgebraContext)
    }

    #[test]
    fn linear_closure_with_adjoint_builds_matrices_and_tracks_statistics() {
        let mut op = make_op();
        op.set_nstates(2);
        op.set_nout(2);
        op.set_nparams(2);

        let y0 = V::from_vec(vec![1.0, 1.0], NalgebraContext);
        let p = V::from_vec(vec![2.0, 3.0], NalgebraContext);
        BuilderOp::calculate_sparsity(&mut op, &y0, 0.0, &p);

        assert_eq!(op.nstates(), 2);
        assert_eq!(op.nout(), 2);
        assert_eq!(op.nparams(), 2);

        let pop = crate::ParameterisedOp::new(&op, &p);
        let matrix = pop.matrix(0.0);
        assert_eq!(matrix.get_index(0, 0), 2.0);
        assert_eq!(matrix.get_index(1, 0), 1.0);
        assert_eq!(matrix.get_index(0, 1), 0.0);
        assert_eq!(matrix.get_index(1, 1), 3.0);
        assert!(pop.sparsity().is_some());

        let mut transpose = M::zeros(2, 2, NalgebraContext);
        pop.transpose_inplace(0.0, &mut transpose);
        assert_eq!(transpose.get_index(0, 0), 2.0);
        assert_eq!(transpose.get_index(1, 0), 0.0);
        assert_eq!(transpose.get_index(0, 1), 0.0);
        assert_eq!(transpose.get_index(1, 1), 3.0);
        assert!(pop.transpose_sparsity().is_some());

        let x = V::from_vec(vec![4.0, 5.0], NalgebraContext);
        let mut y = V::from_vec(vec![1.0, 1.0], NalgebraContext);
        pop.gemv_inplace(&x, 0.0, 0.5, &mut y);
        y.assert_eq_st(&V::from_vec(vec![8.5, 19.5], NalgebraContext), 1e-12);

        let stats = pop.statistics();
        assert!(stats.number_of_calls >= 1);
        assert!(stats.number_of_matrix_evals >= 1);
    }
}
