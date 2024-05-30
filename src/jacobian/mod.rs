use std::collections::HashSet;

use crate::op::{LinearOp, Op};
use crate::vector::Vector;
use crate::Scalar;
use crate::{op::NonLinearOp, Matrix, MatrixSparsityRef, VectorIndex};
use num_traits::{One, Zero};

use self::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy};

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

/// Find the non-zero entries of the Jacobian matrix of a non-linear operator.
pub fn find_non_zeros_nonlinear<F: NonLinearOp + ?Sized>(
    op: &F,
    x: &F::V,
    t: F::T,
) -> Vec<(usize, usize)> {
    let mut v = F::V::zeros(op.nstates());
    let mut col = F::V::zeros(op.nout());
    let mut triplets = Vec::with_capacity(op.nstates());
    for j in 0..op.nstates() {
        v[j] = F::T::NAN;
        op.jac_mul_inplace(x, t, &v, &mut col);
        for i in 0..op.nout() {
            if col[i].is_nan() {
                triplets.push((i, j));
            }
            col[i] = F::T::zero();
        }
        v[j] = F::T::zero();
    }
    triplets
}

/// Find the non-zero entries of the matrix of a linear operator.
pub fn find_non_zeros_linear<F: LinearOp + ?Sized>(op: &F, t: F::T) -> Vec<(usize, usize)> {
    let mut v = F::V::zeros(op.nstates());
    let mut col = F::V::zeros(op.nout());
    let mut triplets = Vec::with_capacity(op.nstates());
    for j in 0..op.nstates() {
        v[j] = F::T::NAN;
        op.call_inplace(&v, t, &mut col);
        for i in 0..op.nout() {
            if col[i].is_nan() {
                triplets.push((i, j));
            }
            col[i] = F::T::zero();
        }
        v[j] = F::T::zero();
    }
    triplets
}

pub struct JacobianColoring<M: Matrix> {
    dst_indices_per_color: Vec<<M::V as Vector>::Index>,
    src_indices_per_color: Vec<<M::V as Vector>::Index>,
    input_indices_per_color: Vec<<M::V as Vector>::Index>,
}

impl<M: Matrix> JacobianColoring<M> {
    pub fn new_from_non_zeros<F: Op<M = M>>(op: &F, non_zeros: Vec<(usize, usize)>) -> Self {
        let sparsity = op
            .sparsity()
            .expect("Jacobian sparsity not defined, cannot use coloring");
        let ncols = op.nstates();
        let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
        let coloring = color_graph_greedy(&graph);
        let max_color = coloring.iter().max().copied().unwrap_or(0);
        let mut dst_indices_per_color = Vec::new();
        let mut src_indices_per_color = Vec::new();
        let mut input_indices_per_color = Vec::new();
        for c in 1..=max_color {
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            for (i, j) in non_zeros.iter() {
                if coloring[*j] == c {
                    rows.push(*i);
                    cols.push(*j);
                }
            }
            let dst_indices = sparsity.get_index(rows.as_slice(), cols.as_slice());
            let src_indices = <M::V as Vector>::Index::from_slice(rows.as_slice());
            let unique_cols: HashSet<_> = HashSet::from_iter(cols.iter().cloned());
            let unique_cols = unique_cols.into_iter().collect::<Vec<_>>();
            let input_indices = <M::V as Vector>::Index::from_slice(unique_cols.as_slice());
            dst_indices_per_color.push(dst_indices);
            src_indices_per_color.push(src_indices);
            input_indices_per_color.push(input_indices);
        }
        Self {
            dst_indices_per_color,
            src_indices_per_color,
            input_indices_per_color,
        }
    }

    //pub fn new_non_linear<F: NonLinearOp<M = M>>(op: &F) -> Self {
    //    let non_zeros = find_non_zeros_nonlinear(op, &F::V::zeros(op.nstates()), F::T::zero());
    //    Self::new_from_non_zeros(op, non_zeros)
    //}

    //pub fn new_linear<F: LinearOp<M = M>>(op: &F) -> Self {
    //    let non_zeros = find_non_zeros_linear(op, F::T::zero());
    //    Self::new_from_non_zeros(op, non_zeros)
    //}

    pub fn jacobian_inplace<F: NonLinearOp<M = M, V = M::V, T = M::T>>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = F::V::zeros(op.nstates());
        let mut col = F::V::zeros(op.nout());
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.jac_mul_inplace(x, t, &v, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::zero());
        }
    }

    pub fn matrix_inplace<F: LinearOp<M = M, V = M::V, T = M::T>>(
        &self,
        op: &F,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = F::V::zeros(op.nstates());
        let mut col = F::V::zeros(op.nout());
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.call_inplace(&v, t, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::zero());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::jacobian::{find_non_zeros_linear, find_non_zeros_nonlinear, JacobianColoring};
    use crate::matrix::sparsity::MatrixSparsityRef;
    use crate::matrix::Matrix;
    use crate::op::linear_closure::LinearClosure;
    use crate::op::{LinearOp, Op};
    use crate::vector::Vector;
    use crate::{
        jacobian::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy},
        op::closure::Closure,
    };
    use crate::{scale, NonLinearOp, SparseColMat};
    use nalgebra::DMatrix;
    use num_traits::{One, Zero};
    use std::ops::MulAssign;

    fn helper_triplets2op_nonlinear<'a, M: Matrix + 'a>(
        triplets: &'a [(usize, usize, M::T)],
        nrows: usize,
        ncols: usize,
    ) -> impl NonLinearOp<M = M, V = M::V, T = M::T> + '_ {
        let nstates = ncols;
        let nout = nrows;
        let f = move |x: &M::V, y: &mut M::V| {
            for (i, j, v) in triplets {
                y[*i] += x[*j] * *v;
            }
        };
        let mut ret = Closure::new(
            move |x: &M::V, _p: &M::V, _t, y: &mut M::V| {
                y.fill(M::T::zero());
                f(x, y);
            },
            move |_x: &M::V, _p: &M::V, _t, v, y: &mut M::V| {
                y.fill(M::T::zero());
                f(v, y);
            },
            nstates,
            nout,
            Rc::new(M::V::zeros(0)),
        );
        let y0 = M::V::zeros(nstates);
        let t0 = M::T::zero();
        ret.calculate_sparsity(&y0, t0);
        ret
    }

    fn helper_triplets2op_linear<'a, M: Matrix + 'a>(
        triplets: &'a [(usize, usize, M::T)],
        nrows: usize,
        ncols: usize,
    ) -> impl LinearOp<M = M, V = M::V, T = M::T> + '_ {
        let nstates = ncols;
        let nout = nrows;
        let f = move |x: &M::V, y: &mut M::V| {
            for (i, j, v) in triplets {
                y[*i] += x[*j] * *v;
            }
        };
        let mut ret = LinearClosure::new(
            move |x: &M::V, _p: &M::V, _t, beta, y: &mut M::V| {
                y.mul_assign(scale(beta));
                f(x, y);
            },
            nstates,
            nout,
            Rc::new(M::V::zeros(0)),
        );
        let t0 = M::T::zero();
        ret.calculate_sparsity(t0);
        ret
    }

    fn find_non_zeros<M: Matrix>() {
        let test_triplets = vec![
            vec![(0, 0, M::T::one()), (1, 1, M::T::one())],
            vec![
                (0, 0, M::T::one()),
                (0, 1, M::T::one()),
                (1, 1, M::T::one()),
            ],
            vec![(1, 1, M::T::one())],
            vec![
                (0, 0, M::T::one()),
                (1, 0, M::T::one()),
                (0, 1, M::T::one()),
                (1, 1, M::T::one()),
            ],
        ];
        for triplets in test_triplets {
            let op = helper_triplets2op_nonlinear::<M>(triplets.as_slice(), 2, 2);
            let non_zeros = find_non_zeros_nonlinear(&op, &M::V::zeros(2), M::T::zero());
            let expect = triplets
                .iter()
                .map(|(i, j, _v)| (*i, *j))
                .collect::<Vec<_>>();
            assert_eq!(non_zeros, expect);
        }
    }

    #[test]
    fn find_non_zeros_dmatrix() {
        find_non_zeros::<DMatrix<f64>>();
    }

    #[test]
    fn find_non_zeros_faer_sparse() {
        find_non_zeros::<SparseColMat<f64>>();
    }

    fn build_coloring<M: Matrix>() {
        let test_triplets = [
            vec![(0, 0, M::T::one()), (1, 1, M::T::one())],
            vec![
                (0, 0, M::T::one()),
                (0, 1, M::T::one()),
                (1, 1, M::T::one()),
            ],
            vec![(1, 1, M::T::one())],
            vec![
                (0, 0, M::T::one()),
                (1, 0, M::T::one()),
                (0, 1, M::T::one()),
                (1, 1, M::T::one()),
            ],
        ];
        let expect = vec![vec![1, 1], vec![1, 2], vec![1, 1], vec![1, 2]];
        for (triplets, expect) in test_triplets.iter().zip(expect) {
            let op = helper_triplets2op_nonlinear::<M>(triplets.as_slice(), 2, 2);
            let non_zeros = find_non_zeros_nonlinear(&op, &M::V::zeros(2), M::T::zero());
            let ncols = op.nstates();
            let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
            let coloring = color_graph_greedy(&graph);

            assert_eq!(coloring, expect);
        }
    }

    #[test]
    fn build_coloring_dmatrix() {
        build_coloring::<DMatrix<f64>>();
    }

    #[test]
    fn build_coloring_faer_sparse() {
        build_coloring::<SparseColMat<f64>>();
    }

    fn matrix_coloring<M: Matrix>() {
        let test_triplets = vec![
            vec![
                (0, 0, M::T::one()),
                (1, 1, M::T::one()),
                (2, 2, M::T::one()),
            ],
            vec![(0, 0, M::T::one()), (1, 1, M::T::one())],
            vec![
                (0, 0, M::T::from(0.9)),
                (1, 0, M::T::from(2.0)),
                (1, 1, M::T::from(1.1)),
                (2, 2, M::T::from(1.4)),
            ],
        ];
        let n = 3;

        // test nonlinear functions
        for triplets in test_triplets.iter() {
            let op = helper_triplets2op_nonlinear::<M>(triplets.as_slice(), n, n);
            let y0 = M::V::zeros(n);
            let t0 = M::T::zero();
            let non_zeros = find_non_zeros_nonlinear(&op, &y0, t0);
            let coloring = JacobianColoring::new_from_non_zeros(&op, non_zeros);
            let mut jac = M::new_from_sparsity(3, 3, op.sparsity().map(|s| s.to_owned()));
            coloring.jacobian_inplace(&op, &y0, t0, &mut jac);
            let mut gemv1 = M::V::zeros(n);
            let v = M::V::from_element(3, M::T::one());
            op.jac_mul_inplace(&y0, t0, &v, &mut gemv1);
            let mut gemv2 = M::V::zeros(n);
            jac.gemv(M::T::one(), &v, M::T::zero(), &mut gemv2);
            gemv1.assert_eq_st(&gemv2, M::T::from(1e-10));
        }

        // test linear functions
        for triplets in test_triplets {
            let op = helper_triplets2op_linear::<M>(triplets.as_slice(), n, n);
            let t0 = M::T::zero();
            let non_zeros = find_non_zeros_linear(&op, t0);
            let coloring = JacobianColoring::new_from_non_zeros(&op, non_zeros);
            let mut jac = M::new_from_sparsity(3, 3, op.sparsity().map(|s| s.to_owned()));
            coloring.matrix_inplace(&op, t0, &mut jac);
            let mut gemv1 = M::V::zeros(n);
            let v = M::V::from_element(3, M::T::one());
            op.gemv_inplace(&v, t0, M::T::zero(), &mut gemv1);
            let mut gemv2 = M::V::zeros(n);
            jac.gemv(M::T::one(), &v, M::T::zero(), &mut gemv2);
            gemv1.assert_eq_st(&gemv2, M::T::from(1e-10));
        }
    }

    #[test]
    fn matrix_coloring_dmatrix() {
        matrix_coloring::<DMatrix<f64>>();
    }

    #[test]
    fn matrix_coloring_faer_sparse() {
        matrix_coloring::<SparseColMat<f64>>();
    }
}
