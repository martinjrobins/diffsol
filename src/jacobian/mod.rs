use crate::op::{LinearOp, Op};
use crate::vector::Vector;
use crate::Scalar;
use crate::{MatrixSparsity, op::NonLinearOp, Matrix, VectorIndex};
use num_traits::{One, Zero};

use self::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy};

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

/// Find the non-zero entries of the Jacobian matrix of a non-linear operator.
/// This is used as the default `find_non_zeros` function for the `NonLinearOp` and `LinearOp` traits.
/// Users can override this function with a more efficient and reliable implementation if desired.
pub fn find_non_zeros_nonlinear<F: NonLinearOp + ?Sized>(op: &F, x: &F::V, t: F::T) -> Vec<(usize, usize)> {
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
/// This is used as the default `find_non_zeros` function for the `NonLinearOp` and `LinearOp` traits.
/// Users can override this function with a more efficient and reliable implementation if desired.
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
    dst_indices_per_color: Vec<<M::Sparsity as MatrixSparsity>::Index>,
    src_indices_per_color: Vec<<M::V as Vector>::Index>,
    input_indices_per_color: Vec<<M::V as Vector>::Index>,
}

impl<M: Matrix> JacobianColoring<M> {
    pub fn new<F: Op<M=M>>(op: &F) -> Self {
        let sparsity = op.sparsity().expect("Jacobian sparsity not defined, cannot use coloring");
        let non_zeros = sparsity.indices();
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
            let input_indices = <M::V as Vector>::Index::from_slice(cols.as_slice());
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

    pub fn jacobian_inplace<F: NonLinearOp<M=M, V=M::V, T=M::T>>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
        y: &mut F::M,
    ) -> Vec<(usize, usize, F::T)> {
        let triplets = Vec::with_capacity(op.nstates());
        let mut v = F::V::zeros(op.nstates());
        let mut col = F::V::zeros(op.nout());
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.jac_mul_inplace(x, t, &v, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::one());
        }
        triplets
    }

    pub fn matrix_inplace<F: LinearOp<M=M, V=M::V, T=M::T>>(
        &self,
        op: &F,
        t: F::T,
        y: &mut F::M,
    ) -> Vec<(usize, usize, F::T)> {
        let triplets = Vec::with_capacity(op.nstates());
        let mut v = F::V::zeros(op.nstates());
        let mut col = F::V::zeros(op.nout());
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.call_inplace(&v, t, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::one());
        }
        triplets
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::op::Op;
    use crate::{
        jacobian::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy},
        op::{closure::Closure, NonLinearOp},
        MatrixSparsity,
    };
    use nalgebra::{DMatrix, DVector};

    fn helper_triplets2op(
        triplets: &[(usize, usize, f64)],
        nrows: usize,
        ncols: usize,
    ) -> impl NonLinearOp<M = DMatrix<f64>, V = DVector<f64>, T = f64> + '_ {
        let nstates = ncols;
        let nout = nrows;
        let f = move |x: &DVector<f64>, y: &mut DVector<f64>| {
            for (i, j, v) in triplets {
                y[*i] += x[*j] * v;
            }
        };
        Closure::new(
            move |x: &DVector<f64>, _p: &DVector<f64>, _t, y| {
                f(x, y);
            },
            move |_x: &DVector<f64>, _p: &DVector<f64>, _t, v, y| {
                f(v, y);
            },
            nstates,
            nout,
            Rc::new(DVector::zeros(0)),
        )
    }
    #[test]
    fn find_non_zeros() {
        let test_triplets = vec![
            vec![(0, 0, 1.0), (1, 1, 1.0)],
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
            vec![(1, 1, 1.0)],
            vec![(0, 0, 1.0), (1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
        ];
        for triplets in test_triplets {
            let op = helper_triplets2op(triplets.as_slice(), 2, 2);
            let non_zeros = op.sparsity().unwrap().indices();
            let expect = triplets
                .iter()
                .map(|(i, j, _v)| (*i, *j))
                .collect::<Vec<_>>();
            assert_eq!(non_zeros, expect);
        }
    }

    #[test]
    fn build_coloring() {
        let test_triplets = [
            vec![(0, 0, 1.0), (1, 1, 1.0)],
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
            vec![(1, 1, 1.0)],
            vec![(0, 0, 1.0), (1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
        ];
        let expect = vec![vec![1, 1], vec![1, 2], vec![1, 1], vec![1, 2]];
        for (triplets, expect) in test_triplets.iter().zip(expect) {
            let op = helper_triplets2op(triplets.as_slice(), 2, 2);

            let non_zeros = op.sparsity().unwrap().indices();
            let ncols = op.nstates();
            let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
            let coloring = color_graph_greedy(&graph);

            assert_eq!(coloring, expect);
        }
    }
}
