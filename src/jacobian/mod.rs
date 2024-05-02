use std::collections::HashSet;

use crate::op::{LinearOp, Op};
use crate::vector::Vector;
use crate::Scalar;
use crate::{op::NonLinearOp, Matrix, MatrixSparsity, VectorIndex};
use num_traits::{One, Zero};

use self::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy};

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

/// Find the non-zero entries of the Jacobian matrix of a non-linear operator.
/// This is used as the default `find_non_zeros` function for the `NonLinearOp` and `LinearOp` traits.
/// Users can override this function with a more efficient and reliable implementation if desired.
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
    use crate::matrix::Matrix;
    use crate::op::linear_closure::LinearClosure;
    use crate::op::{LinearOp, Op};
    use crate::vector::Vector;
    use crate::NonLinearOp;
    use crate::{
        jacobian::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy},
        op::closure::Closure,
    };
    use nalgebra::{DMatrix, DVector};
    use std::ops::MulAssign;

    fn helper_triplets2op_nonlinear(
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
        let mut ret = Closure::new(
            move |x: &DVector<f64>, _p: &DVector<f64>, _t, y: &mut DVector<f64>| {
                y.fill(0.0);
                f(x, y);
            },
            move |_x: &DVector<f64>, _p: &DVector<f64>, _t, v, y: &mut DVector<f64>| {
                y.fill(0.0);
                f(v, y);
            },
            nstates,
            nout,
            Rc::new(DVector::zeros(0)),
        );
        let y0 = DVector::zeros(nstates);
        let t0 = 0.0;
        ret.calculate_sparsity(&y0, t0);
        ret
    }

    fn helper_triplets2op_linear(
        triplets: &[(usize, usize, f64)],
        nrows: usize,
        ncols: usize,
    ) -> impl LinearOp<M = DMatrix<f64>, V = DVector<f64>, T = f64> + '_ {
        let nstates = ncols;
        let nout = nrows;
        let f = move |x: &DVector<f64>, y: &mut DVector<f64>| {
            for (i, j, v) in triplets {
                y[*i] += x[*j] * v;
            }
        };
        let mut ret = LinearClosure::new(
            move |x: &DVector<f64>, _p: &DVector<f64>, _t, beta, y: &mut DVector<f64>| {
                y.mul_assign(beta);
                f(x, y);
            },
            nstates,
            nout,
            Rc::new(DVector::zeros(0)),
        );
        let t0 = 0.0;
        ret.calculate_sparsity(t0);
        ret
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
            let op = helper_triplets2op_nonlinear(triplets.as_slice(), 2, 2);
            let non_zeros = find_non_zeros_nonlinear(&op, &DVector::zeros(2), 0.0);
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
            let op = helper_triplets2op_nonlinear(triplets.as_slice(), 2, 2);
            let non_zeros = find_non_zeros_nonlinear(&op, &DVector::zeros(2), 0.0);
            let ncols = op.nstates();
            let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
            let coloring = color_graph_greedy(&graph);

            assert_eq!(coloring, expect);
        }
    }

    #[test]
    fn matrix_coloring() {
        let test_triplets = vec![
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
            vec![(0, 0, 1.0), (1, 1, 1.0)],
            vec![(0, 0, 0.9), (1, 0, 2.0), (1, 1, 1.1), (2, 2, 1.4)],
        ];
        type V = DVector<f64>;
        type M = DMatrix<f64>;
        let n = 3;

        // test nonlinear functions
        for triplets in test_triplets.iter() {
            let op = helper_triplets2op_nonlinear(triplets.as_slice(), n, n);
            let y0 = V::zeros(n);
            let t0 = 0.0;
            let non_zeros = find_non_zeros_nonlinear(&op, &y0, t0);
            let coloring = JacobianColoring::new_from_non_zeros(&op, non_zeros);
            let mut jac = M::zeros(3, 3);
            coloring.jacobian_inplace(&op, &y0, t0, &mut jac);
            let mut gemv1 = V::zeros(n);
            let v = V::from_element(3, 1.0);
            op.jac_mul_inplace(&y0, t0, &v, &mut gemv1);
            let mut gemv2 = V::zeros(n);
            jac.gemv(1.0, &v, 0.0, &mut gemv2);
            gemv1.assert_eq_st(&gemv2, 1e-10);
        }

        // test linear functions
        for triplets in test_triplets {
            let op = helper_triplets2op_linear(triplets.as_slice(), n, n);
            let t0 = 0.0;
            let non_zeros = find_non_zeros_linear(&op, t0);
            let coloring = JacobianColoring::new_from_non_zeros(&op, non_zeros);
            let mut jac = M::zeros(3, 3);
            coloring.matrix_inplace(&op, t0, &mut jac);
            let mut gemv1 = V::zeros(n);
            let v = V::from_element(3, 1.0);
            op.gemv_inplace(&v, t0, 0.0, &mut gemv1);
            let mut gemv2 = V::zeros(n);
            jac.gemv(1.0, &v, 0.0, &mut gemv2);
            gemv1.assert_eq_st(&gemv2, 1e-10);
        }
    }
}
