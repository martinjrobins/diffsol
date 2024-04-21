use crate::vector::Vector;
use crate::Scalar;
use crate::{matrix::DenseMatrix, op::NonLinearOp, VectorViewMut};
use num_traits::{One, Zero};

use self::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy};

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

/// Find the non-zero entries of the Jacobian matrix of a non-linear operator.
/// This is used as the default `find_non_zeros` function for the `NonLinearOp` and `LinearOp` traits.
/// Users can override this function with a more efficient and reliable implementation if desired.
pub fn find_non_zeros<F: NonLinearOp + ?Sized>(op: &F, x: &F::V, t: F::T) -> Vec<(usize, usize)> {
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

/// Find the non-zero entries of the Jacobian matrix of a non-linear operator.
/// This is used in the default `jacobian` method of the `NonLinearOp` and `LinearOp` traits.
pub fn find_non_zero_entries<F: NonLinearOp + ?Sized>(
    op: &F,
    x: &F::V,
    t: F::T,
) -> Vec<(usize, usize, F::T)> {
    let mut v = F::V::zeros(op.nstates());
    let mut col = F::V::zeros(op.nout());
    let mut triplets = Vec::with_capacity(op.nstates());
    for j in 0..op.nstates() {
        v[j] = F::T::one();
        op.jac_mul_inplace(x, t, &v, &mut col);
        for i in 0..op.nout() {
            if col[i] != F::T::zero() {
                triplets.push((i, j, col[i]));
            }
        }
        v[j] = F::T::zero();
    }
    triplets
}

/// Calculate the dense Jacobian matrix of a non-linear operator, overwrites a given dense matrix `y`.
pub fn jacobian_dense<F: NonLinearOp + ?Sized>(op: &F, x: &F::V, t: F::T, y: &mut F::M)
where
    F::M: DenseMatrix,
{
    let mut v = F::V::zeros(op.nstates());
    let mut col = F::V::zeros(op.nout());
    for j in 0..op.nstates() {
        v[j] = F::T::one();
        // TODO: should be able to just give col_dest here!
        op.jac_mul_inplace(x, t, &v, &mut col);
        let mut col_dest = y.column_mut(j);
        col_dest.copy_from(&col);
        v[j] = F::T::zero();
    }
}

pub struct JacobianColoring {
    cols_per_color: Vec<Vec<usize>>,
    ij_per_color: Vec<Vec<(usize, usize)>>,
}

impl JacobianColoring {
    pub fn new<F: NonLinearOp>(op: &F, x: &F::V, t: F::T) -> Self {
        let non_zeros = op.find_non_zeros(x, t);
        let ncols = op.nstates();
        let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
        let coloring = color_graph_greedy(&graph);
        let max_color = coloring.iter().max().copied().unwrap_or(0);
        let mut cols_per_color = vec![Vec::new(); max_color];
        let mut ij_per_color = vec![Vec::new(); max_color];
        for c in 1..=max_color {
            for (i, j) in non_zeros.iter() {
                if coloring[*j] == c {
                    cols_per_color[c - 1].push(*j);
                    ij_per_color[c - 1].push((*i, *j));
                }
            }
        }
        Self {
            cols_per_color,
            ij_per_color,
        }
    }

    pub fn find_non_zero_entries<F: NonLinearOp>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
    ) -> Vec<(usize, usize, F::T)> {
        let mut triplets = Vec::with_capacity(op.nstates());
        let mut v = F::V::zeros(op.nstates());
        let mut col = F::V::zeros(op.nout());
        for (cols, ijs) in self.cols_per_color.iter().zip(self.ij_per_color.iter()) {
            for j in cols {
                v[*j] = F::T::one();
            }
            op.jac_mul_inplace(x, t, &v, &mut col);
            for (i, j) in ijs {
                if col[*i] != F::T::zero() {
                    triplets.push((*i, *j, col[*i]));
                }
            }
            for j in cols {
                v[*j] = F::T::zero();
            }
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
            let x = DVector::from_vec(vec![1.0, 1.0]);
            let t = 0.0;
            let non_zeros = op.find_non_zeros(&x, t);
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
            let x = DVector::from_vec(vec![1.0, 1.0]);
            let t = 0.0;

            let non_zeros = op.find_non_zeros(&x, t);
            let ncols = op.nstates();
            let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
            let coloring = color_graph_greedy(&graph);

            assert_eq!(coloring, expect);
        }
    }
}
