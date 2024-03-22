use crate::matrix::Matrix;
use crate::op::NonLinearOp;
use crate::vector::Vector;
use crate::Scalar;
use anyhow::Result;
use num_traits::{One, Zero};

use self::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy};

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

pub struct Jacobian<'a, F: NonLinearOp + ?Sized> {
    op: &'a F,
    x: &'a F::V,
    t: F::T,
    coloring: Option<Vec<usize>>,
}

impl<'a, F: NonLinearOp> Jacobian<'a, F> {
    pub fn new(op: &'a F, x: &'a F::V, t: F::T) -> Self {
        let coloring = None;
        Self { op, x, t, coloring }
    }
    pub fn build_coloring(&mut self) {
        let non_zeros = self.find_non_zeros();
        let ncols = self.op.nstates();
        let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
        let coloring = color_graph_greedy(&graph);
        self.coloring = Some(coloring);
    }
    fn find_non_zeros(&self) -> Vec<(usize, usize)> {
        let mut v = F::V::zeros(self.op.nstates());
        let mut col = F::V::zeros(self.op.nout());
        let mut triplets = Vec::with_capacity(self.op.nstates());
        for j in 0..self.op.nstates() {
            v[j] = F::T::NAN;
            self.op.jac_mul_inplace(self.x, self.t, &v, &mut col);
            for i in 0..self.op.nout() {
                if col[i].is_nan() {
                    triplets.push((i, j));
                }
                col[i] = F::T::zero();
            }
            v[j] = F::T::zero();
        }
        triplets
    }
    fn find_non_zero_entries(&self) -> Vec<(usize, usize, F::T)> {
        let mut v = F::V::zeros(self.op.nstates());
        let mut col = F::V::zeros(self.op.nout());
        let mut triplets = Vec::with_capacity(self.op.nstates());
        for j in 0..self.op.nstates() {
            v[j] = F::T::one();
            self.op.jac_mul_inplace(self.x, self.t, &v, &mut col);
            for i in 0..self.op.nout() {
                if col[i] != F::T::zero() {
                    triplets.push((i, j, col[i]));
                }
            }
            v[j] = F::T::zero();
        }
        triplets
    }
    pub fn calc_jacobian_naive(&self) -> F::M {
        let triplets = self.find_non_zero_entries();
        F::M::try_from_triplets(self.op.nstates(), self.op.nout(), triplets).unwrap()
    }
    pub fn calc_jacobian_colored(&self) -> Result<F::M> {
        let _coloring = self
            .coloring
            .as_ref()
            .expect("Coloring not built, call `self.build_coloring()` first");
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::jacobian::Jacobian;
    use crate::op::{closure::Closure, NonLinearOp};
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
            let jacobian = Jacobian::new(&op, &x, t);
            let non_zeros = jacobian.find_non_zeros();
            let expect = triplets
                .iter()
                .map(|(i, j, _v)| (*i, *j))
                .collect::<Vec<_>>();
            assert_eq!(non_zeros, expect);
        }
    }
    
    #[test]
    fn build_coloring() {
        let test_triplets = vec![
            vec![(0, 0, 1.0), (1, 1, 1.0)],
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
            vec![(1, 1, 1.0)],
            vec![(0, 0, 1.0), (1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
        ];
        let expect = vec![
            vec![1, 1],
            vec![1, 2],
            vec![1, 1],
            vec![1, 2],
        ];
        for (triplets, expect) in test_triplets.iter().zip(expect) {
            let op = helper_triplets2op(triplets.as_slice(), 2, 2);
            let x = DVector::from_vec(vec![1.0, 1.0]);
            let t = 0.0;
            let mut jacobian = Jacobian::new(&op, &x, t);
            jacobian.build_coloring();
            let coloring = jacobian.coloring.unwrap();
            assert_eq!(coloring, expect);
        }
    }
}
