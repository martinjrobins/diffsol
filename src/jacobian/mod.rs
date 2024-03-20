use crate::op::NonLinearOp;
use crate::vector::Vector;
use crate::matrix::Matrix;
use num_traits::{One, Zero};
use anyhow::Result;

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

struct Coloring {
    data: Vec<usize>,
}

impl Coloring {
    fn new(data: Vec<usize>) -> Self {
        Self { data }
    }
}


pub struct Jacobian<'a, F: NonLinearOp + ?Sized> {
    op: &'a F,
    x: &'a F::V,
    t: F::T,
    coloring: Option<Coloring>,
}

impl<'a, F: NonLinearOp> Jacobian<'a, F> {
    pub fn new(op: &'a F, x: &'a F::V, t: F::T) -> Self {
        let coloring = None;
        Self { op, x, t, coloring }
    }
    pub fn build_coloring(&mut self) {
        self.coloring = Some(Coloring { data: Vec::new() });
    }
    fn find_non_zeros(&self) -> Vec<(usize, usize, F::T)> {
        let mut v = F::V::zeros(self.op.nstates());
        let mut col = F::V::zeros(self.op.nout());
        let mut triplets = Vec::with_capacity(self.op.nstates());
        for j in 0..self.op.nstates() {
            v[j] = F::T::nan();
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
        let coloring = self.coloring.as_ref().expect("Coloring not built, call `self.build_coloring()` first");
        todo!()
    }
}


