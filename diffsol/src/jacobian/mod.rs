use std::collections::HashSet;

use crate::{
    LinearOp, LinearOpTranspose, Matrix, MatrixSparsity, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, Scalar, Vector, VectorIndex,
};
use num_traits::{One, Zero};

use self::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy};

pub mod coloring;
pub mod graph;
pub mod greedy_coloring;

macro_rules! gen_find_non_zeros_nonlinear {
    ($name:ident, $op_fn:ident, $op_trait:ident, $nrows:ident, $ncols:ident) => {
        /// Find the non-zero entries of the $name matrix of a non-linear operator.
        /// TODO: This function is not efficient for non-host vectors and could be part of the Vector trait
        ///       to allow for more efficient implementations. It's ok for now since this is only used once
        ///       during the setup phase.
        pub fn $name<F: NonLinearOp + $op_trait + ?Sized>(
            op: &F,
            x: &F::V,
            t: F::T,
        ) -> Vec<(usize, usize)> {
            let mut v = F::V::zeros(op.$ncols(), op.context().clone());
            let mut col = F::V::zeros(op.$nrows(), op.context().clone());
            let mut triplets = Vec::with_capacity(op.nstates());
            for j in 0..op.$ncols() {
                v.set_index(j, F::T::NAN);
                op.$op_fn(x, t, &v, &mut col);
                for i in 0..op.nout() {
                    if col.get_index(i).is_nan() {
                        triplets.push((i, j));
                    }
                    col.set_index(i, F::T::zero());
                }
                // OR:
                //col.clone_as_vec().into_iter().for_each(|v| {
                //    if v.is_nan() {
                //        triplets.push((0, 0));
                //    }
                //});
                col.fill(F::T::zero());
                v.set_index(j, F::T::zero());
            }
            triplets
        }
    };
}

gen_find_non_zeros_nonlinear!(
    find_jacobian_non_zeros,
    jac_mul_inplace,
    NonLinearOpJacobian,
    nout,
    nstates
);
gen_find_non_zeros_nonlinear!(
    find_adjoint_non_zeros,
    jac_transpose_mul_inplace,
    NonLinearOpAdjoint,
    nstates,
    nout
);
gen_find_non_zeros_nonlinear!(
    find_sens_non_zeros,
    sens_mul_inplace,
    NonLinearOpSens,
    nstates,
    nparams
);
gen_find_non_zeros_nonlinear!(
    find_sens_adjoint_non_zeros,
    sens_transpose_mul_inplace,
    NonLinearOpSensAdjoint,
    nparams,
    nstates
);

macro_rules! gen_find_non_zeros_linear {
    ($name:ident, $op_fn:ident $(, $op_trait:tt )?) => {
        /// Find the non-zero entries of the $name matrix of a non-linear operator.
        /// TODO: This function is not efficient for non-host vectors and could be part of the Vector trait
        ///       to allow for more efficient implementations. It's ok for now since this is only used once
        ///       during the setup phase.
        pub fn $name<F: LinearOp + ?Sized $(+ $op_trait)?>(op: &F, t: F::T) -> Vec<(usize, usize)> {
            let mut v = F::V::zeros(op.nstates(), op.context().clone());
            let mut col = F::V::zeros(op.nout(), op.context().clone());
            let mut triplets = Vec::with_capacity(op.nstates());
            for j in 0..op.nstates() {
                v.set_index(j, F::T::NAN);
                op.$op_fn(&v, t, &mut col);
                for i in 0..op.nout() {
                    if col.get_index(i).is_nan() {
                        triplets.push((i, j));
                    }
                    col.set_index(i, F::T::zero());
                }
                // OR:
                //col.clone_as_vec().into_iter().for_each(|v| {
                //    if v.is_nan() {
                //        triplets.push((0, 0));
                //    }
                //});
                v.set_index(j, F::T::zero());
            }
            triplets
        }
    };
}

gen_find_non_zeros_linear!(find_matrix_non_zeros, call_inplace);
gen_find_non_zeros_linear!(
    find_transpose_non_zeros,
    call_transpose_inplace,
    LinearOpTranspose
);

use std::cell::RefCell;

pub struct JacobianColoring<M: Matrix> {
    dst_indices_per_color: Vec<<M::V as Vector>::Index>,
    src_indices_per_color: Vec<<M::V as Vector>::Index>,
    input_indices_per_color: Vec<<M::V as Vector>::Index>,
    scratch_v: RefCell<M::V>,
    scratch_col: RefCell<M::V>,
}

impl<M: Matrix> Clone for JacobianColoring<M> {
    fn clone(&self) -> Self {
        Self {
            dst_indices_per_color: self.dst_indices_per_color.clone(),
            src_indices_per_color: self.src_indices_per_color.clone(),
            input_indices_per_color: self.input_indices_per_color.clone(),
            scratch_v: RefCell::new(self.scratch_v.borrow().clone()),
            scratch_col: RefCell::new(self.scratch_col.borrow().clone()),
        }
    }
}

impl<M: Matrix> JacobianColoring<M> {
    pub fn new(sparsity: &impl MatrixSparsity<M>, non_zeros: &[(usize, usize)], ctx: M::C) -> Self {
        let ncols = sparsity.ncols();
        let graph = nonzeros2graph(non_zeros, ncols);
        let coloring = color_graph_greedy(&graph);
        let max_color = coloring.iter().max().copied().unwrap_or(0);
        let mut dst_indices_per_color = Vec::new();
        let mut src_indices_per_color = Vec::new();
        let mut input_indices_per_color = Vec::new();
        for c in 1..=max_color {
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut indices = Vec::new();
            for (i, j) in non_zeros {
                if coloring[*j] == c {
                    rows.push(*i);
                    cols.push(*j);
                    indices.push((*i, *j));
                }
            }
            let dst_indices = sparsity.get_index(indices.as_slice(), ctx.clone());
            let src_indices = <M::V as Vector>::Index::from_vec(rows, ctx.clone());
            let unique_cols: HashSet<_> = HashSet::from_iter(cols.iter().cloned());
            let unique_cols = unique_cols.into_iter().collect::<Vec<_>>();
            let input_indices = <M::V as Vector>::Index::from_vec(unique_cols, ctx.clone());
            dst_indices_per_color.push(dst_indices);
            src_indices_per_color.push(src_indices);
            input_indices_per_color.push(input_indices);
        }
        let scratch_v = RefCell::new(M::V::zeros(sparsity.ncols(), ctx.clone()));
        let scratch_col = RefCell::new(M::V::zeros(sparsity.nrows(), ctx.clone()));
        Self {
            dst_indices_per_color,
            src_indices_per_color,
            input_indices_per_color,
            scratch_v,
            scratch_col,
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

    pub fn jacobian_inplace<F: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = self.scratch_v.borrow_mut();
        let mut col = self.scratch_col.borrow_mut();
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

    pub fn sens_inplace<F: NonLinearOpSens<M = M, V = M::V, T = M::T, C = M::C>>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = self.scratch_v.borrow_mut();
        let mut col = self.scratch_col.borrow_mut();
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.sens_mul_inplace(x, t, &v, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::zero());
        }
    }

    pub fn adjoint_inplace<F: NonLinearOpAdjoint<M = M, V = M::V, T = M::T, C = M::C>>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = self.scratch_v.borrow_mut();
        let mut col = self.scratch_col.borrow_mut();
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.jac_transpose_mul_inplace(x, t, &v, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::zero());
        }
    }

    pub fn sens_adjoint_inplace<F: NonLinearOpSensAdjoint<M = M, V = M::V, T = M::T, C = M::C>>(
        &self,
        op: &F,
        x: &F::V,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = self.scratch_v.borrow_mut();
        let mut col = self.scratch_col.borrow_mut();
        for c in 0..self.dst_indices_per_color.len() {
            let input = &self.input_indices_per_color[c];
            let dst_indices = &self.dst_indices_per_color[c];
            let src_indices = &self.src_indices_per_color[c];
            v.assign_at_indices(input, F::T::one());
            op.sens_transpose_mul_inplace(x, t, &v, &mut col);
            y.set_data_with_indices(dst_indices, src_indices, &col);
            v.assign_at_indices(input, F::T::zero());
        }
    }

    pub fn matrix_inplace<F: LinearOp<M = M, V = M::V, T = M::T, C = M::C>>(
        &self,
        op: &F,
        t: F::T,
        y: &mut F::M,
    ) {
        let mut v = self.scratch_v.borrow_mut();
        let mut col = self.scratch_col.borrow_mut();
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

    use crate::jacobian::{find_jacobian_non_zeros, JacobianColoring};
    use crate::matrix::dense_nalgebra_serial::NalgebraMat;
    use crate::matrix::Matrix;
    use crate::op::linear_closure::LinearClosure;
    use crate::op::ParameterisedOp;
    use crate::vector::Vector;
    use crate::{
        jacobian::{coloring::nonzeros2graph, greedy_coloring::color_graph_greedy},
        op::closure::Closure,
        LinearOp, Op,
    };
    use crate::{scale, FaerSparseMat, NonLinearOpJacobian};
    use num_traits::{FromPrimitive, One, Zero};
    use std::ops::MulAssign;

    #[allow(clippy::type_complexity)]
    fn helper_triplets2op_nonlinear<'a, M: Matrix + 'a>(
        triplets: &'a [(usize, usize, M::T)],
        p: &'a M::V,
        nrows: usize,
        ncols: usize,
    ) -> Closure<
        M,
        impl Fn(&M::V, &M::V, M::T, &mut M::V) + use<'a, M>,
        impl Fn(&M::V, &M::V, M::T, &M::V, &mut M::V) + use<'a, M>,
    > {
        let nstates = ncols;
        let nout = nrows;
        let f = move |x: &M::V, y: &mut M::V| {
            for (i, j, v) in triplets {
                y.set_index(*i, y.get_index(*i) + x.get_index(*j) * *v);
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
            p.len(),
            p.context().clone(),
        );
        let y0 = M::V::zeros(nstates, p.context().clone());
        let t0 = M::T::zero();
        ret.calculate_sparsity(&y0, t0, p);
        ret
    }

    #[allow(clippy::type_complexity)]
    fn helper_triplets2op_linear<'a, M: Matrix + 'a>(
        triplets: &'a [(usize, usize, M::T)],
        p: &'a M::V,
        nrows: usize,
        ncols: usize,
    ) -> LinearClosure<M, impl Fn(&M::V, &M::V, M::T, M::T, &mut M::V) + use<'a, M>> {
        let nstates = ncols;
        let nout = nrows;
        let f = move |x: &M::V, y: &mut M::V| {
            for (i, j, v) in triplets {
                y.set_index(*i, y.get_index(*i) + x.get_index(*j) * *v);
            }
        };
        let mut ret = LinearClosure::new(
            move |x: &M::V, _p: &M::V, _t, beta, y: &mut M::V| {
                y.mul_assign(scale(beta));
                f(x, y);
            },
            nstates,
            nout,
            p.len(),
            p.context().clone(),
        );
        let t0 = M::T::zero();
        ret.calculate_sparsity(t0, p);
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
        let p = M::V::zeros(0, Default::default());
        for triplets in test_triplets {
            let op = helper_triplets2op_nonlinear::<M>(triplets.as_slice(), &p, 2, 2);
            let op = ParameterisedOp::new(&op, &p);
            let non_zeros =
                find_jacobian_non_zeros(&op, &M::V::zeros(2, p.context().clone()), M::T::zero());
            let expect = triplets
                .iter()
                .map(|(i, j, _v)| (*i, *j))
                .collect::<Vec<_>>();
            assert_eq!(non_zeros, expect);
        }
    }

    #[test]
    fn find_non_zeros_dmatrix() {
        find_non_zeros::<NalgebraMat<f64>>();
    }

    #[test]
    fn find_non_zeros_faer_sparse() {
        find_non_zeros::<FaerSparseMat<f64>>();
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
        let p = M::V::zeros(0, Default::default());
        for (triplets, expect) in test_triplets.iter().zip(expect) {
            let op = helper_triplets2op_nonlinear::<M>(triplets.as_slice(), &p, 2, 2);
            let op = ParameterisedOp::new(&op, &p);
            let non_zeros =
                find_jacobian_non_zeros(&op, &M::V::zeros(2, p.context().clone()), M::T::zero());
            let ncols = op.nstates();
            let graph = nonzeros2graph(non_zeros.as_slice(), ncols);
            let coloring = color_graph_greedy(&graph);

            assert_eq!(coloring, expect);
        }
    }

    #[test]
    fn build_coloring_dmatrix() {
        build_coloring::<NalgebraMat<f64>>();
    }

    #[test]
    fn build_coloring_faer_sparse() {
        build_coloring::<FaerSparseMat<f64>>();
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
                (0, 0, M::T::from_f64(0.9).unwrap()),
                (1, 0, M::T::from_f64(2.0).unwrap()),
                (1, 1, M::T::from_f64(1.1).unwrap()),
                (2, 2, M::T::from_f64(1.4).unwrap()),
            ],
        ];
        let n = 3;

        // test nonlinear functions
        let p = M::V::zeros(0, Default::default());
        for triplets in test_triplets.iter() {
            let op = helper_triplets2op_nonlinear::<M>(triplets.as_slice(), &p, n, n);
            let op = ParameterisedOp::new(&op, &p);
            let y0 = M::V::zeros(n, p.context().clone());
            let t0 = M::T::zero();
            let nonzeros = triplets
                .iter()
                .map(|(i, j, _v)| (*i, *j))
                .collect::<Vec<_>>();
            let coloring = JacobianColoring::new(
                &op.jacobian_sparsity().unwrap(),
                &nonzeros,
                p.context().clone(),
            );
            let mut jac = M::new_from_sparsity(3, 3, op.jacobian_sparsity(), p.context().clone());
            coloring.jacobian_inplace(&op, &y0, t0, &mut jac);
            let mut gemv1 = M::V::zeros(n, p.context().clone());
            let v = M::V::from_element(3, M::T::one(), p.context().clone());
            op.jac_mul_inplace(&y0, t0, &v, &mut gemv1);
            let mut gemv2 = M::V::zeros(n, p.context().clone());
            jac.gemv(M::T::one(), &v, M::T::zero(), &mut gemv2);
            gemv1.assert_eq_st(&gemv2, M::T::from_f64(1e-10).unwrap());
        }

        // test linear functions
        let p = M::V::zeros(0, p.context().clone());
        for triplets in test_triplets {
            let op = helper_triplets2op_linear::<M>(triplets.as_slice(), &p, n, n);
            let op = ParameterisedOp::new(&op, &p);
            let t0 = M::T::zero();
            let nonzeros = triplets
                .iter()
                .map(|(i, j, _v)| (*i, *j))
                .collect::<Vec<_>>();
            let coloring =
                JacobianColoring::new(&op.sparsity().unwrap(), &nonzeros, p.context().clone());
            let mut jac = M::new_from_sparsity(3, 3, op.sparsity(), p.context().clone());
            coloring.matrix_inplace(&op, t0, &mut jac);
            let mut gemv1 = M::V::zeros(n, p.context().clone());
            let v = M::V::from_element(3, M::T::one(), p.context().clone());
            op.gemv_inplace(&v, t0, M::T::zero(), &mut gemv1);
            let mut gemv2 = M::V::zeros(n, p.context().clone());
            jac.gemv(M::T::one(), &v, M::T::zero(), &mut gemv2);
            gemv1.assert_eq_st(&gemv2, M::T::from_f64(1e-10).unwrap());
        }
    }

    #[test]
    fn matrix_coloring_dmatrix() {
        matrix_coloring::<NalgebraMat<f64>>();
    }

    #[test]
    fn matrix_coloring_faer_sparse() {
        matrix_coloring::<FaerSparseMat<f64>>();
    }
}
