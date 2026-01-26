use burn::backend::ndarray::NdArrayTensor;
use burn::backend::NdArray;
use burn::tensor::{DType, TensorMetadata};
use burn::{prelude::Backend, tensor::ops::FloatTensor};
use diffsol::matrix::MatrixRef;
use diffsol::{
    DefaultDenseMatrix, DenseMatrix, MatrixCommon, OdeBuilder, OdeSolverMethod, Scalar, ScalarEnum,
    VectorHost, VectorRef, VectorViewHost,
};
use ndarray::{s, Array1, Array2, ArrayRef1, ArrayRef2, ArrayView1, ShapeBuilder};

use crate::equations::Equations;
use crate::rhs::Rhs;
use crate::vector::Vector;

pub trait Solve<V: Vector>: Backend {
    fn solve_check(
        y0: &FloatTensor<Self>,
        params: &FloatTensor<Self>,
        t_eval: &FloatTensor<Self>,
        m: &impl Rhs<Self>,
    ) {
        // make sure y0, params, and t_eval are all on the same device and have the correct dtype
        let device = Self::float_device(&y0);
        let dtype = y0.dtype();
        assert_eq!(
            Self::float_device(&params),
            device,
            "Params tensor is not on the same device as y0 tensor"
        );
        assert_eq!(
            Self::float_device(&t_eval),
            device,
            "t_eval tensor is not on the same device as y0 tensor"
        );
        assert_eq!(
            params.dtype(),
            dtype,
            "Params tensor does not have the same dtype as y0 tensor"
        );
        assert_eq!(
            t_eval.dtype(),
            dtype,
            "t_eval tensor does not have the same dtype as y0 tensor"
        );

        // make sure the input tensors are all of type V::T
        match (dtype, V::T::as_enum()) {
            (DType::F32, ScalarEnum::F32) => {}
            (DType::F64, ScalarEnum::F64) => {}
            _ => panic!("Input tensors must have the same dtype as V::T"),
        }
    }
    fn solve_inner(
        y0: &V,
        params: &FloatTensor<Self>,
        t_eval: &[V::T],
        m: &impl Rhs<Self>,
    ) -> <V as DefaultDenseMatrix>::M
    where
        for<'b> &'b V: VectorRef<V>,
        for<'b> &'b <V as DefaultDenseMatrix>::M: MatrixRef<<V as DefaultDenseMatrix>::M>,
    {
        let eqn = Equations::new(m, y0.clone(), params.clone());
        let problem = OdeBuilder::<<V as DefaultDenseMatrix>::M>::new()
            .build_from_eqn(eqn)
            .unwrap();
        let mut solver = problem.tsit45().unwrap();
        solver.solve_dense(t_eval).unwrap()
    }

    fn solve(
        y0: FloatTensor<Self>,
        params: FloatTensor<Self>,
        t_eval: FloatTensor<Self>,
        m: &impl Rhs<Self>,
    ) -> FloatTensor<Self>
    where
        for<'b> &'b V: VectorRef<V>,
        for<'b> &'b <V as DefaultDenseMatrix>::M: MatrixRef<<V as DefaultDenseMatrix>::M>,
    {
        let ctx = V::C::default();
        let mut y0_v = V::zeros(y0.shape()[0], ctx.clone());
        Self::copy_tensor_to_v(y0, &mut y0_v);
        let ys = Self::solve_inner(&y0_v, &params, Self::tensor_to_vec(t_eval).as_slice(), m);
        Self::m_to_tensor(ys)
    }
    fn tensor_to_vec(t: FloatTensor<Self>) -> Vec<V::T>;
    fn copy_tensor_to_v(t: FloatTensor<Self>, v: &mut V);
    fn v_to_tensor(v: &V) -> FloatTensor<Self>;
    fn m_to_tensor(m: <V as DefaultDenseMatrix>::M) -> FloatTensor<Self>;
}

impl<V: Vector<T = f32>> Solve<V> for NdArray
where
    V: VectorHost,
    for<'a> V::View<'a>: VectorViewHost<'a>,
{
    fn m_to_tensor(m: <V as DefaultDenseMatrix>::M) -> NdArrayTensor {
        let mut ret = Array2::<f32>::zeros((m.nrows(), m.ncols()).f());
        for i in 0..m.ncols() {
            let col = m.column(i);
            let col_slice = col.as_slice();
            let mut ret_col_slice = ret.slice_mut(s![.., i]);
            ret_col_slice
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&col_slice);
        }
        ret.into_dyn().into_shared().into()
    }
    fn tensor_to_vec(t: FloatTensor<Self>) -> Vec<<V>::T> {
        if let NdArrayTensor::F32(t) = t {
            t.into_shared().as_slice().unwrap().to_vec()
        } else {
            panic!("Expected f32 tensor");
        }
    }
    fn copy_tensor_to_v(t: FloatTensor<Self>, v: &mut V) {
        if let NdArrayTensor::F32(t) = t {
            v.as_mut_slice()
                .copy_from_slice(t.into_shared().as_slice().unwrap());
        } else {
            panic!("Expected f32 tensor");
        }
    }
    fn v_to_tensor(v: &V) -> FloatTensor<Self> {
        Array1::<f32>::from_iter(v.as_slice().iter().copied())
            .into_shared()
            .into_dyn()
            .into()
    }
}

#[cfg(test)]
mod tests {
    use diffsol::{FaerMat, FaerVec, Matrix, Vector};
    use ndarray::Array1;

    use super::*;

    #[test]
    fn test_m_to_tensor() {
        let mut m = FaerMat::<f32>::zeros(2, 3, Default::default());

        m.set_index(0, 0, 1.0);
        m.set_index(0, 1, 2.0);
        m.set_index(0, 2, 3.0);
        m.set_index(1, 0, 4.0);
        m.set_index(1, 1, 5.0);
        m.set_index(1, 2, 6.0);

        let t = <NdArray as Solve<FaerVec<f32>>>::m_to_tensor(m);
        if let NdArrayTensor::F32(t) = t {
            let arr = t.into_shared();
            assert_eq!(arr[[0, 0]], 1.0);
            assert_eq!(arr[[0, 1]], 2.0);
            assert_eq!(arr[[0, 2]], 3.0);
            assert_eq!(arr[[1, 0]], 4.0);
            assert_eq!(arr[[1, 1]], 5.0);
            assert_eq!(arr[[1, 2]], 6.0);
        } else {
            panic!("Expected f32 tensor");
        }
    }

    #[test]
    fn test_tensor_to_vec() {
        let arr = Array1::<f32>::from_vec(vec![1.0, 2.0, 3.0]);
        let t: NdArrayTensor = arr.into_dyn().into_shared().into();
        let v = <NdArray as Solve<FaerVec<f32>>>::tensor_to_vec(t);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_copy_tensor_to_v() {
        let arr = Array1::<f32>::from_vec(vec![1.0, 2.0, 3.0]);
        let t: NdArrayTensor = arr.into_dyn().into_shared().into();
        let mut v = FaerVec::<f32>::zeros(3, Default::default());
        <NdArray as Solve<FaerVec<f32>>>::copy_tensor_to_v(t, &mut v);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_copy_v_to_tensor() {
        let v = FaerVec::<f32>::from_slice(&[1.0, 2.0, 3.0], Default::default());
        let t = <NdArray as Solve<FaerVec<f32>>>::v_to_tensor(&v);
        if let NdArrayTensor::F32(t) = t {
            let arr = t.into_shared();
            assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        } else {
            panic!("Expected f32 tensor");
        }
    }
}
