use burn::{
    tensor::{ops::FloatTensor, TensorPrimitive},
    Tensor,
};
use diffsol::{
    ConstantOp, ConstantOpSensAdjoint, DefaultDenseMatrix, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef, Op, UnitCallable,
};
use num_traits::Zero;

use crate::{backend::Solve, rhs::Rhs, vector::Vector};

pub struct Equations<'m, V: Vector, B: Solve<V>, M: Rhs<B>> {
    rhs: &'m M,
    y0: V,
    params: FloatTensor<B>,
    ctx: V::C,
    _marker: std::marker::PhantomData<B>,
}

impl<'m, V: Vector, B: Solve<V>, M: Rhs<B>> Equations<'m, V, B, M> {
    pub fn new(rhs: &'m M, y0: V, params: FloatTensor<B>) -> Self {
        let ctx = y0.context().clone();
        Self {
            rhs,
            y0,
            params,
            ctx,
            _marker: std::marker::PhantomData,
        }
    }
    fn nstates(&self) -> usize {
        self.y0.len()
    }
}
impl<'m, V: Vector, B: Solve<V>, Mod: Rhs<B>> Op for Equations<'m, V, B, Mod> {
    type T = V::T;
    type V = V;
    type M = <V as DefaultDenseMatrix>::M;
    type C = V::C;
    fn nstates(&self) -> usize {
        self.nstates()
    }
    fn nout(&self) -> usize {
        self.nstates()
    }
    fn nparams(&self) -> usize {
        self.rhs.num_params()
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> OdeEquationsRef<'a> for Equations<'m, V, B, Mod> {
    type Mass = UnitCallable<Self::M>;
    type Rhs = RhsRef<'a, 'm, V, B, Mod>;
    type Root = UnitCallable<Self::M>;
    type Init = Init<'a, 'm, V, B, Mod>;
    type Out = UnitCallable<Self::M>;
}

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> OdeEquations for Equations<'m, V, B, Mod> {
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        RhsRef(self)
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        None
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        Init(self)
    }

    fn set_params(&mut self, _p: &Self::V) {}

    fn get_params(&self, _p: &mut Self::V) {}
}

pub struct Init<'a, 'm, V: Vector, B: Solve<V>, M: Rhs<B>>(&'a Equations<'m, V, B, M>);

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> Op for Init<'a, 'm, V, B, Mod> {
    type M = <V as DefaultDenseMatrix>::M;
    type V = V;
    type T = V::T;
    type C = V::C;
    fn nout(&self) -> usize {
        self.0.nout()
    }
    fn nparams(&self) -> usize {
        self.0.nparams()
    }
    fn nstates(&self) -> usize {
        self.0.nstates()
    }
    fn context(&self) -> &Self::C {
        self.0.context()
    }
}

impl<'a, 'm, V: Vector, B: Solve<V>, M: Rhs<B>> ConstantOp for Init<'a, 'm, V, B, M> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(&self.0.y0);
    }
}

impl<'a, 'm, V: Vector, B: Solve<V>, M: Rhs<B>> ConstantOpSensAdjoint for Init<'a, 'm, V, B, M> {
    fn sens_transpose_mul_inplace(&self, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(V::T::zero());
    }
}

pub struct RhsRef<'a, 'm, V: Vector, B: Solve<V>, M: Rhs<B>>(&'a Equations<'m, V, B, M>);

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> Op for RhsRef<'a, 'm, V, B, Mod> {
    type M = <V as DefaultDenseMatrix>::M;
    type V = V;
    type T = V::T;
    type C = V::C;
    fn nout(&self) -> usize {
        self.0.nout()
    }
    fn nparams(&self) -> usize {
        self.0.nparams()
    }
    fn nstates(&self) -> usize {
        self.0.nstates()
    }
    fn context(&self) -> &Self::C {
        self.0.context()
    }
}

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> NonLinearOp for RhsRef<'a, 'm, V, B, Mod> {
    fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
        let x_tensor = B::v_to_tensor(x);
        let params_tensor = Tensor::from_primitive(TensorPrimitive::Float(self.0.params.clone()));
        let x_tensor = Tensor::from_primitive(TensorPrimitive::Float(x_tensor));
        let out_tensor = self.0.rhs.forward(x_tensor, params_tensor);
        let out = out_tensor.into_primitive().tensor();
        B::copy_tensor_to_v(out, y);
    }
}

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> NonLinearOpAdjoint for RhsRef<'a, 'm, V, B, Mod> {
    fn jac_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {}
}

impl<'a, 'm, V: Vector, B: Solve<V>, Mod: Rhs<B>> NonLinearOpSensAdjoint
    for RhsRef<'a, 'm, V, B, Mod>
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {}
}
