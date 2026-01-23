use burn::{Tensor, module::Module, prelude::*};
use diffsol::{ConstantOp, ConstantOpSensAdjoint, NalgebraContext, NalgebraMat, NalgebraVec, NonLinearOp, NonLinearOpAdjoint, NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef, Op, Scalar, UnitCallable};

type T = f32;
type V = NalgebraVec<T>;
type M = NalgebraMat<T>;
type C = NalgebraContext;

pub struct Equations<'m, B, M> 
where 
    B: Backend,
    M: Module<B>,
{
    rhs: &'m M,
    y0: &'m Tensor<B, 1>,
    input: &'m Tensor<B, 1>,
    ctx: C,
}

impl<'m, B, M> Equations<'m, B, M> 
where 
    B: Backend,
    M: Module<B>,
{
    fn new(rhs: &'m M, y0: &'m Tensor<B, 1>, input: &'m Tensor<B, 1>) -> Self {
        Self { rhs, y0, input, ctx: C::default() }
    } 
    fn nstates(&self) -> usize {
        self.y0.shape()[0]
    }
}
impl<'m, B, Mod> Op for Equations<'m, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    type T = T;
    type V = V;
    type M = M;
    type C = C;
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

impl<'a, 'm, B, Mod> OdeEquationsRef<'a> for Equations<'m, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    type Mass = UnitCallable<M>;
    type Rhs = Rhs<'a>;
    type Root = UnitCallable<M>;
    type Init = Init<'a>;
    type Out = UnitCallable<M>;
}

impl<'a, 'm, B, M> OdeEquations for Equations<'m, B, M> 
where 
    B: Backend,
    M: Module<B>,
{
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        Rhs(self)
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        None
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        Init(self)
    }

    fn set_params(&mut self, p: &Self::V) {
    }

    fn get_params(&self, p: &mut Self::V) {
    }
}

struct Init<'a, 'm, B, M>(&'a Equations<'m, B, M>) 
where 
    B: Backend,
    M: Module<B>,
;

impl<'a, 'm, B, Mod> Op for Init<'a, 'm, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    type M = M;
    type V = V;
    type T = T;
    type C = C;
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

impl<'a, 'm, B, M> ConstantOp for Init<'a, 'm, B, M> 
where 
    B: Backend,
    M: Module<B>,
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(&self.0.y0);
    }
}

impl<'a, 'm, B, M> ConstantOpSensAdjoint for Init<'a, 'm, B, M> 
where 
    B: Backend,
    M: Module<B>,
{
    fn sens_transpose_mul_inplace(&self, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(0.0);
    }
}

struct Rhs<'a, 'm, B, M>(&'a Equations<'m, B, M>) 
where 
    B: Backend,
    M: Module<B>,
;

impl<'a, 'm, B, Mod> Op for Rhs<'a, 'm, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    type M = M;
    type V = V;
    type T = T;
    type C = C;
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

impl<'a, 'm, B, Mod> NonLinearOp for Rhs<'a, 'm, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
    }
}

impl<'a, 'm, B, Mod> NonLinearOpAdjoint for Rhs<'a, 'm, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
    }
}

impl<'a, 'm, B, Mod> NonLinearOpSensAdjoint for Rhs<'a, 'm, B, Mod> 
where 
    B: Backend,
    Mod: Module<B>,
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
    }
}