#![allow(dead_code)]
use diffsol::{
    ConstantOp, LinearOp, NalgebraContext, NalgebraMat, NalgebraVec, NonLinearOp, OdeEquations,
    OdeEquationsRef, Op, Vector,
};
type T = f64;
type V = NalgebraVec<f64>;
type M = NalgebraMat<f64>;
type C = NalgebraContext;
struct MyRhs<'a> {
    p: &'a V,
} // implements NonLinearOp
struct MyMass<'a> {
    p: &'a V,
} // implements LinearOp
struct MyInit<'a> {
    p: &'a V,
} // implements ConstantOp
struct MyRoot<'a> {
    p: &'a V,
} // implements NonLinearOp
struct MyOut<'a> {
    p: &'a V,
} // implements NonLinearOp
impl Op for MyRhs<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        2
    }
    fn context(&self) -> &Self::C {
        self.p.context()
    }
}
impl NonLinearOp for MyRhs<'_> {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        y[0] = x[0] * x[0];
    }
}
impl Op for MyMass<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        self.p.context()
    }
}
impl LinearOp for MyMass<'_> {
    fn gemv_inplace(&self, x: &V, _t: T, beta: T, y: &mut V) {
        y[0] = x[0] * beta;
    }
}
impl Op for MyInit<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        self.p.context()
    }
}
impl ConstantOp for MyInit<'_> {
    fn call_inplace(&self, _t: T, y: &mut V) {
        y[0] = 0.1;
    }
}
impl Op for MyRoot<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        self.p.context()
    }
}
impl NonLinearOp for MyRoot<'_> {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        y[0] = x[0] - 1.0;
    }
}
impl Op for MyOut<'_> {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        self.p.context()
    }
}
impl NonLinearOp for MyOut<'_> {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        y[0] = x[0];
    }
}

struct MyProblem {
    p: V,
}

impl MyProblem {
    fn new() -> Self {
        MyProblem {
            p: V::zeros(2, C::default()),
        }
    }
}

impl Op for MyProblem {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        2
    }
    fn context(&self) -> &Self::C {
        self.p.context()
    }
}

impl<'a> OdeEquationsRef<'a> for MyProblem {
    type Rhs = MyRhs<'a>;
    type Mass = MyMass<'a>;
    type Init = MyInit<'a>;
    type Root = MyRoot<'a>;
    type Out = MyOut<'a>;
}

impl OdeEquations for MyProblem {
    fn rhs(&self) -> <MyProblem as OdeEquationsRef<'_>>::Rhs {
        MyRhs { p: &self.p }
    }
    fn mass(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Mass> {
        Some(MyMass { p: &self.p })
    }
    fn init(&self) -> <MyProblem as OdeEquationsRef<'_>>::Init {
        MyInit { p: &self.p }
    }
    fn root(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Root> {
        Some(MyRoot { p: &self.p })
    }
    fn out(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Out> {
        Some(MyOut { p: &self.p })
    }
    fn set_params(&mut self, p: &V) {
        self.p.copy_from(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        p.copy_from(&self.p);
    }
}
use diffsol::OdeBuilder;
fn main() {
    let _problem = OdeBuilder::<M>::new()
        .p(vec![1.0, 10.0])
        .build_from_eqn(MyProblem::new())
        .unwrap();
}
