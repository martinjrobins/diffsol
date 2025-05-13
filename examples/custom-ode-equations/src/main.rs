mod common;
use common::{T, V, M, C};
mod my_rhs;
use my_rhs::MyRhs;
mod my_rhs_impl_op;
mod my_rhs_impl_nonlinear;
mod my_mass;
use my_mass::MyMass;
mod my_mass_impl_op;
mod my_init;
use my_init::MyInit;
mod my_root;
use my_root::MyRoot;
mod my_out;
use my_out::MyOut;





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
