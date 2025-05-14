use crate::{C, M, T, V};
use diffsol::{NonLinearOp, Op, Vector};

pub struct MyOut<'a> {
    pub p: &'a V,
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
