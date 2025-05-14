use crate::{C, M, T, V};
use diffsol::{NonLinearOp, Op, Vector};

pub struct MyRoot<'a> {
    pub p: &'a V,
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
