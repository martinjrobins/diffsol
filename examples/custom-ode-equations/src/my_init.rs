use crate::{C, M, T, V};
use diffsol::{ConstantOp, Op, Vector};

pub struct MyInit<'a> {
    pub p: &'a V,
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
