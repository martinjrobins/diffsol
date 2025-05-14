use crate::{C, M, T, V};
use diffsol::{LinearOp, Op, Vector};

pub struct MyMass<'a> {
    pub p: &'a V,
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
