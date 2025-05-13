use diffsol::{Op, Vector};
use crate::{T, V, M, C, MyEquations};

impl Op for MyEquations {
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