use crate::{MyEquations, C, M, T, V};
use diffsol::{Op, Vector};

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
