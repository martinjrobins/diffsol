use diffsol::NonLinearOp;
use crate::{T, V, MyRhs};

impl NonLinearOp for MyRhs<'_> {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        y[0] = x[0] * x[0];
    }
}