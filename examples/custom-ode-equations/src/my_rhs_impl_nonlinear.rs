use crate::{MyRhs, T, V};
use diffsol::{NonLinearOp, NonLinearOpJacobian};

impl NonLinearOp for MyRhs<'_> {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        let (r, k) = (self.p[0], self.p[1]);
        y[0] = r * x[0] * (1.0 - x[0] / k);
    }
}

impl NonLinearOpJacobian for MyRhs<'_> {
    fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) {
        let (r, k) = (self.p[0], self.p[1]);
        y[0] = r * v[0] * (1.0 - 2.0 * x[0] / k);
    }
}
