use super::Op;
use crate::{Scalar, Vector};
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochOpKind {
    Zero,
    Scalar,
    Diagonal,
    Additive,
    Other,
}

/// Stochastic differential equation (SDE) operations.
///
/// For scalar noise, nprocess is 1.
/// For diagonal noise, y_i only depends on x_i and d_w_i.
/// For additive noise, y_i does not depend on x_i.
pub trait StochOp: Op {
    fn nprocess(&self) -> usize;
    fn process_inplace(&self, x: &Self::V, d_w: &Self::V, t: Self::T, y: &mut [Self::V]);
    fn kind(&self) -> StochOpKind {
        if self.nprocess() == 0 {
            return StochOpKind::Zero;
        }
        if self.nprocess() == 1 {
            return StochOpKind::Scalar;
        }
        let mut y = vec![Self::V::zeros(self.nout(), self.context().clone()); self.nprocess()];
        let mut x = Self::V::zeros(self.nstates(), self.context().clone());
        x.fill(Self::T::NAN);
        let mut d_w = Self::V::zeros(self.nprocess(), self.context().clone());
        d_w.fill(Self::T::one());
        let t = Self::T::zero();
        self.process_inplace(&x, &d_w, t, &mut y);
        // if none of the outputs has nans, it is additive
        if y.iter()
            .all(|y_j| !y_j.clone_as_vec().iter().any(|&val| val.is_nan()))
        {
            return StochOpKind::Additive;
        }

        x.fill(Self::T::one());

        for i in 0..self.nprocess() {
            if i != 0 {
                d_w.set_index(i - 1, Self::T::one());
            }
            d_w.set_index(i, Self::T::NAN);
            self.process_inplace(&x, &d_w, t, &mut y);

            // if any of the y[j] j != i has nans, it is other
            for (j, y_j) in y.iter().enumerate() {
                if j != i {
                    let has_nans = y_j.clone_as_vec().iter().any(|&val| val.is_nan());
                    if has_nans {
                        return StochOpKind::Other;
                    }
                }
            }
        }
        // must be diagonal
        StochOpKind::Diagonal
    }
}

#[cfg(test)]
mod test {
    use crate::{NalgebraContext, NalgebraMat, NalgebraVec, Op, Scale, Vector};
    use num_traits::One;

    use super::{StochOp, StochOpKind};

    struct TestScalar {
        ctx: NalgebraContext,
    }
    impl Op for TestScalar {
        type T = f64;
        type V = NalgebraVec<f64>;
        type C = NalgebraContext;
        type M = NalgebraMat<f64>;

        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            0
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }
    impl StochOp for TestScalar {
        fn nprocess(&self) -> usize {
            1
        }
        fn process_inplace(&self, x: &Self::V, d_w: &Self::V, _t: Self::T, y: &mut [Self::V]) {
            assert_eq!(y.len(), 1);
            y[0] = x + d_w.clone();
        }
    }

    struct TestDiagonal {
        ctx: NalgebraContext,
    }
    impl Op for TestDiagonal {
        type T = f64;
        type V = NalgebraVec<f64>;
        type C = NalgebraContext;
        type M = NalgebraMat<f64>;

        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            0
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }
    impl StochOp for TestDiagonal {
        fn nprocess(&self) -> usize {
            2
        }
        fn process_inplace(&self, x: &Self::V, d_w: &Self::V, _t: Self::T, y: &mut [Self::V]) {
            assert_eq!(y.len(), 2);
            for i in 0..2 {
                y[i] = x.clone() * Scale(d_w[i]);
            }
        }
    }
    struct TestAdditive {
        ctx: NalgebraContext,
    }
    impl Op for TestAdditive {
        type T = f64;
        type V = NalgebraVec<f64>;
        type C = NalgebraContext;
        type M = NalgebraMat<f64>;

        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            0
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }
    impl StochOp for TestAdditive {
        fn nprocess(&self) -> usize {
            2
        }
        fn process_inplace(&self, _x: &Self::V, d_w: &Self::V, _t: Self::T, y: &mut [Self::V]) {
            assert_eq!(y.len(), 2);
            let mut ones = Self::V::zeros(self.nout(), self.context().clone());
            ones.fill(Self::T::one());
            for i in 0..2 {
                y[i] = &ones * Scale(d_w[i]);
            }
        }
    }

    struct TestOther {
        ctx: NalgebraContext,
    }
    impl Op for TestOther {
        type T = f64;
        type V = NalgebraVec<f64>;
        type C = NalgebraContext;
        type M = NalgebraMat<f64>;

        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            0
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
    }
    impl StochOp for TestOther {
        fn nprocess(&self) -> usize {
            2
        }
        fn process_inplace(&self, x: &Self::V, d_w: &Self::V, _t: Self::T, y: &mut [Self::V]) {
            assert_eq!(y.len(), 2);
            for i in 0..2 {
                y[i] = x.clone() * Scale(d_w[i]);
                if i == 1 {
                    y[i] *= Scale(d_w[0]);
                }
            }
        }
    }

    #[test]
    fn test_additive() {
        let op = TestAdditive {
            ctx: NalgebraContext,
        };
        assert_eq!(op.kind(), StochOpKind::Additive);
    }

    #[test]
    fn test_diagonal() {
        let op = TestDiagonal {
            ctx: NalgebraContext,
        };
        assert_eq!(op.kind(), StochOpKind::Diagonal);
    }

    #[test]
    fn test_scalar() {
        let op = TestScalar {
            ctx: NalgebraContext,
        };
        assert_eq!(op.kind(), StochOpKind::Scalar);
    }

    #[test]
    fn test_other() {
        let op = TestOther {
            ctx: NalgebraContext,
        };
        assert_eq!(op.kind(), StochOpKind::Other);
    }
}
