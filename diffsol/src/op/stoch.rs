use super::Op;
use crate::{DefaultDenseMatrix, Scalar, Vector};
use num_traits::{One, Zero};

enum StochOpKind {
    Scalar,
    Diagonal,
    Additive,
    Other,
}

/// Stochastic differential equation (SDE) operations.
/// 
/// In general, this operator computes `F(x, t)`, where `F` is a matrix of size `nstates() x nprocess()`.
/// The matrix `F` is computed by the [Self::call_inplace] method, which returns a dense matrix `y`.
/// The `kind` method returns the type of stochastic operation, either `Scalar`, `Diagonal`, `Additive`, or `Other`,
/// and the `kind` determines how `y` is interpreted.
///
/// For scalar noise, `y` is a matrix with one column, and the noise is applied as `y * dW`, where `dW` is a scalar Wiener increment.
/// For diagonal noise, `y` is a matrix with one column, which is interpreted as the diagonal of the matrix `F(x, t)`. The noise is applied as `F * dW`, where `dW` is a vector of independent Wiener increments.
/// For additive noise, `y` is a full matrix with `nprocess()` columns that does not depend on `x`, and the noise is applied as `F * dW`, where `dW` is a vector of Wiener increments.
/// Diffsol does not support other types of noise, but the `Other` kind is provided for completeness.
pub trait StochOp: Op {
    fn kind(&self) -> StochOpKind;
    fn nprocess(&self) -> usize;
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut <Self::V as DefaultDenseMatrix>::M) where Self::V: DefaultDenseMatrix;
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
