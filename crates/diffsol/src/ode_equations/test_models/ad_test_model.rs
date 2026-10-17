//! A small nonlinear model implementing every derivative operator, for testing derivatives of
//! solver steps against each other: Jacobian and transposed Jacobian products, parameter
//! sensitivities and their transposes, for the right-hand side and the initial condition.
//!
//! With parameters `p = (p0, p1, p2)`:
//! ```text
//! y0' = -p0 y0 + p1 y1 y2
//! y1' =  p0 y0 - p1 y1 y2 - y1^2
//! y2' =  y1^2                         (ODE)
//!   0 =  y0 + y1 + y2 - p2            (DAE: mass diag(1, 1, 0))
//! y(0) = (p2, 0, 0)
//! ```
//! Both forms conserve `y0 + y1 + y2 = p2`.
//!
//! Every operator works lane by lane on batched vectors through [Vector::for_each_batch], with a
//! single-lane state broadcast over the lanes of its argument, as the forward sensitivity
//! equations call them.

use num_traits::{FromPrimitive, One, Zero};

use crate::{
    op::{unit::UnitCallable, ParameterisedOp},
    ConstantOp, ConstantOpSens, ConstantOpSensAdjoint, LinearOp, LinearOpTranspose, Matrix,
    MatrixSparsity, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
    NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef, OdeSolverProblem, Op, Vector,
};

const N: usize = 3;
const NP: usize = 3;

/// Every 3x3 matrix of the model (the Jacobian, the mass matrix, and the parameter derivatives
/// of the right-hand side and the initial condition, since `NP == N`) is given a full pattern, so
/// that it can be assembled into sparse as well as dense matrices.
fn full_sparsity<M: Matrix>() -> Option<M::Sparsity> {
    let indices = (0..N).flat_map(|i| (0..N).map(move |j| (i, j))).collect();
    Some(M::Sparsity::try_from_indices(N, N, indices).unwrap())
}

pub struct AdTestRhs<M: Matrix> {
    p: [M::T; 3],
    dae: bool,
    ctx: M::C,
}

pub struct AdTestInit<M: Matrix> {
    p: [M::T; 3],
    ctx: M::C,
}

pub struct AdTestMass<M: Matrix> {
    ctx: M::C,
}

macro_rules! impl_op {
    ($ty:ident, $nout:expr) => {
        impl<M: Matrix> Op for $ty<M> {
            type T = M::T;
            type V = M::V;
            type M = M;
            type C = M::C;
            fn nstates(&self) -> usize {
                N
            }
            fn nout(&self) -> usize {
                $nout
            }
            fn nparams(&self) -> usize {
                NP
            }
            fn context(&self) -> &Self::C {
                &self.ctx
            }
        }
    };
}
impl_op!(AdTestRhs, N);
impl_op!(AdTestInit, N);
impl_op!(AdTestMass, N);

impl<M: Matrix> NonLinearOp for AdTestRhs<M> {
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        let (p, dae) = (self.p, self.dae);
        y.for_each_batch([x], move |y, [x], _| {
            y[0] = -p[0] * x[0] + p[1] * x[1] * x[2];
            y[1] = p[0] * x[0] - p[1] * x[1] * x[2] - x[1] * x[1];
            y[2] = if dae {
                x[0] + x[1] + x[2] - p[2]
            } else {
                x[1] * x[1]
            };
        });
    }
}

/// `y = J v`
impl<M: Matrix> NonLinearOpJacobian for AdTestRhs<M> {
    fn jacobian_sparsity(&self) -> Option<M::Sparsity> {
        full_sparsity::<M>()
    }
    fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        let (p, dae) = (self.p, self.dae);
        y.for_each_batch([x, v], move |y, [x, v], _| {
            let two = M::T::one() + M::T::one();
            y[0] = -p[0] * v[0] + p[1] * x[2] * v[1] + p[1] * x[1] * v[2];
            y[1] = p[0] * v[0] - (p[1] * x[2] + two * x[1]) * v[1] - p[1] * x[1] * v[2];
            y[2] = if dae {
                v[0] + v[1] + v[2]
            } else {
                two * x[1] * v[1]
            };
        });
    }
}

/// `y = -J^T v`
impl<M: Matrix> NonLinearOpAdjoint for AdTestRhs<M> {
    fn jac_transpose_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        let (p, dae) = (self.p, self.dae);
        y.for_each_batch([x, v], move |y, [x, v], _| {
            let two = M::T::one() + M::T::one();
            // the last row of J is (1, 1, 1) for the DAE and (0, 2 x1, 0) for the ODE
            let (l0, l1, l2) = if dae {
                (v[2], v[2], v[2])
            } else {
                (M::T::zero(), two * x[1] * v[2], M::T::zero())
            };
            y[0] = p[0] * v[0] - p[0] * v[1] - l0;
            y[1] = -p[1] * x[2] * v[0] + (p[1] * x[2] + two * x[1]) * v[1] - l1;
            y[2] = -p[1] * x[1] * v[0] + p[1] * x[1] * v[1] - l2;
        });
    }
}

/// `y = f_p v`
impl<M: Matrix> NonLinearOpSens for AdTestRhs<M> {
    fn sens_sparsity(&self) -> Option<M::Sparsity> {
        full_sparsity::<M>()
    }
    fn sens_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        let dae = self.dae;
        y.for_each_batch([x, v], move |y, [x, v], _| {
            y[0] = -x[0] * v[0] + x[1] * x[2] * v[1];
            y[1] = x[0] * v[0] - x[1] * x[2] * v[1];
            y[2] = if dae { -v[2] } else { M::T::zero() };
        });
    }
}

/// `y = -f_p^T v`
impl<M: Matrix> NonLinearOpSensAdjoint for AdTestRhs<M> {
    fn sens_transpose_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        let dae = self.dae;
        y.for_each_batch([x, v], move |y, [x, v], _| {
            y[0] = x[0] * v[0] - x[0] * v[1];
            y[1] = -x[1] * x[2] * v[0] + x[1] * x[2] * v[1];
            y[2] = if dae { v[2] } else { M::T::zero() };
        });
    }
}

impl<M: Matrix> ConstantOp for AdTestInit<M> {
    fn call_inplace(&self, _t: M::T, y: &mut M::V) {
        let p2 = self.p[2];
        y.for_each_batch([], move |y, [], _| {
            y[0] = p2;
            y[1] = M::T::zero();
            y[2] = M::T::zero();
        });
    }
}

impl<M: Matrix> ConstantOpSens for AdTestInit<M> {
    fn sens_sparsity(&self) -> Option<M::Sparsity> {
        full_sparsity::<M>()
    }
    fn sens_mul_inplace(&self, _t: M::T, v: &M::V, y: &mut M::V) {
        y.for_each_batch([v], |y, [v], _| {
            y[0] = v[2];
            y[1] = M::T::zero();
            y[2] = M::T::zero();
        });
    }
}

impl<M: Matrix> ConstantOpSensAdjoint for AdTestInit<M> {
    fn sens_transpose_mul_inplace(&self, _t: M::T, v: &M::V, y: &mut M::V) {
        y.for_each_batch([v], |y, [v], _| {
            y[0] = M::T::zero();
            y[1] = M::T::zero();
            y[2] = -v[0];
        });
    }
}

impl<M: Matrix> AdTestMass<M> {
    /// `y = diag(1, 1, 0) x + beta y`, which is its own transpose
    fn gemv(x: &M::V, beta: M::T, y: &mut M::V) {
        y.for_each_batch([x], move |y, [x], _| {
            y[0] = x[0] + beta * y[0];
            y[1] = x[1] + beta * y[1];
            y[2] *= beta;
        });
    }
}

impl<M: Matrix> LinearOp for AdTestMass<M> {
    fn sparsity(&self) -> Option<M::Sparsity> {
        full_sparsity::<M>()
    }
    fn gemv_inplace(&self, x: &M::V, _t: M::T, beta: M::T, y: &mut M::V) {
        Self::gemv(x, beta, y);
    }
}

impl<M: Matrix> LinearOpTranspose for AdTestMass<M> {
    fn gemv_transpose_inplace(&self, x: &M::V, _t: M::T, beta: M::T, y: &mut M::V) {
        Self::gemv(x, beta, y);
    }
}

pub struct AdTestEqn<M: Matrix> {
    rhs: AdTestRhs<M>,
    init: AdTestInit<M>,
    mass: Option<AdTestMass<M>>,
    ctx: M::C,
}

impl<M: Matrix> Op for AdTestEqn<M> {
    type T = M::T;
    type V = M::V;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        N
    }
    fn nout(&self) -> usize {
        N
    }
    fn nparams(&self) -> usize {
        NP
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<'a, M: Matrix> OdeEquationsRef<'a> for AdTestEqn<M> {
    type Rhs = &'a AdTestRhs<M>;
    type Mass = &'a AdTestMass<M>;
    type Root = ParameterisedOp<'a, UnitCallable<M>>;
    type Init = &'a AdTestInit<M>;
    type Out = ParameterisedOp<'a, UnitCallable<M>>;
    type Reset = ParameterisedOp<'a, UnitCallable<M>>;
}

impl<M: Matrix> OdeEquations for AdTestEqn<M> {
    fn rhs(&self) -> &AdTestRhs<M> {
        &self.rhs
    }
    fn mass(&self) -> Option<&AdTestMass<M>> {
        self.mass.as_ref()
    }
    fn init(&self) -> &AdTestInit<M> {
        &self.init
    }
    fn set_params(&mut self, p: &M::V) {
        let p = p.clone_as_vec();
        self.rhs.p = [p[0], p[1], p[2]];
        self.init.p = self.rhs.p;
    }
    fn get_params(&self, p: &mut M::V) {
        let params = self.rhs.p;
        p.for_each_batch([], move |p, [], _| p.copy_from_slice(&params));
    }
}

/// The model with parameters `p`, as an ODE or (`dae`) a DAE, at tolerances `rtol` (relative
/// and absolute), with a tight Newton tolerance so that finite differences of solver steps are
/// not swamped by Newton error.
pub fn ad_test_problem<M: Matrix>(
    p: [f64; 3],
    dae: bool,
    rtol: f64,
) -> OdeSolverProblem<AdTestEqn<M>> {
    let ctx = M::C::default();
    let f = |x: f64| M::T::from_f64(x).unwrap();
    let p = p.map(f);
    let eqn = AdTestEqn {
        rhs: AdTestRhs {
            p,
            dae,
            ctx: ctx.clone(),
        },
        init: AdTestInit {
            p,
            ctx: ctx.clone(),
        },
        mass: dae.then(|| AdTestMass { ctx: ctx.clone() }),
        ctx: ctx.clone(),
    };
    let atol = M::V::from_element(N, f(rtol), ctx);
    let ode_options = crate::OdeSolverOptions {
        nonlinear_solver_tolerance: f(1e-6),
        ..Default::default()
    };
    OdeSolverProblem::new(
        eqn,
        f(rtol),
        atol,
        None,
        None,
        None,
        None,
        None,
        None,
        M::T::zero(),
        f(1e-3),
        false,
        Default::default(),
        ode_options,
    )
    .unwrap()
}
