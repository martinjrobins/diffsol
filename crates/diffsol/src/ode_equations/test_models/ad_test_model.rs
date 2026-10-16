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
//! Every operator works lane by lane on batched vectors, with a single-lane state broadcast over
//! the lanes of its argument, as the forward sensitivity equations call them.

use num_traits::{FromPrimitive, One, Zero};

use crate::{
    op::{unit::UnitCallable, ParameterisedOp},
    ConstantOp, ConstantOpSens, ConstantOpSensAdjoint, LinearOp, LinearOpTranspose, Matrix,
    NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint,
    OdeEquations, OdeEquationsRef, OdeSolverProblem, Op, Vector,
};

const N: usize = 3;
const NP: usize = 3;

/// `out = f(x lane, v lane)` for every lane of `v`, where `x` has either one lane or as many as
/// `v`, `v` lanes have `n_in` entries and `out` lanes have `n_out`.
fn map_lanes<V: Vector>(
    x: &V,
    v: &V,
    out: &mut V,
    n_in: usize,
    f: impl Fn(&[V::T], &[V::T]) -> Vec<V::T>,
) {
    let xs = x.clone_as_vec();
    let vs = v.clone_as_vec();
    let nlanes = vs.len() / n_in;
    let x_lanes = xs.len() / N;
    let mut data = Vec::new();
    for l in 0..nlanes {
        let xl = if x_lanes == 1 { 0 } else { l };
        data.extend(f(&xs[xl * N..(xl + 1) * N], &vs[l * n_in..(l + 1) * n_in]));
    }
    out.copy_from(&V::from_vec(data, out.context().clone()));
}

fn mat_vec<T: Copy + Zero + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(
    a: &[[T; 3]; 3],
    v: &[T],
    transpose: bool,
) -> Vec<T> {
    (0..3)
        .map(|i| {
            (0..3).fold(T::zero(), |acc, j| {
                acc + if transpose { a[j][i] } else { a[i][j] } * v[j]
            })
        })
        .collect()
}

pub struct AdTestRhs<M: Matrix> {
    p: Vec<M::T>,
    dae: bool,
    ctx: M::C,
}

impl<M: Matrix> AdTestRhs<M> {
    fn f(&self, y: &[M::T]) -> Vec<M::T> {
        let p = &self.p;
        let last = if self.dae {
            y[0] + y[1] + y[2] - p[2]
        } else {
            y[1] * y[1]
        };
        vec![
            -p[0] * y[0] + p[1] * y[1] * y[2],
            p[0] * y[0] - p[1] * y[1] * y[2] - y[1] * y[1],
            last,
        ]
    }

    fn jac(&self, y: &[M::T]) -> [[M::T; 3]; 3] {
        let p = &self.p;
        let (zero, one) = (M::T::zero(), M::T::one());
        let two = one + one;
        let last = if self.dae {
            [one, one, one]
        } else {
            [zero, two * y[1], zero]
        };
        [
            [-p[0], p[1] * y[2], p[1] * y[1]],
            [p[0], -p[1] * y[2] - two * y[1], -p[1] * y[1]],
            last,
        ]
    }

    fn jac_p(&self, y: &[M::T]) -> [[M::T; 3]; 3] {
        let zero = M::T::zero();
        let last = if self.dae { -M::T::one() } else { zero };
        [
            [-y[0], y[1] * y[2], zero],
            [y[0], -y[1] * y[2], zero],
            [zero, zero, last],
        ]
    }
}

pub struct AdTestInit<M: Matrix> {
    p: Vec<M::T>,
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
        map_lanes(x, x, y, N, |x, _| self.f(x));
    }
}

impl<M: Matrix> NonLinearOpJacobian for AdTestRhs<M> {
    fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        map_lanes(x, v, y, N, |x, v| mat_vec(&self.jac(x), v, false));
    }
}

impl<M: Matrix> NonLinearOpAdjoint for AdTestRhs<M> {
    fn jac_transpose_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        map_lanes(x, v, y, N, |x, v| {
            mat_vec(&self.jac(x), v, true)
                .into_iter()
                .map(|e| -e)
                .collect()
        });
    }
}

impl<M: Matrix> NonLinearOpSens for AdTestRhs<M> {
    fn sens_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        map_lanes(x, v, y, NP, |x, v| mat_vec(&self.jac_p(x), v, false));
    }
}

impl<M: Matrix> NonLinearOpSensAdjoint for AdTestRhs<M> {
    fn sens_transpose_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        map_lanes(x, v, y, N, |x, v| {
            mat_vec(&self.jac_p(x), v, true)
                .into_iter()
                .map(|e| -e)
                .collect()
        });
    }
}

impl<M: Matrix> ConstantOp for AdTestInit<M> {
    fn call_inplace(&self, _t: M::T, y: &mut M::V) {
        let nlanes = y.clone_as_vec().len() / N;
        let lane = [self.p[2], M::T::zero(), M::T::zero()];
        let data = (0..nlanes).flat_map(|_| lane).collect();
        y.copy_from(&M::V::from_vec(data, y.context().clone()));
    }
}

impl<M: Matrix> ConstantOpSens for AdTestInit<M> {
    fn sens_mul_inplace(&self, _t: M::T, v: &M::V, y: &mut M::V) {
        map_lanes(v, v, y, NP, |_, v| vec![v[2], M::T::zero(), M::T::zero()]);
    }
}

impl<M: Matrix> ConstantOpSensAdjoint for AdTestInit<M> {
    fn sens_transpose_mul_inplace(&self, _t: M::T, v: &M::V, y: &mut M::V) {
        map_lanes(v, v, y, N, |_, v| vec![M::T::zero(), M::T::zero(), -v[0]]);
    }
}

impl<M: Matrix> AdTestMass<M> {
    fn gemv(x: &M::V, beta: M::T, y: &mut M::V) {
        let mut mx = y.clone();
        map_lanes(x, x, &mut mx, N, |x, _| vec![x[0], x[1], M::T::zero()]);
        y.axpy(M::T::one(), &mx, beta);
    }
}

impl<M: Matrix> LinearOp for AdTestMass<M> {
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
        self.rhs.p = p.clone_as_vec();
        self.init.p = p.clone_as_vec();
    }
    fn get_params(&self, p: &mut M::V) {
        p.copy_from(&M::V::from_vec(self.rhs.p.clone(), p.context().clone()));
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
    let p: Vec<M::T> = p.iter().map(|&x| f(x)).collect();
    let eqn = AdTestEqn {
        rhs: AdTestRhs {
            p: p.clone(),
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
