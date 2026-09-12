use crate::{
    find_jacobian_non_zeros, ode_solver::problem::OdeSolverSolution, ConstantOp, Context,
    JacobianColoring, Matrix, MatrixSparsity, NonLinearOp, NonLinearOpJacobian, OdeBuilder,
    OdeEquations, OdeEquationsImplicit, OdeEquationsRef, OdeSolverProblem, Op, ParameterisedOp,
    UnitCallable, Vector,
};
use num_traits::{FromPrimitive, One, Zero};

/// The reference solution points, shared by every constructor in this module.
#[allow(clippy::type_complexity)]
fn robertson_soln_data() -> [([f64; 3], f64); 13] {
    [
        ([1.0, 0.0, 0.0], 0.0),
        ([9.851641e-01, 3.386242e-05, 1.480205e-02], 0.4),
        ([9.055097e-01, 2.240338e-05, 9.446793e-02], 4.0),
        ([7.158017e-01, 9.185037e-06, 2.841892e-01], 40.0),
        ([4.505360e-01, 3.223271e-06, 5.494608e-01], 400.0),
        ([1.832299e-01, 8.944378e-07, 8.167692e-01], 4000.0),
        ([3.898902e-02, 1.622006e-07, 9.610108e-01], 40000.0),
        ([4.936383e-03, 1.984224e-08, 9.950636e-01], 400000.0),
        ([5.168093e-04, 2.068293e-09, 9.994832e-01], 4000000.0),
        ([5.202440e-05, 2.081083e-10, 9.999480e-01], 4.0000e+07),
        ([5.201061e-06, 2.080435e-11, 9.999948e-01], 4.0000e+08),
        ([5.258603e-07, 2.103442e-12, 9.999995e-01], 4.0000e+09),
        ([6.934511e-08, 2.773804e-13, 9.999999e-01], 4.0000e+10),
    ]
}

#[cfg(feature = "diffsl")]
#[allow(clippy::type_complexity)]
pub fn robertson_ode_diffsl_problem<
    M: Matrix<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
>() -> (
    OdeSolverProblem<impl crate::OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let code = "
        in_i { k1 = 0.04, k2 = 10000, k3 = 30000000 }
        u_i {
            x = 1,
            y = 0,
            z = 0,
        }
        F_i {
            -k1*x + k2*y*z,
            k1*x - k2*y*z - k3*y*y,
            k3*y*y,
        }";

    let problem = OdeBuilder::<M>::new()
        .p([0.04, 1.0e4, 3.0e7])
        .rtol(1e-4)
        .atol([1.0e-8, 1.0e-6, 1.0e-6])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6, 1e-6])
        .param_rtol(1e-6)
        .param_atol([1e-6, 1e-6, 1e-6])
        .build_from_diffsl::<CG>(code)
        .unwrap();
    let mut soln = soln::<M::V>(problem.context().clone());
    soln.rtol = problem.rtol;
    soln.atol = problem.atol.clone();
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn robertson_ode<M: Matrix + 'static>(
    use_coloring: bool,
    ngroups: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    const N: usize = 3;
    let nstates = N * ngroups;
    let problem = OdeBuilder::<M>::new()
        .p([0.04, 1.0e4, 3.0e7])
        .rtol(1e-4)
        .atol(
            [1.0e-8, 1.0e-14, 1.0e-6]
                .iter()
                .cycle()
                .take(ngroups * N)
                .cloned()
                .collect::<Vec<f64>>(),
        )
        .use_coloring(use_coloring)
        .rhs_implicit(
            //     dy1/dt = -.04*y1 + 1.e4*y2*y3
            //*    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
            //*    dy3/dt = 3.e7*(y2)^2
            move |x: &[M::T], p: &[M::T], _t: M::T, y: &mut [M::T]| {
                for ig in 0..ngroups {
                    let i = ig * N;
                    y[i] = -p[0] * x[i] + p[1] * x[i + 1] * x[i + 2];
                    y[i + 1] =
                        p[0] * x[i] - p[1] * x[i + 1] * x[i + 2] - p[2] * x[i + 1] * x[i + 1];
                    y[i + 2] = p[2] * x[i + 1] * x[i + 1];
                }
            },
            move |x: &[M::T], p: &[M::T], _t: M::T, v: &[M::T], y: &mut [M::T]| {
                for ig in 0..ngroups {
                    let i = ig * N;
                    y[i] = -p[0] * v[i] + p[1] * v[i + 1] * x[i + 2] + p[1] * x[i + 1] * v[i + 2];
                    y[i + 1] = p[0] * v[i]
                        - p[1] * v[i + 1] * x[i + 2]
                        - p[1] * x[i + 1] * v[i + 2]
                        - M::T::from_f64(2.0).unwrap() * p[2] * x[i + 1] * v[i + 1];
                    y[i + 2] = M::T::from_f64(2.0).unwrap() * p[2] * x[i + 1] * v[i + 1];
                }
            },
        )
        .init(
            move |_p: &[M::T], _t: M::T, y: &mut [M::T]| {
                for ig in 0..ngroups {
                    let i = ig * N;
                    y[i] = M::T::one();
                    y[i + 1] = M::T::zero();
                    y[i + 2] = M::T::zero();
                }
            },
            nstates,
        )
        .build()
        .unwrap();

    let mut soln = OdeSolverSolution::default();
    for (values, time) in robertson_soln_data() {
        // expand soln by number of groups
        let mut newvalues = Vec::with_capacity(N * ngroups);
        for _ in 0..ngroups {
            newvalues.extend(values.iter().map(|v| M::T::from_f64(*v).unwrap()));
        }
        soln.push(
            M::V::from_vec(newvalues, problem.context().clone()),
            M::T::from_f64(time).unwrap(),
        );
    }
    (problem, soln)
}

#[cfg(feature = "diffsl")]
fn soln<V: Vector>(ctx: V::C) -> OdeSolverSolution<V> {
    let mut soln = OdeSolverSolution::default();
    for (values, time) in robertson_soln_data() {
        soln.push(
            V::from_vec(
                values.iter().map(|v| V::T::from_f64(*v).unwrap()).collect(),
                ctx.clone(),
            ),
            V::T::from_f64(time).unwrap(),
        );
    }
    soln
}

/* -----------------------------------------------------------------
 * Programmer(s): Scott D. Cohen, Alan C. Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2023, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem, with the coding
 * needed for its solution by CVODE. The problem is from
 * chemical kinetics, and consists of the following three rate
 * equations:
 *    dy1/dt = -.04*y1 + 1.e4*y2*y3
 *    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
 *    dy3/dt = 3.e7*(y2)^2
 * on the interval from t = 0.0 to t = 4.e10, with initial
 * conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
 * While integrating the system, we also use the rootfinding
 * feature to find the points at which y1 = 1e-4 or at which
 * y3 = 0.01. This program solves the problem with the BDF method,
 * Newton iteration with the dense linear solver, and a
 * user-supplied Jacobian routine.
 * It uses a scalar relative tolerance and a vector absolute
 * tolerance. Output is printed in decades from t = .4 to t = 4.e10.
 * Run statistics (optional outputs) are printed at the end.
 * -----------------------------------------------------------------*/

// 3-species kinetics problem

// At t = 2.6391e-01      y =  9.899653e-01    3.470564e-05    1.000000e-02
// rootsfound[] =   0   1
// At t = 4.0000e-01      y =  9.851641e-01    3.386242e-05    1.480205e-02
// At t = 4.0000e+00      y =  9.055097e-01    2.240338e-05    9.446793e-02
// At t = 4.0000e+01      y =  7.158017e-01    9.185037e-06    2.841892e-01
// At t = 4.0000e+02      y =  4.505360e-01    3.223271e-06    5.494608e-01
// At t = 4.0000e+03      y =  1.832299e-01    8.944378e-07    8.167692e-01
// At t = 4.0000e+04      y =  3.898902e-02    1.622006e-07    9.610108e-01
// At t = 4.0000e+05      y =  4.936383e-03    1.984224e-08    9.950636e-01
// At t = 4.0000e+06      y =  5.168093e-04    2.068293e-09    9.994832e-01
// At t = 2.0790e+07      y =  1.000000e-04    4.000397e-10    9.999000e-01
// rootsfound[] =  -1   0
// At t = 4.0000e+07      y =  5.202440e-05    2.081083e-10    9.999480e-01
// At t = 4.0000e+08      y =  5.201061e-06    2.080435e-11    9.999948e-01
// At t = 4.0000e+09      y =  5.258603e-07    2.103442e-12    9.999995e-01
// At t = 4.0000e+10      y =  6.934511e-08    2.773804e-13    9.999999e-01
//
// Final Statistics:
// Current time                 = 41154661313.59952
// Steps                        = 542
// Error test fails             = 22
// NLS step fails               = 0
// Initial step size            = 8.236259832589498e-14
// Last step size               = 4747036977.219155
// Current step size            = 4747036977.219155
// Last method order            = 4
// Current method order         = 4
// Stab. lim. order reductions  = 0
// RHS fn evals                 = 754
// NLS iters                    = 751
// NLS fails                    = 3
// NLS iters per step           = 1.385608856088561
// LS setups                    = 107
// Jac fn evals                 = 11
// LS RHS fn evals              = 0
// Prec setup evals             = 0
// Prec solves                  = 0
// LS iters                     = 0
// LS fails                     = 0
// Jac-times setups             = 0
// Jac-times evals              = 0
// LS iters per NLS iter        = 0
// Jac evals per NLS iter       = 0.01464713715046605
// Prec evals per NLS iter      = 0
// Root fn evals                = 570

// ============================================================
// Element-parallel implementation
// ============================================================

/// States in the Robertson system, and parameters (the three rate constants).
const RE_NSTATES: usize = 3;
const RE_NPARAMS: usize = 3;

/// `dy/dt` for one state of the Robertson system.
///
/// The element form of the body in [`robertson_ode`]: `i` is the state being written, and the
/// whole 3-state lane plus the parameter lane are readable. This is what lets
/// [`Vector::for_each_elem`] run it a thread per `(lane, state)` on a device backend.
fn robertson_elem_rhs<T: crate::Scalar>(y: &mut T, x: &[T], p: &[T], i: usize) {
    *y = match i {
        0 => -p[0] * x[0] + p[1] * x[1] * x[2],
        1 => p[0] * x[0] - p[1] * x[1] * x[2] - p[2] * x[1] * x[1],
        _ => p[2] * x[1] * x[1],
    };
}

/// One state of `J v` for [`robertson_elem_rhs`].
fn robertson_elem_jac_mul<T: crate::Scalar>(y: &mut T, x: &[T], p: &[T], v: &[T], i: usize) {
    // `T::from_f64(2.0).unwrap()` would put a panic path in the kernel
    let two = T::one() + T::one();
    *y = match i {
        0 => -p[0] * v[0] + p[1] * v[1] * x[2] + p[1] * x[1] * v[2],
        1 => p[0] * v[0] - p[1] * v[1] * x[2] - p[1] * x[1] * v[2] - two * p[2] * x[1] * v[1],
        _ => two * p[2] * x[1] * v[1],
    };
}

/// The Robertson kinetics with an element-parallel right-hand side.
///
/// Batching takes the place of [`robertson_ode`]'s `ngroups`: every lane is an independent copy
/// of the 3-state system, so the lane count is the group count.
pub struct RobertsonOdeElem<M: Matrix> {
    p: M::V,
    rhs_sparsity: Option<M::Sparsity>,
    rhs_coloring: Option<JacobianColoring<M>>,
    ctx: M::C,
}

impl<M: Matrix> RobertsonOdeElem<M> {
    pub fn new(p: M::V, t0: M::T) -> Self {
        let ctx = p.context().clone();
        let mut ret = Self {
            p,
            rhs_sparsity: None,
            rhs_coloring: None,
            ctx,
        };
        let y0 = RobertsonOdeElemInit { eqn: &ret }.call(t0);
        let rhs = RobertsonOdeElemRhs { eqn: &ret };
        let non_zeros = find_jacobian_non_zeros(&rhs, &y0, t0);
        ret.rhs_sparsity = Some(
            MatrixSparsity::try_from_indices(rhs.nout(), rhs.nstates(), non_zeros.clone()).unwrap(),
        );
        ret.rhs_coloring = Some(JacobianColoring::new(
            ret.rhs_sparsity.as_ref().unwrap(),
            &non_zeros,
            ret.ctx.clone(),
        ));
        ret
    }
}

pub struct RobertsonOdeElemRhs<'a, M: Matrix> {
    eqn: &'a RobertsonOdeElem<M>,
}

pub struct RobertsonOdeElemInit<'a, M: Matrix> {
    eqn: &'a RobertsonOdeElem<M>,
}

macro_rules! impl_robertson_elem_op {
    ($name:ident) => {
        impl<M: Matrix> Op for $name<'_, M> {
            type M = M;
            type V = M::V;
            type T = M::T;
            type C = M::C;

            fn nstates(&self) -> usize {
                RE_NSTATES
            }
            fn nout(&self) -> usize {
                RE_NSTATES
            }
            fn nparams(&self) -> usize {
                RE_NPARAMS
            }
            fn context(&self) -> &Self::C {
                &self.eqn.ctx
            }
        }
    };
}

impl_robertson_elem_op!(RobertsonOdeElemRhs);
impl_robertson_elem_op!(RobertsonOdeElemInit);

impl<M: Matrix> NonLinearOp for RobertsonOdeElemRhs<'_, M> {
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        y.for_each_elem(
            [x, &self.eqn.p],
            |y: &mut M::T, [x, p]: [&[M::T]; 2], _lane: usize, i: usize| {
                robertson_elem_rhs(y, x, p, i)
            },
        );
    }
}

impl<M: Matrix> NonLinearOpJacobian for RobertsonOdeElemRhs<'_, M> {
    fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        y.for_each_elem(
            [x, &self.eqn.p, v],
            |y: &mut M::T, [x, p, v]: [&[M::T]; 3], _lane: usize, i: usize| {
                robertson_elem_jac_mul(y, x, p, v, i)
            },
        );
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.eqn.rhs_coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
    fn jacobian_sparsity(&self) -> Option<M::Sparsity> {
        self.eqn.rhs_sparsity.clone()
    }
}

impl<M: Matrix> ConstantOp for RobertsonOdeElemInit<'_, M> {
    fn call_inplace(&self, _t: M::T, y: &mut M::V) {
        y.for_each_elem(
            [],
            |y: &mut M::T, _: [&[M::T]; 0], _lane: usize, i: usize| {
                *y = if i == 0 { M::T::one() } else { M::T::zero() };
            },
        );
    }
}

impl<M: Matrix> Op for RobertsonOdeElem<M> {
    type M = M;
    type V = M::V;
    type T = M::T;
    type C = M::C;

    fn nstates(&self) -> usize {
        RE_NSTATES
    }
    fn nout(&self) -> usize {
        RE_NSTATES
    }
    fn nparams(&self) -> usize {
        RE_NPARAMS
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<'a, M: Matrix> OdeEquationsRef<'a> for RobertsonOdeElem<M> {
    type Rhs = RobertsonOdeElemRhs<'a, M>;
    type Init = RobertsonOdeElemInit<'a, M>;
    type Mass = ParameterisedOp<'a, UnitCallable<M>>;
    type Root = ParameterisedOp<'a, UnitCallable<M>>;
    type Out = ParameterisedOp<'a, UnitCallable<M>>;
    type Reset = ParameterisedOp<'a, UnitCallable<M>>;
}

impl<M: Matrix> OdeEquations for RobertsonOdeElem<M> {
    fn rhs(&self) -> RobertsonOdeElemRhs<'_, M> {
        RobertsonOdeElemRhs { eqn: self }
    }
    fn init(&self) -> RobertsonOdeElemInit<'_, M> {
        RobertsonOdeElemInit { eqn: self }
    }
    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        None
    }
    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        None
    }
    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn set_params(&mut self, p: &Self::V) {
        self.p.copy_from(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        p.copy_from(&self.p);
    }
}

/// [`robertson_ode`] with an element-parallel right-hand side, over `nbatch` lanes.
///
/// The three rate constants stay parameters (`nparams == 3`), so forward sensitivities can be
/// added later; they are held at the context's lane count and read out of the parameter lane by
/// the kernel.
#[allow(clippy::type_complexity)]
pub fn robertson_ode_elem_problem<M: Matrix + 'static>(
    nbatch: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let ctx = M::C::default().clone_with_nbatch(nbatch).unwrap();
    let per_lane = |values: [f64; 3]| {
        let mut v = Vec::with_capacity(3 * nbatch);
        for _ in 0..nbatch {
            v.extend(values.iter().map(|x| M::T::from_f64(*x).unwrap()));
        }
        M::V::from_vec(v, ctx.clone())
    };
    let p = per_lane([0.04, 1.0e4, 3.0e7]);
    let atol = per_lane([1.0e-8, 1.0e-14, 1.0e-6]);
    let rtol = M::T::from_f64(1e-4).unwrap();
    let t0 = M::T::zero();
    let h0 = M::T::one();
    let eqn = RobertsonOdeElem::new(p, t0);
    let problem = OdeSolverProblem::new(
        eqn,
        rtol,
        atol,
        None,
        None,
        None,
        None,
        None,
        None,
        t0,
        h0,
        false,
        Default::default(),
        Default::default(),
    )
    .unwrap();
    let mut soln = OdeSolverSolution::<M::V> {
        atol: per_lane([1.0e-8, 1.0e-14, 1.0e-6]),
        rtol,
        ..Default::default()
    };
    for (values, time) in robertson_soln_data() {
        soln.push(
            per_lane([values[0], values[1], values[2]]),
            M::T::from_f64(time).unwrap(),
        );
    }
    (problem, soln)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix::dense_nalgebra_serial::NalgebraMat, NalgebraVec};

    /// The element right-hand side and the group right-hand side of [`robertson_ode`] are two
    /// copies of the same maths, so check them against each other at `ngroups == 1`.
    #[test]
    fn test_elem_matches_groups() {
        type M = NalgebraMat<f64>;
        let (elem, _) = robertson_ode_elem_problem::<M>(1);
        let (groups, _) = robertson_ode::<M>(false, 1);

        let y0 = elem.eqn.init().call(0.0);
        let y0_groups = groups.eqn.init().call(0.0);
        y0.assert_eq_st(&y0_groups, 1e-14);

        let x = NalgebraVec::from_vec(vec![0.7, 1e-5, 0.3], *elem.context());
        elem.eqn
            .rhs()
            .call(&x, 0.0)
            .assert_eq_st(&groups.eqn.rhs().call(&x, 0.0), 1e-10);

        let v = NalgebraVec::from_vec(vec![0.1, 0.2, 0.3], *elem.context());
        elem.eqn
            .rhs()
            .jac_mul(&x, 0.0, &v)
            .assert_eq_st(&groups.eqn.rhs().jac_mul(&x, 0.0, &v), 1e-10);
    }
}
