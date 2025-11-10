use crate::{
    matrix::Matrix, ode_solver::problem::OdeSolverSolution, MatrixHost, OdeBuilder,
    OdeEquationsImplicit, OdeEquationsImplicitSens, OdeSolverProblem, Op, Vector,
};
use num_traits::{FromPrimitive, One, Zero};

#[cfg(feature = "diffsl")]
#[allow(clippy::type_complexity)]
pub fn robertson_diffsl_problem<
    M: MatrixHost<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
>() -> (
    OdeSolverProblem<impl crate::OdeEquationsImplicitAdjoint<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let code = "
        in = [k1, k2, k3]
        k1 { 0.04 }
        k2 { 10000 }
        k3 { 30000000 }
        u_i {
            x = 1,
            y = 0,
            z = 0,
        }
        dudt_i {
            dxdt = 1,
            dydt = 0,
            dzdt = 0,
        }
        M_i {
            dxdt,
            dydt,
            0,
        }
        F_i {
            -k1 * x + k2 * y * z,
            k1 * x - k2 * y * z - k3 * y * y,
            1 - x - y - z,
        }
        out_i {
            x,
            y,
            z,
        }";

    let problem = OdeBuilder::<M>::new()
        .p([0.04, 1.0e4, 3.0e7])
        .rtol(1e-4)
        .atol([1.0e-8, 1.0e-6, 1.0e-6])
        .build_from_diffsl::<CG>(code)
        .unwrap();
    let mut soln = soln::<M::V>(problem.context().clone());
    soln.rtol = problem.rtol;
    soln.atol = problem.atol.clone();
    (problem, soln)
}

//*      dy1/dt = -.04*y1 + 1.e4*y2*y3
//*      dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
//*         0   = y1 + y2 + y3 - 1
fn robertson_rhs<M: MatrixHost>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = -p[0] * x[0] + p[1] * x[1] * x[2];
    y[1] = p[0] * x[0] - p[1] * x[1] * x[2] - p[2] * x[1] * x[1];
    y[2] = x[0] + x[1] + x[2] - M::T::one();
}
fn robertson_jac_mul<M: MatrixHost>(x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y[0] = -p[0] * v[0] + p[1] * v[1] * x[2] + p[1] * x[1] * v[2];
    y[1] = p[0] * v[0]
        - p[1] * v[1] * x[2]
        - p[1] * x[1] * v[2]
        - M::T::from_f64(2.0).unwrap() * p[2] * x[1] * v[1];
    y[2] = v[0] + v[1] + v[2];
}

fn robertson_sens_mul<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y[0] = -v[0] * x[0] + v[1] * x[1] * x[2];
    y[1] = v[0] * x[0] - v[1] * x[1] * x[2] - v[2] * x[1] * x[1];
    y[2] = M::T::zero();
}

fn robertson_mass<M: MatrixHost>(x: &M::V, _p: &M::V, _t: M::T, beta: M::T, y: &mut M::V) {
    y[0] = x[0] + beta * y[0];
    y[1] = x[1] + beta * y[1];
    y[2] = beta * y[2];
}

fn robertson_init<M: MatrixHost>(_p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = M::T::one();
    y[1] = M::T::zero();
    y[2] = M::T::zero();
}

fn robertson_init_sens<M: Matrix>(_p: &M::V, _t: M::T, _v: &M::V, y: &mut M::V) {
    y.fill(M::T::zero());
}

#[allow(clippy::type_complexity)]
pub fn robertson<M: MatrixHost>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .p([0.04, 1.0e4, 3.0e7])
        .rtol(1e-4)
        .atol([1.0e-8, 1.0e-6, 1.0e-6])
        .use_coloring(use_coloring)
        .rhs_implicit(robertson_rhs::<M>, robertson_jac_mul::<M>)
        .mass(robertson_mass::<M>)
        .init(robertson_init::<M>, 3)
        .build()
        .unwrap();

    let ctx = problem.context().clone();
    (problem, soln(ctx))
}

fn soln<V: Vector>(ctx: V::C) -> OdeSolverSolution<V> {
    let mut soln = OdeSolverSolution::default();
    let data = vec![
        (vec![1.0, 0.0, 0.0], 0.0),
        (vec![9.8517e-01, 3.3864e-05, 1.4794e-02], 0.4),
        (vec![9.0553e-01, 2.2406e-05, 9.4452e-02], 4.0),
        (vec![7.1579e-01, 9.1838e-06, 2.8420e-01], 40.0),
        (vec![4.5044e-01, 3.2218e-06, 5.4956e-01], 400.0),
        (vec![1.8320e-01, 8.9444e-07, 8.1680e-01], 4000.0),
        (vec![3.8992e-02, 1.6221e-07, 9.6101e-01], 40000.0),
        (vec![4.9369e-03, 1.9842e-08, 9.9506e-01], 400000.0),
        (vec![5.1674e-04, 2.0684e-09, 9.9948e-01], 4000000.0),
        (vec![5.2009e-05, 2.0805e-10, 9.9995e-01], 4.0000e+07),
        (vec![5.2012e-06, 2.0805e-11, 9.9999e-01], 4.0000e+08),
        (vec![5.1850e-07, 2.0740e-12, 1.0000e+00], 4.0000e+09),
        (vec![4.8641e-08, 1.9456e-13, 1.0000e+00], 4.0000e+10),
    ];

    for (values, time) in data {
        soln.push(
            V::from_vec(
                values
                    .into_iter()
                    .map(|v| V::T::from_f64(v).unwrap())
                    .collect(),
                ctx.clone(),
            ),
            V::T::from_f64(time).unwrap(),
        );
    }
    soln
}

#[allow(clippy::type_complexity)]
pub fn robertson_sens<M: MatrixHost + 'static>() -> (
    OdeSolverProblem<impl OdeEquationsImplicitSens<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::<M>::new()
        .atol([1e-8, 1e-6, 1e-6])
        .rtol(1e-4)
        .p([0.04, 1.0e4, 3.0e7])
        .turn_off_sensitivities_error_control()
        .rhs_sens_implicit(
            robertson_rhs::<M>,
            robertson_jac_mul::<M>,
            robertson_sens_mul::<M>,
        )
        .init_sens(robertson_init::<M>, robertson_init_sens::<M>, 3)
        .mass(robertson_mass::<M>)
        .build()
        .unwrap();

    let mut soln = OdeSolverSolution::default();
    let data = vec![
        (vec![1.0, 0.0, 0.0], 0.0),
        (vec![9.8517e-01, 3.3864e-05, 1.4794e-02], 0.4),
        (vec![9.0553e-01, 2.2406e-05, 9.4452e-02], 4.0),
        (vec![7.1579e-01, 9.1838e-06, 2.8420e-01], 40.0),
        (vec![4.5044e-01, 3.2218e-06, 5.4956e-01], 400.0),
        (vec![1.8320e-01, 8.9444e-07, 8.1680e-01], 4000.0),
        (vec![3.8992e-02, 1.6221e-07, 9.6101e-01], 40000.0),
        (vec![4.9369e-03, 1.9842e-08, 9.9506e-01], 400000.0),
        (vec![5.1674e-04, 2.0684e-09, 9.9948e-01], 4000000.0),
        (vec![5.2009e-05, 2.0805e-10, 9.9995e-01], 4.0000e+07),
        (vec![5.2012e-06, 2.0805e-11, 9.9999e-01], 4.0000e+08),
        (vec![5.1850e-07, 2.0740e-12, 1.0000e+00], 4.0000e+09),
        (vec![4.8641e-08, 1.9456e-13, 1.0000e+00], 4.0000e+10),
    ];

    for (values, time) in data {
        soln.push(
            M::V::from_vec(
                values
                    .into_iter()
                    .map(|v| M::T::from_f64(v).unwrap())
                    .collect(),
                problem.eqn.context().clone(),
            ),
            M::T::from_f64(time).unwrap(),
        );
    }

    (problem, soln)
}

/* -----------------------------------------------------------------
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
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
 * This simple example problem for IDA, due to Robertson,
 * is from chemical kinetics, and consists of the following three
 * equations:
 *
 *      dy1/dt = -.04*y1 + 1.e4*y2*y3
 *      dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
 *         0   = y1 + y2 + y3 - 1
 *
 * on the interval from t = 0.0 to t = 4.e10, with initial
 * conditions: y1 = 1, y2 = y3 = 0.
 *
 * While integrating the system, we also use the rootfinding
 * feature to find the points at which y1 = 1e-4 or at which
 * y3 = 0.01.
 *
 * The problem is solved with IDA using the DENSE linear
 * solver, with a user-supplied Jacobian. Output is printed at
 * t = .4, 4, 40, ..., 4e10.
 * -----------------------------------------------------------------*/

// Output from Sundials IDA serial example problem for Robertson kinetics:
//
//idaRoberts_dns: Robertson kinetics DAE serial example problem for IDA
//         Three equation chemical kinetics problem.
//
//Linear solver: DENSE, with user-supplied Jacobian.
//Tolerance parameters:  rtol = 0.0001   atol = 1e-08 1e-06 1e-06
//Initial conditions y0 = (1 0 0)
//Constraints and id not used.
//
//-----------------------------------------------------------------------
//  t             y1           y2           y3      | nst  k      h
//-----------------------------------------------------------------------
//2.6402e-01   9.8997e-01   3.4706e-05   1.0000e-02 |  27  2   4.4012e-02
//    rootsfound[] =   0   1
//4.0000e-01   9.8517e-01   3.3864e-05   1.4794e-02 |  29  3   8.8024e-02
//4.0000e+00   9.0553e-01   2.2406e-05   9.4452e-02 |  43  4   6.3377e-01
//4.0000e+01   7.1579e-01   9.1838e-06   2.8420e-01 |  68  4   3.1932e+00
//4.0000e+02   4.5044e-01   3.2218e-06   5.4956e-01 |  95  4   3.3201e+01
//4.0000e+03   1.8320e-01   8.9444e-07   8.1680e-01 | 126  3   3.1458e+02
//4.0000e+04   3.8992e-02   1.6221e-07   9.6101e-01 | 161  5   2.5058e+03
//4.0000e+05   4.9369e-03   1.9842e-08   9.9506e-01 | 202  3   2.6371e+04
//4.0000e+06   5.1674e-04   2.0684e-09   9.9948e-01 | 250  3   1.7187e+05
//2.0788e+07   1.0000e-04   4.0004e-10   9.9990e-01 | 280  5   1.0513e+06
//    rootsfound[] =  -1   0
//4.0000e+07   5.2009e-05   2.0805e-10   9.9995e-01 | 293  4   2.3655e+06
//4.0000e+08   5.2012e-06   2.0805e-11   9.9999e-01 | 325  4   2.6808e+07
//4.0000e+09   5.1850e-07   2.0740e-12   1.0000e+00 | 348  3   7.4305e+08
//4.0000e+10   4.8641e-08   1.9456e-13   1.0000e+00 | 362  2   7.5480e+09
//
//Final Statistics:
//Current time                 = 41226212070.53522
//Steps                        = 362
//Error test fails             = 15
//NLS step fails               = 0
//Initial step size            = 2.164955286048077e-05
//Last step size               = 7548045540.281308
//Current step size            = 7548045540.281308
//Last method order            = 2
//Current method order         = 2
//Residual fn evals            = 537
//IC linesearch backtrack ops  = 0
//NLS iters                    = 537
//NLS fails                    = 5
//NLS iters per step           = 1.483425414364641
//LS setups                    = 60
//Jac fn evals                 = 60
//LS residual fn evals         = 0
//Prec setup evals             = 0
//Prec solves                  = 0
//LS iters                     = 0
//LS fails                     = 0
//Jac-times setups             = 0
//Jac-times evals              = 0
//LS iters per NLS iter        = 0
//Jac evals per NLS iter       = 0.111731843575419
//Prec evals per NLS iter      = 0
//Root fn evals                = 404
