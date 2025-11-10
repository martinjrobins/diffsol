//
//heatres: heat equation system residual function
//This uses 5-point central differencing on the interior points, and
//includes algebraic equations for the boundary values.
//So for each interior point, the residual component has the form
//   res_i = u'_i - (central difference)_i
//while for each boundary point, it is res_i = u_i.

use crate::{
    ode_solver::problem::OdeSolverSolution, scalar::Scalar, Matrix, MatrixHost, OdeBuilder,
    OdeEquationsImplicit, OdeSolverProblem, Vector,
};
use nalgebra::ComplexField;
use num_traits::{FromPrimitive, One, Zero};

#[cfg(feature = "diffsl")]
use crate::{ConstantOp, LinearOp, NonLinearOpJacobian, OdeEquations};

#[cfg(feature = "diffsl")]
#[allow(clippy::type_complexity)]
pub fn heat2d_diffsl_problem<
    M: MatrixHost<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
    const MGRID: usize,
>() -> (
    OdeSolverProblem<impl crate::OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    use crate::VectorHost;
    let (problem, _soln) = head2d_problem::<M, MGRID>();
    let u0 = problem.eqn.init().call(0.0);
    let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
    let mass = problem.eqn.mass().unwrap().matrix(0.0);
    let init = problem.eqn.init().call(0.0);
    let init_diffsl = init
        .as_slice()
        .iter()
        .map(|v| format!("            {}", *v))
        .collect::<Vec<_>>()
        .join(",\n");
    let jac_diffsl = jac
        .triplet_iter()
        .map(|(i, j, v)| format!("            ({i}, {j}): {v}"))
        .collect::<Vec<_>>()
        .join(",\n");

    let mass_ones = mass.triplet_iter().map(|(i, _j, _v)| i).collect::<Vec<_>>();
    let mut mass_diffsl = Vec::new();
    for i in 0..MGRID * MGRID {
        // check if i in mass_ones
        if mass_ones.contains(&i) {
            mass_diffsl.push(format!("            ({i}, {i}): 1"));
        } else {
            mass_diffsl.push(format!("            ({i}, {i}): 0"));
        }
    }
    let mass_diffsl = mass_diffsl.join(",\n");

    let code = format!(
        "
        in = []
        D_ij {{
{}
        }}
        Mass_ij {{
{}
        }}
        init_i {{
{}
        }}
        u_i {{
            y = init_i,
        }}
        dudt_i {{
            (0:{n}): dydt = 0,
        }}
        M_i {{
            Mass_ij * dydt_j,
        }}
        F_i {{
            D_ij * y_j,
        }}
        out_i {{
            {dx2} * y_j * y_j,
        }}",
        jac_diffsl,
        mass_diffsl,
        init_diffsl,
        n = MGRID * MGRID,
        dx2 = (1.0 / (MGRID as f64 - 1.0)).powi(2),
    );

    let problem = OdeBuilder::<M>::new()
        .rtol(1e-7)
        .atol([1e-7])
        .build_from_diffsl::<CG>(code.as_str())
        .unwrap();
    let soln = soln::<M>(problem.context().clone());
    (problem, soln)
}

fn heat2d_rhs<M: MatrixHost, const MGRID: usize>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    // Initialize y to x, to take care of boundary equations.
    y.copy_from(x);
    let mm = M::T::from_f64(MGRID as f64).unwrap();
    let four = M::T::from_f64(4.0).unwrap();

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] =
                coeff * (x[loc - 1] + x[loc + 1] + x[loc - MGRID] + x[loc + MGRID] - four * x[loc]);
        }
    }
}

fn heat2d_jac_mul<M: MatrixHost, const MGRID: usize>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    // Initialize y to x, to take care of boundary equations.
    y.copy_from(v);
    let mm = M::T::from_f64(MGRID as f64).unwrap();
    let four = M::T::from_f64(4.0).unwrap();

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] =
                coeff * (v[loc - 1] + v[loc + 1] + v[loc - MGRID] + v[loc + MGRID] - four * v[loc]);
        }
    }
}

fn heat2d_init<M: MatrixHost, const MGRID: usize>(_p: &M::V, _t: M::T, uu: &mut M::V) {
    let mm = M::T::from_f64(MGRID as f64).unwrap();
    let bval = M::T::zero();
    let one = M::T::one();
    let dx = one / (mm - one);
    let mm1 = MGRID - 1;
    let sixteen = M::T::from_f64(16.0).unwrap();

    /* Initialize uu on all grid points. */
    for j in 0..MGRID {
        let yfact = dx * M::T::from_f64(j as f64).unwrap();
        let offset = MGRID * j;
        for i in 0..MGRID {
            let xfact = dx * M::T::from_f64(i as f64).unwrap();
            let loc = offset + i;
            uu[loc] = sixteen * xfact * (one - xfact) * yfact * (one - yfact);
        }
    }

    /* Finally, set values of u at boundary points. */
    for j in 0..MGRID {
        let offset = MGRID * j;
        for i in 0..MGRID {
            let loc = offset + i;
            if j == 0 || j == mm1 || i == 0 || i == mm1 {
                uu[loc] = bval;
            }
        }
    }
}

fn heat2d_mass<M: MatrixHost, const MGRID: usize>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    let mm = MGRID;
    let mm1 = mm - 1;
    for j in 0..mm {
        let offset = mm * j;
        for i in 0..mm {
            let loc = offset + i;
            if j == 0 || j == mm1 || i == 0 || i == mm1 {
                y[loc] *= beta;
            } else {
                y[loc] = x[loc] + beta * y[loc];
            }
        }
    }
}

fn heat2d_out<M: MatrixHost, const MGRID: usize>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    let dx = M::T::one() / (M::T::from_f64(MGRID as f64).unwrap() - M::T::one());
    let norm = x.norm(2);
    y[0] = (norm * dx).powi(2);
}

fn heat2d_out_jac_mul<M: Matrix, const MGRID: usize>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    _y: &mut M::V,
) {
    unimplemented!()
}

fn _pde_solution<T: Scalar>(x: T, y: T, t: T, max_terms: usize) -> T {
    let mut u = T::zero();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    let four = T::from_f64(4.0).unwrap();
    let two = T::from_f64(2.0).unwrap();
    let sixteen = T::from_f64(16.0).unwrap();

    for n in 1..=max_terms {
        let nt = T::from_f64(n as f64).unwrap();
        for m in 1..=max_terms {
            let mt = T::from_f64(m as f64).unwrap();
            let ii = (-pi * mt * (pi * mt).sin() - two * (pi * mt).cos() + two)
                / (pi.powi(3) * mt.powi(3));
            let jj = (-pi * nt * (pi * nt).sin() - two * (pi * nt).cos() + two)
                / (pi.powi(3) * nt.powi(3));
            let coefficient = four * sixteen * ii * jj;

            let sin_term = (nt * pi * x).sin() * (mt * pi * y).sin();
            let exp_term = ((-(nt * pi).powi(2) - (mt * pi).powi(2)) * t).exp();

            u += coefficient * sin_term * exp_term;
        }
    }

    u
}

#[allow(clippy::type_complexity)]
pub fn head2d_problem<M: MatrixHost + 'static, const MGRID: usize>() -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let nstates = MGRID * MGRID;
    let problem = OdeBuilder::<M>::new()
        .rtol(1e-7)
        .atol([1e-7])
        .rhs_implicit(heat2d_rhs::<M, MGRID>, heat2d_jac_mul::<M, MGRID>)
        .mass(heat2d_mass::<M, MGRID>)
        .init(heat2d_init::<M, MGRID>, nstates)
        .out_implicit(heat2d_out::<M, MGRID>, heat2d_out_jac_mul::<M, MGRID>, 1)
        .build()
        .unwrap();
    let ctx = problem.context().clone();

    (problem, soln::<M>(ctx))
}

fn soln<M: Matrix>(ctx: M::C) -> OdeSolverSolution<M::V> {
    let mut soln = OdeSolverSolution {
        solution_points: Vec::new(),
        sens_solution_points: None,
        rtol: M::T::from_f64(1e-5).unwrap(),
        atol: M::V::from_element(1, M::T::from_f64(1e-5).unwrap(), ctx.clone()),
        negative_time: false,
    };
    let data = vec![
        (vec![0.28435774340267284], 0.0),
        (vec![0.19195491512700597], 0.01),
        (vec![0.12979676270145094], 0.02),
        (vec![0.05939666913561712], 0.04),
        (vec![0.012441804151214689], 0.08),
        (vec![0.0005459318925793768], 0.16),
        (vec![1.05130465235137e-6], 0.32),
        (vec![3.983888966577838e-12], 0.64),
        (vec![7.015395128730499e-16], 1.28),
        (vec![5.06965159341517e-17], 2.56),
        (vec![1.735145399301106e-18], 5.12),
        (vec![3.259034338585213e-17], 10.24),
    ];
    for (values, time) in data {
        let values = M::V::from_vec(
            values
                .iter()
                .map(|v| M::T::from_f64(*v).unwrap())
                .collect::<Vec<_>>(),
            ctx.clone(),
        );
        let time = M::T::from_f64(time).unwrap();
        soln.push(values, time);
    }
    soln
}

#[cfg(test)]
mod tests {
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat, ConstantOp, LinearOp, NonLinearOpJacobian,
        OdeEquations,
    };

    use super::*;

    #[test]
    fn test_jacobian() {
        //let jac = heat2d_jacobian::<nalgebra::DMatrix<f64>, 10>();
        let (problem, _soln) = head2d_problem::<NalgebraMat<f64>, 10>();
        let u0 = problem.eqn.init().call(0.0);
        let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
        insta::assert_yaml_snapshot!(jac.data.to_string());
    }

    #[test]
    fn test_mass() {
        let (problem, _soln) = head2d_problem::<NalgebraMat<f64>, 10>();
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        insta::assert_yaml_snapshot!(mass.data.to_string());
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn test_mass_diffsl() {
        use crate::{FaerSparseMat, FaerVec};
        use diffsl::CraneliftJitModule;

        let (problem, _soln) = heat2d_diffsl_problem::<FaerSparseMat<f64>, CraneliftJitModule, 5>();
        let u = FaerVec::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
            *problem.context(),
        );
        let mut y = FaerVec::zeros(25, *problem.context());
        problem.eqn.mass().unwrap().call_inplace(&u, 0.0, &mut y);
        let expect = FaerVec::from_vec(
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 12.0, 13.0, 14.0, 0.0, 0.0,
                17.0, 18.0, 19.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            *problem.context(),
        );
        y.assert_eq_st(&expect, 1.0e-10);
    }

    #[test]
    fn test_soln() {
        let (_problem, _soln) = head2d_problem::<NalgebraMat<f64>, 10>();
    }
}
