//
//heatres: heat equation system residual function
//This uses 5-point central differencing on the interior points, and
//includes algebraic equations for the boundary values.
//So for each interior point, the residual component has the form
//   res_i = u'_i - (central difference)_i
//while for each boundary point, it is res_i = u_i.

use crate::{
    ode_solver::problem::OdeSolverSolution, scalar::Scalar, Matrix, OdeBuilder, OdeEquations,
    OdeSolverProblem, Vector,
};
use num_traits::{One, Zero};

fn heat2d_rhs<M: Matrix, const MGRID: usize>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    // Initialize y to x, to take care of boundary equations.
    y.copy_from(x);
    let mm = M::T::from(MGRID as f64);

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] = coeff
                * (x[loc - 1] + x[loc + 1] + x[loc - MGRID] + x[loc + MGRID]
                    - M::T::from(4.0) * x[loc]);
        }
    }
}

fn heat2d_jac_mul<M: Matrix, const MGRID: usize>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    // Initialize y to x, to take care of boundary equations.
    y.copy_from(v);
    let mm = M::T::from(MGRID as f64);

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] = coeff
                * (v[loc - 1] + v[loc + 1] + v[loc - MGRID] + v[loc + MGRID]
                    - M::T::from(4.0) * v[loc]);
        }
    }
}

fn heat2d_init<M: Matrix, const MGRID: usize>(_p: &M::V, _t: M::T) -> M::V {
    let mm = M::T::from(MGRID as f64);
    let mut uu = M::V::zeros(MGRID * MGRID);
    let bval = M::T::zero();
    let one = M::T::one();
    let dx = one / (mm - one);
    let mm1 = MGRID - 1;

    /* Initialize uu on all grid points. */
    for j in 0..MGRID {
        let yfact = dx * M::T::from(j as f64);
        let offset = MGRID * j;
        for i in 0..MGRID {
            let xfact = dx * M::T::from(i as f64);
            let loc = offset + i;
            uu[loc] = M::T::from(16.0) * xfact * (one - xfact) * yfact * (one - yfact);
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
    uu
}

fn heat2d_mass<M: Matrix, const MGRID: usize>(
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

fn heat2d_out<M: Matrix, const MGRID: usize>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    let mut max_y = M::T::zero();
    for j in 0..MGRID {
        let offset = MGRID * j;
        for i in 0..MGRID {
            let loc = offset + i;
            if x[loc] > max_y {
                max_y = x[loc];
            }
        }
    }
    y[0] = max_y;
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
    let pi = T::from(std::f64::consts::PI);
    let four = T::from(4.0);
    let two = T::from(2.0);
    let sixteen = T::from(16.0);

    for n in 1..=max_terms {
        let nt = T::from(n as f64);
        for m in 1..=max_terms {
            let mt = T::from(m as f64);
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

pub fn head2d_problem<M: Matrix + 'static, const MGRID: usize>() -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let problem = OdeBuilder::new()
        .rtol(1e-7)
        .atol([1e-7])
        .build_ode_with_mass_and_out(
            heat2d_rhs::<M, MGRID>,
            heat2d_jac_mul::<M, MGRID>,
            heat2d_mass::<M, MGRID>,
            heat2d_init::<M, MGRID>,
            heat2d_out::<M, MGRID>,
            heat2d_out_jac_mul::<M, MGRID>,
            1,
        )
        .unwrap();

    let mut soln = OdeSolverSolution{
        solution_points: Vec::new(),
        sens_solution_points: None,
        rtol: M::T::from(1e-5),
        atol: M::V::from_element(1, M::T::from(1e-5)),
    };
    let data = vec![
        (vec![9.75461e-01], 0.0),
        (vec![8.24056e-01], 0.01),
        (vec![6.88097e-01], 0.02),
        (vec![4.70961e-01], 0.04),
        (vec![2.16312e-01], 0.08),
        (vec![4.53210e-02], 0.16),
        (vec![1.98864e-03], 0.32),
        (vec![3.83238e-06], 0.64),
        (vec![0.0], 1.28),
        (vec![0.0], 2.56),
        (vec![0.0], 5.12),
        (vec![0.0], 10.24),
    ];
    for (values, time) in data {
        //let time = M::T::from(time);
        //let mut soln_at_t = Vec::with_capacity(MGRID * MGRID);
        //let mut max_u = M::T::zero();
        //let one = M::T::one();
        //let dx = one / (M::T::from(MGRID as f64) - one);
        //for j in 0..MGRID {
        //    let y = dx * M::T::from(j as f64);
        //    for i in 0..MGRID {
        //        let x = dx * M::T::from(i as f64);
        //        let u = pde_solution(x, y, time, 100);
        //        if u > max_u {
        //            max_u = u;
        //        }
        //        soln_at_t.push(u);
        //    }
        //}
        //assert!((M::T::from(values[0]) - max_u).abs() < M::T::from(1e-2));
        let values = M::V::from_vec(values.iter().map(|v| M::T::from(*v)).collect::<Vec<_>>());
        let time = M::T::from(time);
        soln.push(values, time);
    }
    (problem, soln)
}

#[cfg(test)]
mod tests {
    use crate::{ConstantOp, LinearOp, NonLinearOp};

    use super::*;

    #[test]
    fn test_jacobian() {
        //let jac = heat2d_jacobian::<nalgebra::DMatrix<f64>, 10>();
        let (problem, _soln) = head2d_problem::<nalgebra::DMatrix<f64>, 10>();
        let u0 = problem.eqn.init().call(0.0);
        let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
        insta::assert_yaml_snapshot!(jac.to_string());
    }

    #[test]
    fn test_mass() {
        let (problem, _soln) = head2d_problem::<nalgebra::DMatrix<f64>, 10>();
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        insta::assert_yaml_snapshot!(mass.to_string());
    }

    #[test]
    fn test_soln() {
        let (_problem, _soln) = head2d_problem::<nalgebra::DMatrix<f64>, 10>();
    }
}
