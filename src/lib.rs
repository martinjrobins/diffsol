
pub trait Scalar: nalgebra::Scalar + From<f64> + Display + SimdRealField + ComplexField + Copy + ClosedSub + From<f64> + ClosedMul + ClosedDiv + ClosedAdd + Signed + PartialOrd + Pow<Self, Output=Self> + Pow<i32, Output=Self> {
    const EPSILON: Self;
    const INFINITY: Self;
}

type IndexType = usize;


impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
}


pub mod vector;
pub mod matrix;
pub mod linear_solver;
pub mod op;
pub mod nonlinear_solver;
pub mod ode_solver;
pub mod solver;

use std::fmt::Display;

use nalgebra::{ClosedSub, ClosedMul, ClosedDiv, ClosedAdd, SimdRealField, ComplexField};
use num_traits::{Signed, Pow};
use vector::{Vector, VectorView, VectorViewMut, VectorIndex, VectorRef};
use nonlinear_solver::{NonLinearSolver, newton::NewtonNonlinearSolver};
use op::{NonLinearOp, LinearOp, ConstantOp};
use matrix::{Matrix, MatrixViewMut, MatrixCommon};
use solver::SolverProblem;
use linear_solver::{lu::LU, LinearSolver};
use ode_solver::{OdeSolverProblem, OdeSolverState, bdf::Bdf, OdeSolverMethod};


#[cfg(test)]
#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{OdeSolverProblem, Bdf, OdeSolverState, OdeSolverMethod};

    // WARNING: if this test fails and you make a change to the code, you should update the README.md file as well!!!
    #[test]
    fn test_readme() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        let p = V::from_vec(vec![0.04.into(), 1.0e4.into(), 3.0e7.into()]);
        let mut problem = OdeSolverProblem::new_ode(
            | x: &V, p: &V, _t: T, y: &mut V | {
                y[0] = -p[0] * x[0] + p[1] * x[1] * x[2];
                y[1] = p[0] * x[0] - p[1] * x[1] * x[2] - p[2] * x[1] * x[1];
                y[2] = p[2] * x[1] * x[1];
            },
            | x: &V, p: &V, _t: T, v: &V, y: &mut V | {
                y[0] = -p[0] * v[0] + p[1] * v[1] * x[2] + p[1] * x[1] * v[2];
                y[1] = p[0] * v[0] - p[1] * v[1] * x[2] - p[1] * x[1] * v[2]  - 2.0 * p[2] * x[1] * v[1];
                y[2] = 2.0 * p[2] * x[1] * v[1];
            },
            | _p: &V, _t: T | {
                V::from_vec(vec![1.0, 0.0, 0.0])
            },
            p,
        );
        problem.rtol = 1.0e-4;
        problem.atol = Rc::new(V::from_vec(vec![1.0e-8, 1.0e-6, 1.0e-6]));

        let mut solver = Bdf::default();

        let t = 0.4;
        let _y = solver.solve(&problem, t).unwrap();

        let mut state = OdeSolverState::new(&problem);
        while state.t <= t {
            solver.step(&mut state).unwrap();
        }
        let _y = solver.interpolate(&state, t);
    }
}

    



 