#[cfg(feature = "diffsl-llvm10")]
pub extern crate diffsl10_0 as diffsl;
#[cfg(feature = "diffsl-llvm11")]
pub extern crate diffsl11_0 as diffsl;
#[cfg(feature = "diffsl-llvm12")]
pub extern crate diffsl12_0 as diffsl;
#[cfg(feature = "diffsl-llvm13")]
pub extern crate diffsl13_0 as diffsl;
#[cfg(feature = "diffsl-llvm14")]
pub extern crate diffsl14_0 as diffsl;
#[cfg(feature = "diffsl-llvm15")]
pub extern crate diffsl15_0 as diffsl;
#[cfg(feature = "diffsl-llvm16")]
pub extern crate diffsl16_0 as diffsl;
#[cfg(feature = "diffsl-llvm17")]
pub extern crate diffsl17_0 as diffsl;
#[cfg(feature = "diffsl-llvm4")]
pub extern crate diffsl4_0 as diffsl;
#[cfg(feature = "diffsl-llvm5")]
pub extern crate diffsl5_0 as diffsl;
#[cfg(feature = "diffsl-llvm6")]
pub extern crate diffsl6_0 as diffsl;
#[cfg(feature = "diffsl-llvm7")]
pub extern crate diffsl7_0 as diffsl;
#[cfg(feature = "diffsl-llvm8")]
pub extern crate diffsl8_0 as diffsl;
#[cfg(feature = "diffsl-llvm9")]
pub extern crate diffsl9_0 as diffsl;

pub trait Scalar:
    nalgebra::Scalar
    + From<f64>
    + Display
    + SimdRealField
    + ComplexField
    + Copy
    + ClosedSub
    + From<f64>
    + ClosedMul
    + ClosedDiv
    + ClosedAdd
    + Signed
    + PartialOrd
    + Pow<Self, Output = Self>
    + Pow<i32, Output = Self>
{
    const EPSILON: Self;
    const INFINITY: Self;
}

type IndexType = usize;

impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
}

pub mod linear_solver;
pub mod matrix;
pub mod nonlinear_solver;
pub mod ode_solver;
pub mod op;
pub mod solver;
pub mod vector;

use std::fmt::Display;

use linear_solver::{lu::LU, LinearSolver};
use matrix::{DenseMatrix, Matrix, MatrixViewMut};
use nalgebra::{ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, ComplexField, SimdRealField};
use nonlinear_solver::{newton::NewtonNonlinearSolver, NonLinearSolver};
use num_traits::{Pow, Signed};
pub use ode_solver::{
    bdf::Bdf, equations::OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
};
use op::{LinearOp, NonLinearOp};
use solver::SolverProblem;
use vector::{Vector, VectorIndex, VectorRef, VectorView, VectorViewMut};

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::{vector::Vector, Bdf, OdeSolverMethod, OdeSolverProblem, OdeSolverState};

    // WARNING: if this test fails and you make a change to the code, you should update the README.md file as well!!!
    #[test]
    fn test_readme() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        let p = V::from_vec(vec![0.04, 1.0e4, 3.0e7]);
        let mut problem = OdeSolverProblem::new_ode(
            |x: &V, p: &V, _t: T, y: &mut V| {
                y[0] = -p[0] * x[0] + p[1] * x[1] * x[2];
                y[1] = p[0] * x[0] - p[1] * x[1] * x[2] - p[2] * x[1] * x[1];
                y[2] = p[2] * x[1] * x[1];
            },
            |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
                y[0] = -p[0] * v[0] + p[1] * v[1] * x[2] + p[1] * x[1] * v[2];
                y[1] = p[0] * v[0]
                    - p[1] * v[1] * x[2]
                    - p[1] * x[1] * v[2]
                    - 2.0 * p[2] * x[1] * v[1];
                y[2] = 2.0 * p[2] * x[1] * v[1];
            },
            |_p: &V, _t: T| V::from_vec(vec![1.0, 0.0, 0.0]),
            p,
        );
        problem.rtol = 1.0e-4;
        problem.atol = Rc::new(V::from_vec(vec![1.0e-8, 1.0e-6, 1.0e-6]));

        let mut solver = Bdf::default();

        let t = 0.4;
        let y = solver.solve(&problem, t).unwrap();

        let mut state = OdeSolverState::new(&problem);
        solver.set_problem(&mut state, problem);
        while state.t <= t {
            solver.step(&mut state).unwrap();
        }
        let y2 = solver.interpolate(&state, t);

        y2.assert_eq(&y, 1e-6);
    }
}
