//! # DiffSol
//!
//! DiffSol is a library for solving differential equations. It provides a simple interface to solve ODEs and semi-explicit DAEs.
//!
//! ## Getting Started
//!
//! To create a new problem, use the [OdeBuilder] struct. You can set the initial time, initial step size, relative tolerance, absolute tolerance, and parameters,
//! or leave them at their default values. Then, call the [OdeBuilder::build_ode] method with the ODE equations, or the [OdeBuilder::build_ode_with_mass] method
//! with the ODE equations and the mass matrix equations.
//!
//! You will also need to choose a matrix type to use. DiffSol can use the [nalgebra](https://nalgebra.org) `DMatrix` type, or any other type that implements the
//! [Matrix] trait. You can also use the [sundials](https://computation.llnl.gov/projects/sundials) library for the matrix and vector types (see [SundialsMatrix]).
//!
//! To solve the problem, you need to choose a solver. DiffSol provides a pure rust [Bdf] solver, or you can use the [SundialsIda] solver from the sundials library (requires the `sundials` feature).
//! See the [OdeSolverMethod] trait for a more detailed description of the available methods on the solver.
//!
//! ```rust
//! use diffsol::{OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod};
//! type M = nalgebra::DMatrix<f64>;
//!
//! let problem = OdeBuilder::new()
//!   .rtol(1e-6)
//!   .p([0.1])
//!   .build_ode::<M, _, _, _>(
//!     // dy/dt = -ay
//!     |x, p, t, y| {
//!       y[0] = -p[0] * x[0];
//!     },
//!     // Jv = -av
//!     |x, p, t, v, y| {
//!       y[0] = -p[0] * v[0];
//!     },
//!     // y(0) = 1
//!    |p, t| {
//!       nalgebra::DVector::from_vec(vec![1.0])
//!    },
//!   ).unwrap();
//!
//! let mut solver = Bdf::default();
//! let t = 0.4;
//! let mut state = OdeSolverState::new(&problem);
//! solver.set_problem(&mut state, &problem);
//! while state.t <= t {
//!     solver.step(&mut state).unwrap();
//! }
//! let y = solver.interpolate(&state, t);
//! ```

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

pub mod jacobian;
pub mod linear_solver;
pub mod matrix;
pub mod nonlinear_solver;
pub mod ode_solver;
pub mod op;
pub mod scalar;
pub mod solver;
pub mod vector;

pub use linear_solver::lu::LU;
use linear_solver::LinearSolver;

#[cfg(feature = "sundials")]
pub use matrix::sundials::SundialsMatrix;

#[cfg(feature = "sundials")]
pub use vector::sundials::SundialsVector;

#[cfg(feature = "sundials")]
pub use linear_solver::sundials::SundialsLinearSolver;

#[cfg(feature = "sundials")]
pub use ode_solver::sundials::SundialsIda;

use matrix::{DenseMatrix, Matrix, MatrixViewMut};
pub use nonlinear_solver::newton::NewtonNonlinearSolver;
use nonlinear_solver::NonLinearSolver;
pub use ode_solver::{
    bdf::Bdf, builder::OdeBuilder, equations::OdeEquations, method::OdeSolverMethod,
    method::OdeSolverState, problem::OdeSolverProblem,
};
use op::NonLinearOp;
use scalar::{IndexType, Scalar, Scale};
use solver::SolverProblem;
use vector::{Vector, VectorIndex, VectorRef, VectorView, VectorViewMut};

pub use scalar::scale;

#[cfg(test)]
mod tests {

    use crate::{
        ode_solver::builder::OdeBuilder, vector::Vector, Bdf, OdeSolverMethod, OdeSolverState,
    };

    // WARNING: if this test fails and you make a change to the code, you should update the README.md file as well!!!
    #[test]
    fn test_readme() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        let problem = OdeBuilder::new()
            .p([0.04, 1.0e4, 3.0e7])
            .rtol(1e-4)
            .atol([1.0e-8, 1.0e-6, 1.0e-6])
            .build_ode(
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
            )
            .unwrap();

        let mut solver = Bdf::default();

        let t = 0.4;
        let y = solver.solve(&problem, t).unwrap();

        let mut state = OdeSolverState::new(&problem);
        solver.set_problem(&mut state, &problem);
        while state.t <= t {
            solver.step(&mut state).unwrap();
        }
        let y2 = solver.interpolate(&state, t);

        y2.assert_eq(&y, 1e-6);
    }
}
