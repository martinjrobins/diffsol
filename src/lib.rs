//! # DiffSol
//!
//! DiffSol is a library for solving differential equations. It provides a simple interface to solve ODEs and semi-explicit DAEs.
//!
//! ## Solving ODEs
//!
//! To create a new problem, use the [OdeBuilder] struct. You can set the initial time, initial step size, relative tolerance, absolute tolerance, and parameters,
//! or leave them at their default values. Then, call the [OdeBuilder::build_ode] method with the ODE equations, or the [OdeBuilder::build_ode_with_mass] method
//! with the ODE equations and the mass matrix equations.
//!
//! You will also need to choose a matrix type to use. DiffSol can use the [nalgebra](https://nalgebra.org) `DMatrix` type, or any other type that implements the
//! [Matrix] trait. You can also use the [sundials](https://computation.llnl.gov/projects/sundials) library for the matrix and vector types (see [SundialsMatrix]).
//!
//! To solve the problem, you need to choose a solver. DiffSol provides the following solvers:
//! - A Backwards Difference Formulae [Bdf] solver, suitable for stiff problems and singular mass matrices.
//! - A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver [Sdirk]. You can use your own butcher tableau using [Tableau] or use one of the provided ([Tableau::tr_bdf2], [Tableau::esdirk34]).
//! - A BDF solver that wraps the IDA solver solver from the sundials library ([SundialsIda], requires the `sundials` feature).
//!
//! See the [OdeSolverMethod] trait for a more detailed description of the available methods on each solver.
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
//! let state = OdeSolverState::new(&problem);
//! solver.set_problem(state, &problem);
//! while solver.state().unwrap().t <= t {
//!     solver.step().unwrap();
//! }
//! let y = solver.interpolate(t);
//! ```
//!
//! ## DiffSL
//!
//! DiffSL is a domain-specific language for specifying differential equations <https://github.com/martinjrobins/diffsl>. It uses the LLVM compiler framwork
//! to compile the equations to efficient machine code and uses the EnzymeAD library to compute the jacobian. You can use DiffSL with DiffSol by enabling one of the `diffsl-llvm*` features
//! corresponding to the version of LLVM you have installed, and using the [OdeBuilder::build_diffsl] method.
//!
//! For more information on the DiffSL language, see the [DiffSL documentation](https://martinjrobins.github.io/diffsl/)
//!
//! ## Custom ODE problems
//!
//! The [OdeBuilder] struct can be used to create an ODE problem from a set of closures.
//! If this is not suitable for your problem or you want more control over how your equations are implemented, you can also implement the [OdeEquations] trait manually.
//!
//! ## Nonlinear and linear solvers
//!
//! DiffSol provides generic nonlinear and linear solvers that are used internally by the ODE solver. You can use the solvers provided by DiffSol, or implement your own following the provided traits.
//! The linear solver trait is [LinearSolver], and the nonlinear solver trait is [NonLinearSolver]. The [SolverProblem] struct is used to define the problem to solve.
//!
//! The provided linear solvers are:
//! - [NalgebraLU]: a direct solver that uses the LU decomposition implemented in the [nalgebra](https://nalgebra.org) library.
//! - [SundialsLinearSolver]: a linear solver that uses the [sundials](https://computation.llnl.gov/projects/sundials) library (requires the `sundials` feature).
//!
//! The provided nonlinear solvers are:
//! - [NewtonNonlinearSolver]: a nonlinear solver that uses the Newton method.
//!
//! ## Jacobian and Mass matrix calculation
//!
//! Via [OdeEquations], the user provides the action of the jacobian on a vector `J(x) v`. By default DiffSol uses this to generate a jacobian matrix for the ODE solver.
//! Generally this requires `n` evaluations of the jacobian action for a system of size `n`, so it is often more efficient if the user can provide the jacobian matrix directly
//! by implementing the [OdeEquations::jacobian_matrix] and the [OdeEquations::mass_matrix] (is applicable) functions.
//!
//! DiffSol also provides an experimental feature to calculate sparse jacobians more efficiently by automatically detecting the sparsity pattern of the jacobian and using
//! colouring \[1\] to reduce the number of jacobian evaluations. You can enable this feature by enabling [OdeBuilder::use_coloring()] option when building the ODE problem.
//!
//! \[1\] Gebremedhin, A. H., Manne, F., & Pothen, A. (2005). What color is your Jacobian? Graph coloring for computing derivatives. SIAM review, 47(4), 629-705.
//!
//! ## Matrix and vector types
//!
//! When solving ODEs, you will need to choose a matrix and vector type to use. DiffSol uses the following types:
//! - [nalgebra::DMatrix] and [nalgebra::DVector] from the [nalgebra](https://nalgebra.org) library.
//! - [SundialsMatrix] and [SundialsVector] from the [sundials](https://computation.llnl.gov/projects/sundials) library (requires the `sundials` feature).
//!
//! If you wish to use your own matrix and vector types, you will need to implement the following traits:
//! - For matrices: [Matrix], [MatrixView], [MatrixViewMut], [DenseMatrix], and [MatrixCommon].
//! - For vectors: [Vector], [VectorIndex], [VectorView], [VectorViewMut], and [VectorCommon].
//!

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

use linear_solver::LinearSolver;
pub use linear_solver::{FaerLU, NalgebraLU};

#[cfg(feature = "sundials")]
pub use matrix::sundials::SundialsMatrix;

#[cfg(feature = "sundials")]
pub use vector::sundials::SundialsVector;

#[cfg(feature = "sundials")]
pub use linear_solver::sundials::SundialsLinearSolver;

#[cfg(feature = "sundials")]
pub use ode_solver::sundials::SundialsIda;

use matrix::{DenseMatrix, Matrix, MatrixCommon, MatrixSparsity, MatrixView, MatrixViewMut};
pub use nonlinear_solver::newton::NewtonNonlinearSolver;
use nonlinear_solver::NonLinearSolver;
pub use ode_solver::{
    bdf::Bdf, builder::OdeBuilder, equations::OdeEquations, method::OdeSolverMethod,
    method::OdeSolverState, problem::OdeSolverProblem, sdirk::Sdirk, tableau::Tableau,
};
use op::{closure::Closure, linear_closure::LinearClosure, LinearOp, NonLinearOp, Op};
use scalar::{IndexType, Scalar, Scale};
use solver::SolverProblem;
use vector::{Vector, VectorCommon, VectorIndex, VectorRef, VectorView, VectorViewMut};

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
            .build_ode_dense(
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

        let state = OdeSolverState::new(&problem);
        solver.set_problem(state, &problem);
        while solver.state().unwrap().t <= t {
            solver.step().unwrap();
        }
        let y2 = solver.interpolate(t).unwrap();

        y2.assert_eq_st(&y, 1e-6);
    }
    #[test]
    fn test_readme_faer() {
        type T = f64;
        type V = faer::Col<f64>;
        type M = faer::Mat<f64>;
        let problem = OdeBuilder::new()
            .p([0.04, 1.0e4, 3.0e7])
            .rtol(1e-4)
            .atol([1.0e-8, 1.0e-6, 1.0e-6])
            .build_ode_dense(
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

        let mut solver = Bdf::<M, _, _>::default();

        let t = 0.4;
        let y = solver.solve(&problem, t).unwrap();

        let state = OdeSolverState::new(&problem);
        solver.set_problem(state, &problem);
        while solver.state().unwrap().t <= t {
            solver.step().unwrap();
        }
        let y2 = solver.interpolate(t).unwrap();

        y2.assert_eq_st(&y, 1e-6);
    }

    // y2.assert_eq(&y, 1e-6);
}
