//! # DiffSol
//!
//! DiffSol is a library for solving differential equations. It provides a simple interface to solve ODEs with optional mass matrices,
//! where the user can provide the equations either as closures or via strings in a domain-specific language.
//!
//! ## Solving ODEs
//!
//! The simplest way to create a new problem is to use the [OdeBuilder] struct. You can set the initial time, initial step size, relative tolerance, absolute tolerance, and parameters,
//! or leave them at their default values. Then, call one of the `build_*` functions (e.g. [OdeBuilder::build_ode], [OdeBuilder::build_ode_with_mass], [OdeBuilder::build_diffsl]) to create a [OdeSolverProblem].
//!
//! You will also need to choose a matrix type to use. DiffSol can use the [nalgebra](https://nalgebra.org) `DMatrix` type, the [faer](https://github.com/sarah-ek/faer-rs) `Mat` type, or any other type that implements the
//! [Matrix] trait.
//!
//! ## Initial state
//!
//! The solver state is held in [OdeSolverState], and contains a state vector, the gradient of the state vector, the time, and the step size. You can intitialise a new state using [OdeSolverState::new],
//! or create an uninitialised state using [OdeSolverState::new_without_initialise] and intitialise it manually or using the [OdeSolverState::set_consistent] and [OdeSolverState::set_step_size] methods.
//!
//! ## The solver
//!
//! To solve the problem given the initial state, you need to choose a solver. DiffSol provides the following solvers:
//! - A Backwards Difference Formulae [Bdf] solver, suitable for stiff problems and singular mass matrices.
//! - A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver [Sdirk]. You can use your own butcher tableau using [Tableau] or use one of the provided ([Tableau::tr_bdf2], [Tableau::esdirk34]).
//!
//! See the [OdeSolverMethod] trait for a more detailed description of the available methods on each solver. Possible workflows are:
//! - Use the [OdeSolverMethod::step] method to step the solution forward in time with an internal time step chosen by the solver to meet the error tolerances.
//! - Use the [OdeSolverMethod::interpolate] method to interpolate the solution between the last two time steps.
//! - Use the [OdeSolverMethod::set_stop_time] method to stop the solver at a specific time (i.e. this will override the internal time step so that the solver stops at the specified time).
//! - Alternatively, use the convenience functions [OdeSolverMethod::solve] or [OdeSolverMethod::solve_dense] that will both initialise the problem and solve the problem up to a specific time or a sequence of times.
//!
//! ## DiffSL
//!
//! DiffSL is a domain-specific language for specifying differential equations <https://github.com/martinjrobins/diffsl>. It uses the LLVM compiler framwork
//! to compile the equations to efficient machine code and uses the EnzymeAD library to compute the jacobian.
//!
//! You can use DiffSL with DiffSol using the [DiffSlContext] struct and [OdeBuilder::build_diffsl] method. You need to enable one of the `diffsl-llvm*` features
//! corresponding to the version of LLVM you have installed. E.g. to use your LLVM 10 installation, enable the `diffsl-llvm10` feature.
//!
//! For more information on the DiffSL language, see the [DiffSL documentation](https://martinjrobins.github.io/diffsl/)
//!
//! ## Custom ODE problems
//!
//! The [OdeBuilder] struct is the easiest way to create a problem, and can be used to create an ODE problem from a set of closures or the DiffSL language.
//! However, if this is not suitable for your problem or you want more control over how your equations are implemented, you can use your own structs to define the problem and wrap them in an [OdeSolverEquations] struct.
//! See the [OdeSolverEquations] struct for more information.
//!
//! ## Sparsity pattern for Jacobians and Mass matrices
//!
//! Via an implementation of [OdeEquations], the user provides the action of the jacobian on a vector `J(x) v`. By default DiffSol uses this to generate a jacobian matrix for the ODE solver.
//! For sparse jacobians, DiffSol will attempt to detect the sparsity pattern of the jacobian using this function and use a sparse matrix representation internally.
//! It attempts to determine the sparsity pattern of the jacobian (i.e. its non-zero values) by passing in `NaNs` for the input vector `x` and checking which elements
//! of the output vector `J(x) v` are also `NaN`, using the fact that `NaN`s propagate through most operations. However, this method is not foolproof and will fail if,
//! for example, your jacobian function uses any control flow that depends on the input vector. If this is the case, you can provide the jacobian matrix directly by
//! implementing the optional [NonLinearOp::jacobian_inplace] and the [LinearOp::matrix_inplace] (if applicable) functions, or by providing a sparsity pattern using the [Op::sparsity] function.
//!
//! ## Events / Root finding
//!
//! DiffSol provides a simple way to detect user-provided events during the integration of the ODEs. You can use this by providing a closure that has a zero-crossing at the event you want to detect, using the [OdeBuilder::build_ode_with_root] builder,
//! or by providing a [NonLinearOp] that has a zero-crossing at the event you want to detect. To use the root finding feature while integrating with the solver, you can use the return value of [OdeSolverMethod::step] to check if an event has been detected.
//!
//! ## Forward Sensitivity Analysis
//!
//! DiffSol provides a way to compute the forward sensitivity of the solution with respect to the parameters. You can use this by using the [OdeBuilder::build_ode_with_sens] or [OdeBuilder::build_ode_with_mass_and_sens] builder functions.
//! Note that by default the sensitivity equations are not included in the error control for the solvers, you can change this by using the [OdeBuilder::sensitivities_error_control] method.
//!
//! To obtain the sensitivity solution via interpolation, you can use the [OdeSolverMethod::interpolate_sens] method. Otherwise the sensitivity vectors are stored in the [OdeSolverState] struct.
//!
//! ## Nonlinear and linear solvers
//!
//! DiffSol provides generic nonlinear and linear solvers that are used internally by the ODE solver. You can use the solvers provided by DiffSol, or implement your own following the provided traits.
//! The linear solver trait is [LinearSolver], and the nonlinear solver trait is [NonLinearSolver]. The [SolverProblem] struct is used to define the problem to solve.
//!
//! The provided linear solvers are:
//! - [NalgebraLU]: a direct solver that uses the LU decomposition implemented in the [nalgebra](https://nalgebra.org) library.
//! - [FaerLU]: a direct solver that uses the LU decomposition implemented in the [faer](https://github.com/sarah-ek/faer-rs) library.
//! - [FaerSparseLU]: a sparse direct solver that uses the sparse LU decomposition implemented in the [faer](https://github.com/sarah-ek/faer-rs).
//!
//! The provided nonlinear solvers are:
//! - [NewtonNonlinearSolver]: a nonlinear solver that uses the Newton method.
//!
//! ## Matrix and vector types
//!
//! When solving ODEs, you will need to choose a matrix and vector type to use. DiffSol uses the following types:
//! - [nalgebra::DMatrix] and [nalgebra::DVector] from the [nalgebra](https://nalgebra.org) library.
//! - [faer::Mat] and [faer::Col] from the [faer](https://github.com/sarah-ek/faer-rs) library.
//! - [nalgebra_sparse::CscMatrix] from the nalgebra-sparse library.
//! - [SparseColMat], which is a thin wrapper around the [faer::sparse::SparseColMat] type from faer.
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

#[cfg(feature = "sundials")]
pub mod sundials_sys;

use linear_solver::LinearSolver;
pub use linear_solver::{faer::sparse_lu::FaerSparseLU, FaerLU, NalgebraLU};

pub use matrix::sparse_faer::SparseColMat;

#[cfg(feature = "sundials")]
pub use matrix::sundials::SundialsMatrix;

#[cfg(feature = "sundials")]
pub use vector::sundials::SundialsVector;

#[cfg(feature = "sundials")]
pub use linear_solver::sundials::SundialsLinearSolver;

#[cfg(feature = "sundials")]
pub use ode_solver::sundials::SundialsIda;

#[cfg(feature = "suitesparse")]
pub use linear_solver::suitesparse::klu::KLU;

#[cfg(feature = "diffsl")]
pub use ode_solver::diffsl::DiffSlContext;

pub use jacobian::{find_non_zeros_linear, find_non_zeros_nonlinear, JacobianColoring};
pub use matrix::{default_solver::DefaultSolver, Matrix};
use matrix::{
    sparsity::Dense, sparsity::DenseRef, sparsity::MatrixSparsity, sparsity::MatrixSparsityRef,
    DenseMatrix, MatrixCommon, MatrixRef, MatrixView, MatrixViewMut,
};
pub use nonlinear_solver::newton::NewtonNonlinearSolver;
use nonlinear_solver::{
    convergence::Convergence, convergence::ConvergenceStatus, newton::newton_iteration,
    root::RootFinder, NonLinearSolver,
};
pub use ode_solver::{
    bdf::Bdf, builder::OdeBuilder, equations::OdeEquations, equations::OdeSolverEquations,
    method::OdeSolverMethod, method::OdeSolverState, method::OdeSolverStopReason,
    problem::OdeSolverProblem, sdirk::Sdirk, sens_equations::SensEquations,
    sens_equations::SensInit, sens_equations::SensRhs, solution::OdeSolution, tableau::Tableau,
};
pub use op::{
    closure::Closure, constant_closure::ConstantClosure, linear_closure::LinearClosure,
    unit::UnitCallable, ConstantOp, LinearOp, NonLinearOp, Op,
};
use op::{
    closure_no_jac::ClosureNoJac, closure_with_sens::ClosureWithSens,
    constant_closure_with_sens::ConstantClosureWithSens, init::InitOp,
    linear_closure_with_sens::LinearClosureWithSens,
};
use scalar::{IndexType, Scalar, Scale};
use solver::SolverProblem;
pub use vector::DefaultDenseMatrix;
use vector::{Vector, VectorCommon, VectorIndex, VectorRef, VectorView, VectorViewMut};

pub use scalar::scale;
