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
//! To view the state within a solver, you can use the [OdeSolverMethod::state] or [OdeSolverMethod::state_mut] methods. These will return references to the state using either the [StateRef] or [StateRefMut] structs
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
//! Via an implementation of [OdeEquationsImplicit], the user provides the action of the jacobian on a vector `J(x) v`. By default DiffSol uses this to generate a jacobian matrix for the ODE solver.
//! For sparse jacobians, DiffSol will attempt to detect the sparsity pattern of the jacobian using this function and use a sparse matrix representation internally.
//! It attempts to determine the sparsity pattern of the jacobian (i.e. its non-zero values) by passing in `NaNs` for the input vector `x` and checking which elements
//! of the output vector `J(x) v` are also `NaN`, using the fact that `NaN`s propagate through most operations. However, this method is not foolproof and will fail if,
//! for example, your jacobian function uses any control flow that depends on the input vector. If this is the case, you can provide the jacobian matrix directly by
//! implementing the optional [NonLinearOpJacobian::jacobian_inplace] and the [LinearOp::matrix_inplace] (if applicable) functions,
//! or by providing a sparsity pattern using the [Op::sparsity] function.
//!
//! ## Events / Root finding
//!
//! DiffSol provides a simple way to detect user-provided events during the integration of the ODEs. You can use this by providing a closure that has a zero-crossing at the event you want to detect, using the [OdeBuilder::build_ode_with_root] builder,
//! or by providing a [NonLinearOp] that has a zero-crossing at the event you want to detect. To use the root finding feature while integrating with the solver, you can use the return value of [OdeSolverMethod::step] to check if an event has been detected.
//!
//! ## Forward Sensitivity Analysis
//!
//! DiffSol provides a way to compute the forward sensitivity of the solution with respect to the parameters. To use this your equations struct must implement the [OdeEquationsSens] trait.
//! Note that by default the sensitivity equations are included in the error control for the solvers, you can change this by setting tolerances using the [OdeBuilder::sens_atol] and [[OdeBuilder::sens_rtol]] methods.
//! You will also need to use [SensitivitiesOdeSolverMethod::set_problem_with_sensitivities] to set the problem with sensitivities.
//!
//! To obtain the sensitivity solution via interpolation, you can use the [OdeSolverMethod::interpolate_sens] method. Otherwise the sensitivity vectors are stored in the [OdeSolverState] struct.
//!
//! ## Checkpointing
//!
//! You can checkpoint the solver at a set of times using the [OdeSolverMethod::checkpoint] method. This will store the state of the solver at the given times, and subsequently use the [OdeSolverMethod::set_problem]
//! method to restore the solver to the state at the given time.
//!
//! ## Interpolation
//!
//! The [HermiteInterpolator] struct provides a way to interpolate a solution between a sequence of steps. If the number of steps in your solution is too large to fit in memory,
//! you can instead use checkpointing to store the solution at a reduced set of times and dynamically interpolate between these checkpoints using the [Checkpointing] struct
//! (at the cost of recomputing the solution between the checkpoints).  
//!
//! ## Quadrature and Output functions
//!
//! The [OdeSolverEquations::Out] associated type can be used to define an output function. DiffSol will optionally integrate this function over the solution trajectory by
//! using the [OdeBuilder::integrate_out] method. By default, the output integration is added to the error control of the solver, and the tolerances can be
//! adjusted using the [OdeBuilder::out_atol] and [OdeBuilder::out_rtol] methods. It can be removed from the error control by setting the tolerances to `None`.
//!
//! ## Adjoint Sensitivity Analysis
//!
//! If you require the partial gradient of the output function with respect to the parameters and your parameter vector is sufficiently large, then it is more efficient
//! to use the adjoint sensitivity method. This method uses a lagrange multiplier to derive a set of adjoint ode equations that are solved backwards in time,
//! and then used to compute the sensitivities of the output function. Checkpointing is typically used to store the forward solution at a set of times as theses are required
//! to solve the adjoint equations.
//!
//! To use the adjoint sensitivity method, your equations struct must implement the [OdeEquationsAdjoint] trait. When you compute the forward solution, use checkpointing
//! to store the solution at a set of times. From this you should obtain a `Vec<OdeSolverState>` (that can be the start and end of the solution), and
//! a [HermiteInterpolator] that can be used to interpolate the solution between the last two checkpoints. You can then use the [AdjointOdeSolverMethod::into_adjoint_solver]
//! method to create an adjoint solver from the forward solver, and then use this solver to step the adjoint equations backwards in time. Once the adjoint equations have been solved,
//! the sensitivities of the output function will be stored in the [StateRef::sg] field of the adjoint solver state. If your parameters are used to calculate the initial conditions
//! of the forward problem, then you will need to use the [AdjointEquations::correct_sg_for_init] method to correct the sensitivities for the initial conditions.
//!
//! ## Nonlinear and linear solvers
//!
//! DiffSol provides generic nonlinear and linear solvers that are used internally by the ODE solver. You can use the solvers provided by DiffSol, or implement your own following the provided traits.
//! The linear solver trait is [LinearSolver], and the nonlinear solver trait is [NonLinearSolver].
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
#[cfg(feature = "diffsl-cranelift")]
pub extern crate diffsl_no_llvm as diffsl;

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

pub use linear_solver::LinearSolver;
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

pub use jacobian::{
    find_adjoint_non_zeros, find_jacobian_non_zeros, find_matrix_non_zeros,
    find_sens_adjoint_non_zeros, find_sens_non_zeros, find_transpose_non_zeros, JacobianColoring,
};
pub use matrix::{default_solver::DefaultSolver, Matrix};
use matrix::{
    sparsity::Dense, sparsity::DenseRef, sparsity::MatrixSparsity, sparsity::MatrixSparsityRef,
    DenseMatrix, MatrixCommon, MatrixRef, MatrixView, MatrixViewMut,
};
use nonlinear_solver::{
    convergence::Convergence, convergence::ConvergenceStatus, root::RootFinder,
};
pub use nonlinear_solver::{newton::NewtonNonlinearSolver, NonLinearSolver};
use ode_solver::jacobian_update::JacobianUpdate;
pub use ode_solver::state::{StateRef, StateRefMut};
pub use ode_solver::{
    adjoint_equations::AdjointContext, adjoint_equations::AdjointEquations,
    adjoint_equations::AdjointInit, adjoint_equations::AdjointRhs, bdf::Bdf, bdf::BdfAdj,
    bdf_state::BdfState, builder::OdeBuilder, checkpointing::Checkpointing,
    checkpointing::HermiteInterpolator, equations::AugmentedOdeEquations,
    equations::AugmentedOdeEquationsImplicit, equations::NoAug, equations::OdeEquations,
    equations::OdeEquationsAdjoint, equations::OdeEquationsImplicit, equations::OdeEquationsSens,
    equations::OdeSolverEquations, method::AdjointOdeSolverMethod, method::OdeSolverMethod,
    method::OdeSolverStopReason, method::SensitivitiesOdeSolverMethod, problem::OdeSolverProblem,
    sdirk::Sdirk, sdirk::SdirkAdj, sdirk_state::SdirkState, sens_equations::SensEquations,
    sens_equations::SensInit, sens_equations::SensRhs, state::OdeSolverState, tableau::Tableau,
};
pub use op::constant_op::{ConstantOp, ConstantOpSens, ConstantOpSensAdjoint};
pub use op::linear_op::{LinearOp, LinearOpSens, LinearOpTranspose};
pub use op::nonlinear_op::{
    NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint,
};
pub use op::{
    closure::Closure, closure_with_adjoint::ClosureWithAdjoint, constant_closure::ConstantClosure,
    constant_closure_with_adjoint::ConstantClosureWithAdjoint, linear_closure::LinearClosure,
    unit::UnitCallable, Op,
};
use op::{
    closure_no_jac::ClosureNoJac, closure_with_sens::ClosureWithSens,
    constant_closure_with_sens::ConstantClosureWithSens, init::InitOp,
};
use scalar::{IndexType, Scalar, Scale};
pub use vector::DefaultDenseMatrix;
use vector::{Vector, VectorCommon, VectorIndex, VectorRef, VectorView, VectorViewMut};

pub use scalar::scale;

pub mod error;
