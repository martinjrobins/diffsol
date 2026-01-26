//! # Diffsol
//!
//! Diffsol is a library for solving differential equations. It provides a simple interface to solve ODEs with optional mass matrices,
//! where the user can provide the equations either as closures or via strings in a domain-specific language.
//!
//! ## Solving ODEs
//!
//! The simplest way to create a new problem is to use the [OdeBuilder] struct. You can set many configuration options such as the initial time ([OdeBuilder::t0]), initial step size ([OdeBuilder::h0]),
//! relative tolerance ([OdeBuilder::rtol]), absolute tolerance ([OdeBuilder::atol]), parameters ([OdeBuilder::p]) and equations ([OdeBuilder::rhs_implicit], [OdeBuilder::init], [OdeBuilder::mass] etc.)
//! or leave them at their default values. Then, call the [OdeBuilder::build] function to create a [OdeSolverProblem].
//!
//! You will also need to choose a matrix type to use. Diffsol can use the [nalgebra](https://nalgebra.org) `DMatrix` type, the [faer](https://github.com/sarah-ek/faer-rs) `Mat` type, or any other type that implements the
//! [Matrix] trait.
//!
//! ## Initial state
//!
//! The solver state is held in [OdeSolverState], and contains a state vector, the gradient of the state vector, the time, and the step size. The [OdeSolverProblem] class has a collection of methods to create and initialise
//! a new state for each solver ([OdeSolverProblem::bdf_state], [OdeSolverProblem::rk_state], [OdeSolverProblem::rk_state_and_consistent]). Or you can manually intitialise a new state using [OdeSolverState::new],
//! or create an uninitialised state using [OdeSolverState::new_without_initialise] and intitialise it manually or using the [OdeSolverState::set_consistent] and [OdeSolverState::set_step_size] methods.
//!
//! To view the state within a solver, you can use the [OdeSolverMethod::state] or [OdeSolverMethod::state_mut] methods. These will return references to the state using either the [StateRef] or [StateRefMut] structs
//!
//! ## The solver
//!
//! To solve the problem given the initial state, you need to choose a solver. Diffsol provides the following solvers:
//! - A Backwards Difference Formulae [Bdf] solver, suitable for stiff problems and singular mass matrices.
//! - A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver [Sdirk]. You can use your own butcher tableau using [Tableau] or use one of the provided ([Tableau::tr_bdf2], [Tableau::esdirk34]).
//!
//! The easiest way to create a solver is to use one of the provided methods on the [OdeSolverProblem] struct ([OdeSolverProblem::bdf_solver], [OdeSolverProblem::tr_bdf2_solver], [OdeSolverProblem::esdirk34_solver]).
//! These create a new solver from a provided state and problem. Alternatively, you can create both the solver and the state at once using [OdeSolverProblem::bdf], [OdeSolverProblem::tr_bdf2], [OdeSolverProblem::esdirk34].
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
//! You can use DiffSL with Diffsol using the [DiffSlContext] and [DiffSl] structs and [OdeBuilder::build_from_eqn] method. You need to enable one of the `diffsl-llvm*` features
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
//! Via an implementation of [OdeEquationsImplicit], the user provides the action of the jacobian on a vector `J(x) v`. By default Diffsol uses this to generate a jacobian matrix for the ODE solver.
//! For sparse jacobians, Diffsol will attempt to detect the sparsity pattern of the jacobian using this function and use a sparse matrix representation internally.
//! It attempts to determine the sparsity pattern of the jacobian (i.e. its non-zero values) by passing in `NaNs` for the input vector `x` and checking which elements
//! of the output vector `J(x) v` are also `NaN`, using the fact that `NaN`s propagate through most operations. However, this method is not foolproof and will fail if,
//! for example, your jacobian function uses any control flow that depends on the input vector. If this is the case, you can provide the jacobian matrix directly by
//! implementing the optional [NonLinearOpJacobian::jacobian_inplace] and the [LinearOp::matrix_inplace] (if applicable) functions,
//! or by providing a sparsity pattern using the [NonLinearOpJacobian::jacobian_sparsity] and [LinearOp::sparsity] functions.
//!
//! ## Events / Root finding
//!
//! Diffsol provides a simple way to detect user-provided events during the integration of the ODEs. You can use this by providing a closure that has a zero-crossing at the event you want to detect, using the [OdeBuilder::root] method,
//! or by providing a [NonLinearOp] that has a zero-crossing at the event you want to detect. To use the root finding feature while integrating with the solver, you can use the return value of [OdeSolverMethod::step] to check if an event has been detected.
//!
//! ## Forward Sensitivity Analysis
//!
//! Diffsol provides a way to compute the forward sensitivity of the solution with respect to the parameters. You can provide the requires equations to the builder using [OdeBuilder::rhs_sens_implicit] and [OdeBuilder::init_sens],
//! or your equations struct must implement the [OdeEquationsImplicitSens] trait,
//! Note that by default the sensitivity equations are included in the error control for the solvers, you can change this by setting tolerances using the [OdeBuilder::sens_atol] and [OdeBuilder::sens_rtol] methods.
//!
//! The easiest way to obtain the sensitivity solution is to use the [SensitivitiesOdeSolverMethod::solve_dense_sensitivities] method, which will solve the forward problem and the sensitivity equations simultaneously and return the result.
//! If you are manually stepping the solver, you can use the [OdeSolverMethod::interpolate_sens] method to obtain the sensitivity solution at a given time. Otherwise the sensitivity vectors are stored in the [OdeSolverState] struct.
//!
//! ## Checkpointing
//!
//! You can checkpoint the solver at the current internal time [OdeSolverMethod::checkpoint] method.
//!
//! ## Interpolation
//!
//! The [HermiteInterpolator] struct provides a way to interpolate a solution between a sequence of steps. If the number of steps in your solution is too large to fit in memory,
//! you can instead use checkpointing to store the solution at a reduced set of times and dynamically interpolate between these checkpoints using the [Checkpointing] struct
//! (at the cost of recomputing the solution between the checkpoints).  
//!
//! ## Quadrature and Output functions
//!
//! The [OdeSolverEquations::Out] associated type can be used to define an output function. Diffsol will optionally integrate this function over the solution trajectory by
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
//! To provide the builder with the required equations, you can use the [OdeBuilder::rhs_adjoint_implicit], [OdeBuilder::init_adjoint], and [OdeBuilder::out_adjoint_implicit] methods,
//! or your equations struct must implement the [OdeEquationsAdjoint] trait.
//!
//! To obtain the adjoint solution, it is first required to generate a checkpointing struct using either [OdeSolverMethod::solve_with_checkpointing] or [OdeSolverMethod::solve_dense_with_checkpointing] methods,
//! which will solve the forwards problem. Then you can create an adjoint solver using the functions on the problem struct, for example [OdeSolverProblem::bdf_solver_adjoint]. Once you have created your
//! adjoint solver, then use the [AdjointOdeSolverMethod::solve_adjoint_backwards_pass] method to solve the adjoint equations backwards in time and obtain your solution.
//! The gradients of your outputs wrt the parameters are stored in the [StateRef::sg] field.
//!
//! If you wish to manually do the timestepping, then the best place to start is by looking at the source code for the [AdjointOdeSolverMethod::solve_adjoint_backwards_pass] method. During the solution of the forwards problem
//! you will need to use checkpointing to store the solution at a set of times, and you can see how this is done in the [OdeSolverMethod::solve_with_checkpointing] method.
//!
//! ## Nonlinear and linear solvers
//!
//! Diffsol provides generic nonlinear and linear solvers that are used internally by the ODE solver. You can use the solvers provided by Diffsol, or implement your own following the provided traits.
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
//! When solving ODEs, you will need to choose a matrix and vector type to use. Diffsol uses the following types:
//! - [NalgebraVec] and [NalgebraMat] (wrappers around [nalgebra::DMatrix] and [nalgebra::DVector] from the [nalgebra](https://nalgebra.org) library).
//! - [FaerVec], [FaerMat] and [FaerSparseMat] (wrappers around [faer::Mat], [faer::Col] and [faer::sparse::SparseColMat] from the [faer](https://github.com/sarah-ek/faer-rs) library).
//!
//! If you wish to use your own matrix and vector types, you will need to implement the following traits:
//! - For matrices: [Matrix], [MatrixView], [MatrixViewMut], [DenseMatrix], and [MatrixCommon].
//! - For vectors: [Vector], [VectorIndex], [VectorView], [VectorViewMut], and [VectorCommon].
//!

#[cfg(feature = "diffsl")]
pub use diffsl::execution::module::{CodegenModule, CodegenModuleCompile, CodegenModuleJit};
#[cfg(feature = "diffsl")]
pub use diffsl::execution::scalar::Scalar as DiffSlScalar;
#[cfg(feature = "diffsl-cranelift")]
pub use diffsl::CraneliftJitModule;
#[cfg(feature = "diffsl-llvm")]
pub use diffsl::LlvmModule;

/// Context objects for managing device resources for vectors and matrices (e.g. device streams, threading pools, etc.).
///
/// This module provides context types that encapsulate information about where data is stored and computed
/// (CPU, GPU, etc.). Different backends like nalgebra and faer may require different context implementations.
/// The [Context] trait defines the interface that must be implemented.
pub mod context;

/// Jacobian computation and coloring algorithms for efficient Jacobian evaluation.
///
/// This module provides utilities for:
/// - Detecting the sparsity pattern of Jacobian matrices using NaN propagation
/// - Computing efficient graph colorings of the Jacobian sparsity pattern
/// - Using the coloring to compute sparse Jacobians with fewer function evaluations
///
/// The [JacobianColoring] struct is the main entry point for computing Jacobians efficiently.
pub mod jacobian;

/// Linear solver implementations and traits.
///
/// This module defines the [LinearSolver] trait for solving linear systems and provides implementations:
/// - Direct solvers: [NalgebraLU], [FaerLU], [FaerSparseLU]
/// - Optional sparse solvers: `KLU` (requires `suitesparse` feature)
/// - GPU solvers: `CudaLU` (requires `cuda` feature)
///
/// The linear solver is a critical component used internally by nonlinear solvers to solve Newton systems.
pub mod linear_solver;

/// Matrix types and operations.
///
/// This module defines the [Matrix] trait and related traits for matrix operations:
/// - [DenseMatrix] for dense column-major matrices
/// - [MatrixView] and [MatrixViewMut] for borrowed views
/// - Sparsity detection and handling
///
/// Implementations are provided for:
/// - Dense matrices: [NalgebraMat], [FaerMat]
/// - Sparse matrices: [FaerSparseMat]
/// - GPU matrices: `CudaMat` (requires `cuda` feature)
pub mod matrix;

/// Nonlinear solver implementations and traits.
///
/// This module defines the [NonLinearSolver] trait and provides the [NewtonNonlinearSolver] implementation.
/// It also includes:
/// - [LineSearch] implementations for globalization ([NoLineSearch], [BacktrackingLineSearch])
/// - Root finding algorithms via [RootFinder]
/// - Convergence testing via [Convergence]
///
/// Nonlinear solvers are used internally by ODE solvers to solve implicit equations.
pub mod nonlinear_solver;

/// ODE equations and traits.
///
/// This module defines the [OdeEquations] trait and specialized variants:
/// - [OdeEquationsImplicit] for implicit ODEs with mass matrices
/// - [OdeEquationsImplicitSens] for forward sensitivity equations
/// - [OdeEquationsAdjoint] for adjoint sensitivity equations
///
/// It also provides implementations:
/// - [DiffSl] for equations specified in the DiffSL domain-specific language
/// - [SensEquations] and [AdjointEquations] for sensitivity computations
///
/// All the test equations used in Diffsol's test suite are also provided here.
pub mod ode_equations;

/// ODE solver implementations and traits.
///
/// This module provides the complete ODE solving interface including:
/// - [OdeSolverMethod] trait with implementations: [Bdf], [Sdirk], [ExplicitRk]
/// - [OdeSolverProblem] for problem setup (equations, parameters, tolerances, solver options etc.)
/// - [OdeSolverState] for managing solution state (including state vector, sensitivities, time, step size etc.)
/// - [OdeBuilder] for convenient problem construction (builds and configures [OdeSolverProblem])
/// - [Checkpointing] and [HermiteInterpolator] for solution interpolation
pub mod ode_solver;

/// Operator types and traits (e.g. non-linear, linear and constant operators; as well as their jacobians).
///
/// This module defines fundamental operator traits, for example:
/// - [NonLinearOp] and variants for Jacobians and sensitivities
/// - [LinearOp] for linear operators
/// - [ConstantOp] for constants
///
/// It also provides concrete implementations, for example:
/// - [Closure] for wrapping user-provided closures
/// - [LinearClosure] for linear operators
/// - [ConstantClosure] for constants
/// - [MatrixOp] for explicit matrix operators
pub mod op;

/// Scalar types and traits.
///
/// This module defines the [Scalar] trait that all floating-point types used in DiffSol must implement.
/// It aggregates requirements from nalgebra, faer, and num_traits to ensure compatibility with linear algebra operations.
///
/// Implementations are provided for `f32` and `f64`.
/// GPU scalar types are available via `ScalarCuda` (requires `cuda` feature).
pub mod scalar;

/// Vector types and traits.
///
/// This module defines the [Vector] trait and related traits for vector operations:
/// - [VectorView] and [VectorViewMut] for borrowed views
/// - [VectorIndex] for index collections
/// - [VectorHost] for CPU-resident vectors with direct access
///
/// Implementations are provided for:
/// - [NalgebraVec] using nalgebra vectors
/// - [FaerVec] using faer vectors
/// - `CudaVec` for GPU computation (requires `cuda` feature)
pub mod vector;

/// Error types and handling.
///
/// This module defines the [DiffsolError] enum and specialized error variants
/// for different failure modes in ODE solving, including parsing, compilation,
/// and numerical errors.
pub mod error;

pub use error::DiffsolError;

#[cfg(feature = "sundials")]
pub mod sundials_sys;

pub use linear_solver::LinearSolver;
pub use linear_solver::{faer::sparse_lu::FaerSparseLU, FaerLU, NalgebraLU};

pub use context::{faer::FaerContext, nalgebra::NalgebraContext, Context};

#[cfg(feature = "suitesparse")]
pub use linear_solver::suitesparse::klu::KLU;

#[cfg(feature = "diffsl")]
pub use ode_equations::diffsl::{DiffSl, DiffSlContext};

pub use jacobian::{
    find_adjoint_non_zeros, find_jacobian_non_zeros, find_matrix_non_zeros,
    find_sens_adjoint_non_zeros, find_sens_non_zeros, find_transpose_non_zeros, JacobianColoring,
};
use matrix::extract_block::ColMajBlock;
pub use matrix::{
    default_solver::DefaultSolver, dense_faer_serial::FaerMat, dense_nalgebra_serial::NalgebraMat,
    sparse_faer::FaerSparseMat, DenseMatrix, Matrix, MatrixCommon,
};

use matrix::{
    sparsity::Dense, sparsity::DenseRef, sparsity::MatrixSparsity, sparsity::MatrixSparsityRef,
    MatrixHost, MatrixRef, MatrixView, MatrixViewMut,
};
use nonlinear_solver::{
    convergence::Convergence, convergence::ConvergenceStatus, root::RootFinder,
};
pub use nonlinear_solver::{
    line_search::{BacktrackingLineSearch, LineSearch, NoLineSearch},
    newton::NewtonNonlinearSolver,
    NonLinearSolver,
};
pub use ode_equations::{
    adjoint_equations::AdjointContext, adjoint_equations::AdjointEquations,
    adjoint_equations::AdjointInit, adjoint_equations::AdjointRhs, sens_equations::SensEquations,
    sens_equations::SensInit, sens_equations::SensRhs, AugmentedOdeEquations,
    AugmentedOdeEquationsImplicit, NoAug, OdeEquations, OdeEquationsAdjoint, OdeEquationsImplicit,
    OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, OdeEquationsRef, OdeEquationsStoch,
    OdeSolverEquations,
};
use ode_solver::jacobian_update::JacobianUpdate;
pub use ode_solver::sde::SdeSolverMethod;
pub use ode_solver::state::{StateRef, StateRefMut};
pub use ode_solver::{
    adjoint::AdjointOdeSolverMethod,
    bdf::Bdf,
    bdf_state::BdfState,
    builder::OdeBuilder,
    checkpointing::Checkpointing,
    checkpointing::HermiteInterpolator,
    config::{
        BdfConfig, ExplicitRkConfig, OdeSolverConfig, OdeSolverConfigMut, OdeSolverConfigRef,
        SdirkConfig,
    },
    explicit_rk::ExplicitRk,
    method::AugmentedOdeSolverMethod,
    method::OdeSolverMethod,
    method::OdeSolverStopReason,
    problem::{InitialConditionSolverOptions, OdeSolverOptions, OdeSolverProblem},
    sdirk::Sdirk,
    sdirk_state::RkState,
    sensitivities::SensitivitiesOdeSolverMethod,
    state::OdeSolverState,
    tableau::Tableau,
};
pub use op::constant_op::{ConstantOp, ConstantOpSens, ConstantOpSensAdjoint};
pub use op::linear_op::{LinearOp, LinearOpSens, LinearOpTranspose};
pub use op::nonlinear_op::{
    NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint,
};
pub use op::stoch::{StochOp, StochOpKind};
pub use op::{
    closure::Closure, closure_with_adjoint::ClosureWithAdjoint, constant_closure::ConstantClosure,
    constant_closure_with_adjoint::ConstantClosureWithAdjoint, linear_closure::LinearClosure,
    matrix::MatrixOp, unit::UnitCallable, BuilderOp, Op, ParameterisedOp,
};
use op::{
    closure_no_jac::ClosureNoJac, closure_with_sens::ClosureWithSens,
    constant_closure_with_sens::ConstantClosureWithSens, init::InitOp,
};
pub use scalar::{IndexType, Scalar, Scale, ScalarEnum};
pub use vector::DefaultDenseMatrix;
pub use vector::{
    faer_serial::{FaerVec, FaerVecIndex, FaerVecMut, FaerVecRef},
    nalgebra_serial::{NalgebraVec, NalgebraVecMut, NalgebraVecRef},
    Vector, VectorCommon, VectorHost, VectorViewHost, VectorIndex, VectorRef, VectorView, VectorViewMut,
};

#[cfg(feature = "cuda")]
pub use context::cuda::CudaContext;
#[cfg(feature = "cuda")]
pub use linear_solver::cuda::lu::CudaLU;
#[cfg(feature = "cuda")]
pub use matrix::cuda::{CudaMat, CudaMatMut, CudaMatRef};
#[cfg(feature = "cuda")]
pub use scalar::cuda::{CudaType, ScalarCuda};
#[cfg(feature = "cuda")]
pub use vector::cuda::{CudaIndex, CudaVec, CudaVecMut, CudaVecRef};

pub use scalar::scale;
