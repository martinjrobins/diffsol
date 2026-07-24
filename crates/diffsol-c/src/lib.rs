//! # Diffsol-c
//!
//! Diffsol-c is a companion crate to [Diffsol](https://crates.io/crates/diffsol) that
//! provides two higher-level APIs on top of the core `diffsol` library:
//!
//! 1. A **dynamic dispatch** API that wraps the generic solver in runtime-dispatched types,
//!    allowing you to select the matrix backend, scalar type, linear solver, ODE solver method,
//!    and JIT backend at runtime rather than at compile time.
//! 2. A **C FFI** API that exposes the dynamic dispatch API via `extern "C"` functions,
//!    enabling integration with C, Wasm, Matlab and other languages
//!    with a C-compatible interface.
//!
//! ## Dynamic Dispatch API
//!
//! The core of the dynamic dispatch API is the [OdeWrapper] struct. It wraps the internal
//! solver state in an `Arc<Mutex<...>>` so that it can be shared across threads and across
//! FFI boundaries. Unlike the core `diffsol` crate — which requires you to specify the matrix
//! type, code generation backend, ODE equation type, linear solver, and solver method all as
//! generic parameters — `OdeWrapper` erases these type parameters behind trait objects so you
//! can choose them at runtime.
//!
//! To create an `OdeWrapper`, use one of the feature-gated constructors:
//! - `new_jit` if you have DiffSL code to JIT-compile (requires the
//!   `diffsl-cranelift` or `diffsl-llvm` feature).
//! - `new_external` if you have pre-compiled DiffSL symbols (requires the
//!   `external` feature).
//! - `new_external_dynamic` if you have a shared library with pre-compiled
//!   DiffSL symbols (requires the `diffsl-external-dynamic` feature).
//!
//! Once created, the `OdeWrapper` provides methods to configure solver tolerances and options
//! (e.g. [OdeWrapper::set_rtol], [OdeWrapper::set_atol], [OdeWrapper::set_t0],
//! [OdeWrapper::set_h0]), query equation metadata (e.g. [OdeWrapper::get_nstates],
//! [OdeWrapper::get_nparams], [OdeWrapper::get_nout]), and access separate option handles
//! via [OdeWrapper::get_options] (returning an [OdeSolverOptions]) and
//! [OdeWrapper::get_ic_options] (returning an [InitialConditionSolverOptions]).
//!
//! Solving is done via the following methods:
//! - [OdeWrapper::solve] — adaptive-time integration to a final time.
//! - [OdeWrapper::solve_dense] — integration to a specified set of time points.
//! - [OdeWrapper::solve_fwd_sens] — forward sensitivity analysis.
//! - [OdeWrapper::solve_continuous_adjoint] — continuous adjoint sensitivity analysis.
//! - [OdeWrapper::solve_adjoint_fwd] / [OdeWrapper::solve_adjoint_bkwd] — discrete adjoint
//!   sensitivity analysis (forward + backward pass).
//!
//! Each solve method returns a [SolutionWrapper], from which you can extract the solution
//! values via [SolutionWrapper::get_ys], [SolutionWrapper::get_ts], and
//! [SolutionWrapper::get_sens]. These methods return [HostArray] objects, which are
//! read-only views of the Rust-allocated data that can be safely accessed from the host
//! language without copying.
//!
//! The adoint solve methods involve an [AdjointCheckpointWrapper] that stores forward-pass
//! checkpoint data required for the backward pass.
//!
//! ### Runtime configuration types
//!
//! The dynamic dispatch API uses the following enums to configure the solver at runtime:
//! - [OdeSolverType] — the ODE solver method ([OdeSolverType::Bdf],
//!   [OdeSolverType::Esdirk34], [OdeSolverType::TrBdf2], [OdeSolverType::Tsit45]).
//! - [MatrixType] — the matrix backend ([MatrixType::NalgebraDense],
//!   [MatrixType::FaerDense], [MatrixType::FaerSparse]).
//! - [ScalarType] — the floating-point type ([ScalarType::F32], [ScalarType::F64]).
//! - [LinearSolverType] — the linear solver for implicit methods
//!   ([LinearSolverType::Default], [LinearSolverType::Lu], [LinearSolverType::Klu]).
//! - [JitBackendType] — the JIT backend for compiling DiffSL code
//!   (`Cranelift` or `Llvm` depending on enabled features).
//!
//! ### Error handling
//!
//! All operations that can fail in the dynamic dispatch API return
//! `Result<_, [DiffsolRtError]>`, a runtime error type wrapping the core
//! [DiffsolError](diffsol::error::DiffsolError).
//!
//! ## C FFI API
//!
//! The C FFI API is exposed via `extern "C"` functions across several modules suffixed `_c`.
//! These functions wrap the dynamic dispatch API using raw pointers and integer error codes
//! so that they can be called from C or any language with C interop. The public `_c` modules
//! are:
//! - [ode_c] — create, configure, solve, and destroy [OdeWrapper] handles.
//! - [solution_wrapper_c] — extract solution data and destroy [SolutionWrapper] handles.
//! - [ode_options_c] — get/set solver convergence and performance options.
//! - [initial_condition_options_c] — get/set initial condition solver options.
//! - [matrix_type_c], [scalar_type_c], [linear_solver_type_c],
//!   [ode_solver_type_c], [jit_c] — query and convert enum values.
//! - [host_array_c] — inspect and free [HostArray] data.
//! - [error_c] — retrieve and clear thread-local error messages.
//! - [string_c] — allocate and free Rust-owned strings from C.
//!
//! All C FFI functions follow a common pattern:
//! - Return `i32` (0 = success, negative = error).
//! - Store error details in a thread-local variable retrievable via functions in [error_c].
//! - Use raw pointers for ownership transfer (caller allocates/frees via dedicated functions).
//!
//! See [c_api_utils] for the internal macros and helper functions used by the C FFI layer.
//!
//! ## Feature flags
//!
//! The crate requires at least one of the following features to be enabled:
//! - `diffsl-cranelift` — enable the Cranelift JIT backend for DiffSL code.
//! - `diffsl-llvm` — enable the LLVM JIT backend for DiffSL code (requires an LLVM
//!   installation; use a version-specific feature like `diffsl-llvm21`).
//! - `external` — enable statically linked pre-compiled DiffSL symbols.
//! - `external-dynamic` — enable dynamically loaded pre-compiled DiffSL symbols.
//!
//! Additional features:
//! - `suitesparse` — enable the KLU sparse linear solver.
//! - `sundials` — enable SUNDIALS support (via the core `diffsol` crate).
//! - `cuda` — enable CUDA GPU support (via the core `diffsol` crate).
//!
//! ## Utility functions
//!
//! The [utils] module provides a few standalone functions:
//! - [utils::version] — returns the crate version.
//! - [utils::is_klu_available] — checks whether KLU was compiled in.
//! - [utils::is_sens_available] — checks whether sensitivity analysis is supported on this platform.

// Configure dlmalloc as the global allocator for WASM targets
// dlmalloc is a maintained alternative to the unmaintained wee_alloc
#[cfg(target_arch = "wasm32")]
use dlmalloc::GlobalDlmalloc;

#[cfg(not(any(
    feature = "external",
    feature = "external-dynamic",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm"
)))]
compile_error!(
    "diffsol-c requires one of `external`, `external-dynamic`, `diffsl-cranelift`, or `diffsl-llvm`."
);

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: GlobalDlmalloc = GlobalDlmalloc;

#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm16"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm17"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm18"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm17"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm18"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm18"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm18", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm18", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm18", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm19", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm19", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm20", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");

/// Adjoint checkpoint wrapper for storing and retrieving forward-pass checkpoint data.
///
/// This module provides [AdjointCheckpointWrapper], a thread-safe wrapper around the
/// internal `AdjointCheckpoint` trait object. It is created during the forward pass of a
/// discrete adjoint solve (via [OdeWrapper::solve_adjoint_fwd](crate::ode::OdeWrapper::solve_adjoint_fwd))
/// and consumed during the backward pass (via [OdeWrapper::solve_adjoint_bkwd](crate::ode::OdeWrapper::solve_adjoint_bkwd)).
///
/// The internal `AdjointCheckpoint` trait erases the concrete matrix, codegen, and solver tag
/// types, allowing the checkpoint to be shared between the forward and backward solvers which
/// may use different ODE solver methods (e.g. BDF forward, ESDIRK backward).
///
/// The `AdjointCheckpointData<M, CG, Tag>` struct stores the concrete [CheckpointingPath](diffsol::CheckpointingPath)
/// plus the forward solver metadata (method, linear solver, parameters).
pub mod adjoint_checkpoint;
/// Macros and helper functions for the C FFI layer.
///
/// This module provides the internal infrastructure used by the `*_c` modules to implement
/// `extern "C"` functions. It defines:
/// - Return code constants: `DIFFSOL_OK` (0), `DIFFSOL_ERR` (-1), `DIFFSOL_BAD_ARG` (-2).
/// - Conversion traits `CMapTo<Out>` / `CMapFrom<In>` for mapping between Rust and C types.
/// - Helper functions for common patterns: `map_get`, `map_set`, `invalid_arg_at`, `error_at`,
///   `null_err_at`, `valid_f64_ptr`.
/// - Convenience macros: `c_invalid_arg!`, `c_error!`, `c_null_err!`, `c_getter_simple!`,
///   `c_setter_simple!`.
///
/// This module is not intended for external use; it exists to share boilerplate across the
/// `_c` modules.
pub mod c_api_utils;
/// Runtime error type for the dynamic dispatch API.
///
/// This module defines [DiffsolRtError], a newtype wrapper around
/// [DiffsolError](diffsol::error::DiffsolError). It implements
/// `Display`, `std::error::Error`, and `From<DiffsolError>`. All operations in the
/// dynamic dispatch API that can fail return `Result<_, DiffsolRtError>`.
pub mod error;
/// C FFI error handling.
///
/// This module provides thread-local error storage and `extern "C"` functions for retrieving
/// and clearing error information from C code:
/// - `diffsol_error_code` — returns `DIFFSOL_OK`, `DIFFSOL_ERR`, or `DIFFSOL_BAD_ARG`.
/// - `diffsol_last_error_message` — returns a pointer to the last error message.
/// - `diffsol_last_error_file` / `diffsol_last_error_line` — return the source location of the last error.
/// - `diffsol_clear_last_error` — clears the thread-local error state.
#[path = "error_c.rs"]
pub mod error_c;
/// Read-only array allocated in Rust, safe to access from the host language without copying.
///
/// This module defines [HostArray], a bridge between Rust-owned numeric arrays
/// and the host language (Python, C, Wasm/JS). A `HostArray` stores a raw data pointer together
/// with shape, strides, and scalar type metadata. It optionally owns the underlying allocation
/// via a `Box<dyn Any>`, ensuring the data remains valid for as long as the `HostArray` exists.
///
/// `HostArray` is the return type of
/// [SolutionWrapper::get_ys],
/// [SolutionWrapper::get_ts], and
/// [SolutionWrapper::get_sens].
///
/// # Conversion traits
/// - [ToHostArray] — convert owned Rust types (e.g. `Vec<T>`,
///   `nalgebra::DMatrix<T>`, `faer::Mat<T>`) into a `HostArray`.
/// - [FromHostArray] — convert a `HostArray` back into a Rust type (e.g.
///   `Vec<f64>`, `Vec<Vec<f64>>`, `ndarray::ArrayView2`).
///
/// # Key methods
/// - [HostArray::new], [HostArray::new_vector],
///   [HostArray::alloc_vector],
///   [HostArray::new_col_major] — constructors.
/// - [HostArray::as_array] — view as `ndarray::ArrayView2`.
/// - [HostArray::as_slice] — view as `&[T]`.
/// - `HostArray::data_ptr`, `HostArray::ndim`,
///   `HostArray::dim`, `HostArray::stride`,
///   `HostArray::dtype` — metadata accessors.
pub mod host_array;
/// C FFI functions for inspecting and freeing [HostArray] objects.
///
/// Provides `extern "C"` functions for querying `HostArray` metadata (dimensions, strides,
/// data type, raw pointer) and for freeing a `HostArray` previously returned by this library.
#[path = "host_array_c.rs"]
pub mod host_array_c;
/// Options for the initial condition solver used before integration.
///
/// This module defines [InitialConditionSolverOptions], a handle obtained via
/// [OdeWrapper::get_ic_options](crate::ode::OdeWrapper::get_ic_options). It provides
/// getters and setters for configuring the nonlinear root-finding solver that computes
/// the initial state of the ODE system:
/// - Line search settings: [InitialConditionSolverOptions::get_use_linesearch],
///   [InitialConditionSolverOptions::get_max_linesearch_iterations].
/// - Newton iteration limits: [InitialConditionSolverOptions::get_max_newton_iterations],
///   [InitialConditionSolverOptions::get_max_linear_solver_setups].
/// - Step control: [InitialConditionSolverOptions::get_step_reduction_factor],
///   [InitialConditionSolverOptions::get_armijo_constant].
///
/// Also includes [InitialConditionSolverOptionsSnapshot], a serializable snapshot used
/// for serde round-tripping of solver configuration.
pub mod initial_condition_options;
/// C FFI functions for getting and setting [InitialConditionSolverOptions].
///
/// Provides `extern "C"` getter/setter functions for each option field exposed by
/// `InitialConditionSolverOptions`, using `i32` for boolean values and `f64`/`usize`
/// for numeric fields.
#[path = "initial_condition_options_c.rs"]
pub mod initial_condition_options_c;
/// JIT backend type for compiling DiffSL code at runtime.
///
/// This module defines [JitBackendType], an enum representing the available JIT
/// compilation backends for DiffSL code:
/// - `Cranelift` — available when the `diffsl-cranelift` feature is enabled.
/// - `Llvm` — available when the `diffsl-llvm` feature is enabled.
///
/// The function [default_enabled_jit_backend] returns the preferred backend based on which
/// features are compiled in (LLVM is preferred over Cranelift if both are available).
pub mod jit;
/// C FFI functions for querying [JitBackendType] enum values.
///
/// Provides `extern "C"` functions to enumerate available JIT backends, check whether a
/// backend value is valid, and get the name string of a backend.
#[path = "jit_c.rs"]
pub mod jit_c;
/// Linear solver type for implicit ODE solvers.
///
/// This module defines [LinearSolverType], an enum specifying which linear solver to use
/// inside implicit ODE methods (BDF, ESDIRK, TR-BDF2):
/// - [LinearSolverType::Default] — the solver's default choice (typically LU).
/// - [LinearSolverType::Lu] — LU decomposition (dense or sparse as appropriate).
/// - [LinearSolverType::Klu] — KLU sparse linear solver (requires `suitesparse` feature).
pub mod linear_solver_type;
/// C FFI functions for querying and converting [LinearSolverType] enum values.
///
/// Provides `extern "C"` functions to map between `i32` values and the `LinearSolverType`
/// enum variants, and to get the name of a linear solver type as a C string.
#[path = "linear_solver_type_c.rs"]
pub mod linear_solver_type_c;
/// Matrix backend type for the ODE solver.
///
/// This module defines [MatrixType], an enum specifying the matrix/linear algebra backend:
/// - [MatrixType::NalgebraDense] — dense matrix using the [nalgebra](https://nalgebra.org) crate.
/// - [MatrixType::FaerDense] — dense matrix using the [faer](https://github.com/sarah-ek/faer-rs) crate.
/// - [MatrixType::FaerSparse] — sparse matrix using the [faer](https://github.com/sarah-ek/faer-rs) crate.
pub mod matrix_type;
/// C FFI functions for querying and converting [MatrixType] enum values.
///
/// Provides `extern "C"` functions to map between `i32` values and the `MatrixType` enum
/// variants, and to get the name of a matrix type as a C string.
#[path = "matrix_type_c.rs"]
pub mod matrix_type_c;
/// Dynamic dispatch ODE wrapper — the primary entry point for the dynamic dispatch API.
///
/// This module defines [OdeWrapper], a thread-safe handle around an internal ODE solver.
/// `OdeWrapper` erases the generic type parameters of the core `diffsol` solver (matrix type,
/// codegen backend, linear solver, solver method) so that they can be chosen at runtime rather
/// than compile time. The internal state is stored behind an `Arc<Mutex<...>>` for safe sharing
/// across threads.
///
/// # Constructors
/// - `new_jit` — JIT-compile DiffSL code (requires `diffsl-cranelift` or `diffsl-llvm`).
/// - `new_external` — use statically linked pre-compiled DiffSL symbols (requires `external`).
/// - `new_external_dynamic` — load pre-compiled DiffSL symbols from a shared library
///   (requires `diffsl-external-dynamic`).
///
/// # Configuration
/// Tolerances: [OdeWrapper::set_rtol], [OdeWrapper::set_atol], [OdeWrapper::set_t0],
/// [OdeWrapper::set_h0], [OdeWrapper::set_integrate_out].
/// Sensitivity tolerances: [OdeWrapper::set_sens_rtol], [OdeWrapper::set_sens_atol],
/// [OdeWrapper::set_out_rtol], [OdeWrapper::set_out_atol],
/// [OdeWrapper::set_param_rtol], [OdeWrapper::set_param_atol].
/// Solver selection: [OdeWrapper::set_ode_solver], [OdeWrapper::set_linear_solver].
///
/// # Solving
/// - [OdeWrapper::solve] — adaptive-time integration.
/// - [OdeWrapper::solve_dense] — integration to specified time points.
/// - [OdeWrapper::solve_fwd_sens] — forward sensitivity analysis.
/// - [OdeWrapper::solve_continuous_adjoint] — continuous adjoint.
/// - [OdeWrapper::solve_adjoint_fwd] / [OdeWrapper::solve_adjoint_bkwd] — discrete adjoint.
///
/// # Low-level evaluation
/// - [OdeWrapper::y0] — compute initial state.
/// - [OdeWrapper::rhs] — evaluate the RHS function.
/// - [OdeWrapper::rhs_jac_mul] — evaluate the Jacobian-vector product.
///
/// # Serialization
/// `OdeWrapper` implements `Serialize`/`Deserialize` for JIT-backed solvers.
pub mod ode;
/// C FFI functions for creating, configuring, solving, and destroying [OdeWrapper] objects.
///
/// Provides `extern "C"` functions for the full lifecycle of an ODE solver:
/// - `diffsol_ode_wrapper_new_*` — constructors (feature-gated by JIT backend or external).
/// - `diffsol_ode_wrapper_get_*` / `diffsol_ode_wrapper_set_*` — getters/setters for tolerances,
///   time settings, solver type, and linear solver.
/// - `diffsol_ode_wrapper_y0`, `diffsol_ode_wrapper_rhs`, `diffsol_ode_wrapper_rhs_jac_mul` —
///   low-level RHS evaluation.
/// - `diffsol_ode_wrapper_solve`, `diffsol_ode_wrapper_solve_dense`,
///   `diffsol_ode_wrapper_solve_fwd_sens`, `diffsol_ode_wrapper_solve_continuous_adjoint`,
///   `diffsol_ode_wrapper_solve_adjoint_fwd`, `diffsol_ode_wrapper_solve_adjoint_bkwd` —
///   solve functions returning [SolutionWrapper] pointers.
/// - `diffsol_ode_wrapper_free` — free an `OdeWrapper`.
/// - Serialization: `diffsol_ode_wrapper_serialize`, `diffsol_ode_wrapper_deserialize`.
#[path = "ode_c.rs"]
pub mod ode_c;
/// ODE solver convergence and performance options for the dynamic dispatch API.
///
/// This module defines [OdeSolverOptions], a handle obtained via
/// [OdeWrapper::get_options](crate::ode::OdeWrapper::get_options). It provides getters and
/// setters for tuning the nonlinear solver and time-stepping behaviour of the ODE integrator:
/// - Nonlinear solver: [OdeSolverOptions::get_max_nonlinear_solver_iterations],
///   [OdeSolverOptions::get_max_error_test_failures].
/// - Jacobian updates: [OdeSolverOptions::get_update_jacobian_after_steps],
///   [OdeSolverOptions::get_update_rhs_jacobian_after_steps],
///   [OdeSolverOptions::get_threshold_to_update_jacobian],
///   [OdeSolverOptions::get_threshold_to_update_rhs_jacobian].
/// - Step size: [OdeSolverOptions::get_min_timestep].
///
/// Also includes [OdeSolverOptionsSnapshot], a serializable snapshot used for serde
/// round-tripping of solver configuration.
pub mod ode_options;
/// C FFI functions for getting and setting [OdeSolverOptions].
///
/// Provides `extern "C"` getter/setter functions for each option field exposed by
/// `OdeSolverOptions`, using `f64` for floating-point values and `usize` for integer fields.
#[path = "ode_options_c.rs"]
pub mod ode_options_c;
mod ode_solver_tag;
/// ODE solver method type.
///
/// This module defines [OdeSolverType], an enum specifying which ODE integration method to use:
/// - [OdeSolverType::Bdf] — Backward Differentiation Formula (stiff ODEs, singular mass matrices).
/// - [OdeSolverType::Esdirk34] — Explicit Singly Diagonally Implicit Runge-Kutta (moderately stiff).
/// - [OdeSolverType::TrBdf2] — Trapezoidal BDF of order 2 (moderately stiff).
/// - [OdeSolverType::Tsit45] — Tsitouras 4/5th order Explicit Runge-Kutta (non-stiff, explicit).
///
/// The internal `solve` methods on this enum dispatch to the appropriate diffsol solver type
/// via tag structs defined in the private `ode_solver_tag` module.
pub mod ode_solver_type;
/// C FFI functions for querying and converting [OdeSolverType] enum values.
///
/// Provides `extern "C"` functions to map between `i32` values and the `OdeSolverType` enum
/// variants, and to get the name of an ODE solver type as a C string.
#[path = "ode_solver_type_c.rs"]
pub mod ode_solver_type_c;
/// Floating-point scalar type for diffisol.
///
/// This module defines [ScalarType], an enum identifying the floating-point type used
/// by the solver:
/// - [ScalarType::F32] — 32-bit single precision.
/// - [ScalarType::F64] — 64-bit double precision.
///
/// It also defines the [Scalar] trait (combining [DiffSlScalar](diffsol::DiffSlScalar)
/// with [ToScalarType]) and the [ToScalarType] trait for mapping Rust types (`f32`, `f64`)
/// to their `ScalarType` variant.
pub mod scalar_type;
/// C FFI functions for querying and converting [ScalarType] enum values.
///
/// Provides `extern "C"` functions to map between `i32` values and the `ScalarType` enum
/// variants (`F32` / `F64`), and to get the name of a scalar type as a C string.
#[path = "scalar_type_c.rs"]
pub mod scalar_type_c;
/// Internal trait used by [SolutionWrapper] for type-erased solution access.
///
/// The `Solution` trait provides the object-safe interface that `SolutionWrapper` wraps.
/// It is implemented for [diffsol::Solution] for all compatible vector
/// types `V`. The trait is `pub(crate)` — the public API is exposed through `SolutionWrapper`.
pub mod solution;
/// Solution wrapper for the dynamic dispatch API.
///
/// This module defines [SolutionWrapper], a thread-safe handle around a type-erased ODE
/// solution. Returned by all solve methods on [OdeWrapper], it
/// provides methods to extract the solution data:
/// - [SolutionWrapper::get_ys] — the state values at each time point (returns [HostArray]).
/// - [SolutionWrapper::get_ts] — the time points (returns [HostArray]).
/// - [SolutionWrapper::get_sens] — the sensitivity matrices (returns `Vec<HostArray>`).
///
/// `SolutionWrapper` implements `Serialize`, serializing as `{ts, ys, sensitivities}`.
pub mod solution_wrapper;
/// C FFI functions for extracting data from and freeing [SolutionWrapper] objects.
///
/// Provides `extern "C"` functions to retrieve the time points, state values, and sensitivities
/// from a `SolutionWrapper`, and to free the wrapper when no longer needed.
#[path = "solution_wrapper_c.rs"]
pub mod solution_wrapper_c;
/// Core dynamic dispatch trait and its implementation.
///
/// Defines the `Solve` trait, the object-safe trait that erases the generic type parameters
/// of the core `diffsol` solver. Each combination of matrix type, codegen backend, and solver
/// method implements this trait. The concrete `GenericSolve<M, CG>` struct provides the
/// implementation.
///
/// Also provides factory functions (`solve_factory_jit`, `solve_factory_external`,
/// `solve_factory_external_dynamic`) that create the appropriate `Box<dyn Solve>` for a given
/// configuration. These are called by [OdeWrapper]'s constructors.
///
/// All items in this module are `pub(crate)`; the public API is exposed through `OdeWrapper`.
pub mod solve;
/// Internal macros for generating ODE solver option accessor methods.
///
/// This module provides macros used by the `Solve` trait implementation
/// to automatically generate getter/setter methods for initial condition and ODE solver options.
/// The macros are `pub(crate)` and intended only for internal use within the `diffsol-c` crate.
pub mod solve_macros;
/// Internal serialization utilities for the dynamic dispatch solver types.
///
/// Defines the `SolveSerialization<M>` trait and implementations for various codegen backends
/// (LLVM, Cranelift, external). Used by [OdeWrapper] during
/// serialization/deserialization. All items are `pub(crate)`.
mod solve_serialization;
/// C FFI memory management helpers for string and byte buffer allocation.
///
/// Provides `extern "C"` functions for allocating and freeing Rust-owned memory from C code:
/// - `diffsol_alloc_string` / `diffsol_free_string` — allocate/free Rust-owned C strings.
/// - `diffsol_alloc` / `diffsol_free` — allocate/free byte buffers.
///
/// These are used by the other `_c` modules when returning dynamically sized data (e.g.
/// serialized equation blobs) to C callers.
#[path = "string_c.rs"]
pub mod string_c;
/// Utility functions for querying library capabilities.
///
/// Provides standalone functions useful for runtime capability checks:
/// - [version] — returns the crate version string.
/// - [is_klu_available] — returns `true` if the `suitesparse` feature (KLU sparse linear solver) was enabled at compile time.
/// - [is_sens_available] — returns `true` if sensitivity analysis is supported on the current platform (not Windows).
pub mod utils;
/// Internal validation of linear solver compatibility with matrix types.
///
/// Provides `validate_linear_solver::<M>(linear_solver)` which checks that a given
/// [LinearSolverType] is compatible with a
/// matrix type `M`. For example, KLU is only valid for `FaerSparseMat<f64>` when the
/// `suitesparse` feature is enabled. Uses `LuValidator<M>` and `KluValidator<M>` traits
/// to associate each matrix type with its valid linear solver implementations.
pub mod valid_linear_solver;

#[cfg(test)]
mod test_support;

pub use adjoint_checkpoint::AdjointCheckpointWrapper;
pub use error::DiffsolRtError;
pub use host_array::{FromHostArray, HostArray, ToHostArray};
pub use initial_condition_options::{
    InitialConditionSolverOptions, InitialConditionSolverOptionsSnapshot,
};
pub use jit::{default_enabled_jit_backend, JitBackendType};
pub use linear_solver_type::LinearSolverType;
pub use matrix_type::MatrixType;
pub use ode::OdeWrapper;
pub use ode_options::{OdeSolverOptions, OdeSolverOptionsSnapshot};
pub use ode_solver_type::OdeSolverType;
pub use scalar_type::{Scalar, ScalarType, ToScalarType};
pub use solution_wrapper::SolutionWrapper;
pub use utils::{is_klu_available, is_sens_available, version};
