//! ODE equations defined by C callbacks.
//!
//! [`OdeCallbacks`] implements [`OdeEquations`](diffsol::OdeEquations). It also implements each
//! flavour that the solvers need: implicit, forward sensitivities and adjoint. The equations
//! come from a table of C function pointers, [`OdeCallbacksDesc`]. Thus a C host (or a different
//! FFI host) can define its own equations.
//!
//! There is one callback for each closure of [`diffsol::OdeBuilder`]. Each callback has the
//! arguments of its closure, plus a `user_data` pointer as the last argument. Only `rhs` and
//! `init` are necessary.
//!
//! # Callback conventions
//!
//! Each pointer points to a contiguous array of the scalar type `T` (`double` or `float`). The
//! table gives the array lengths. The counts are fields of [`OdeCallbacksDesc`].
//!
//! | callback | `x` | `p` | `v` | `y` (output) |
//! |---|---|---|---|---|
//! | `rhs`, `reset` | nstates | nparams | - | nstates |
//! | `rhs_jac`, `reset_jac` (`y = J v`) | nstates | nparams | nstates | nstates |
//! | `rhs_sens`, `reset_sens` (`y = J_p v`) | nstates | nparams | nparams | nstates |
//! | `rhs_adjoint`, `reset_adjoint` (`y = -J^T v`) | nstates | nparams | nstates | nstates |
//! | `rhs_sens_adjoint`, `reset_sens_adjoint` (`y = -J_p^T v`) | nstates | nparams | nstates | nparams |
//! | `root` | nstates | nparams | - | nroots |
//! | `root_jac` / `root_sens` | nstates | nparams | nstates / nparams | nroots |
//! | `root_adjoint` / `root_sens_adjoint` | nstates | nparams | nroots | nstates / nparams |
//! | `out` | nstates | nparams | - | nout |
//! | `out_jac` / `out_sens` | nstates | nparams | nstates / nparams | nout |
//! | `out_adjoint` / `out_sens_adjoint` | nstates | nparams | nout | nstates / nparams |
//! | `init` | - | nparams | - | nstates |
//! | `init_sens` / `init_sens_adjoint` | - | nparams | nparams / nstates | nstates / nparams |
//! | `mass` / `mass_adjoint` (`y = M v + beta y`, `M^T` for the adjoint) | - | nparams | nstates | nstates |
//!
//! `J` is the Jacobian of the function with respect to the state `x`. `J_p` is the Jacobian
//! with respect to the parameters `p`.
//!
//! Each `*_adjoint` and `*_sens_adjoint` callback calculates the *negative* transposed product.
//! For example, `out_adjoint` calculates `y = -J^T v`, and `init_sens_adjoint` calculates
//! `y = -(dy0/dp)^T v`. `mass_adjoint` is the only exception: it calculates `y = M^T v + beta y`.
//!
//! Each callback has the same meaning as the related [`diffsol::OdeBuilder`] closure. For
//! example, `rhs_adjoint` is the `rhs_adjoint` closure of
//! [`diffsol::OdeBuilder::rhs_adjoint_implicit`].
//!
//! # Return codes
//!
//! Each callback returns an `int`. The value `0` means success. A different value means
//! failure. When a callback fails, its output is filled with NaN.
//!
//! The solver can recover from a failure. For example, it can reject a step and try a smaller
//! step. Thus `rhs` can return an error for a state that is not in its domain.
//!
//! If the solve succeeds, the callback failures are ignored.
//!
//! If the solve fails, its error is the error of the solver. If a callback failed during the
//! solve, the message also gives the first failure, for example
//! `(a callback failed during the solve: callback 'rhs' returned error code 7)`. This failure is
//! often the cause. But it can also be an earlier failure that the solver recovered from.
//!
//! # Sparsity patterns
//!
//! For a sparse matrix type (`FaerSparse`), the construction finds the sparsity pattern of each
//! matrix one time. It calls each derivative callback with NaN in the inputs `x` and `v`. Each
//! output that is NaN is a non-zero. This is the same method that [`diffsol::OdeBuilder`] uses.
//!
//! The probe uses the initial state at `t = 0`, with all parameters zero. The patterns do not
//! change after this, also when the parameters change. Thus, for a sparse matrix type:
//! - each callback must accept NaN inputs, and the parameters `p = 0`;
//! - the pattern of each derivative must not depend on the values of `x`, `p` or `t`. For
//!   example, do not use an `if` on an input value to decide whether to write an output.
//!
//! If a callback does not obey these rules, the matrix can be incorrect. Then the solver can
//! converge slowly or fail. The dense matrix types do not use a pattern.
//!
//! # Ownership of `user_data`
//!
//! Each callback gets `user_data` as its last argument, with no change.
//!
//! If `free_user_data` is set, [`OdeCallbacks`] owns `user_data` from the start of its
//! construction. This is also true if the construction fails. [`OdeCallbacks`] calls
//! `free_user_data(user_data)` one time, when it is dropped.
//!
//! If `free_user_data` is not set, `user_data` must stay valid until the equations are dropped.
//!
//! # Threads
//!
//! The callbacks are called one at a time. They are never called at the same time from two
//! threads.
//!
//! But a solver can move between threads. An `OdeWrapper` is `Send` and `Sync`. Thus the
//! callbacks run on the thread that uses the solver at that time. `free_user_data` runs on the
//! thread that drops the solver. If `user_data` must stay on one thread (for example a Python
//! object that needs the GIL, or a JNI local reference), use the solver and drop it only on that
//! thread.
use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    fmt,
};

use diffsol::{
    error::DiffsolError,
    jacobian::{
        find_adjoint_non_zeros, find_constant_sens_non_zeros, find_jacobian_non_zeros,
        find_matrix_non_zeros, find_sens_adjoint_non_zeros, find_sens_non_zeros, JacobianColoring,
    },
    matrix::sparsity::MatrixSparsity,
    op::OpStatistics,
    ConstantOp, ConstantOpSens, ConstantOpSensAdjoint, LinearOp, LinearOpTranspose, Matrix,
    NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint,
    OdeEquations, OdeEquationsRef, Op, Scalar, Vector,
};

use num_traits::Zero;

use crate::{
    error::DiffsolRtError,
    linear_solver_type::LinearSolverType,
    matrix_type::MatrixType,
    ode::OdeWrapper,
    ode_solver_type::OdeSolverType,
    scalar_type::ToScalarType,
    solve::{GenericSolve, Solve},
    solve_eqn::{EqnRequirement, SolveEqn},
};

/// `y = F(x, p, t)`: used for `rhs`, `root`, `out` and `reset`.
pub type OdeCallbackFn<T> =
    unsafe extern "C" fn(x: *const T, p: *const T, t: T, y: *mut T, user_data: *mut c_void) -> i32;

/// `y = A(x, p, t) v`: used for every `*_jac`, `*_sens`, `*_adjoint` and `*_sens_adjoint`
/// callback of `rhs`, `root`, `out` and `reset`.
pub type OdeCallbackMulFn<T> = unsafe extern "C" fn(
    x: *const T,
    p: *const T,
    t: T,
    v: *const T,
    y: *mut T,
    user_data: *mut c_void,
) -> i32;

/// `y = y0(p, t)`: the initial state.
pub type OdeCallbackInitFn<T> =
    unsafe extern "C" fn(p: *const T, t: T, y: *mut T, user_data: *mut c_void) -> i32;

/// `y = A(p, t) v`: used for `init_sens` and `init_sens_adjoint`.
pub type OdeCallbackInitMulFn<T> =
    unsafe extern "C" fn(p: *const T, t: T, v: *const T, y: *mut T, user_data: *mut c_void) -> i32;

/// `y = M(p, t) v + beta y`: used for `mass` (and `mass_adjoint`, with `M^T`).
pub type OdeCallbackMassFn<T> = unsafe extern "C" fn(
    v: *const T,
    p: *const T,
    t: T,
    beta: T,
    y: *mut T,
    user_data: *mut c_void,
) -> i32;

/// Frees `user_data`, see the [module documentation](self).
pub type OdeCallbackFreeFn = unsafe extern "C" fn(user_data: *mut c_void);

/// The C callbacks (and the problem dimensions) that define an [`OdeCallbacks`].
///
/// A `NULL` function pointer means the callback is not provided. Only `rhs` and `init` are
/// required; see the [module documentation](self) for the meaning and argument lengths of each
/// callback.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OdeCallbacksDesc<T> {
    /// Number of states.
    pub nstates: usize,
    /// Number of parameters.
    pub nparams: usize,
    /// Number of outputs of `out`, which must be 0 exactly when `out` is not provided.
    pub nout: usize,
    /// Number of root functions of `root`, which must be 0 exactly when `root` is not provided.
    pub nroots: usize,
    /// Passed as the last argument of every callback.
    pub user_data: *mut c_void,
    /// Called once with `user_data` when the equations are dropped. May be `NULL`.
    pub free_user_data: Option<OdeCallbackFreeFn>,

    pub rhs: Option<OdeCallbackFn<T>>,
    pub rhs_jac: Option<OdeCallbackMulFn<T>>,
    pub rhs_sens: Option<OdeCallbackMulFn<T>>,
    pub rhs_adjoint: Option<OdeCallbackMulFn<T>>,
    pub rhs_sens_adjoint: Option<OdeCallbackMulFn<T>>,

    pub init: Option<OdeCallbackInitFn<T>>,
    pub init_sens: Option<OdeCallbackInitMulFn<T>>,
    pub init_sens_adjoint: Option<OdeCallbackInitMulFn<T>>,

    pub mass: Option<OdeCallbackMassFn<T>>,
    pub mass_adjoint: Option<OdeCallbackMassFn<T>>,

    pub root: Option<OdeCallbackFn<T>>,
    pub root_jac: Option<OdeCallbackMulFn<T>>,
    pub root_sens: Option<OdeCallbackMulFn<T>>,
    pub root_adjoint: Option<OdeCallbackMulFn<T>>,
    pub root_sens_adjoint: Option<OdeCallbackMulFn<T>>,

    pub out: Option<OdeCallbackFn<T>>,
    pub out_jac: Option<OdeCallbackMulFn<T>>,
    pub out_sens: Option<OdeCallbackMulFn<T>>,
    pub out_adjoint: Option<OdeCallbackMulFn<T>>,
    pub out_sens_adjoint: Option<OdeCallbackMulFn<T>>,

    pub reset: Option<OdeCallbackFn<T>>,
    pub reset_jac: Option<OdeCallbackMulFn<T>>,
    pub reset_sens: Option<OdeCallbackMulFn<T>>,
    pub reset_adjoint: Option<OdeCallbackMulFn<T>>,
    pub reset_sens_adjoint: Option<OdeCallbackMulFn<T>>,
}

/// [`OdeCallbacksDesc`] for `double` precision.
pub type DiffsolOdeCallbacksF64 = OdeCallbacksDesc<f64>;
/// [`OdeCallbacksDesc`] for `float` precision.
pub type DiffsolOdeCallbacksF32 = OdeCallbacksDesc<f32>;

impl<T> OdeCallbacksDesc<T> {
    /// A descriptor with only the required callbacks set, and no `user_data`.
    pub fn new(
        nstates: usize,
        nparams: usize,
        rhs: OdeCallbackFn<T>,
        init: OdeCallbackInitFn<T>,
    ) -> Self {
        Self {
            nstates,
            nparams,
            nout: 0,
            nroots: 0,
            user_data: std::ptr::null_mut(),
            free_user_data: None,
            rhs: Some(rhs),
            rhs_jac: None,
            rhs_sens: None,
            rhs_adjoint: None,
            rhs_sens_adjoint: None,
            init: Some(init),
            init_sens: None,
            init_sens_adjoint: None,
            mass: None,
            mass_adjoint: None,
            root: None,
            root_jac: None,
            root_sens: None,
            root_adjoint: None,
            root_sens_adjoint: None,
            out: None,
            out_jac: None,
            out_sens: None,
            out_adjoint: None,
            out_sens_adjoint: None,
            reset: None,
            reset_jac: None,
            reset_sens: None,
            reset_adjoint: None,
            reset_sens_adjoint: None,
        }
    }

    /// Check the descriptor is self-consistent, returning a description of the first problem.
    fn validate(&self) -> Result<(), String> {
        if self.rhs.is_none() {
            return Err("the 'rhs' callback is required".to_string());
        }
        if self.init.is_none() {
            return Err("the 'init' callback is required".to_string());
        }
        if self.nstates == 0 {
            return Err("nstates must be greater than 0".to_string());
        }
        if self.root.is_some() != (self.nroots > 0) {
            return Err(
                "nroots must be greater than 0 exactly when the 'root' callback is provided"
                    .to_string(),
            );
        }
        if self.out.is_some() != (self.nout > 0) {
            return Err(
                "nout must be greater than 0 exactly when the 'out' callback is provided"
                    .to_string(),
            );
        }
        if self.reset.is_some() && self.root.is_none() {
            return Err("the 'reset' callback requires the 'root' callback".to_string());
        }
        if self.mass_adjoint.is_some() && self.mass.is_none() {
            return Err("the 'mass_adjoint' callback requires the 'mass' callback".to_string());
        }
        // the derivative callbacks of root, out and reset, in the order of `SUFFIXES`
        const SUFFIXES: [&str; 4] = ["jac", "sens", "adjoint", "sens_adjoint"];
        let derived = [
            (
                "root",
                self.root.is_some(),
                [
                    self.root_jac.is_some(),
                    self.root_sens.is_some(),
                    self.root_adjoint.is_some(),
                    self.root_sens_adjoint.is_some(),
                ],
            ),
            (
                "out",
                self.out.is_some(),
                [
                    self.out_jac.is_some(),
                    self.out_sens.is_some(),
                    self.out_adjoint.is_some(),
                    self.out_sens_adjoint.is_some(),
                ],
            ),
            (
                "reset",
                self.reset.is_some(),
                [
                    self.reset_jac.is_some(),
                    self.reset_sens.is_some(),
                    self.reset_adjoint.is_some(),
                    self.reset_sens_adjoint.is_some(),
                ],
            ),
        ];
        for (base, has_base, has_derived) in derived {
            for (suffix, has) in SUFFIXES.into_iter().zip(has_derived) {
                if has && !has_base {
                    return Err(format!(
                        "the '{base}_{suffix}' callback requires the '{base}' callback"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Free `user_data` with `free_user_data`, if set.
    ///
    /// # Safety
    /// Must be called at most once per `user_data`, and only if nothing else owns it.
    pub(crate) unsafe fn free_user_data(&self) {
        if let Some(free) = self.free_user_data {
            unsafe { free(self.user_data) };
        }
    }
}

/// A scalar type that C callbacks can use: `f64` or `f32`.
///
/// This trait is sealed. Users cannot implement it for other types.
pub trait CallbackScalar: crate::scalar_type::Scalar + sealed::Sealed {}

mod sealed {
    use crate::{
        error::DiffsolRtError, linear_solver_type::LinearSolverType, matrix_type::MatrixType,
        ode::OdeWrapper, ode_callbacks::OdeCallbacksDesc, ode_solver_type::OdeSolverType,
    };

    pub trait Sealed: Sized {
        /// Construct a solver from `callbacks`, with the scalar type `Self`.
        ///
        /// # Safety
        /// `callbacks` must satisfy the conditions of
        /// [`OdeCallbacks::new`](crate::ode_callbacks::OdeCallbacks::new).
        unsafe fn new_ode(
            callbacks: OdeCallbacksDesc<Self>,
            matrix_type: MatrixType,
            linear_solver: LinearSolverType,
            ode_solver: OdeSolverType,
        ) -> Result<OdeWrapper, DiffsolRtError>;
    }
}

/// Build a [Solve] for the matrix type `matrix_type` and the scalar type `$t` from `callbacks`.
///
/// The caller must satisfy the safety conditions of [OdeCallbacks::new].
macro_rules! solve_for_matrix_type {
    ($t:ty, $callbacks:expr, $matrix_type:expr) => {{
        let callbacks = $callbacks;
        // SAFETY: the caller guarantees the conditions of `OdeCallbacks::new`
        match $matrix_type {
            MatrixType::NalgebraDense => {
                let eqn = unsafe {
                    OdeCallbacks::<diffsol::NalgebraMat<$t>>::new(callbacks, Default::default())
                };
                Box::new(GenericSolve::from_eqn(eqn?)?) as Box<dyn Solve>
            }
            MatrixType::FaerDense => {
                let eqn = unsafe {
                    OdeCallbacks::<diffsol::FaerMat<$t>>::new(callbacks, Default::default())
                };
                Box::new(GenericSolve::from_eqn(eqn?)?)
            }
            MatrixType::FaerSparse => {
                let eqn = unsafe {
                    OdeCallbacks::<diffsol::FaerSparseMat<$t>>::new(callbacks, Default::default())
                };
                Box::new(GenericSolve::from_eqn(eqn?)?)
            }
        }
    }};
}

macro_rules! impl_callback_scalar {
    ($t:ty) => {
        impl sealed::Sealed for $t {
            unsafe fn new_ode(
                callbacks: OdeCallbacksDesc<$t>,
                matrix_type: MatrixType,
                linear_solver: LinearSolverType,
                ode_solver: OdeSolverType,
            ) -> Result<OdeWrapper, DiffsolRtError> {
                let solve = solve_for_matrix_type!($t, callbacks, matrix_type);
                OdeWrapper::build(
                    String::new(),
                    <$t as ToScalarType>::scalar_type(),
                    solve,
                    None,
                    linear_solver,
                    ode_solver,
                )
            }
        }

        impl CallbackScalar for $t {}
    };
}

impl_callback_scalar!(f64);
impl_callback_scalar!(f32);

impl OdeWrapper {
    /// Construct an ODE solver from equations that C callbacks give.
    ///
    /// The scalar type `T` (`f64` or `f32`) of the callbacks is also the scalar type of the
    /// solver. [crate::ode_callbacks] gives the callback conventions.
    ///
    /// The solver takes ownership of `callbacks.user_data`. If `callbacks.free_user_data` is
    /// set, the solver calls it one time. It calls it when the solver is dropped. If this
    /// function fails, it calls it before it returns.
    ///
    /// # Safety
    /// `callbacks` must satisfy the conditions of [OdeCallbacks::new].
    pub unsafe fn new_callbacks<T: CallbackScalar>(
        callbacks: OdeCallbacksDesc<T>,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        // SAFETY: the caller guarantees the conditions of `OdeCallbacks::new`
        unsafe { T::new_ode(callbacks, matrix_type, linear_solver, ode_solver) }
    }
}

/// An error recorded while evaluating the callbacks of an [`OdeCallbacks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeCallbackError {
    /// The callback returned a non-zero status code.
    Failed { callback: &'static str, code: i32 },
    /// The solver needed a callback that was not provided.
    Missing { callback: &'static str },
}

impl fmt::Display for OdeCallbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeCallbackError::Failed { callback, code } => {
                write!(f, "callback '{callback}' returned error code {code}")
            }
            OdeCallbackError::Missing { callback } => {
                f.write_str(&missing_message("evaluating the equations", callback))
            }
        }
    }
}

/// The message for a missing callback `name`, the same whether the missing callback is found
/// by the check before a solve or when the solver calls it.
fn missing_message(operation: &str, name: &str) -> String {
    format!("OdeCallbacks: {operation} requires the '{name}' callback")
}

/// A sparsity pattern together with the coloring used to evaluate a matrix with that pattern.
struct Pattern<M: Matrix> {
    sparsity: M::Sparsity,
    coloring: JacobianColoring<M>,
}

impl<M: Matrix> Pattern<M> {
    fn new(nrows: usize, ncols: usize, non_zeros: Vec<(usize, usize)>, ctx: &M::C) -> Self {
        let sparsity = MatrixSparsity::try_from_indices(nrows, ncols, non_zeros.clone())
            .expect("invalid sparsity pattern");
        let coloring = JacobianColoring::new(&sparsity, &non_zeros, ctx.clone());
        Self { sparsity, coloring }
    }
}

/// The patterns of the matrices of one non-linear operator (`rhs`, `root`, `out` or `reset`).
struct NonLinearPatterns<M: Matrix> {
    jac: Option<Pattern<M>>,
    adjoint: Option<Pattern<M>>,
    sens: Option<Pattern<M>>,
    sens_adjoint: Option<Pattern<M>>,
}

impl<M: Matrix> Default for NonLinearPatterns<M> {
    fn default() -> Self {
        Self {
            jac: None,
            adjoint: None,
            sens: None,
            sens_adjoint: None,
        }
    }
}

/// Every sparsity pattern of an [`OdeCallbacks`]. Only computed for sparse matrix types, where
/// the matrices cannot be built without them.
struct Patterns<M: Matrix> {
    rhs: NonLinearPatterns<M>,
    root: NonLinearPatterns<M>,
    out: NonLinearPatterns<M>,
    reset: NonLinearPatterns<M>,
    init_sens: Option<Pattern<M>>,
    mass: Option<Pattern<M>>,
    mass_transpose: Option<Pattern<M>>,
}

impl<M: Matrix> Default for Patterns<M> {
    fn default() -> Self {
        Self {
            rhs: NonLinearPatterns::default(),
            root: NonLinearPatterns::default(),
            out: NonLinearPatterns::default(),
            reset: NonLinearPatterns::default(),
            init_sens: None,
            mass: None,
            mass_transpose: None,
        }
    }
}

/// ODE equations defined by C callbacks, see the [module documentation](self).
pub struct OdeCallbacks<M: Matrix> {
    callbacks: OdeCallbacksDesc<M::T>,
    p: M::V,
    ctx: M::C,
    patterns: Patterns<M>,
    statistics: RefCell<OpStatistics>,
    error: Cell<Option<OdeCallbackError>>,
}

impl<M: Matrix> Drop for OdeCallbacks<M> {
    fn drop(&mut self) {
        // SAFETY: `OdeCallbacks` took ownership of `user_data` on construction, and is only
        // dropped once
        unsafe { self.callbacks.free_user_data() };
    }
}

impl<M: Matrix> OdeCallbacks<M> {
    /// Create the equations from `callbacks`. All parameters are zero at the start.
    ///
    /// This function takes ownership of `callbacks.user_data`, also if it returns an error. See
    /// the [module documentation](self).
    ///
    /// For sparse matrix types, this function finds the sparsity pattern of each derivative
    /// callback. To do this, it calls the callback with NaN inputs at the initial state and
    /// `t = 0`.
    ///
    /// # Safety
    /// - Each function pointer in `callbacks` must be null or a valid function with the
    ///   signature that the [module documentation](self) gives.
    /// - Each function must stay valid until the equations are dropped.
    /// - Each function must read and write only the array lengths in the table.
    /// - If `free_user_data` is set, the equations own `user_data`. Do not free `user_data`
    ///   yourself, and do not give it to other equations.
    /// - If `free_user_data` is not set, `user_data` must stay valid until the equations are
    ///   dropped.
    pub unsafe fn new(callbacks: OdeCallbacksDesc<M::T>, ctx: M::C) -> Result<Self, DiffsolError> {
        let mut eqn = Self {
            p: M::V::zeros(callbacks.nparams, ctx.clone()),
            callbacks,
            ctx,
            patterns: Patterns::default(),
            statistics: RefCell::new(OpStatistics::default()),
            error: Cell::new(None),
        };
        // `eqn` owns `user_data` from here on, so it is freed even if validation fails
        eqn.callbacks
            .validate()
            .map_err(|msg| DiffsolError::Other(format!("Invalid ODE callbacks: {msg}")))?;
        if M::is_sparse() {
            eqn.patterns = eqn.find_patterns();
            if let Some(err) = eqn.take_callback_error() {
                return Err(DiffsolError::Other(format!(
                    "Failed to find the sparsity pattern of the ODE callbacks: {err}"
                )));
            }
        }
        Ok(eqn)
    }

    /// Take (and clear) the first error recorded while evaluating the callbacks.
    pub fn take_callback_error(&self) -> Option<OdeCallbackError> {
        self.error.take()
    }

    fn record(&self, err: OdeCallbackError, y: &mut [M::T]) {
        if self.error.get().is_none() {
            self.error.set(Some(err));
        }
        y.fill(M::T::NAN);
    }

    fn check_code(&self, callback: &'static str, code: i32, y: &mut [M::T]) {
        if code != 0 {
            self.record(OdeCallbackError::Failed { callback, code }, y);
        }
    }

    /// Call the callback `f` on one batch lane `y` with `call`. If `f` is missing or fails, record
    /// the error and fill `y` with NaN.
    fn run<F>(
        &self,
        name: &'static str,
        f: Option<F>,
        y: &mut [M::T],
        call: impl FnOnce(F, &mut [M::T]) -> i32,
    ) {
        match f {
            Some(f) => {
                let code = call(f, y);
                self.check_code(name, code, y);
            }
            None => self.record(OdeCallbackError::Missing { callback: name }, y),
        }
    }

    // SAFETY (for each `unsafe` call below): the slices have the lengths documented for the
    // callback, and `user_data` is the pointer the caller gave with the callbacks.

    /// Evaluate `y = f(x, p, t)` for each batch lane.
    fn eval(
        &self,
        name: &'static str,
        f: Option<OdeCallbackFn<M::T>>,
        x: &M::V,
        t: M::T,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        y.for_each_batch_host([x, &self.p], |y, [x, p], _| {
            self.run(name, f, y, |f, y| unsafe {
                f(x.as_ptr(), p.as_ptr(), t, y.as_mut_ptr(), ud)
            })
        });
    }

    /// Evaluate `y = f(x, p, t, v)` for each batch lane.
    fn eval_mul(
        &self,
        name: &'static str,
        f: Option<OdeCallbackMulFn<M::T>>,
        x: &M::V,
        t: M::T,
        v: &M::V,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        y.for_each_batch_host([x, &self.p, v], |y, [x, p, v], _| {
            self.run(name, f, y, |f, y| unsafe {
                f(x.as_ptr(), p.as_ptr(), t, v.as_ptr(), y.as_mut_ptr(), ud)
            })
        });
    }

    /// Evaluate `y = f(p, t)` for each batch lane.
    fn eval_init(
        &self,
        name: &'static str,
        f: Option<OdeCallbackInitFn<M::T>>,
        t: M::T,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        y.for_each_batch_host([&self.p], |y, [p], _| {
            self.run(name, f, y, |f, y| unsafe {
                f(p.as_ptr(), t, y.as_mut_ptr(), ud)
            })
        });
    }

    /// Evaluate `y = f(p, t, v)` for each batch lane.
    fn eval_init_mul(
        &self,
        name: &'static str,
        f: Option<OdeCallbackInitMulFn<M::T>>,
        t: M::T,
        v: &M::V,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        y.for_each_batch_host([&self.p, v], |y, [p, v], _| {
            self.run(name, f, y, |f, y| unsafe {
                f(p.as_ptr(), t, v.as_ptr(), y.as_mut_ptr(), ud)
            })
        });
    }

    /// Evaluate `y = f(v, p, t, beta, y)` for each batch lane.
    fn eval_mass(
        &self,
        name: &'static str,
        f: Option<OdeCallbackMassFn<M::T>>,
        v: &M::V,
        t: M::T,
        beta: M::T,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        y.for_each_batch_host([v, &self.p], |y, [v, p], _| {
            self.run(name, f, y, |f, y| unsafe {
                f(v.as_ptr(), p.as_ptr(), t, beta, y.as_mut_ptr(), ud)
            })
        });
    }

    /// Find the sparsity patterns of the matrices that diffsol builds, for each provided
    /// derivative callback. These are the same matrices that `OdeBuilder` finds patterns for:
    /// all four of the rhs, the sensitivities of root and reset, and the sensitivities and
    /// adjoints of out.
    fn find_patterns(&self) -> Patterns<M> {
        let t0 = M::T::zero();
        let y0 = self.init().call(t0);
        let cb = &self.callbacks;
        let ctx = &self.ctx;
        let rhs = nonlinear_patterns(
            &self.rhs(),
            &y0,
            t0,
            [cb.rhs_jac, cb.rhs_adjoint, cb.rhs_sens, cb.rhs_sens_adjoint],
            ctx,
        );
        let root = self
            .root()
            .map(|root| nonlinear_patterns(&root, &y0, t0, [None, None, cb.root_sens, None], ctx));
        let out = self.out().map(|out| {
            let callbacks = [None, cb.out_adjoint, cb.out_sens, cb.out_sens_adjoint];
            nonlinear_patterns(&out, &y0, t0, callbacks, ctx)
        });
        let reset = self.reset().map(|reset| {
            nonlinear_patterns(&reset, &y0, t0, [None, None, cb.reset_sens, None], ctx)
        });
        let init_sens = (cb.init_sens.is_some() && cb.nparams > 0).then(|| {
            let init = self.init();
            let non_zeros = find_constant_sens_non_zeros(&init, t0);
            Pattern::new(init.nout(), cb.nparams, non_zeros, ctx)
        });
        let n = cb.nstates;
        let (mass, mass_transpose) = match self.mass() {
            Some(mass) => {
                let non_zeros = find_matrix_non_zeros(&mass, t0);
                let transpose = cb
                    .mass_adjoint
                    .map(|_| Pattern::new(n, n, transposed(&non_zeros), ctx));
                (Some(Pattern::new(n, n, non_zeros, ctx)), transpose)
            }
            None => (None, None),
        };
        Patterns {
            rhs,
            root: root.unwrap_or_default(),
            out: out.unwrap_or_default(),
            reset: reset.unwrap_or_default(),
            init_sens,
            mass,
            mass_transpose,
        }
    }

    /// Check the callbacks needed by `requirement` were provided. Each row of the tables below is
    /// `(needed, name, provided)`: the callback `name` is needed if `needed` is true.
    fn check(&self, requirement: EqnRequirement) -> Result<(), DiffsolRtError> {
        let cb = &self.callbacks;
        let (has_out, has_root, has_reset) =
            (cb.out.is_some(), cb.root.is_some(), cb.reset.is_some());
        // implicit methods need the rhs Jacobian. Tsit45, the only explicit method, does not
        // support a mass matrix, so a mass matrix needs no separate row.
        let needs_rhs_jac = |method| !matches!(method, OdeSolverType::Tsit45);
        let (operation, rows) = match requirement {
            EqnRequirement::Solve(method) => (
                "this solver",
                vec![(needs_rhs_jac(method), "rhs_jac", cb.rhs_jac.is_some())],
            ),
            EqnRequirement::FwdSens => (
                "a forward sensitivity solve",
                vec![
                    (true, "rhs_jac", cb.rhs_jac.is_some()),
                    (true, "rhs_sens", cb.rhs_sens.is_some()),
                    (true, "init_sens", cb.init_sens.is_some()),
                    (has_out, "out_jac", cb.out_jac.is_some()),
                    (has_out, "out_sens", cb.out_sens.is_some()),
                    (has_root, "root_jac", cb.root_jac.is_some()),
                    (has_root, "root_sens", cb.root_sens.is_some()),
                    (has_reset, "reset_jac", cb.reset_jac.is_some()),
                    (has_reset, "reset_sens", cb.reset_sens.is_some()),
                ],
            ),
            EqnRequirement::Adjoint(method) => (
                "an adjoint solve",
                vec![
                    (needs_rhs_jac(method), "rhs_jac", cb.rhs_jac.is_some()),
                    (true, "rhs_adjoint", cb.rhs_adjoint.is_some()),
                    (true, "rhs_sens_adjoint", cb.rhs_sens_adjoint.is_some()),
                    (true, "init_sens_adjoint", cb.init_sens_adjoint.is_some()),
                    (cb.mass.is_some(), "mass_adjoint", cb.mass_adjoint.is_some()),
                    (has_out, "out_adjoint", cb.out_adjoint.is_some()),
                    (has_out, "out_sens_adjoint", cb.out_sens_adjoint.is_some()),
                    (has_root, "root_jac", cb.root_jac.is_some()),
                    (has_root, "root_adjoint", cb.root_adjoint.is_some()),
                    (
                        has_root,
                        "root_sens_adjoint",
                        cb.root_sens_adjoint.is_some(),
                    ),
                    (has_reset, "reset_jac", cb.reset_jac.is_some()),
                    (has_reset, "reset_adjoint", cb.reset_adjoint.is_some()),
                    (
                        has_reset,
                        "reset_sens_adjoint",
                        cb.reset_sens_adjoint.is_some(),
                    ),
                ],
            ),
        };
        match rows
            .into_iter()
            .find(|&(needed, _, provided)| needed && !provided)
        {
            Some((_, name, _)) => Err(DiffsolError::Other(missing_message(operation, name)).into()),
            None => Ok(()),
        }
    }
}

/// The pattern of the transpose of a matrix with the non-zeros `non_zeros`.
fn transposed(non_zeros: &[(usize, usize)]) -> Vec<(usize, usize)> {
    non_zeros.iter().map(|&(i, j)| (j, i)).collect()
}

/// Find the patterns of the matrices of a non-linear operator, for each of the
/// `[jac, adjoint, sens, sens_adjoint]` callbacks that is provided. The pattern of an adjoint
/// matrix is the transpose of the forward pattern when that was found, otherwise it is probed.
fn nonlinear_patterns<M, F>(
    op: &F,
    y0: &M::V,
    t0: M::T,
    callbacks: [Option<OdeCallbackMulFn<M::T>>; 4],
    ctx: &M::C,
) -> NonLinearPatterns<M>
where
    M: Matrix,
    F: NonLinearOpJacobian<M = M, V = M::V, T = M::T, C = M::C>
        + NonLinearOpAdjoint<M = M, V = M::V, T = M::T, C = M::C>
        + NonLinearOpSens<M = M, V = M::V, T = M::T, C = M::C>
        + NonLinearOpSensAdjoint<M = M, V = M::V, T = M::T, C = M::C>,
{
    let [has_jac, has_adjoint, has_sens, has_sens_adjoint] = callbacks.map(|f| f.is_some());
    let (nstates, nout, nparams) = (op.nstates(), op.nout(), op.nparams());
    let has_params = nparams > 0;
    let jac = has_jac.then(|| find_jacobian_non_zeros(op, y0, t0));
    let sens = (has_sens && has_params).then(|| find_sens_non_zeros(op, y0, t0));
    let adjoint = has_adjoint.then(|| match &jac {
        Some(jac) => transposed(jac),
        None => find_adjoint_non_zeros(op, y0, t0),
    });
    let sens_adjoint = (has_sens_adjoint && has_params).then(|| match &sens {
        Some(sens) => transposed(sens),
        None => find_sens_adjoint_non_zeros(op, y0, t0),
    });
    NonLinearPatterns {
        jac: jac.map(|nz| Pattern::new(nout, nstates, nz, ctx)),
        adjoint: adjoint.map(|nz| Pattern::new(nstates, nout, nz, ctx)),
        sens: sens.map(|nz| Pattern::new(nout, nparams, nz, ctx)),
        sens_adjoint: sens_adjoint.map(|nz| Pattern::new(nparams, nout, nz, ctx)),
    }
}

impl<M: Matrix> Op for OdeCallbacks<M> {
    type T = M::T;
    type V = M::V;
    type M = M;
    type C = M::C;

    fn context(&self) -> &Self::C {
        &self.ctx
    }
    fn nstates(&self) -> usize {
        self.callbacks.nstates
    }
    fn nout(&self) -> usize {
        if self.callbacks.out.is_some() {
            self.callbacks.nout
        } else {
            self.callbacks.nstates
        }
    }
    fn nparams(&self) -> usize {
        self.callbacks.nparams
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

/// The right-hand side `F(x, p, t)` of an [`OdeCallbacks`].
pub struct OdeCallbacksRhs<'a, M: Matrix>(&'a OdeCallbacks<M>);
/// The root function of an [`OdeCallbacks`].
pub struct OdeCallbacksRoot<'a, M: Matrix>(&'a OdeCallbacks<M>);
/// The output function of an [`OdeCallbacks`].
pub struct OdeCallbacksOut<'a, M: Matrix>(&'a OdeCallbacks<M>);
/// The reset function of an [`OdeCallbacks`].
pub struct OdeCallbacksReset<'a, M: Matrix>(&'a OdeCallbacks<M>);
/// The initial condition of an [`OdeCallbacks`].
pub struct OdeCallbacksInit<'a, M: Matrix>(&'a OdeCallbacks<M>);
/// The mass matrix of an [`OdeCallbacks`].
pub struct OdeCallbacksMass<'a, M: Matrix>(&'a OdeCallbacks<M>);

macro_rules! impl_op_for_view {
    ($view:ident, |$eqn:ident| $nout:expr $(, $stats:ident)?) => {
        impl<M: Matrix> Op for $view<'_, M> {
            type T = M::T;
            type V = M::V;
            type M = M;
            type C = M::C;

            fn context(&self) -> &Self::C {
                &self.0.ctx
            }
            fn nstates(&self) -> usize {
                self.0.callbacks.nstates
            }
            #[allow(clippy::misnamed_getters)]
            fn nout(&self) -> usize {
                let $eqn = &self.0.callbacks;
                $nout
            }
            fn nparams(&self) -> usize {
                self.0.callbacks.nparams
            }
            $(
                fn $stats(&self) -> OpStatistics {
                    self.0.statistics.borrow().clone()
                }
            )?
        }
    };
}

impl_op_for_view!(OdeCallbacksRhs, |cb| cb.nstates, statistics);
impl_op_for_view!(OdeCallbacksRoot, |cb| cb.nroots);
impl_op_for_view!(OdeCallbacksOut, |cb| cb.nout);
impl_op_for_view!(OdeCallbacksReset, |cb| cb.nstates);
impl_op_for_view!(OdeCallbacksInit, |cb| cb.nstates);
impl_op_for_view!(OdeCallbacksMass, |cb| cb.nstates);

/// Implement the non-linear operator traits for a view of `rhs`, `root`, `out` or `reset`.
/// `$stats` says whether to count evaluations in the equations' statistics (only the rhs does).
macro_rules! impl_nonlinear_view {
    ($view:ident, $f:ident, $stats:literal) => {
        paste::paste! {
        impl<M: Matrix> NonLinearOp for $view<'_, M> {
            fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
                if $stats {
                    self.0.statistics.borrow_mut().increment_call();
                }
                self.0.eval(stringify!($f), self.0.callbacks.$f, x, t, y);
            }
        }

        impl<M: Matrix> NonLinearOpJacobian for $view<'_, M> {
            fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
                if $stats {
                    self.0.statistics.borrow_mut().increment_jac_mul();
                }
                self.0
                    .eval_mul(stringify!([<$f _jac>]), self.0.callbacks.[<$f _jac>], x, t, v, y);
            }
            fn jacobian_inplace(&self, x: &M::V, t: M::T, y: &mut M) {
                if $stats {
                    self.0.statistics.borrow_mut().increment_matrix();
                }
                match self.0.patterns.$f.jac.as_ref() {
                    Some(pattern) => pattern.coloring.jacobian_inplace(self, x, t, y),
                    None => self._default_jacobian_inplace(x, t, y),
                }
            }
            fn jacobian_sparsity(&self) -> Option<M::Sparsity> {
                self.0
                    .patterns
                    .$f
                    .jac
                    .as_ref()
                    .map(|p| p.sparsity.clone())
            }
        }

        impl<M: Matrix> NonLinearOpSens for $view<'_, M> {
            fn sens_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
                self.0
                    .eval_mul(stringify!([<$f _sens>]), self.0.callbacks.[<$f _sens>], x, t, v, y);
            }
            fn sens_inplace(&self, x: &M::V, t: M::T, y: &mut M) {
                match self.0.patterns.$f.sens.as_ref() {
                    Some(pattern) => pattern.coloring.sens_inplace(self, x, t, y),
                    None => self._default_sens_inplace(x, t, y),
                }
            }
            fn sens_sparsity(&self) -> Option<M::Sparsity> {
                self.0
                    .patterns
                    .$f
                    .sens
                    .as_ref()
                    .map(|p| p.sparsity.clone())
            }
        }

        impl<M: Matrix> NonLinearOpAdjoint for $view<'_, M> {
            fn jac_transpose_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
                if $stats {
                    self.0.statistics.borrow_mut().increment_jac_adj_mul();
                }
                self.0
                    .eval_mul(stringify!([<$f _adjoint>]), self.0.callbacks.[<$f _adjoint>], x, t, v, y);
            }
            fn adjoint_inplace(&self, x: &M::V, t: M::T, y: &mut M) {
                match self.0.patterns.$f.adjoint.as_ref() {
                    Some(pattern) => pattern.coloring.adjoint_inplace(self, x, t, y),
                    None => self._default_adjoint_inplace(x, t, y),
                }
            }
            fn adjoint_sparsity(&self) -> Option<M::Sparsity> {
                self.0
                    .patterns
                    .$f
                    .adjoint
                    .as_ref()
                    .map(|p| p.sparsity.clone())
            }
        }

        impl<M: Matrix> NonLinearOpSensAdjoint for $view<'_, M> {
            fn sens_transpose_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
                self.0.eval_mul(
                    stringify!([<$f _sens_adjoint>]),
                    self.0.callbacks.[<$f _sens_adjoint>],
                    x,
                    t,
                    v,
                    y,
                );
            }
            fn sens_adjoint_inplace(&self, x: &M::V, t: M::T, y: &mut M) {
                match self.0.patterns.$f.sens_adjoint.as_ref() {
                    Some(pattern) => pattern.coloring.sens_adjoint_inplace(self, x, t, y),
                    None => self._default_sens_adjoint_inplace(x, t, y),
                }
            }
            fn sens_adjoint_sparsity(&self) -> Option<M::Sparsity> {
                self.0
                    .patterns
                    .$f
                    .sens_adjoint
                    .as_ref()
                    .map(|p| p.sparsity.clone())
            }
        }
        }
    };
}

impl_nonlinear_view!(OdeCallbacksRhs, rhs, true);
impl_nonlinear_view!(OdeCallbacksRoot, root, false);
impl_nonlinear_view!(OdeCallbacksOut, out, false);
impl_nonlinear_view!(OdeCallbacksReset, reset, false);

impl<M: Matrix> ConstantOp for OdeCallbacksInit<'_, M> {
    fn call_inplace(&self, t: M::T, y: &mut M::V) {
        self.0.eval_init("init", self.0.callbacks.init, t, y);
    }
}

impl<M: Matrix> ConstantOpSens for OdeCallbacksInit<'_, M> {
    fn sens_mul_inplace(&self, t: M::T, v: &M::V, y: &mut M::V) {
        self.0
            .eval_init_mul("init_sens", self.0.callbacks.init_sens, t, v, y);
    }
    fn sens_inplace(&self, t: M::T, y: &mut M) {
        match self.0.patterns.init_sens.as_ref() {
            Some(pattern) => pattern.coloring.constant_sens_inplace(self, t, y),
            None => self._default_sens_inplace(t, y),
        }
    }
    fn sens_sparsity(&self) -> Option<M::Sparsity> {
        self.0
            .patterns
            .init_sens
            .as_ref()
            .map(|p| p.sparsity.clone())
    }
}

impl<M: Matrix> ConstantOpSensAdjoint for OdeCallbacksInit<'_, M> {
    fn sens_transpose_mul_inplace(&self, t: M::T, v: &M::V, y: &mut M::V) {
        self.0.eval_init_mul(
            "init_sens_adjoint",
            self.0.callbacks.init_sens_adjoint,
            t,
            v,
            y,
        );
    }
}

impl<M: Matrix> LinearOp for OdeCallbacksMass<'_, M> {
    fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
        self.0
            .eval_mass("mass", self.0.callbacks.mass, x, t, beta, y);
    }
    fn matrix_inplace(&self, t: M::T, y: &mut M) {
        match self.0.patterns.mass.as_ref() {
            Some(pattern) => pattern.coloring.matrix_inplace(self, t, y),
            None => self._default_matrix_inplace(t, y),
        }
    }
    fn sparsity(&self) -> Option<M::Sparsity> {
        self.0.patterns.mass.as_ref().map(|p| p.sparsity.clone())
    }
}

impl<M: Matrix> LinearOpTranspose for OdeCallbacksMass<'_, M> {
    fn gemv_transpose_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
        self.0
            .eval_mass("mass_adjoint", self.0.callbacks.mass_adjoint, x, t, beta, y);
    }
    fn transpose_inplace(&self, t: M::T, y: &mut M) {
        match self.0.patterns.mass_transpose.as_ref() {
            Some(pattern) => pattern.coloring.transpose_inplace(self, t, y),
            None => self._default_transpose_inplace(t, y),
        }
    }
    fn transpose_sparsity(&self) -> Option<M::Sparsity> {
        self.0
            .patterns
            .mass_transpose
            .as_ref()
            .map(|p| p.sparsity.clone())
    }
}

impl<'a, M: Matrix> OdeEquationsRef<'a> for OdeCallbacks<M> {
    type Mass = OdeCallbacksMass<'a, M>;
    type Rhs = OdeCallbacksRhs<'a, M>;
    type Root = OdeCallbacksRoot<'a, M>;
    type Init = OdeCallbacksInit<'a, M>;
    type Out = OdeCallbacksOut<'a, M>;
    type Reset = OdeCallbacksReset<'a, M>;
}

impl<M: Matrix> OdeEquations for OdeCallbacks<M> {
    fn rhs(&self) -> OdeCallbacksRhs<'_, M> {
        OdeCallbacksRhs(self)
    }
    fn mass(&self) -> Option<OdeCallbacksMass<'_, M>> {
        self.callbacks.mass.map(|_| OdeCallbacksMass(self))
    }
    fn root(&self) -> Option<OdeCallbacksRoot<'_, M>> {
        self.callbacks.root.map(|_| OdeCallbacksRoot(self))
    }
    fn out(&self) -> Option<OdeCallbacksOut<'_, M>> {
        self.callbacks.out.map(|_| OdeCallbacksOut(self))
    }
    fn reset(&self) -> Option<OdeCallbacksReset<'_, M>> {
        self.callbacks.reset.map(|_| OdeCallbacksReset(self))
    }
    fn init(&self) -> OdeCallbacksInit<'_, M> {
        OdeCallbacksInit(self)
    }
    fn set_params(&mut self, p: &M::V) {
        self.p.copy_from(p);
    }
    fn get_params(&self, p: &mut M::V) {
        p.copy_from(&self.p);
    }
}

impl<M: Matrix> SolveEqn for OdeCallbacks<M> {
    fn check_requirement(&self, requirement: EqnRequirement) -> Result<(), DiffsolRtError> {
        self.check(requirement)
    }

    fn take_error(&self) -> Option<String> {
        self.take_callback_error().map(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use diffsol::{
        ConstantOp, Context, LinearOp, LinearOpTranspose, Matrix, NalgebraContext, NalgebraMat,
        NonLinearOp, NonLinearOpJacobian, OdeEquations, Vector,
    };

    use super::{OdeCallbackError, OdeCallbacks, OdeCallbacksDesc};

    type M = NalgebraMat<f64>;

    unsafe extern "C" fn rhs(
        x: *const f64,
        p: *const f64,
        _t: f64,
        y: *mut f64,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe { *y = -*p * *x };
        0
    }

    unsafe extern "C" fn rhs_jac(
        _x: *const f64,
        p: *const f64,
        _t: f64,
        v: *const f64,
        y: *mut f64,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe { *y = -*p * *v };
        0
    }

    unsafe extern "C" fn failing_rhs(
        _x: *const f64,
        _p: *const f64,
        _t: f64,
        _y: *mut f64,
        _ud: *mut c_void,
    ) -> i32 {
        7
    }

    unsafe extern "C" fn init(_p: *const f64, _t: f64, y: *mut f64, _ud: *mut c_void) -> i32 {
        unsafe { *y = 1.0 };
        0
    }

    unsafe extern "C" fn count_free(ud: *mut c_void) {
        unsafe { &*(ud as *const AtomicUsize) }.fetch_add(1, Ordering::SeqCst);
    }

    // a 2 state system with the non-symmetric mass matrix [[1, 2], [0, 3]]
    unsafe extern "C" fn rhs2(
        x: *const f64,
        _p: *const f64,
        _t: f64,
        y: *mut f64,
        _ud: *mut c_void,
    ) -> i32 {
        let (x, y) = unsafe {
            (
                std::slice::from_raw_parts(x, 2),
                std::slice::from_raw_parts_mut(y, 2),
            )
        };
        y.copy_from_slice(&[-x[0], -x[1]]);
        0
    }

    unsafe extern "C" fn init2(_p: *const f64, _t: f64, y: *mut f64, _ud: *mut c_void) -> i32 {
        unsafe { std::slice::from_raw_parts_mut(y, 2) }.copy_from_slice(&[1.0, 1.0]);
        0
    }

    unsafe extern "C" fn mass2(
        v: *const f64,
        _p: *const f64,
        _t: f64,
        beta: f64,
        y: *mut f64,
        _ud: *mut c_void,
    ) -> i32 {
        let (v, y) = unsafe {
            (
                std::slice::from_raw_parts(v, 2),
                std::slice::from_raw_parts_mut(y, 2),
            )
        };
        y[0] = v[0] + 2.0 * v[1] + beta * y[0];
        y[1] = 3.0 * v[1] + beta * y[1];
        0
    }

    unsafe extern "C" fn mass2_adjoint(
        v: *const f64,
        _p: *const f64,
        _t: f64,
        beta: f64,
        y: *mut f64,
        _ud: *mut c_void,
    ) -> i32 {
        let (v, y) = unsafe {
            (
                std::slice::from_raw_parts(v, 2),
                std::slice::from_raw_parts_mut(y, 2),
            )
        };
        y[0] = v[0] + beta * y[0];
        y[1] = 2.0 * v[0] + 3.0 * v[1] + beta * y[1];
        0
    }

    fn desc() -> OdeCallbacksDesc<f64> {
        OdeCallbacksDesc::new(1, 1, rhs, init)
    }

    fn build(desc: OdeCallbacksDesc<f64>) -> Result<OdeCallbacks<M>, diffsol::DiffsolError> {
        // SAFETY: the test callbacks are valid and use the documented lengths
        unsafe { OdeCallbacks::<M>::new(desc, NalgebraContext::default()) }
    }

    fn build_err(desc: OdeCallbacksDesc<f64>) -> String {
        match build(desc) {
            Ok(_) => panic!("expected the descriptor to be rejected"),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn evaluates_callbacks() {
        let mut d = desc();
        d.rhs_jac = Some(rhs_jac);
        let mut eqn = build(d).unwrap();
        let ctx = NalgebraContext::default();
        eqn.set_params(&ctx.vector_from_vec(vec![2.0]));
        let x = ctx.vector_from_vec(vec![3.0]);
        eqn.rhs()
            .call(&x, 0.0)
            .assert_eq_st(&ctx.vector_from_vec(vec![-6.0]), 1e-12);
        eqn.rhs()
            .jac_mul(&x, 0.0, &x)
            .assert_eq_st(&ctx.vector_from_vec(vec![-6.0]), 1e-12);
        eqn.init()
            .call(0.0)
            .assert_eq_st(&ctx.vector_from_vec(vec![1.0]), 1e-12);
        assert!(eqn.mass().is_none());
        assert!(eqn.root().is_none());
        assert!(eqn.out().is_none());
        assert!(eqn.reset().is_none());
        assert!(eqn.take_callback_error().is_none());
    }

    #[test]
    fn records_failures_and_missing_callbacks() {
        let mut d = desc();
        d.rhs = Some(failing_rhs);
        let eqn = build(d).unwrap();
        let x = NalgebraContext::default().vector_from_vec(vec![3.0]);
        let y = eqn.rhs().call(&x, 0.0);
        assert!(y.get_index(0).is_nan());
        assert_eq!(
            eqn.take_callback_error(),
            Some(OdeCallbackError::Failed {
                callback: "rhs",
                code: 7
            })
        );
        assert!(eqn.take_callback_error().is_none());

        let y = eqn.rhs().jac_mul(&x, 0.0, &x);
        assert!(y.get_index(0).is_nan());
        assert_eq!(
            eqn.take_callback_error(),
            Some(OdeCallbackError::Missing {
                callback: "rhs_jac"
            })
        );
    }

    #[test]
    fn rejects_invalid_descriptors() {
        let mut d = desc();
        d.rhs = None;
        assert!(build_err(d).contains("'rhs' callback is required"));

        let mut d = desc();
        d.init = None;
        assert!(build_err(d).contains("'init' callback is required"));

        let mut d = desc();
        d.nstates = 0;
        assert!(build_err(d).contains("nstates"));

        let mut d = desc();
        d.nout = 2;
        assert!(build_err(d).contains("nout"));

        let mut d = desc();
        d.out = Some(rhs);
        assert!(build_err(d).contains("nout"));

        let mut d = desc();
        d.nroots = 1;
        assert!(build_err(d).contains("nroots"));

        let mut d = desc();
        d.out_jac = Some(rhs_jac);
        assert!(build_err(d).contains("'out_jac' callback requires the 'out' callback"));

        let mut d = desc();
        d.root = Some(rhs);
        d.nroots = 1;
        d.reset_jac = Some(rhs_jac);
        assert!(build_err(d).contains("'reset_jac' callback requires the 'reset' callback"));
    }

    #[test]
    fn frees_user_data_exactly_once() {
        let count = AtomicUsize::new(0);
        let mut d = desc();
        d.user_data = &count as *const AtomicUsize as *mut c_void;
        d.free_user_data = Some(count_free);

        drop(build(d).unwrap());
        assert_eq!(count.load(Ordering::SeqCst), 1);

        d.rhs = None;
        assert!(build(d).is_err());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    /// The mass matrix and its transpose are built correctly from their colorings for a
    /// sparse matrix type, also when the mass matrix is not symmetric.
    #[test]
    fn sparse_mass_and_transpose_matrices() {
        type S = diffsol::FaerSparseMat<f64>;
        let mut desc = OdeCallbacksDesc::new(2, 0, rhs2, init2);
        desc.mass = Some(mass2);
        desc.mass_adjoint = Some(mass2_adjoint);
        // SAFETY: the test callbacks are valid and use the documented lengths
        let eqn = unsafe { OdeCallbacks::<S>::new(desc, Default::default()) }.unwrap();
        let mass = eqn.mass().unwrap();
        let dense = |m: &S| {
            let mut d = [[0.0; 2]; 2];
            let (indices, values) = m.triplet_iter();
            for ((i, j), v) in indices.zip(values) {
                d[i][j] += v;
            }
            d
        };
        let m = mass.matrix(0.0);
        assert_eq!(dense(&m), [[1.0, 2.0], [0.0, 3.0]]);
        let mut mt = S::new_from_sparsity(2, 2, mass.transpose_sparsity(), Default::default());
        mass.transpose_inplace(0.0, &mut mt);
        assert_eq!(dense(&mt), [[1.0, 0.0], [2.0, 3.0]]);
    }
}
