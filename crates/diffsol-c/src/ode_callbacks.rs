//! ODE equations defined by C callbacks.
//!
//! [`OdeCallbacks`] implements [`diffsol::OdeEquations`]. It also implements each
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
//! The library calls a callback **one time for all the lanes** of the equations. Each vector
//! argument is a pointer to a [`TensorView`] (an input) or a [`TensorViewMut`] (the output `y`).
//! The views show the data of the solver with no copy. Each view has the C layout of
//! [`RawLayout`](crate::tensor::RawLayout): the fields `ptr`, `dtype`, `ndim`, `shape` and
//! `strides` (in bytes). `dtype` is the scalar type of the descriptor (`double` or `float`).
//!
//! Each view has 3 axes, `[n, P, B]`. The table gives `n`. The counts are fields of
//! [`OdeCallbacksDesc`].
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
//! # Lanes
//!
//! The output `y` has `P * B` lanes, as the grid `[P, B]`. Lane `(k, b)` of each argument has
//! the values that the callback uses to calculate lane `(k, b)` of `y`. Element `i` of lane
//! `(k, b)` is at the byte offset `i * strides[0] + k * strides[1] + b * strides[2]` from `ptr`.
//! The values of one lane are contiguous in memory: `strides[0]` is the size of one value.
//!
//! - `B` is `nbatch`, the number of lanes of the equations (independent copies of the system,
//!   each with its own parameters and state).
//! - `P` is 1 in a normal solve. In a sensitivity or an adjoint solve, the solver can calculate
//!   more than one product for each lane `b` in one call (for example one product for each
//!   parameter). Then `P > 1`.
//!
//! An argument that has the same values for each `k` (for example `x` and `p` in a forward
//! sensitivity solve) has `strides[1] == 0`. An argument that has the same values for all lanes
//! has `strides[1] == strides[2] == 0`. Thus the callback does not need to know which arguments
//! are broadcast: it uses the strides. The output `y` never has a stride of 0. Do not write to
//! an input.
//!
//! A simple callback loops over `b`, `k` and `i`. A callback for a GPU can calculate all the
//! lanes in one kernel.
//!
//! The `root` and `reset` callbacks are not available with more than one lane (`nbatch > 1`).
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
//! failure. When a callback fails, all the lanes of its output are filled with NaN.
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
    ConstantOp, ConstantOpSens, ConstantOpSensAdjoint, Context, LinearOp, LinearOpTranspose,
    Matrix, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
    NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef, Op, Scalar, Vector,
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
    tensor::{TensorLayout, TensorLayoutMut, TensorView, TensorViewMut},
};

// Each callback type is a nullable function pointer, the same as a function pointer type in C:
// `None` is `NULL`, and means that the callback is not given. Each vector argument is a view of
// all the lanes; the module documentation gives the layout.

/// `y = F(x, p, t)`: used for `rhs`, `root`, `out` and `reset`.
pub type OdeCallbackFn<T> = Option<
    unsafe extern "C" fn(
        x: *const TensorView<'_>,
        p: *const TensorView<'_>,
        t: T,
        y: *mut TensorViewMut<'_>,
        user_data: *mut c_void,
    ) -> i32,
>;

/// `y = A(x, p, t) v`: used for every `*_jac`, `*_sens`, `*_adjoint` and `*_sens_adjoint`
/// callback of `rhs`, `root`, `out` and `reset`.
pub type OdeCallbackMulFn<T> = Option<
    unsafe extern "C" fn(
        x: *const TensorView<'_>,
        p: *const TensorView<'_>,
        t: T,
        v: *const TensorView<'_>,
        y: *mut TensorViewMut<'_>,
        user_data: *mut c_void,
    ) -> i32,
>;

/// `y = y0(p, t)`: the initial state.
pub type OdeCallbackInitFn<T> = Option<
    unsafe extern "C" fn(
        p: *const TensorView<'_>,
        t: T,
        y: *mut TensorViewMut<'_>,
        user_data: *mut c_void,
    ) -> i32,
>;

/// `y = A(p, t) v`: used for `init_sens` and `init_sens_adjoint`.
pub type OdeCallbackInitMulFn<T> = Option<
    unsafe extern "C" fn(
        p: *const TensorView<'_>,
        t: T,
        v: *const TensorView<'_>,
        y: *mut TensorViewMut<'_>,
        user_data: *mut c_void,
    ) -> i32,
>;

/// `y = M(p, t) v + beta y`: used for `mass` (and `mass_adjoint`, with `M^T`). `y` is an input
/// and the output.
pub type OdeCallbackMassFn<T> = Option<
    unsafe extern "C" fn(
        v: *const TensorView<'_>,
        p: *const TensorView<'_>,
        t: T,
        beta: T,
        y: *mut TensorViewMut<'_>,
        user_data: *mut c_void,
    ) -> i32,
>;

/// Frees `user_data`, see the [module documentation](self).
pub type OdeCallbackFreeFn = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

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
    pub free_user_data: OdeCallbackFreeFn,

    pub rhs: OdeCallbackFn<T>,
    pub rhs_jac: OdeCallbackMulFn<T>,
    pub rhs_sens: OdeCallbackMulFn<T>,
    pub rhs_adjoint: OdeCallbackMulFn<T>,
    pub rhs_sens_adjoint: OdeCallbackMulFn<T>,

    pub init: OdeCallbackInitFn<T>,
    pub init_sens: OdeCallbackInitMulFn<T>,
    pub init_sens_adjoint: OdeCallbackInitMulFn<T>,

    pub mass: OdeCallbackMassFn<T>,
    pub mass_adjoint: OdeCallbackMassFn<T>,

    pub root: OdeCallbackFn<T>,
    pub root_jac: OdeCallbackMulFn<T>,
    pub root_sens: OdeCallbackMulFn<T>,
    pub root_adjoint: OdeCallbackMulFn<T>,
    pub root_sens_adjoint: OdeCallbackMulFn<T>,

    pub out: OdeCallbackFn<T>,
    pub out_jac: OdeCallbackMulFn<T>,
    pub out_sens: OdeCallbackMulFn<T>,
    pub out_adjoint: OdeCallbackMulFn<T>,
    pub out_sens_adjoint: OdeCallbackMulFn<T>,

    pub reset: OdeCallbackFn<T>,
    pub reset_jac: OdeCallbackMulFn<T>,
    pub reset_sens: OdeCallbackMulFn<T>,
    pub reset_adjoint: OdeCallbackMulFn<T>,
    pub reset_sens_adjoint: OdeCallbackMulFn<T>,
}

// SAFETY: `user_data` is the only field that is not `Send`. The caller of the C API agrees to
// the thread contract in the module documentation: the callbacks and `free_user_data` can run
// on any thread, but never at the same time. Thus it is safe to move the descriptor to a
// different thread. The descriptor is not `Sync`, because two threads must not call the
// callbacks at the same time.
unsafe impl<T: Send> Send for OdeCallbacksDesc<T> {}

/// [`OdeCallbacksDesc`] for `double` precision.
pub type DiffsolOdeCallbacksF64 = OdeCallbacksDesc<f64>;
/// [`OdeCallbacksDesc`] for `float` precision.
pub type DiffsolOdeCallbacksF32 = OdeCallbacksDesc<f32>;

impl<T> OdeCallbacksDesc<T> {
    /// Check that `nbatch` is at least 1, and that the largest buffer of the solver (one
    /// vector for each lane and each parameter) has a size that does not overflow.
    pub(crate) fn check_nbatch(&self, nbatch: usize) -> Result<(), String> {
        if nbatch == 0 {
            return Err("nbatch must be greater than 0".to_string());
        }
        let len = self.nstates.max(self.nout).max(self.nparams).max(1);
        let bytes = nbatch
            .checked_mul(self.nparams.max(1))
            .and_then(|n| n.checked_mul(len))
            .and_then(|n| n.checked_mul(std::mem::size_of::<T>()));
        if bytes.is_none_or(|bytes| bytes > isize::MAX as usize) {
            return Err(format!("nbatch {nbatch} is too large for this ODE"));
        }
        Ok(())
    }

    /// A descriptor with only the required callbacks set, and no `user_data`.
    pub fn new(
        nstates: usize,
        nparams: usize,
        rhs: unsafe extern "C" fn(
            x: *const TensorView<'_>,
            p: *const TensorView<'_>,
            t: T,
            y: *mut TensorViewMut<'_>,
            user_data: *mut c_void,
        ) -> i32,
        init: unsafe extern "C" fn(
            p: *const TensorView<'_>,
            t: T,
            y: *mut TensorViewMut<'_>,
            user_data: *mut c_void,
        ) -> i32,
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
            nbatch: usize,
        ) -> Result<OdeWrapper, DiffsolRtError>;
    }
}

/// Return the default context of `C` with `nbatch` lanes.
fn batch_context<C: Context>(nbatch: usize) -> Result<C, DiffsolError> {
    Ok(C::default().clone_with_nbatch(nbatch)?)
}

/// Build a [Solve] for the matrix type `matrix_type` and the scalar type `$t` from `callbacks`
/// with `nbatch` lanes.
///
/// The caller must satisfy the safety conditions of [OdeCallbacks::new].
macro_rules! solve_for_matrix_type {
    ($t:ty, $callbacks:expr, $matrix_type:expr, $nbatch:expr) => {{
        let callbacks = $callbacks;
        let nbatch = $nbatch;
        // SAFETY: the caller guarantees the conditions of `OdeCallbacks::new`
        match $matrix_type {
            MatrixType::NalgebraDense => {
                let eqn = unsafe {
                    OdeCallbacks::<diffsol::NalgebraMat<$t>>::new(callbacks, batch_context(nbatch)?)
                };
                Box::new(GenericSolve::from_eqn(eqn?)?) as Box<dyn Solve>
            }
            MatrixType::FaerDense => {
                let eqn = unsafe {
                    OdeCallbacks::<diffsol::FaerMat<$t>>::new(callbacks, batch_context(nbatch)?)
                };
                Box::new(GenericSolve::from_eqn(eqn?)?)
            }
            MatrixType::FaerSparse => {
                let eqn = unsafe {
                    OdeCallbacks::<diffsol::FaerSparseMat<$t>>::new(
                        callbacks,
                        batch_context(nbatch)?,
                    )
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
                nbatch: usize,
            ) -> Result<OdeWrapper, DiffsolRtError> {
                let solve = solve_for_matrix_type!($t, callbacks, matrix_type, nbatch);
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
    /// The solver integrates `nbatch` lanes (independent copies of the system, each with its
    /// own parameters and state) together. `nbatch` must be at least 1.
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
        nbatch: usize,
    ) -> Result<Self, DiffsolRtError> {
        if let Err(message) = callbacks.check_nbatch(nbatch) {
            // ownership of `user_data` has passed to us, so free it on every failure path
            // SAFETY: nothing else has taken ownership of `user_data` yet
            unsafe { callbacks.free_user_data() };
            return Err(DiffsolError::Other(message).into());
        }
        // SAFETY: the caller guarantees the conditions of `OdeCallbacks::new`
        unsafe { T::new_ode(callbacks, matrix_type, linear_solver, ode_solver, nbatch) }
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

/// A matrix type that [`OdeCallbacks`] can use: its vectors give views of their data
/// ([`TensorLayoutMut`]), so the callbacks can read and write all the lanes with no copy.
pub trait CallbackMatrix: Matrix<V: TensorLayoutMut> {}

impl<M: Matrix<V: TensorLayoutMut>> CallbackMatrix for M {}

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

impl<M: CallbackMatrix> OdeCallbacks<M> {
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
    /// - Each function must read and write only the values that the views show, and must not
    ///   write to an input view.
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
        // the solver needs all lanes to find the same root at the same time, and it cannot
        // report an error if they do not
        if eqn.ctx.nbatch() > 1 && (eqn.callbacks.root.is_some() || eqn.callbacks.reset.is_some()) {
            return Err(DiffsolError::Other(
                "Invalid ODE callbacks: the 'root' and 'reset' callbacks are not available with \
                 more than one lane (nbatch > 1)"
                    .to_string(),
            ));
        }
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

    /// Record `err` (if it is the first error) and fill all the lanes of `y` with NaN.
    fn record(&self, err: OdeCallbackError, y: &mut M::V) {
        if self.error.get().is_none() {
            self.error.set(Some(err));
        }
        y.fill(M::T::NAN);
    }

    /// Call the callback `f` one time for all the lanes of `y`, with `call`. `call` gets a view
    /// of each vector of `inputs` and a view of `y`, all with the `[n, P, B]` layout of the module
    /// documentation. If `f` is missing or fails, record the error and fill `y` with NaN.
    fn eval_lanes<F, const N: usize>(
        &self,
        name: &'static str,
        f: Option<F>,
        inputs: [&M::V; N],
        y: &mut M::V,
        call: impl FnOnce(F, [*const TensorView<'_>; N], *mut TensorViewMut<'_>) -> i32,
    ) {
        let Some(f) = f else {
            return self.record(OdeCallbackError::Missing { callback: name }, y);
        };
        // `y` has `P * B` lanes. The inputs have `P * B`, `B` or 1 lanes, and diffsol repeats
        // each lane of an input with `B` lanes over `P` adjacent lanes of `y`
        let nlanes = y.context().nbatch();
        let nbatch = self.ctx.nbatch();
        let (ninner, nouter) = if nlanes.is_multiple_of(nbatch) {
            (nlanes / nbatch, nbatch)
        } else {
            (nlanes, 1)
        };
        // SAFETY: each layout shows the data of a vector that stays borrowed until the callback
        // returns. The inputs are only read, and `y` is mutably borrowed, so nothing else reads
        // or writes its data during the call
        let views = inputs.map(|v| unsafe {
            TensorView::from_layout(v.tensor_layout().batch_lanes(ninner, nouter))
        });
        let code = {
            let y_layout = y.tensor_layout_mut().batch_lanes(ninner, nouter);
            let mut y_view = unsafe { TensorViewMut::from_layout(y_layout) };
            // the view goes out of scope before `y` is used again
            call(f, views.each_ref().map(|v| v as *const _), &mut y_view)
        };
        if code != 0 {
            self.record(
                OdeCallbackError::Failed {
                    callback: name,
                    code,
                },
                y,
            );
        }
    }

    // SAFETY (for each `unsafe` call below): the views have the layouts in the module
    // documentation, and `user_data` is the pointer the caller gave with the callbacks.

    /// Evaluate `y = f(x, p, t)` for all the lanes.
    fn eval(&self, name: &'static str, f: OdeCallbackFn<M::T>, x: &M::V, t: M::T, y: &mut M::V) {
        let ud = self.callbacks.user_data;
        self.eval_lanes(name, f, [x, &self.p], y, |f, [x, p], y| unsafe {
            f(x, p, t, y, ud)
        });
    }

    /// Evaluate `y = f(x, p, t, v)` for all the lanes.
    fn eval_mul(
        &self,
        name: &'static str,
        f: OdeCallbackMulFn<M::T>,
        x: &M::V,
        t: M::T,
        v: &M::V,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        self.eval_lanes(name, f, [x, &self.p, v], y, |f, [x, p, v], y| unsafe {
            f(x, p, t, v, y, ud)
        });
    }

    /// Evaluate `y = f(p, t)` for all the lanes.
    fn eval_init(&self, name: &'static str, f: OdeCallbackInitFn<M::T>, t: M::T, y: &mut M::V) {
        let ud = self.callbacks.user_data;
        self.eval_lanes(name, f, [&self.p], y, |f, [p], y| unsafe { f(p, t, y, ud) });
    }

    /// Evaluate `y = f(p, t, v)` for all the lanes.
    fn eval_init_mul(
        &self,
        name: &'static str,
        f: OdeCallbackInitMulFn<M::T>,
        t: M::T,
        v: &M::V,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        self.eval_lanes(name, f, [&self.p, v], y, |f, [p, v], y| unsafe {
            f(p, t, v, y, ud)
        });
    }

    /// Evaluate `y = f(v, p, t, beta, y)` for all the lanes.
    fn eval_mass(
        &self,
        name: &'static str,
        f: OdeCallbackMassFn<M::T>,
        v: &M::V,
        t: M::T,
        beta: M::T,
        y: &mut M::V,
    ) {
        let ud = self.callbacks.user_data;
        self.eval_lanes(name, f, [v, &self.p], y, |f, [v, p], y| unsafe {
            f(v, p, t, beta, y, ud)
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
    callbacks: [OdeCallbackMulFn<M::T>; 4],
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
        impl<M: CallbackMatrix> NonLinearOp for $view<'_, M> {
            fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
                if $stats {
                    self.0.statistics.borrow_mut().increment_call();
                }
                self.0.eval(stringify!($f), self.0.callbacks.$f, x, t, y);
            }
        }

        impl<M: CallbackMatrix> NonLinearOpJacobian for $view<'_, M> {
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

        impl<M: CallbackMatrix> NonLinearOpSens for $view<'_, M> {
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

        impl<M: CallbackMatrix> NonLinearOpAdjoint for $view<'_, M> {
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

        impl<M: CallbackMatrix> NonLinearOpSensAdjoint for $view<'_, M> {
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

impl<M: CallbackMatrix> ConstantOp for OdeCallbacksInit<'_, M> {
    fn call_inplace(&self, t: M::T, y: &mut M::V) {
        self.0.eval_init("init", self.0.callbacks.init, t, y);
    }
}

impl<M: CallbackMatrix> ConstantOpSens for OdeCallbacksInit<'_, M> {
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

impl<M: CallbackMatrix> ConstantOpSensAdjoint for OdeCallbacksInit<'_, M> {
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

impl<M: CallbackMatrix> LinearOp for OdeCallbacksMass<'_, M> {
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

impl<M: CallbackMatrix> LinearOpTranspose for OdeCallbacksMass<'_, M> {
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

impl<'a, M: CallbackMatrix> OdeEquationsRef<'a> for OdeCallbacks<M> {
    type Mass = OdeCallbacksMass<'a, M>;
    type Rhs = OdeCallbacksRhs<'a, M>;
    type Root = OdeCallbacksRoot<'a, M>;
    type Init = OdeCallbacksInit<'a, M>;
    type Out = OdeCallbacksOut<'a, M>;
    type Reset = OdeCallbacksReset<'a, M>;
}

impl<M: CallbackMatrix> OdeEquations for OdeCallbacks<M> {
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

impl<M: CallbackMatrix> SolveEqn for OdeCallbacks<M> {
    fn check_requirement(&self, requirement: EqnRequirement) -> Result<(), DiffsolRtError> {
        self.check(requirement)
    }

    fn take_error(&self) -> Option<String> {
        self.take_callback_error().map(|err| err.to_string())
    }
}

/// Call `f` one time for each lane of the output `y` of a callback, with the same lane of each
/// input. This is a simple way to write a callback in Rust: `f` gets one lane of each view as a
/// slice, so it does not use the strides. It is the equivalent of
/// [`diffsol::Vector::for_each_batch`] for the views of a callback.
///
/// The last argument of `f` is the index of the lane of `y`, `b * P + k` for the lane `(k, b)`
/// (the lane index of diffsol). The batch lane is `lane / P`, where `P` is `y.shape()[1]`.
///
/// ```ignore
/// unsafe extern "C" fn rhs(
///     x: *const TensorView<'_>,
///     p: *const TensorView<'_>,
///     _t: f64,
///     y: *mut TensorViewMut<'_>,
///     _user_data: *mut c_void,
/// ) -> i32 {
///     unsafe { for_each_batch([x, p], y, |[x, p], y: &mut [f64], _lane| y[0] = -p[0] * x[0]) };
///     0
/// }
/// ```
///
/// # Panics
/// If a view does not have 3 axes, the scalar type of a view is not `T`, or an input does not
/// have the lanes `[P, B]` of `y`.
///
/// # Safety
/// The pointers must be the arguments that the library gave to the callback, and the call must
/// be during the callback.
pub unsafe fn for_each_batch<T: crate::scalar_type::Scalar, const N: usize>(
    inputs: [*const TensorView<'_>; N],
    y: *mut TensorViewMut<'_>,
    mut f: impl FnMut([&[T]; N], &mut [T], usize),
) {
    // SAFETY: the caller gives the views of the callback, which are valid during the call
    let (inputs, y) = unsafe { (inputs.map(|v| &*v), &mut *y) };
    let dtype = T::scalar_type();
    assert!(
        y.shape().len() == 3 && y.dtype() == dtype,
        "y must have 3 axes and the scalar type {dtype:?}"
    );
    let lanes = [y.shape()[1], y.shape()[2]];
    for (i, v) in inputs.iter().enumerate() {
        assert!(
            v.shape().len() == 3 && v.dtype() == dtype && v.shape()[1..] == lanes,
            "input {i} must have 3 axes, the scalar type {dtype:?} and the lanes {lanes:?} of y"
        );
    }
    for b in 0..lanes[1] {
        for k in 0..lanes[0] {
            // the checks above make the scalar types and the indices correct
            let lane_inputs = inputs.map(|v| v.column::<T>(&[k, b]).unwrap());
            f(
                lane_inputs,
                y.column_mut::<T>(&[k, b]).unwrap(),
                b * lanes[0] + k,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use diffsol::{
        ConstantOp, Context, LinearOp, LinearOpTranspose, Matrix, NalgebraContext, NalgebraMat,
        NonLinearOp, NonLinearOpJacobian, OdeEquations, Vector,
    };

    use super::{for_each_batch, OdeCallbackError, OdeCallbacks, OdeCallbacksDesc};
    use crate::tensor::{TensorView, TensorViewMut};

    type M = NalgebraMat<f64>;
    type View<'a> = *const TensorView<'a>;
    type ViewMut<'a> = *mut TensorViewMut<'a>;

    unsafe extern "C" fn rhs(
        x: View<'_>,
        p: View<'_>,
        _t: f64,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe { for_each_batch([x, p], y, |[x, p], y: &mut [f64], _| y[0] = -p[0] * x[0]) };
        0
    }

    unsafe extern "C" fn rhs_jac(
        _x: View<'_>,
        p: View<'_>,
        _t: f64,
        v: View<'_>,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe { for_each_batch([p, v], y, |[p, v], y: &mut [f64], _| y[0] = -p[0] * v[0]) };
        0
    }

    unsafe extern "C" fn failing_rhs(
        _x: View<'_>,
        _p: View<'_>,
        _t: f64,
        _y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        7
    }

    unsafe extern "C" fn init(p: View<'_>, _t: f64, y: ViewMut<'_>, _ud: *mut c_void) -> i32 {
        unsafe { for_each_batch([p], y, |_, y: &mut [f64], _| y.fill(1.0)) };
        0
    }

    unsafe extern "C" fn count_free(ud: *mut c_void) {
        unsafe { &*(ud as *const AtomicUsize) }.fetch_add(1, Ordering::SeqCst);
    }

    // a 2 state system with the non-symmetric mass matrix [[1, 2], [0, 3]]
    unsafe extern "C" fn rhs2(
        x: View<'_>,
        _p: View<'_>,
        _t: f64,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([x], y, |[x], y: &mut [f64], _| {
                y.copy_from_slice(&[-x[0], -x[1]])
            })
        };
        0
    }

    unsafe extern "C" fn init2(p: View<'_>, _t: f64, y: ViewMut<'_>, _ud: *mut c_void) -> i32 {
        unsafe { for_each_batch([p], y, |_, y: &mut [f64], _| y.fill(1.0)) };
        0
    }

    unsafe extern "C" fn mass2(
        v: View<'_>,
        _p: View<'_>,
        _t: f64,
        beta: f64,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([v], y, |[v], y: &mut [f64], _| {
                y[0] = v[0] + 2.0 * v[1] + beta * y[0];
                y[1] = 3.0 * v[1] + beta * y[1];
            })
        };
        0
    }

    unsafe extern "C" fn mass2_adjoint(
        v: View<'_>,
        _p: View<'_>,
        _t: f64,
        beta: f64,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        unsafe {
            for_each_batch([v], y, |[v], y: &mut [f64], _| {
                y[0] = v[0] + beta * y[0];
                y[1] = 2.0 * v[0] + 3.0 * v[1] + beta * y[1];
            })
        };
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

    /// A failure fills all the lanes of the output with NaN.
    #[test]
    fn failures_fill_all_lanes_with_nan() {
        let mut d = desc();
        d.rhs = Some(failing_rhs);
        let ctx = NalgebraContext::default().clone_with_nbatch(3).unwrap();
        // SAFETY: the test callbacks are valid and use the documented lengths
        let eqn = unsafe { OdeCallbacks::<M>::new(d, ctx) }.unwrap();
        let x = ctx.vector_from_vec(vec![1.0, 2.0, 3.0]);
        let y = eqn.rhs().call(&x, 0.0);
        assert!(y.clone_as_vec().iter().all(|y| y.is_nan()));
    }

    /// The layouts that a callback saw: `(shape, strides)` of each argument, `y` last.
    type Seen = Vec<([usize; 3], [usize; 3])>;

    thread_local! {
        static SEEN: RefCell<Seen> = const { RefCell::new(Vec::new()) };
    }

    fn layout(shape: &[usize], strides: &[usize]) -> ([usize; 3], [usize; 3]) {
        (shape.try_into().unwrap(), strides.try_into().unwrap())
    }

    /// `y = diag(p) v` with 2 states and 2 parameters, and record the layouts.
    unsafe extern "C" fn record_jac(
        x: View<'_>,
        p: View<'_>,
        _t: f64,
        v: View<'_>,
        y: ViewMut<'_>,
        _ud: *mut c_void,
    ) -> i32 {
        let seen = [x, p, v].map(|v| unsafe { layout((*v).shape(), (*v).strides()) });
        let y_seen = unsafe { layout((*y).shape(), (*y).strides()) };
        SEEN.with(|s| *s.borrow_mut() = [&seen[..], &[y_seen]].concat());
        unsafe {
            for_each_batch([p, v], y, |[p, v], y: &mut [f64], _| {
                y[0] = p[0] * v[0];
                y[1] = p[1] * v[1];
            })
        };
        0
    }

    /// In a forward sensitivity solve, the state and the parameters have `nbatch` lanes, and
    /// `v` and `y` have one lane for each batch lane and each parameter. The callback sees the
    /// state and the parameters with a stride of 0 on the parameter axis.
    #[test]
    fn grouped_broadcast_gives_strides_of_zero() {
        let (nbatch, nparams) = (2, 2);
        let mut d = OdeCallbacksDesc::new(2, nparams, rhs2, init2);
        d.rhs_jac = Some(record_jac);
        let ctx = NalgebraContext::default()
            .clone_with_nbatch(nbatch)
            .unwrap();
        let aug_ctx = ctx.clone_with_nbatch(nbatch * nparams).unwrap();
        // SAFETY: the test callbacks are valid and use the documented lengths
        let mut eqn = unsafe { OdeCallbacks::<M>::new(d, ctx) }.unwrap();
        // lane `b` has the parameters `[10 b + 1, 10 b + 2]`
        eqn.set_params(&ctx.vector_from_vec(vec![1.0, 2.0, 11.0, 12.0]));
        let x = ctx.vector_from_vec(vec![0.0; 4]);
        // lane `b * nparams + k` of `v` is `[b, k]`
        let v_values: Vec<f64> = (0..nbatch)
            .flat_map(|b| (0..nparams).flat_map(move |k| [b as f64, k as f64]))
            .collect();
        let v = aug_ctx.vector_from_vec(v_values);
        let mut y = aug_ctx.vector_from_vec(vec![0.0; 8]);
        eqn.rhs().jac_mul_inplace(&x, 0.0, &v, &mut y);
        assert!(eqn.take_callback_error().is_none());

        let s = std::mem::size_of::<f64>();
        let c = 2 * s;
        let seen = SEEN.with(|s| s.borrow().clone());
        let shape = [2, nparams, nbatch];
        assert_eq!(
            seen,
            vec![
                (shape, [s, 0, c]),
                (shape, [s, 0, c]),
                (shape, [s, c, nparams * c]),
                (shape, [s, c, nparams * c]),
            ]
        );
        let expected: Vec<f64> = (0..nbatch)
            .flat_map(|b| {
                let p = [10.0 * b as f64 + 1.0, 10.0 * b as f64 + 2.0];
                (0..nparams).flat_map(move |k| [p[0] * b as f64, p[1] * k as f64])
            })
            .collect();
        assert_eq!(y.clone_as_vec(), expected);
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
