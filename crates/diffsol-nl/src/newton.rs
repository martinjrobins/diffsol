use diffsol_la::{
    Context, IndexType, LinearOp as LaLinearOp, LinearSolver, Matrix, Scalar, Vector,
};
use num_traits::{One, Zero};
use std::collections::HashMap;

use crate::{
    convergence::{Convergence, ConvergenceStatus},
    error::{NlError, NonLinearSolverError},
    line_search::LineSearch,
    non_linear_solver_error,
    nonlinear_op::{NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian},
    nonlinear_solver::NonLinearSolver,
};

#[allow(clippy::too_many_arguments)]
pub fn newton_iteration<V: Vector>(
    xn: &mut V,
    tmp: &mut V,
    error_y: &V,
    fun: impl Fn(&V, &mut V),
    linear_solver: impl Fn(&mut V) -> Result<(), NlError>,
    convergence: &mut Convergence<V>,
    line_search: &mut impl LineSearch<V>,
) -> Result<(), NlError> {
    convergence.reset();
    line_search.reset();
    for _ in 0..convergence.max_iter() {
        let res =
            line_search.take_optimal_step(xn, tmp, error_y, &fun, &linear_solver, convergence)?;
        // xn = xn + alpha * delta_n, where alpha is determined by line search, return status

        match res {
            ConvergenceStatus::Continue => continue,
            ConvergenceStatus::Converged => return Ok(()),
            ConvergenceStatus::Diverged => return Err(non_linear_solver_error!(NewtonDiverged)),
        }
    }
    Err(non_linear_solver_error!(NewtonMaxIterations))
}

/// Preconditioned Richardson iteration for the linear problem `A x = b`, from `x = 0`
///
/// The first update is `x_1 = P^-1 b`, the error of every later update is measured
/// relative to it
///
/// With `rhs_scale` (a batched scalar, one value per lane), each lane is solved for `b`
/// divided by its largest magnitude, and the solution multiplied back by it at the end. The
/// problem is linear, so this changes only the convergence test, which then no longer depends
/// on the scale of `b`. This is for right-hand sides, such as cotangents, whose units are not
/// those of the tolerances in `convergence`.
#[allow(clippy::too_many_arguments)]
pub fn richardson<V: Vector>(
    b: &V,
    x: &mut V,
    delta: &mut V,
    scale: &mut V,
    mut rhs_scale: Option<&mut V>,
    mul: impl Fn(&V, &mut V),
    mul_is_negative: bool,
    precondition: impl Fn(&mut V) -> Result<(), NlError>,
    convergence: &mut Convergence<V>,
) -> Result<(), NlError> {
    x.copy_from(b);
    // if rhs_scale is passed, normalise b by its largest magnitude
    if let Some(rhs_scale) = rhs_scale.as_deref_mut() {
        V::reduce_elem(
            rhs_scale,
            [b],
            V::T::zero(),
            |[b], _lane, i| num_traits::abs(b[i]),
            |a, c| a.max(c),
        );
        // hold -1/max|b|, the multiplier of b in the update below. A zero b needs no scaling,
        // and a NaN or infinite one must reach the solution unscaled, so the solve fails
        rhs_scale.for_each_elem([], |s, [], _lane, _i| {
            *s = if *s > V::T::zero() && *s < V::T::INFINITY {
                -V::T::one() / *s
            } else {
                -V::T::one()
            };
        });
        x.for_each_elem([rhs_scale], |x, [s], _lane, _i| *x = -*x * s[0]);
    }
    // x_1 = P^-1 b, the update from x_0 = 0
    precondition(x)?;
    scale.copy_from(x);
    convergence.reset();
    let sign = if mul_is_negative {
        -V::T::one()
    } else {
        V::T::one()
    };
    for _ in 0..convergence.max_iter() {
        // delta = P^-1 (A x - b), the update with its sign reversed
        mul(x, delta);
        match rhs_scale.as_deref() {
            Some(rhs_scale) => delta.batched_axpy(rhs_scale, b, sign),
            None => delta.axpy(-V::T::one(), b, sign),
        }
        precondition(delta)?;
        x.sub_assign(&*delta);
        let norm = convergence.norm(delta, scale);
        match convergence.check_new_iteration(norm) {
            ConvergenceStatus::Continue => continue,
            ConvergenceStatus::Converged => {
                if let Some(rhs_scale) = rhs_scale {
                    x.for_each_elem([rhs_scale], |x, [s], _lane, _i| *x = -*x / s[0]);
                }
                return Ok(());
            }
            ConvergenceStatus::Diverged => return Err(non_linear_solver_error!(NewtonDiverged)),
        }
    }
    Err(non_linear_solver_error!(NewtonMaxIterations))
}

/// A borrowing adapter that presents a [NonLinearOpJacobian] (evaluated at a fixed
/// state `x`) as a [`diffsol_la::LinearOp`].
///
/// This is the bridge that allows the [NewtonNonlinearSolver] to be implemented on
/// top of the time-unaware [`diffsol_la::LinearSolver`] backends.
struct JacobianRef<'a, C: NonLinearOpJacobian> {
    op: &'a C,
    x: Option<&'a C::V>,
}

impl<'a, C: NonLinearOpJacobian> JacobianRef<'a, C> {
    /// Create an adapter used only to query the sparsity pattern (no state set).
    fn sparsity_only(op: &'a C) -> Self {
        Self { op, x: None }
    }

    /// Create an adapter that evaluates the Jacobian at `x`.
    fn at(op: &'a C, x: &'a C::V) -> Self {
        Self { op, x: Some(x) }
    }
}

impl<C: NonLinearOpJacobian> LaLinearOp for JacobianRef<'_, C> {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    type C = C::C;

    fn nrows(&self) -> IndexType {
        self.op.nout()
    }

    fn ncols(&self) -> IndexType {
        self.op.nstates()
    }

    fn context(&self) -> &Self::C {
        self.op.context()
    }

    fn matrix_inplace(&self, y: &mut Self::M) {
        let x = self.x.expect("JacobianRef: state x not set");
        self.op.jacobian_inplace(x, y);
    }

    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.jacobian_sparsity()
    }
}

/// The work vector in `tmps` for a solve of `x`, with its lane count.
fn work_vector<'a, V: Vector>(tmps: &'a mut HashMap<usize, V>, x: &V) -> &'a mut V {
    tmps.entry(x.context().nbatch())
        .or_insert_with(|| V::zeros(x.len(), x.context().clone()))
}

pub struct NewtonNonlinearSolver<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>> {
    linear_solver: Ls,
    line_search: Lsearch,
    is_jacobian_set: bool,
    /// Work vectors, one per lane count the solver has been asked for.
    tmps: HashMap<usize, M::V>,
    /// Second work vectors, for the error scale of [richardson].
    scales: HashMap<usize, M::V>,
    /// Per-lane scales of the right-hand side of [richardson], one batched scalar per lane count.
    rhs_scales: HashMap<usize, M::V>,
}

impl<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>>
    NewtonNonlinearSolver<M, Ls, Lsearch>
{
    pub fn new(linear_solver: Ls, line_search: Lsearch) -> Self {
        Self {
            linear_solver,
            line_search,
            is_jacobian_set: false,
            tmps: HashMap::new(),
            scales: HashMap::new(),
            rhs_scales: HashMap::new(),
        }
    }
    pub fn linear_solver(&self) -> &Ls {
        &self.linear_solver
    }
}

impl<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>> Default
    for NewtonNonlinearSolver<M, Ls, Lsearch>
{
    fn default() -> Self {
        Self::new(Ls::default(), Lsearch::default())
    }
}

impl<M: Matrix, Ls: LinearSolver<M>, Lsearch: LineSearch<M::V>> NonLinearSolver<M>
    for NewtonNonlinearSolver<M, Ls, Lsearch>
{
    fn clear_jacobian(&mut self) {
        self.is_jacobian_set = false;
    }

    fn is_jacobian_set(&self) -> bool {
        self.is_jacobian_set
    }

    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C) {
        self.linear_solver
            .set_sparsity(&JacobianRef::sparsity_only(op));
        self.is_jacobian_set = false;
        self.tmps.clear();
        self.scales.clear();
    }

    fn reset_jacobian<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &C::V,
    ) {
        self.linear_solver
            .set_linearisation(&JacobianRef::at(op, x));
        self.is_jacobian_set = true;
    }

    fn solve_linearised_in_place(&self, x: &mut M::V) -> Result<(), NlError> {
        self.linear_solver.solve_in_place(x).map_err(Into::into)
    }

    fn solve_jacobian_in_place<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x_lin: &M::V,
        b: &M::V,
        x: &mut M::V,
        convergence: &mut Convergence<M::V>,
    ) -> Result<(), NlError> {
        if !self.is_jacobian_set {
            return Err(non_linear_solver_error!(JacobianNotReset));
        }
        let delta = work_vector(&mut self.tmps, x);
        let scale = work_vector(&mut self.scales, x);
        let mul = |x: &M::V, y: &mut M::V| op.jac_mul_inplace(x_lin, x, y);
        let precondition = |x: &mut M::V| self.linear_solver.solve_in_place(x).map_err(Into::into);
        richardson(
            b,
            x,
            delta,
            scale,
            None,
            mul,
            false,
            precondition,
            convergence,
        )
    }

    fn solve_jacobian_transpose_in_place<
        C: NonLinearOpAdjoint<V = M::V, T = M::T, M = M, C = M::C>,
    >(
        &mut self,
        op: &C,
        x_lin: &M::V,
        b: &M::V,
        x: &mut M::V,
        convergence: &mut Convergence<M::V>,
    ) -> Result<(), NlError> {
        if !self.is_jacobian_set {
            return Err(non_linear_solver_error!(JacobianNotReset));
        }
        let delta = work_vector(&mut self.tmps, x);
        let scale = work_vector(&mut self.scales, x);
        let rhs_scale = self
            .rhs_scales
            .entry(x.context().nbatch())
            .or_insert_with(|| M::V::zeros(1, x.context().clone()));
        // the op gives -J^T x
        let mul = |x: &M::V, y: &mut M::V| op.jac_transpose_mul_inplace(x_lin, x, y);
        let precondition = |x: &mut M::V| {
            self.linear_solver
                .solve_transpose_in_place(x)
                .map_err(Into::into)
        };
        // the right-hand side is a cotangent, whose units are not those of the state the
        // tolerances are for, so the solve is normalised by its scale
        richardson(
            b,
            x,
            delta,
            scale,
            Some(rhs_scale),
            mul,
            true,
            precondition,
            convergence,
        )
    }

    fn solve_in_place<C: NonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        xn: &mut M::V,
        error_y: &M::V,
        convergence: &mut Convergence<M::V>,
    ) -> Result<(), NlError> {
        if !self.is_jacobian_set {
            return Err(non_linear_solver_error!(JacobianNotReset));
        }
        if xn.len() != op.nstates() {
            let error = NonLinearSolverError::WrongStateLength {
                expected: op.nstates(),
                found: xn.len(),
            };
            return Err(NlError::from(error));
        }
        let tmp = work_vector(&mut self.tmps, xn);
        let linear_solver = |x: &mut C::V| self.linear_solver.solve_in_place(x).map_err(Into::into);
        let fun = |x: &C::V, y: &mut C::V| op.call_inplace(x, y);
        newton_iteration(
            xn,
            tmp,
            error_y,
            fun,
            linear_solver,
            convergence,
            &mut self.line_search,
        )
    }
}
