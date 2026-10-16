//! Preconditioned Richardson iteration for `A x = b`.
//!
//! This is not a [LinearSolver](super::LinearSolver): that trait only sees its operator through
//! [LinearOp::matrix_inplace](crate::LinearOp::matrix_inplace) and solves with a factorisation
//! of exactly that matrix. Richardson instead applies the exact operator matrix-free and
//! preconditions with a different, approximate operator `P`, typically an older factorisation
//! held by a [LinearSolver](super::LinearSolver). Failing to converge is an expected outcome,
//! telling the caller that `P` is too far from `A` and should be refreshed.

use num_traits::{One, Zero};

use crate::{
    error::{LaError, LinearSolverError},
    Scalar, Vector,
};

/// Options for [richardson].
#[derive(Clone, Debug)]
pub struct RichardsonOptions<T: Scalar> {
    /// Converged once the estimated error of `x` is at most `rtol * ‖x‖`.
    pub rtol: T,
    /// Maximum number of iterations, each one application of `A` and one of `P⁻¹`.
    pub max_iter: usize,
    /// Diverged once an update shrinks by less than this factor, `‖δ_{k+1}‖ > max_rate ‖δ_k‖`.
    pub max_rate: T,
}

impl<T: Scalar> Default for RichardsonOptions<T> {
    fn default() -> Self {
        Self {
            rtol: T::from_f64(1e-10).unwrap(),
            max_iter: 10,
            max_rate: T::from_f64(0.9).unwrap(),
        }
    }
}

/// Solve `A x = b` by the preconditioned Richardson iteration `x ← x + P⁻¹(b − A x)`, starting
/// from the given `x`.
///
/// - `apply(x, y)` sets `y = A x`. Pass `Aᵀ` here (and a transposed `precond`) to solve the
///   transposed problem.
/// - `precond(r)` overwrites `r` with `P⁻¹ r`.
/// - `tmp` is a work vector shaped like `x`.
///
/// The iteration converges when `P⁻¹ A` is close to the identity. It stops once the update
/// `δ_k` satisfies `‖δ_k‖ ≤ rtol ‖x‖`, or, once a contraction rate `ρ = ‖δ_k‖ / ‖δ_{k-1}‖` is
/// known, when the error estimate `ρ / (1 − ρ) ‖δ_k‖` does. With `P = A` it stops after two
/// iterations. Norms are 2-norms over the whole vector, including every batch.
///
/// Returns the number of iterations taken, or
/// [LinearSolverError::RichardsonNotConverged] if the updates stop contracting by at least
/// `max_rate` or `max_iter` is reached. Errors from `precond` are passed through.
pub fn richardson<V: Vector>(
    apply: impl Fn(&V, &mut V),
    precond: impl Fn(&mut V) -> Result<(), LaError>,
    b: &V,
    x: &mut V,
    tmp: &mut V,
    opts: &RichardsonOptions<V::T>,
) -> Result<usize, LaError> {
    let mut prev_norm: Option<V::T> = None;
    for iter in 0..opts.max_iter {
        // tmp = P⁻¹ (b − A x)
        apply(x, tmp);
        tmp.axpy(V::T::one(), b, -V::T::one());
        precond(tmp)?;
        x.axpy(V::T::one(), tmp, V::T::one());

        let delta_norm = tmp.norm(2);
        let tol = opts.rtol * x.norm(2);
        if delta_norm <= tol || delta_norm == V::T::zero() {
            return Ok(iter + 1);
        }
        if let Some(prev) = prev_norm {
            let rate = delta_norm / prev;
            if rate > opts.max_rate {
                break;
            }
            if rate / (V::T::one() - rate) * delta_norm <= tol {
                return Ok(iter + 1);
            }
        }
        prev_norm = Some(delta_norm);
    }
    Err(LaError::from(LinearSolverError::RichardsonNotConverged))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{richardson, RichardsonOptions};
    use crate::error::{LaError, LinearSolverError};
    use crate::linear_solver::tests::DiagonalOp;
    use crate::{LinearSolver, Matrix, Vector};
    use num_traits::{FromPrimitive, One, Zero};

    /// `A_0 = [[4, 1, 0], [1, 3, 1], [0, 2, 5]]`, not symmetric so a transposed solve differs.
    fn a_entries(eps: f64) -> (Vec<(usize, usize)>, Vec<f64>) {
        let indices = vec![(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)];
        let values = [4.0, 1.0, 1.0, 3.0, 2.0, 1.0, 5.0];
        // perturb every entry by a different multiple of eps
        let values = values
            .iter()
            .enumerate()
            .map(|(i, &v)| v + eps * (1.0 + i as f64) * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        (indices, values)
    }

    fn matrix<M: Matrix>(eps: f64, transpose: bool) -> M {
        let (indices, values) = a_entries(eps);
        let indices = indices
            .into_iter()
            .map(|(i, j)| if transpose { (j, i) } else { (i, j) })
            .collect();
        let values = values.iter().map(|&v| M::T::from_f64(v).unwrap()).collect();
        M::try_from_triplets(3, 3, indices, values, Default::default()).unwrap()
    }

    fn factorised<M: Matrix, LS: LinearSolver<M>>(eps: f64) -> LS {
        let op = DiagonalOp {
            matrix: matrix::<M>(eps, false),
        };
        let mut ls = LS::default();
        ls.set_sparsity(&op);
        ls.set_linearisation(&op);
        ls
    }

    /// Solve `A x = b` and `Aᵀ x = b` with Richardson, preconditioned by a factorisation of
    /// `A + eps E`, and check against the exact solution `x = (1, 2, 3)`. Returns the iteration
    /// counts, or the error.
    fn solve<M: Matrix, LS: LinearSolver<M>>(
        eps: f64,
        opts: &RichardsonOptions<M::T>,
    ) -> Result<(usize, usize), LaError> {
        let f = |x: f64| M::T::from_f64(x).unwrap();
        let p = factorised::<M, LS>(eps);
        let expected = M::V::from_vec(vec![f(1.0), f(2.0), f(3.0)], Default::default());
        let mut counts = [0; 2];
        for (count, transpose) in counts.iter_mut().zip([false, true]) {
            let a = matrix::<M>(0.0, transpose);
            let mut b = M::V::zeros(3, Default::default());
            a.gemv(M::T::one(), &expected, M::T::zero(), &mut b);
            let mut x = M::V::zeros(3, Default::default());
            let mut tmp = M::V::zeros(3, Default::default());
            *count = richardson(
                |x: &M::V, y: &mut M::V| a.gemv(M::T::one(), x, M::T::zero(), y),
                |r: &mut M::V| {
                    if transpose {
                        p.solve_transpose_in_place(r)
                    } else {
                        p.solve_in_place(r)
                    }
                },
                &b,
                &mut x,
                &mut tmp,
                opts,
            )?;
            x.assert_eq_st(&expected, opts.rtol * M::T::from_f64(1000.0).unwrap());
        }
        Ok((counts[0], counts[1]))
    }

    pub fn test_richardson<M: Matrix, LS: LinearSolver<M>>() {
        let mut opts = RichardsonOptions::<M::T>::default();
        // f32 can't reach the f64 default tolerance
        if std::mem::size_of::<M::T>() == 4 {
            opts.rtol = M::T::from_f64(1e-5).unwrap();
        }

        // an exact preconditioner converges in two iterations: one to solve, one to see that the
        // update has vanished
        let (n, nt) = solve::<M, LS>(0.0, &opts).unwrap();
        assert!(n <= 2 && nt <= 2, "exact P took {n} / {nt} iterations");

        // a nearby preconditioner still converges, just in more iterations
        let (n, nt) = solve::<M, LS>(0.05, &opts).unwrap();
        assert!(
            n > 2 && nt > 2,
            "perturbed P took only {n} / {nt} iterations"
        );

        // a preconditioner far from A makes the iteration diverge
        let err = solve::<M, LS>(1.5, &opts).unwrap_err();
        assert!(
            matches!(
                err,
                LaError::LinearSolverError(LinearSolverError::RichardsonNotConverged)
            ),
            "{err}"
        );

        // too few iterations to converge is an error too
        opts.max_iter = 1;
        assert!(solve::<M, LS>(0.05, &opts).is_err());
    }
}
