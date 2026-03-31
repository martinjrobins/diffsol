// Solver method Python enum. This is used to select the overarching solver
// stragegy like bdf or esdirk34 in diffsol.

use diffsol::error::DiffsolError;
use diffsol::{
    AdjointOdeSolverMethod, Checkpointing, DefaultSolver, DenseMatrix, MatrixCommon,
    OdeSolverState, Op, SensitivitiesOdeSolverMethod, VectorViewMut,
};
use diffsol::{
    DefaultDenseMatrix, DiffSl, LinearSolver, Matrix, OdeSolverMethod, OdeSolverProblem, Vector,
    VectorHost, VectorRef, matrix::MatrixRef,
};
use nalgebra::ComplexField;
use ndarray::ArrayView2;
use num_traits::{FromPrimitive, Zero}; // for generic nums in _solve_sum_squares_adj

use crate::jit::JitModule;
use crate::scalar_type::Scalar;
use crate::solution::GenericState;
use crate::utils::is_sens_available;
use crate::{
    linear_solver_type::LinearSolverType,
    valid_linear_solver::{KluValidator, LuValidator},
};

/// Enumerates the possible ODE solver methods for diffsol. See the solver descriptions in the diffsol documentation (https://github.com/martinjrobins/diffsol) for more details.
///
/// :attr bdf: Backward Differentiation Formula (BDF) method for stiff ODEs and singular mass matrices
/// :attr esdirk34: Explicit Singly Diagonally Implicit Runge-Kutta (ESDIRK) method for moderately stiff ODEs and singular mass matrices.
/// :attr tr_bdf2: Trapezoidal Backward Differentiation Formula of order 2 (TR-BDF2) method for moderately stiff ODEs and singular mass matrices.
/// :attr tsit45: Tsitouras 4/5th order Explicit Runge-Kutta (TSIT45) method for non-stiff ODEs. This is an explicit method, it cannot handle singular mass matrices and does not require a linear solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OdeSolverType {
    Bdf,
    Esdirk34,
    TrBdf2,
    Tsit45,
}

impl OdeSolverType {
    pub(crate) fn solve<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule<M::T>>>,
        final_time: M::T,
        initial_state: Option<GenericState<M::V>>,
    ) -> Result<
        (
            <M::V as DefaultDenseMatrix>::M,
            Vec<M::T>,
            GenericState<M::V>,
        ),
        DiffsolError,
    >
    where
        M: Matrix<T: Scalar>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        use crate::solution::GenericState;
        match self {
            OdeSolverType::Bdf => {
                let mut solver = match initial_state {
                    Some(GenericState::Bdf(state)) => {
                        let mut solver = problem.bdf_solver::<LS>(state)?;
                        solver.state_mut();
                        solver
                    }
                    Some(GenericState::Rk(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected a BDF state for bdf method".to_string(),
                        ));
                    }
                    None => problem.bdf::<LS>()?,
                };
                let (ys, ts, _) = solver.solve(final_time)?;
                Ok((ys, ts, GenericState::Bdf(solver.into_state())))
            }
            OdeSolverType::Esdirk34 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.esdirk34_solver::<LS>(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for esdirk34 method".to_string(),
                        ));
                    }
                    None => problem.esdirk34::<LS>()?,
                };
                let (ys, ts, _) = solver.solve(final_time)?;
                Ok((ys, ts, GenericState::Rk(solver.into_state())))
            }
            OdeSolverType::TrBdf2 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.tr_bdf2_solver::<LS>(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tr_bdf2 method".to_string(),
                        ));
                    }
                    None => problem.tr_bdf2::<LS>()?,
                };
                let (ys, ts, _) = solver.solve(final_time)?;
                Ok((ys, ts, GenericState::Rk(solver.into_state())))
            }
            OdeSolverType::Tsit45 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.tsit45_solver(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tsit45 method".to_string(),
                        ));
                    }
                    None => problem.tsit45()?,
                };
                let (ys, ts, _) = solver.solve(final_time)?;
                Ok((ys, ts, GenericState::Rk(solver.into_state())))
            }
        }
    }

    pub(crate) fn solve_dense<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule<M::T>>>,
        t_eval: &[M::T],
        initial_state: Option<GenericState<M::V>>,
    ) -> Result<(<M::V as DefaultDenseMatrix>::M, GenericState<M::V>), DiffsolError>
    where
        M: Matrix<T: Scalar>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                let mut solver = match initial_state {
                    Some(GenericState::Bdf(state)) => problem.bdf_solver::<LS>(state)?,
                    Some(GenericState::Rk(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected a BDF state for bdf method".to_string(),
                        ));
                    }
                    None => problem.bdf::<LS>()?,
                };
                let (ys, _) = solver.solve_dense(t_eval)?;
                Ok((ys, GenericState::Bdf(solver.into_state())))
            }
            OdeSolverType::Esdirk34 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.esdirk34_solver::<LS>(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for esdirk34 method".to_string(),
                        ));
                    }
                    None => problem.esdirk34::<LS>()?,
                };
                let (ys, _) = solver.solve_dense(t_eval)?;
                Ok((ys, GenericState::Rk(solver.into_state())))
            }
            OdeSolverType::TrBdf2 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.tr_bdf2_solver::<LS>(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tr_bdf2 method".to_string(),
                        ));
                    }
                    None => problem.tr_bdf2::<LS>()?,
                };
                let (ys, _) = solver.solve_dense(t_eval)?;
                Ok((ys, GenericState::Rk(solver.into_state())))
            }
            OdeSolverType::Tsit45 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.tsit45_solver(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tsit45 method".to_string(),
                        ));
                    }
                    None => problem.tsit45()?,
                };
                let (ys, _) = solver.solve_dense(t_eval)?;
                Ok((ys, GenericState::Rk(solver.into_state())))
            }
        }
    }

    fn check_sens_available() -> Result<(), DiffsolError> {
        if !is_sens_available() {
            return Err(DiffsolError::Other(
                "Sensitivity analysis is not supported on Windows, please use a linux or macOS system.".to_string(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_fwd_sens<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule<M::T>>>,
        t_eval: &[M::T],
        initial_state: Option<GenericState<M::V>>,
    ) -> Result<
        (
            <M::V as DefaultDenseMatrix>::M,
            Vec<<M::V as DefaultDenseMatrix>::M>,
            GenericState<M::V>,
        ),
        DiffsolError,
    >
    where
        M: Matrix<T: Scalar> + DefaultSolver,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => {
                let mut solver = match initial_state {
                    Some(GenericState::Bdf(state)) => {
                        let mut solver = problem.bdf_solver_sens::<LS>(state)?;
                        solver.state_mut();
                        solver
                    }
                    Some(GenericState::Rk(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected a BDF state for bdf method".to_string(),
                        ));
                    }
                    None => problem.bdf_sens::<LS>()?,
                };
                let (ys, sens, _) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericState::Bdf(solver.into_state())))
            }
            OdeSolverType::Esdirk34 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.esdirk34_solver_sens::<LS>(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for esdirk34 method".to_string(),
                        ));
                    }
                    None => problem.esdirk34_sens::<LS>()?,
                };
                let (ys, sens, _) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericState::Rk(solver.into_state())))
            }
            OdeSolverType::TrBdf2 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.tr_bdf2_solver_sens::<LS>(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tr_bdf2 method".to_string(),
                        ));
                    }
                    None => problem.tr_bdf2_sens::<LS>()?,
                };
                let (ys, sens, _) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericState::Rk(solver.into_state())))
            }
            OdeSolverType::Tsit45 => {
                let mut solver = match initial_state {
                    Some(GenericState::Rk(state)) => problem.tsit45_solver_sens(state)?,
                    Some(GenericState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tsit45 method".to_string(),
                        ));
                    }
                    None => problem.tsit45_sens()?,
                };
                let (ys, sens, _) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericState::Rk(solver.into_state())))
            }
        }
    }

    pub(crate) fn solve_sum_squares_adj<'a, M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule<M::T>>>,
        data: ArrayView2<'a, M::T>,
        t_eval: &[M::T],
        backwards_method: OdeSolverType,
        backwards_linear_solver: LinearSolverType,
    ) -> Result<(M::T, M::V), DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + LuValidator<M> + KluValidator<M>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => self._solve_sum_squares_adj(
                problem.bdf::<LS>()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            OdeSolverType::Esdirk34 => self._solve_sum_squares_adj(
                problem.esdirk34::<LS>()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            OdeSolverType::TrBdf2 => self._solve_sum_squares_adj(
                problem.tr_bdf2::<LS>()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            OdeSolverType::Tsit45 => self._solve_sum_squares_adj(
                problem.tsit45()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
        }
    }

    pub(crate) fn _solve_sum_squares_adj<'data, 'solver, M, S>(
        &self,
        mut solver: S,
        data: ArrayView2<'data, M::T>,
        t_eval: &[M::T],
        backwards_method: OdeSolverType,
        backwards_linear_solver: LinearSolverType,
    ) -> Result<(M::T, M::V), DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + LuValidator<M> + KluValidator<M>,
        M::V: VectorHost + DefaultDenseMatrix,
        S: OdeSolverMethod<'solver, DiffSl<M, JitModule<M::T>>>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let (chk, ys) = solver.solve_dense_with_checkpointing(t_eval, None)?;
        let eqn = solver.problem().eqn();
        let ctx = eqn.context();
        let mut g_m = <M::V as DefaultDenseMatrix>::M::zeros(eqn.nout(), t_eval.len(), ctx.clone());
        let mut y = M::T::zero();
        for j in 0..g_m.ncols() {
            let ys_col = ys.column(j);
            // TODO: can we avoid this allocation? (I can't see how right now)
            let mut tmp = M::V::from_slice(data.column(j).as_slice().unwrap(), ctx.clone());
            // tmp = 2 * ys_col - 2 * tmp
            tmp.axpy_v(
                M::T::from_f64(2.0).unwrap(),
                &ys_col,
                M::T::from_f64(-2.0).unwrap(),
            );
            g_m.column_mut(j).copy_from(&tmp);

            // y = (1/4) * dot(tmp, tmp) + y
            y += M::T::from_f64(1.0 / 4.0).unwrap() * tmp.norm(2).powi(2);
        }
        let mut y_sens = match backwards_linear_solver {
            LinearSolverType::Default => backwards_method
                .solve_adjoint_backwards::<M, <M as DefaultSolver>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
            LinearSolverType::Lu => backwards_method
                .solve_adjoint_backwards::<M, <M as LuValidator<M>>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
            LinearSolverType::Klu => backwards_method
                .solve_adjoint_backwards::<M, <M as KluValidator<M>>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
        };
        Ok((y, y_sens.pop().unwrap()))
    }

    pub(crate) fn solve_adjoint_backwards<'solver, M, LS, S>(
        &self,
        problem: &'solver OdeSolverProblem<DiffSl<M, JitModule<M::T>>>,
        checkpointing: Checkpointing<'solver, DiffSl<M, JitModule<M::T>>, S>,
        g_m: &<M::V as DefaultDenseMatrix>::M,
        t_eval: &[M::T],
        nout_override: Option<usize>,
    ) -> Result<Vec<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver,
        M::V: VectorHost + DefaultDenseMatrix,
        S: OdeSolverMethod<'solver, DiffSl<M, JitModule<M::T>>>,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => problem
                .bdf_solver_adjoint::<LS, _>(checkpointing, nout_override)?
                .solve_adjoint_backwards_pass(t_eval, &[g_m])
                .map(|res| res.into_common().sg),
            OdeSolverType::Esdirk34 => problem
                .esdirk34_solver_adjoint::<LS, _>(checkpointing, nout_override)?
                .solve_adjoint_backwards_pass(t_eval, &[g_m])
                .map(|res| res.into_common().sg),
            OdeSolverType::TrBdf2 => problem
                .tr_bdf2_solver_adjoint::<LS, _>(checkpointing, nout_override)?
                .solve_adjoint_backwards_pass(t_eval, &[g_m])
                .map(|res| res.into_common().sg),
            OdeSolverType::Tsit45 => Err(DiffsolError::Other(
                "Tsit45 solver does not support adjoint sensitivity analysis.".to_string(),
            )),
        }
    }
}
