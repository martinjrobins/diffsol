// Solver method Python enum. This is used to select the overarching solver
// stragegy like bdf or esdirk34 in diffsol.

use diffsol::error::DiffsolError;
use diffsol::{
    AdjointOdeSolverMethod, Checkpointing, CodegenModule, DefaultSolver, DenseMatrix, MatrixCommon,
    OdeEquations, OdeSolverState, OdeSolverStopReason, Op, SensitivitiesOdeSolverMethod, Solution,
    VectorViewMut,
};
use diffsol::{
    DefaultDenseMatrix, DiffSl, LinearSolver, Matrix, OdeSolverMethod, OdeSolverProblem, Vector,
    VectorHost, VectorRef, matrix::MatrixRef,
};
use nalgebra::ComplexField;
use ndarray::ArrayView2;
use num_traits::{FromPrimitive, Zero}; // for generic nums in _solve_sum_squares_adj
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::scalar_type::Scalar;
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OdeSolverType {
    Bdf,
    Esdirk34,
    TrBdf2,
    Tsit45,
}

impl OdeSolverType {
    pub(crate) fn solve<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        final_time: M::T,
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                let solver = problem.bdf::<LS>()?;
                let mut soln = Solution::new(final_time);
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::Esdirk34 => {
                let solver = problem.esdirk34::<LS>()?;
                let mut soln = Solution::new(final_time);
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::TrBdf2 => {
                let solver = problem.tr_bdf2::<LS>()?;
                let mut soln = Solution::new(final_time);
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::Tsit45 => {
                let solver = problem.tsit45()?;
                let mut soln = Solution::new(final_time);
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
        }
    }

    pub(crate) fn solve_dense<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                let solver = problem.bdf::<LS>()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::Esdirk34 => {
                let solver = problem.esdirk34::<LS>()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::TrBdf2 => {
                let solver = problem.tr_bdf2::<LS>()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::Tsit45 => {
                let solver = problem.tsit45()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln(&mut soln)?;
                Ok(soln)
            }
        }
    }

    pub(crate) fn solve_hybrid<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        final_time: M::T,
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                let mut soln = Solution::new(final_time);
                let mut solver = problem.bdf::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.bdf_solver::<LS>(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::Esdirk34 => {
                let mut soln = Solution::new(final_time);
                let mut solver = problem.esdirk34::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.esdirk34_solver::<LS>(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::TrBdf2 => {
                let mut soln = Solution::new(final_time);
                let mut solver = problem.tr_bdf2::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.tr_bdf2_solver::<LS>(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::Tsit45 => {
                let mut soln = Solution::new(final_time);
                let mut solver = problem.tsit45()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.tsit45_solver(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
        }
    }

    pub(crate) fn solve_hybrid_dense<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            OdeSolverType::Bdf => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.bdf::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.bdf_solver::<LS>(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::Esdirk34 => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.esdirk34::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.esdirk34_solver::<LS>(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::TrBdf2 => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.tr_bdf2::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.tr_bdf2_solver::<LS>(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::Tsit45 => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.tsit45()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.tsit45_solver(state)?;
                    restarted_solver.reset()?;
                    solver = restarted_solver;
                }
                Ok(soln)
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
    pub(crate) fn solve_fwd_sens<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => {
                let solver = problem.bdf_sens::<LS>()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln_sensitivities(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::Esdirk34 => {
                let solver = problem.esdirk34_sens::<LS>()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln_sensitivities(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::TrBdf2 => {
                let solver = problem.tr_bdf2_sens::<LS>()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln_sensitivities(&mut soln)?;
                Ok(soln)
            }
            OdeSolverType::Tsit45 => {
                let solver = problem.tsit45_sens()?;
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                solver.solve_soln_sensitivities(&mut soln)?;
                Ok(soln)
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_hybrid_fwd_sens<M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        t_eval: &[M::T],
    ) -> Result<Solution<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            OdeSolverType::Bdf => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.bdf_sens::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln_sensitivities(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.bdf_solver_sens::<LS>(state)?;
                    restarted_solver.reset_with_sens()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::Esdirk34 => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.esdirk34_sens::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln_sensitivities(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.esdirk34_solver_sens::<LS>(state)?;
                    restarted_solver.reset_with_sens()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::TrBdf2 => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.tr_bdf2_sens::<LS>()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln_sensitivities(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.tr_bdf2_solver_sens::<LS>(state)?;
                    restarted_solver.reset_with_sens()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
            OdeSolverType::Tsit45 => {
                let mut soln = Solution::new_dense(t_eval.to_vec())?;
                let mut solver = problem.tsit45_sens()?;
                while !soln.is_complete() {
                    solver = solver.solve_soln_sensitivities(&mut soln)?;
                    let root_idx = match soln.stop_reason {
                        Some(OdeSolverStopReason::RootFound(_, root_idx))
                            if !soln.is_complete() =>
                        {
                            root_idx
                        }
                        _ => continue,
                    };
                    let state = solver.into_state();
                    problem.eqn.set_model_index(root_idx);
                    let mut restarted_solver = problem.tsit45_solver_sens(state)?;
                    restarted_solver.reset_with_sens()?;
                    solver = restarted_solver;
                }
                Ok(soln)
            }
        }
    }

    pub(crate) fn solve_sum_squares_adj<'a, M, CG, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, CG>>,
        data: ArrayView2<'a, M::T>,
        t_eval: &[M::T],
        backwards_method: OdeSolverType,
        backwards_linear_solver: LinearSolverType,
    ) -> Result<(M::T, M::V), DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + LuValidator<M> + KluValidator<M>,
        CG: CodegenModule,
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

    pub(crate) fn _solve_sum_squares_adj<'data, 'solver, M, CG, S>(
        &self,
        mut solver: S,
        data: ArrayView2<'data, M::T>,
        t_eval: &[M::T],
        backwards_method: OdeSolverType,
        backwards_linear_solver: LinearSolverType,
    ) -> Result<(M::T, M::V), DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver + LuValidator<M> + KluValidator<M>,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        S: OdeSolverMethod<'solver, DiffSl<M, CG>>,
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
                .solve_adjoint_backwards::<M, CG, <M as DefaultSolver>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
            LinearSolverType::Lu => backwards_method
                .solve_adjoint_backwards::<M, CG, <M as LuValidator<M>>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
            LinearSolverType::Klu => backwards_method
                .solve_adjoint_backwards::<M, CG, <M as KluValidator<M>>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
        };
        Ok((y, y_sens.pop().unwrap()))
    }

    pub(crate) fn solve_adjoint_backwards<'solver, M, CG, LS, S>(
        &self,
        problem: &'solver OdeSolverProblem<DiffSl<M, CG>>,
        checkpointing: Checkpointing<'solver, DiffSl<M, CG>, S>,
        g_m: &<M::V as DefaultDenseMatrix>::M,
        t_eval: &[M::T],
        nout_override: Option<usize>,
    ) -> Result<Vec<M::V>, DiffsolError>
    where
        M: Matrix<T: Scalar> + DefaultSolver,
        CG: CodegenModule,
        M::V: VectorHost + DefaultDenseMatrix,
        S: OdeSolverMethod<'solver, DiffSl<M, CG>>,
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
