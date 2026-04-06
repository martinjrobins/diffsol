// Solver method Python enum. This is used to select the overarching solver
// stragegy like bdf or esdirk34 in diffsol.

use diffsol::error::DiffsolError;
use diffsol::{
    matrix::MatrixRef, DefaultDenseMatrix, DiffSl, LinearSolver, Matrix, OdeSolverMethod,
    OdeSolverProblem, Vector, VectorHost, VectorRef,
};
use diffsol::{
    AdjointOdeSolverMethod, Checkpointing, CodegenModule, DefaultSolver, DenseMatrix, MatrixCommon,
    OdeEquations, OdeSolverState, OdeSolverStopReason, Op, SensitivitiesOdeSolverMethod, Solution,
    VectorViewMut,
};
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
            let norm = tmp.norm(2);
            y += M::T::from_f64(1.0 / 4.0).unwrap() * norm * norm;
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

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod tests {
    use diffsol::Vector;
    use diffsol::{
        CodegenModuleCompile, CodegenModuleJit, DefaultSolver, DenseMatrix, OdeBuilder,
        OdeSolverProblem,
    };
    use ndarray::Array2;

    use crate::linear_solver_type::LinearSolverType;
    use crate::test_support::{
        assert_close, hybrid_logistic_diffsl_code, hybrid_logistic_state, hybrid_logistic_state_dr,
        logistic_diffsl_code, logistic_integral, logistic_state, logistic_state_dr, LOGISTIC_X0,
    };
    use crate::valid_linear_solver::LuValidator;

    use super::OdeSolverType;

    type M = diffsol::NalgebraMat<f64>;

    fn build_problem<CG>(code: &str) -> OdeSolverProblem<diffsol::DiffSl<M, CG>>
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        OdeBuilder::<M>::new()
            .p([2.0])
            .rtol(1e-6)
            .atol([1e-6])
            .build_from_diffsl::<CG>(code)
            .unwrap()
    }

    fn assert_dense_solution_matches_expected(
        soln: &diffsol::Solution<diffsol::NalgebraVec<f64>>,
        t_eval: &[f64],
        expected: impl Fn(f64) -> f64,
    ) {
        assert_eq!(soln.ts, t_eval);
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                soln.ys.get_index(0, i),
                expected(t),
                5e-4,
                &format!("solution[{i}]"),
            );
        }
    }

    fn test_all_solver_variants<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve::<M, CG, <M as DefaultSolver>::LS>(&mut problem, 1.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 1.0, 5e-4, "solve final time");
            assert_close(
                soln.ys.get_index(0, soln.ts.len() - 1),
                logistic_state(LOGISTIC_X0, 2.0, 1.0),
                5e-4,
                "solve final value",
            );

            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve_dense::<M, CG, <M as DefaultSolver>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                logistic_state(LOGISTIC_X0, 2.0, t)
            });
        }
    }

    fn test_all_hybrid_solver_variants<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_hybrid::<M, CG, <M as DefaultSolver>::LS>(&mut problem, 2.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 2.0, 5e-4, "hybrid final time");
            assert_close(
                soln.ys.get_index(0, soln.ts.len() - 1),
                hybrid_logistic_state(2.0, 2.0),
                5e-4,
                "hybrid final value",
            );

            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_hybrid_dense::<M, CG, <M as DefaultSolver>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                hybrid_logistic_state(2.0, t)
            });
        }
    }

    fn test_all_solver_variants_with_lu<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, 1.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 1.0, 5e-4, "lu solve final time");

            let mut problem = build_problem::<CG>(logistic_diffsl_code());
            let soln = method
                .solve_dense::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                logistic_state(LOGISTIC_X0, 2.0, t)
            });
        }
    }

    fn test_all_hybrid_solver_variants_with_lu<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_hybrid::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, 2.0)
                .unwrap();
            assert_close(*soln.ts.last().unwrap(), 2.0, 5e-4, "lu hybrid final time");

            let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_hybrid_dense::<M, CG, <M as LuValidator<M>>::LS>(&mut problem, &t_eval)
                .unwrap();
            assert_dense_solution_matches_expected(&soln, &t_eval, |t| {
                hybrid_logistic_state(2.0, t)
            });
        }
    }

    fn assert_direct_hybrid_restart_path_for_method<CG>(method: OdeSolverType)
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    {
        let t_eval = [0.5, 1.0, 1.25, 1.5, 2.0];

        let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
        let soln = method
            .solve_hybrid::<M, CG, <M as DefaultSolver>::LS>(&mut problem, 2.0)
            .unwrap();
        assert_close(
            *soln.ts.last().unwrap(),
            2.0,
            5e-4,
            "direct hybrid restart final time",
        );
        assert_close(
            soln.ys.get_index(0, soln.ts.len() - 1),
            hybrid_logistic_state(2.0, 2.0),
            5e-4,
            "direct hybrid restart final value",
        );

        let mut problem = build_problem::<CG>(hybrid_logistic_diffsl_code());
        let soln = method
            .solve_hybrid_dense::<M, CG, <M as DefaultSolver>::LS>(&mut problem, &t_eval)
            .unwrap();
        assert_dense_solution_matches_expected(&soln, &t_eval, |t| hybrid_logistic_state(2.0, t));
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_all_sensitivity_solver_variants() {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let soln = method
                .solve_fwd_sens::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    logistic_state_dr(LOGISTIC_X0, 2.0, t),
                    5e-4,
                    &format!("fwd_sens[{i}]"),
                );
            }

            let mut problem = build_problem::<diffsol::LlvmModule>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_hybrid_fwd_sens::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    hybrid_logistic_state_dr(2.0, t),
                    5e-4,
                    &format!("hybrid_fwd_sens[{i}]"),
                );
            }
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_lu_sensitivity_and_adjoint_solver_variants() {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let soln = method
                .solve_fwd_sens::<M, diffsol::LlvmModule, <M as LuValidator<M>>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    logistic_state_dr(LOGISTIC_X0, 2.0, t),
                    5e-4,
                    &format!("lu fwd_sens[{i}]"),
                );
            }
        }

        let adjoint_t_eval = [0.0, 0.25, 0.5, 1.0];
        let data = Array2::from_shape_vec(
            (1, adjoint_t_eval.len()),
            adjoint_t_eval
                .iter()
                .map(|&t| logistic_integral(LOGISTIC_X0, 2.0, t))
                .collect(),
        )
        .unwrap();

        let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
        let (objective, gradient) = OdeSolverType::Bdf
            .solve_sum_squares_adj::<M, diffsol::LlvmModule, <M as LuValidator<M>>::LS>(
                &mut problem,
                data.view(),
                &adjoint_t_eval,
                OdeSolverType::TrBdf2,
                LinearSolverType::Lu,
            )
            .unwrap();
        assert!(objective.is_finite());
        assert_eq!(gradient.len(), 1);
        assert!(gradient.get_index(0).is_finite());
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_direct_hybrid_sensitivity_restart_paths() {
        let t_eval = [0.25, 0.5, 1.0];
        for method in [
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(hybrid_logistic_diffsl_code());
            let soln = method
                .solve_hybrid_fwd_sens::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    &t_eval,
                )
                .unwrap();
            for (i, &t) in t_eval.iter().enumerate() {
                assert_close(
                    soln.y_sens[0].get_index(0, i),
                    hybrid_logistic_state_dr(2.0, t),
                    5e-4,
                    &format!("direct hybrid fwd sens[{i}]"),
                );
            }
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_adjoint_backwards_methods_and_klu_branch() {
        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let data = Array2::from_shape_vec(
            (1, t_eval.len()),
            t_eval
                .iter()
                .map(|&t| logistic_integral(LOGISTIC_X0, 2.0, t))
                .collect(),
        )
        .unwrap();

        for backwards_method in [OdeSolverType::Esdirk34, OdeSolverType::TrBdf2] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let (objective, gradient) = OdeSolverType::Bdf
                .solve_sum_squares_adj::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    data.view(),
                    &t_eval,
                    backwards_method,
                    LinearSolverType::Klu,
                )
                .unwrap();
            assert!(objective.is_finite());
            assert_eq!(gradient.len(), 1);
            assert!(gradient.get_index(0).is_finite());
        }

        let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
        let err = OdeSolverType::Bdf
            .solve_sum_squares_adj::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                &mut problem,
                data.view(),
                &t_eval,
                OdeSolverType::Tsit45,
                LinearSolverType::Default,
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("Tsit45 solver does not support adjoint sensitivity analysis"));
    }

    #[cfg(feature = "diffsl-llvm")]
    fn test_all_adjoint_solver_variants() {
        let t_eval = [0.0, 0.25, 0.5, 1.0];
        let data = Array2::from_shape_vec(
            (1, t_eval.len()),
            t_eval
                .iter()
                .map(|&t| logistic_integral(LOGISTIC_X0, 2.0, t))
                .collect(),
        )
        .unwrap();

        for method in [
            OdeSolverType::Bdf,
            OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45,
        ] {
            let mut problem = build_problem::<diffsol::LlvmModule>(logistic_diffsl_code());
            let (objective, gradient) = method
                .solve_sum_squares_adj::<M, diffsol::LlvmModule, <M as DefaultSolver>::LS>(
                    &mut problem,
                    data.view(),
                    &t_eval,
                    OdeSolverType::Bdf,
                    crate::linear_solver_type::LinearSolverType::Default,
                )
                .unwrap();
            assert!(objective.is_finite());
            assert_eq!(gradient.len(), 1);
            assert!(gradient.get_index(0).is_finite());
        }
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn runtime_dispatch_solves_all_variants_for_cranelift() {
        test_all_solver_variants::<diffsol::CraneliftJitModule>();
        test_all_solver_variants_with_lu::<diffsol::CraneliftJitModule>();
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn runtime_dispatch_solves_all_hybrid_variants_for_cranelift() {
        test_all_hybrid_solver_variants::<diffsol::CraneliftJitModule>();
        test_all_hybrid_solver_variants_with_lu::<diffsol::CraneliftJitModule>();
        assert_direct_hybrid_restart_path_for_method::<diffsol::CraneliftJitModule>(
            OdeSolverType::Esdirk34,
        );
        assert_direct_hybrid_restart_path_for_method::<diffsol::CraneliftJitModule>(
            OdeSolverType::TrBdf2,
        );
        assert_direct_hybrid_restart_path_for_method::<diffsol::CraneliftJitModule>(
            OdeSolverType::Tsit45,
        );
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_variants_for_llvm() {
        test_all_solver_variants::<diffsol::LlvmModule>();
        test_all_solver_variants_with_lu::<diffsol::LlvmModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_hybrid_variants_for_llvm() {
        test_all_hybrid_solver_variants::<diffsol::LlvmModule>();
        test_all_hybrid_solver_variants_with_lu::<diffsol::LlvmModule>();
        assert_direct_hybrid_restart_path_for_method::<diffsol::LlvmModule>(
            OdeSolverType::Esdirk34,
        );
        assert_direct_hybrid_restart_path_for_method::<diffsol::LlvmModule>(OdeSolverType::TrBdf2);
        assert_direct_hybrid_restart_path_for_method::<diffsol::LlvmModule>(OdeSolverType::Tsit45);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_forward_sensitivity_variants_for_llvm() {
        test_all_sensitivity_solver_variants();
        test_lu_sensitivity_and_adjoint_solver_variants();
        test_direct_hybrid_sensitivity_restart_paths();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn runtime_dispatch_solves_all_adjoint_variants_for_llvm() {
        test_all_adjoint_solver_variants();
        test_adjoint_backwards_methods_and_klu_branch();
    }
}
