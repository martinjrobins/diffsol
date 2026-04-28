use crate::{
    error::OdeSolverError, ode_solver_error, DefaultDenseMatrix, DenseMatrix, DiffsolError, Matrix,
    MatrixCommon, OdeSolverStopReason, Scalar,
};

pub(crate) enum SolutionMode<T: Scalar> {
    /// The solution is being computed at a number of time points specified by the user.
    /// The usize is the index of the next time point to compute (i.e. the number of time points computed so far).
    Tevals(usize),
    /// The solution is being computed until a final time specified by the user.
    /// The T is the final time.
    Tfinal(T),
}

pub(crate) const INITIAL_NCOLS: usize = 10;

/// Stores ODE solve output and continuation state for staged integrations.
///
/// A `Solution` can be reused across multiple calls to
/// [`crate::OdeSolverMethod::solve_soln`], allowing you to stop, inspect
/// `stop_reason`, update model parameters or state, and then continue from where
/// you left off.
///
/// Use [`Solution::new`] to collect adaptive internal timesteps up to a final
/// time, or [`Solution::new_dense`] to fill user-provided evaluation times.
///
/// # Example
/// ```
/// use diffsol::{
///     OdeBuilder, OdeSolverMethod, Solution, NalgebraLU, NalgebraMat,
///     NalgebraVec,
/// };
///
/// type M = NalgebraMat<f64>;
/// type V = NalgebraVec<f64>;
/// type LS = NalgebraLU<f64>;
///
/// let problem = OdeBuilder::<M>::new()
///     .p([0.1, 1.0])
///     .rhs_implicit(
///         |x, p, _t, y| {
///             y[0] = -p[0] * x[0];
///         },
///         |_x, p, _t, v, y| {
///             y[0] = -p[0] * v[0];
///         },
///     )
///     .init(|p, _t, y| {
///         y[0] = p[1];
///     }, 1)
///     .build()
///     .unwrap();
///
/// let mut state = problem.bdf_state::<LS>().unwrap();
/// let mut soln = Solution::<V>::new(1.0_f64);
///
/// while !soln.is_complete() {
///     state = problem
///         .bdf_solver::<LS>(state)
///         .unwrap()
///         .solve_soln(&mut soln)
///         .unwrap()
///         .into_state();
/// }
///
/// assert!(soln.is_complete());
/// assert!(!soln.ts.is_empty());
/// ```
pub struct Solution<V: DefaultDenseMatrix> {
    pub ts: Vec<V::T>,
    pub ys: <V as DefaultDenseMatrix>::M,
    pub y_sens: Vec<<V as DefaultDenseMatrix>::M>,
    pub stop_reason: Option<OdeSolverStopReason<V::T>>,
    pub(crate) tmp_nout: V,
    pub(crate) tmp_nparams: V,
    pub(crate) tmp_nstates: V,
    pub(crate) tmp_nsens: Vec<V>,
    pub(crate) mode: SolutionMode<V::T>,
}

impl<V: DefaultDenseMatrix> Solution<V> {
    pub fn is_complete(&self) -> bool {
        match self.mode {
            SolutionMode::Tevals(next_col) => next_col >= self.ts.len(),
            SolutionMode::Tfinal(t_final) => self.ts.last().map(|&t| t >= t_final).unwrap_or(false),
        }
    }

    pub fn truncate(&mut self) {
        if let Some(OdeSolverStopReason::RootFound(t_root, _)) = self.stop_reason {
            let ncols = self
                .ts
                .iter()
                .position(|&t| t > t_root)
                .unwrap_or(self.ts.len());
            self.ts.truncate(ncols);
            self.ys.resize_cols(ncols);
            for sens in &mut self.y_sens {
                sens.resize_cols(ncols);
            }
        }
    }

    pub fn new(t_final: V::T) -> Self {
        let ctx = V::C::default();
        Self {
            ts: Vec::new(),
            ys: <V as DefaultDenseMatrix>::M::zeros(0, 0, ctx.clone()),
            y_sens: Vec::new(),
            stop_reason: None,
            tmp_nout: V::zeros(0, ctx.clone()),
            tmp_nparams: V::zeros(0, ctx.clone()),
            tmp_nstates: V::zeros(0, ctx.clone()),
            tmp_nsens: Vec::new(),
            mode: SolutionMode::Tfinal(t_final),
        }
    }

    pub fn new_dense(t_evals: Vec<V::T>) -> Result<Self, DiffsolError> {
        // check t_eval is increasing
        if t_evals.windows(2).any(|w| w[0] > w[1]) {
            return Err(ode_solver_error!(InvalidTEval));
        }
        let ctx = V::C::default();
        Ok(Self {
            ts: t_evals,
            ys: <V as DefaultDenseMatrix>::M::zeros(0, 0, ctx.clone()),
            y_sens: Vec::new(),
            stop_reason: None,
            tmp_nout: V::zeros(0, ctx.clone()),
            tmp_nparams: V::zeros(0, ctx.clone()),
            tmp_nstates: V::zeros(0, ctx.clone()),
            tmp_nsens: Vec::new(),
            mode: SolutionMode::Tevals(0),
        })
    }

    pub(crate) fn ensure_ode_allocation(
        &mut self,
        ctx: &V::C,
        nrows: usize,
        nout: usize,
        nstates: usize,
    ) -> Result<(), DiffsolError> {
        match self.mode {
            SolutionMode::Tfinal(_) => {
                if self.ys.nrows() == 0 && self.ys.ncols() == 0 {
                    self.ys =
                        <V as DefaultDenseMatrix>::M::zeros(nrows, INITIAL_NCOLS, ctx.clone());
                } else if self.ys.nrows() != nrows {
                    return Err(ode_solver_error!(
                        Other,
                        "Solution is incompatible with the current equations: output size changed"
                    ));
                }
            }
            SolutionMode::Tevals(_) => {
                if self.ys.nrows() == 0 && self.ys.ncols() == 0 {
                    self.ys =
                        <V as DefaultDenseMatrix>::M::zeros(nrows, self.ts.len(), ctx.clone());
                } else if self.ys.nrows() != nrows || self.ys.ncols() != self.ts.len() {
                    return Err(ode_solver_error!(
                        Other,
                        "Solution is incompatible with the current equations: output size changed"
                    ));
                }
            }
        }

        if self.tmp_nout.len() == 0 {
            self.tmp_nout = V::zeros(nout, ctx.clone());
        } else if self.tmp_nout.len() != nout {
            return Err(ode_solver_error!(
                Other,
                "Solution is incompatible with the current equations: output size changed"
            ));
        }

        match self.mode {
            SolutionMode::Tfinal(_) => {
                if self.tmp_nstates.len() != 0 && self.tmp_nstates.len() != nstates {
                    return Err(ode_solver_error!(
                        Other,
                        "Solution is incompatible with the current equations: state size changed"
                    ));
                }
            }
            SolutionMode::Tevals(_) => {
                if self.tmp_nstates.len() == 0 {
                    self.tmp_nstates = V::zeros(nstates, ctx.clone());
                } else if self.tmp_nstates.len() != nstates {
                    return Err(ode_solver_error!(
                        Other,
                        "Solution is incompatible with the current equations: state size changed"
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_sens_allocation(
        &mut self,
        ctx: &V::C,
        nrows: usize,
        nout: usize,
        nout_params: usize,
        nstates: usize,
        nparams: usize,
    ) -> Result<(), DiffsolError> {
        self.ensure_ode_allocation(ctx, nrows, nout, nstates)?;

        if self.y_sens.is_empty() {
            self.y_sens =
                vec![
                    <V as DefaultDenseMatrix>::M::zeros(nrows, self.ts.len(), ctx.clone());
                    nparams
                ];
        } else if self.y_sens.len() != nparams
            || self
                .y_sens
                .iter()
                .any(|m| m.nrows() != nrows || m.ncols() != self.ts.len())
        {
            return Err(ode_solver_error!(
                Other,
                "Solution is incompatible with the current equations: sensitivity size changed"
            ));
        }

        if self.tmp_nparams.len() == 0 {
            self.tmp_nparams = V::zeros(nout_params, ctx.clone());
        } else if self.tmp_nparams.len() != nout_params {
            return Err(ode_solver_error!(
                Other,
                "Solution is incompatible with the current equations: output sensitivity size changed"
            ));
        }

        if self.tmp_nsens.is_empty() {
            self.tmp_nsens = vec![V::zeros(nstates, ctx.clone()); nparams];
        } else if self.tmp_nsens.len() != nparams
            || self.tmp_nsens.iter().any(|v| v.len() != nstates)
        {
            return Err(ode_solver_error!(
                Other,
                "Solution is incompatible with the current equations: sensitivity size changed"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        error::{DiffsolError, OdeSolverError},
        matrix::{dense_nalgebra_serial::NalgebraMat, MatrixCommon},
        ode_equations::test_models::exponential_decay::{
            exponential_decay_problem, exponential_decay_problem_with_root,
        },
        NalgebraLU, NalgebraVec, OdeBuilder, OdeSolverMethod,
    };

    use super::Solution;

    fn assert_exponential_decay_values(soln: &Solution<NalgebraVec<f64>>, k: f64) {
        for (i, &t) in soln.ts.iter().enumerate() {
            let expected = f64::exp(-k * t);
            for row in 0..soln.ys.nrows() {
                let got = soln.ys[(row, i)];
                assert!(
                    (got - expected).abs() < 2e-3,
                    "mismatch at row={row}, t={t}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_solution_new_with_solve_soln_sets_complete() {
        let (problem, _soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let t_final = 10.0_f64;
        let k = 0.1_f64;
        let mut state = problem.bdf_state::<NalgebraLU<f64>>().unwrap();

        let mut soln = Solution::new(t_final);
        assert!(!soln.is_complete());
        while !soln.is_complete() {
            state = problem
                .bdf_solver::<NalgebraLU<f64>>(state)
                .unwrap()
                .solve_soln(&mut soln)
                .unwrap()
                .into_state();
        }
        assert!(soln.is_complete());
        assert!(!soln.ts.is_empty());
        assert!((soln.ts.last().copied().unwrap() - t_final).abs() < 1e-8);
        assert_exponential_decay_values(&soln, k);
    }

    #[test]
    fn test_solution_new_dense_with_solve_soln_sets_complete() {
        let (problem, _soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let k = 0.1_f64;
        let t_eval = (0..=10).map(|i| i as f64).collect::<Vec<_>>();
        let mut state = problem.bdf_state::<NalgebraLU<f64>>().unwrap();

        let mut soln = Solution::new_dense(t_eval.clone()).unwrap();
        assert!(!soln.is_complete());
        while !soln.is_complete() {
            state = problem
                .bdf_solver::<NalgebraLU<f64>>(state)
                .unwrap()
                .solve_soln(&mut soln)
                .unwrap()
                .into_state();
        }
        assert!(soln.is_complete());
        assert_eq!(soln.ts, t_eval);
        assert_exponential_decay_values(&soln, k);
    }

    #[test]
    fn test_solution_new_with_out_eqn_sets_complete() {
        let (problem, _soln) =
            exponential_decay_problem_with_root::<NalgebraMat<f64>>(false, false);
        let t_final = 2.0_f64;
        let k = 0.1_f64;
        let mut state = problem.bdf_state::<NalgebraLU<f64>>().unwrap();

        let mut soln = Solution::new(t_final);
        assert!(!soln.is_complete());
        while !soln.is_complete() {
            state = problem
                .bdf_solver::<NalgebraLU<f64>>(state)
                .unwrap()
                .solve_soln(&mut soln)
                .unwrap()
                .into_state();
        }
        assert!(soln.is_complete());
        assert!(!soln.ts.is_empty());
        assert!((soln.ts.last().copied().unwrap() - t_final).abs() < 1e-8);
        assert_exponential_decay_values(&soln, k);
    }

    #[test]
    fn test_solution_new_dense_with_out_eqn_sets_complete() {
        let (problem, _soln) =
            exponential_decay_problem_with_root::<NalgebraMat<f64>>(false, false);
        let k = 0.1_f64;
        let t_eval = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut state = problem.bdf_state::<NalgebraLU<f64>>().unwrap();

        let mut soln = Solution::new_dense(t_eval.clone()).unwrap();
        assert!(!soln.is_complete());
        while !soln.is_complete() {
            state = problem
                .bdf_solver::<NalgebraLU<f64>>(state)
                .unwrap()
                .solve_soln(&mut soln)
                .unwrap()
                .into_state();
        }
        assert!(soln.is_complete());
        assert_eq!(soln.ts, t_eval);
        assert_exponential_decay_values(&soln, k);
    }

    #[test]
    fn test_solution_new_dense_errors_on_non_increasing_t_evals() {
        let t_eval = vec![0.0, 1.0, 0.5, 2.0];
        let err = Solution::<NalgebraVec<f64>>::new_dense(t_eval);
        assert!(matches!(
            err,
            Err(DiffsolError::OdeSolverError(OdeSolverError::InvalidTEval))
        ));
    }

    #[test]
    fn test_solution_solve_soln_errors_on_incompatible_equations() {
        type M = NalgebraMat<f64>;
        type LS = NalgebraLU<f64>;

        let problem1 = OdeBuilder::<M>::new()
            .p([0.1])
            .rhs_implicit(
                |x, p, _t, y| y[0] = -p[0] * x[0],
                |_x, p, _t, v, y| y[0] = -p[0] * v[0],
            )
            .init(|_p, _t, y| y[0] = 1.0, 1)
            .build()
            .unwrap();

        let problem2 = OdeBuilder::<M>::new()
            .p([0.1])
            .rhs_implicit(
                |x, p, _t, y| {
                    y[0] = -p[0] * x[0];
                    y[1] = -p[0] * x[1];
                },
                |_x, p, _t, v, y| {
                    y[0] = -p[0] * v[0];
                    y[1] = -p[0] * v[1];
                },
            )
            .init(
                |_p, _t, y| {
                    y[0] = 1.0;
                    y[1] = 1.0;
                },
                2,
            )
            .build()
            .unwrap();

        let mut soln = Solution::<NalgebraVec<f64>>::new_dense(vec![0.0, 1.0]).unwrap();
        problem1.bdf::<LS>().unwrap().solve_soln(&mut soln).unwrap();

        let err = match problem2.bdf::<LS>().unwrap().solve_soln(&mut soln) {
            Ok(_) => panic!("expected incompatible solution error"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            DiffsolError::OdeSolverError(OdeSolverError::Other(_))
        ));
    }
}
