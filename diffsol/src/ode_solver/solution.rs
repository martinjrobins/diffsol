use crate::{
    error::OdeSolverError, ode_solver_error, Context, DefaultDenseMatrix, DiffsolError,
    OdeEquations, OdeSolverStopReason, Op, Scalar,
};

pub(crate) enum SolutionMode<T: Scalar> {
    /// The solution is being computed at a number of time points specified by the user.
    /// The usize is the index of the next time point to compute (i.e. the number of time points computed so far).
    Tevals(usize),
    /// The solution is being computed until a final time specified by the user.
    /// The T is the final time.
    Tfinal(T),
}

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
/// };
///
/// type M = NalgebraMat<f64>;
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
/// let mut soln = Solution::new(1.0_f64, problem.eqn());
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
    pub fn new(t_final: V::T, eq: &impl OdeEquations<T = V::T, V = V, C = V::C>) -> Self {
        let nrows = if eq.out().is_some() {
            eq.out().unwrap().nout()
        } else {
            eq.rhs().nstates()
        };
        const INITIAL_NCOLS: usize = 10;
        let ret = eq.context().dense_mat_zeros::<V>(nrows, INITIAL_NCOLS);

        let tmp_nout = if let Some(out) = eq.out() {
            V::zeros(out.nout(), eq.context().clone())
        } else {
            V::zeros(0, eq.context().clone())
        };
        let tmp_nstates = V::zeros(0, eq.context().clone());
        Self {
            ts: Vec::new(),
            ys: ret,
            y_sens: Vec::new(),
            stop_reason: None,
            tmp_nout,
            tmp_nparams: V::zeros(0, eq.context().clone()),
            tmp_nstates,
            tmp_nsens: Vec::new(),
            mode: SolutionMode::Tfinal(t_final),
        }
    }

    pub fn new_dense(
        t_evals: Vec<V::T>,
        eq: &impl OdeEquations<T = V::T, V = V, C = V::C>,
    ) -> Result<Self, DiffsolError> {
        let nrows = if eq.out().is_some() {
            eq.out().unwrap().nout()
        } else {
            eq.rhs().nstates()
        };
        let ret = eq.context().dense_mat_zeros::<V>(nrows, t_evals.len());

        // check t_eval is increasing
        if t_evals.windows(2).any(|w| w[0] > w[1]) {
            return Err(ode_solver_error!(InvalidTEval));
        }
        let tmp_nout = if let Some(out) = eq.out() {
            V::zeros(out.nout(), eq.context().clone())
        } else {
            V::zeros(0, eq.context().clone())
        };
        let tmp_nstates = V::zeros(eq.rhs().nstates(), eq.context().clone());
        Ok(Self {
            ts: t_evals,
            ys: ret,
            y_sens: Vec::new(),
            stop_reason: None,
            tmp_nout,
            tmp_nparams: V::zeros(0, eq.context().clone()),
            tmp_nstates,
            tmp_nsens: Vec::new(),
            mode: SolutionMode::Tevals(0),
        })
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
        NalgebraLU, NalgebraVec, OdeSolverMethod,
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

        let mut soln = Solution::new(t_final, problem.eqn());
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

        let mut soln = Solution::new_dense(t_eval.clone(), problem.eqn()).unwrap();
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

        let mut soln = Solution::new(t_final, problem.eqn());
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

        let mut soln = Solution::new_dense(t_eval.clone(), problem.eqn()).unwrap();
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
        let (problem, _soln) = exponential_decay_problem::<NalgebraMat<f64>>(false);
        let t_eval = vec![0.0, 1.0, 0.5, 2.0];
        let err = Solution::new_dense(t_eval, problem.eqn());
        assert!(matches!(
            err,
            Err(DiffsolError::OdeSolverError(OdeSolverError::InvalidTEval))
        ));
    }
}
