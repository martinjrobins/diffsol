use crate::{Context, DefaultDenseMatrix, DiffsolError, OdeEquations, OdeSolverStopReason, Op, Scalar, error::OdeSolverError, ode_solver_error};


pub(crate) enum SolutionMode<T: Scalar> {
    /// The solution is being computed at a number of time points specified by the user. 
    /// The usize is the index of the next time point to compute (i.e. the number of time points computed so far).
    Tevals(usize),
    /// The solution is being computed until a final time specified by the user.
    /// The T is the final time.
    Tfinal(T),
}


pub struct Solution<V: DefaultDenseMatrix> {
    pub ts: Vec<V::T>,
    pub ys: <V as DefaultDenseMatrix>::M,
    pub y_sens: Vec<<V as DefaultDenseMatrix>::M>,
    pub stop_reason: Option<OdeSolverStopReason<V::T>>,
    pub(crate) tmp_nout: V,
    pub(crate) tmp_nstates: V,
    pub(crate) mode: SolutionMode<V::T>,
}

impl<V: DefaultDenseMatrix> Solution<V> 
{
    pub fn is_complete(&self) -> bool {
        match self.mode {
            SolutionMode::Tevals(next_col) => next_col >= self.ts.len(),
            SolutionMode::Tfinal(t_final) => self.ts.last().map(|&t| t >= t_final).unwrap_or(false)
        }
    }
    pub fn new(t_final: V::T, eq: &impl OdeEquations<T=V::T, V=V, C=V::C>) -> Self {
        let nrows = if eq.out().is_some() {
            eq.out().unwrap().nout()
        } else {
            eq.rhs().nstates()
        };
        const INITIAL_NCOLS: usize = 10;
        let ret = eq
            .context()
            .dense_mat_zeros::<V>(nrows, INITIAL_NCOLS);

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
            tmp_nstates,
            mode: SolutionMode::Tfinal(t_final),
        }
    }
    
    pub fn new_dense(t_evals: Vec<V::T>, eq: &impl OdeEquations<T=V::T, V=V, C=V::C>) -> Result<Self, DiffsolError> {
        let nrows = if eq.out().is_some() {
            eq.out().unwrap().nout()
        } else {
            eq.rhs().nstates()
        };
        let ret = eq
            .context()
            .dense_mat_zeros::<V>(nrows, t_evals.len());

        // check t_eval is increasing
        if t_evals.windows(2).any(|w| w[0] > w[1]) {
            return Err(ode_solver_error!(InvalidTEval));
        }
        let tmp_nout = if let Some(out) = eq.out() {
            V::zeros(out.nout(), eq.context().clone())
        } else {
            V::zeros(0, eq.context().clone())
        };
        let tmp_nstates = V::zeros(
            eq.rhs().nstates(),
            eq.context().clone(),
        );
        Ok(Self {
            ts: t_evals,
            ys: ret,
            y_sens: Vec::new(),
            stop_reason: None,
            tmp_nout,
            tmp_nstates,
            mode: SolutionMode::Tevals(0),
        })
    }
}