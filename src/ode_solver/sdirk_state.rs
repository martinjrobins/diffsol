use crate::{error::DiffsolError, error::OdeSolverError, ode_solver_error, DenseMatrix, OdeEquations, OdeSolverProblem, OdeSolverState, Op, Vector};

#[derive(Clone)]
pub struct SdirkState<
    V: Vector,
    M: DenseMatrix<T = V::T, V = V>,
> {
    pub(crate) order: usize,
    pub(crate) diff: M,
    pub(crate) sdiff: Vec<M>,
    pub(crate) y: V,
    pub(crate) dy: V,
    pub(crate) s: Vec<V>,
    pub(crate) ds: Vec<V>,
    pub(crate) t: V::T,
    pub(crate) h: V::T,
}

impl<V, M> SdirkState<V, M>
where 
    V: Vector,
    M: DenseMatrix<T = V::T, V = V>,
{
}

impl<V, M> OdeSolverState<V> for SdirkState<V, M>
where 
    V: Vector,
    M: DenseMatrix<T = V::T, V = V>,
{
    fn set_problem<Eqn: OdeEquations>(&mut self, ode_problem: &OdeSolverProblem<Eqn>) -> Result<(), DiffsolError> {
        let not_initialised = self.diff.ncols() == 0;
        let nstates = ode_problem.eqn.rhs().nstates();
        let nparams = ode_problem.eqn.rhs().nparams();
        if not_initialised {
            self.diff = M::zeros(nstates, self.order);
            self.sdiff = vec![M::zeros(nstates, self.order); nparams];
        }
        if self.diff.nrows() != nstates || self.sdiff[0].nrows() != nstates || self.sdiff.len() != nparams {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        Ok(())
    }

    fn new_internal_state(y: V, dy: V, s: Vec<V>, ds: Vec<V>, t: <V>::T, h: <V>::T) -> Self {
        Self {
            order: 0,
            diff: M::zeros(0, 0),
            sdiff: Vec::new(),
            y,
            dy,
            s,
            ds,
            t,
            h,
        }
    }

    fn s(&self) -> &[V] {
        self.s.as_slice()
    }
    fn s_mut(&mut self) -> &mut [V] {
        &mut self.s
    }
    fn ds_mut(&mut self) -> &mut [V] {
        &mut self.ds
    }
    fn ds(&self) -> &[V] {
        self.ds.as_slice()
    }
    fn y(&self) -> &V {
        &self.y
    }

    fn y_mut(&mut self) -> &mut V {
        &mut self.y
    }

    fn dy(&self) -> &V {
        &self.dy
    }

    fn dy_mut(&mut self) -> &mut V {
        &mut self.dy
    }

    fn t(&self) -> V::T {
        self.t
    }

    fn t_mut(&mut self) -> &mut V::T {
        &mut self.t
    }

    fn h(&self) -> V::T {
        self.h
    }

    fn h_mut(&mut self) -> &mut V::T {
        &mut self.h
    }
}