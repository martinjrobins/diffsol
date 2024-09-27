use crate::{
    error::DiffsolError, scalar::IndexType, scale, DenseMatrix, OdeEquations, OdeSolverProblem,
    OdeSolverState, Op, Vector, VectorViewMut,
};
use std::ops::MulAssign;

#[derive(Clone)]
pub struct BdfState<V: Vector, M: DenseMatrix<T = V::T, V = V>> {
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

impl<V, M> BdfState<V, M>
where
    V: Vector,
    M: DenseMatrix<T = V::T, V = V>,
{
    pub(crate) const MAX_ORDER: IndexType = 5;

    pub fn initialise_diff_to_first_order(&mut self, has_sens: bool) {
        self.order = 1usize;

        self.diff.column_mut(0).copy_from(&self.y);
        self.diff.column_mut(1).copy_from(&self.dy);
        self.diff.column_mut(1).mul_assign(scale(self.h));
        let nparams = self.s.len();
        if has_sens {
            for i in 0..nparams {
                let sdiff = &mut self.sdiff[i];
                let s = &self.s[i];
                let ds = &self.ds[i];
                sdiff.column_mut(0).copy_from(s);
                sdiff.column_mut(1).copy_from(ds);
                sdiff.column_mut(1).mul_assign(scale(self.h));
            }
        }
    }
}

impl<V, M> OdeSolverState<V> for BdfState<V, M>
where
    V: Vector,
    M: DenseMatrix<T = V::T, V = V>,
{
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        let not_initialised = self.diff.ncols() == 0;
        let nstates = ode_problem.eqn.rhs().nstates();
        let nparams = ode_problem.eqn.rhs().nparams();
        let has_sens = ode_problem.with_sensitivity;
        if not_initialised {
            self.diff = M::zeros(nstates, Self::MAX_ORDER + 3);
            if has_sens {
                self.sdiff = vec![M::zeros(nstates, Self::MAX_ORDER + 3); nparams];
            }
            self.initialise_diff_to_first_order(has_sens);
        }
        Ok(())
    }

    fn new_internal_state(y: V, dy: V, s: Vec<V>, ds: Vec<V>, t: <V>::T, h: <V>::T) -> Self {
        Self {
            order: 1,
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
    fn s_ds_mut(&mut self) -> (&mut [V], &mut [V]) {
        (&mut self.s, &mut self.ds)
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

    fn y_dy_mut(&mut self) -> (&mut V, &mut V) {
        (&mut self.y, &mut self.dy)
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
