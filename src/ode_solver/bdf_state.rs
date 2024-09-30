use crate::{
    error::DiffsolError, ode_solver_error, scalar::IndexType, scale, AugmentedOdeEquations, DenseMatrix, OdeEquations, OdeSolverProblem, OdeSolverState, Op, Vector, VectorViewMut, error::OdeSolverError,
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
    pub(crate) diff_initialised: bool,
    pub(crate) sdiff_initialised: bool,
}

impl<V, M> BdfState<V, M>
where
    V: Vector,
    M: DenseMatrix<T = V::T, V = V>,
{
    pub(crate) const MAX_ORDER: IndexType = 5;

    pub fn initialise_diff_to_first_order(&mut self) {
        self.order = 1usize;
        self.diff.column_mut(0).copy_from(&self.y);
        self.diff.column_mut(1).copy_from(&self.dy);
        self.diff.column_mut(1).mul_assign(scale(self.h));
        self.diff_initialised = true;
    }
    
    pub fn initialise_sdiff_to_first_order(&mut self) {
        let naug = self.sdiff.len();
        for i in 0..naug {
            let sdiff = &mut self.sdiff[i];
            let s = &self.s[i];
            let ds = &self.ds[i];
            sdiff.column_mut(0).copy_from(s);
            sdiff.column_mut(1).copy_from(ds);
            sdiff.column_mut(1).mul_assign(scale(self.h));
        }
        self.sdiff_initialised = true;
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
        let nstates = ode_problem.eqn.rhs().nstates();
        if self.diff.nrows() != nstates {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.diff_initialised {
            self.initialise_diff_to_first_order();
        }
        Ok(())
    }
    
    fn set_augmented_problem<Eqn: OdeEquations, AugmentedEqn: AugmentedOdeEquations<Eqn>>(
        &mut self,
        ode_problem: &OdeSolverProblem<Eqn>,
        augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError> {
        let naug = augmented_eqn.max_index();
        let nstates = ode_problem.eqn.rhs().nstates();
        if self.sdiff.len() != naug || self.sdiff[0].nrows() != nstates {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.sdiff_initialised {
            self.initialise_sdiff_to_first_order();
        }
        Ok(())
    }



    fn new_internal_state(y: V, dy: V, s: Vec<V>, ds: Vec<V>, t: <V>::T, h: <V>::T, naug: usize) -> Self {
        let nstates = y.len();
        let diff = M::zeros(nstates, Self::MAX_ORDER + 3);
        let sdiff = vec![M::zeros(nstates, Self::MAX_ORDER + 3); naug];
        Self {
            order: 1,
            diff,
            sdiff,
            y,
            dy,
            s,
            ds,
            t,
            h,
            diff_initialised: false,
            sdiff_initialised: false,
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
