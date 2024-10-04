use crate::{
    error::DiffsolError, error::OdeSolverError, ode_solver_error, scalar::IndexType, scale,
    AugmentedOdeEquations, DenseMatrix, OdeEquations, OdeSolverProblem, OdeSolverState, Op, Vector,
    VectorViewMut,
};
use std::ops::MulAssign;

#[derive(Clone)]
pub struct BdfState<V: Vector, M: DenseMatrix<T = V::T, V = V>> {
    pub(crate) order: usize,
    pub(crate) diff: M,
    pub(crate) sdiff: Vec<M>,
    pub(crate) gdiff: M,
    pub(crate) sgdiff: Vec<M>,
    pub(crate) y: V,
    pub(crate) dy: V,
    pub(crate) g: V,
    pub(crate) dg: V,
    pub(crate) s: Vec<V>,
    pub(crate) ds: Vec<V>,
    pub(crate) sg: Vec<V>,
    pub(crate) dsg: Vec<V>,
    pub(crate) t: V::T,
    pub(crate) h: V::T,
    pub(crate) diff_initialised: bool,
    pub(crate) sdiff_initialised: bool,
    pub(crate) gdiff_initialised: bool,
    pub(crate) sgdiff_initialised: bool,
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

    pub fn initialise_gdiff_to_first_order(&mut self) {
        self.gdiff.column_mut(0).copy_from(&self.g);
        self.gdiff.column_mut(1).copy_from(&self.dg);
        self.gdiff.column_mut(1).mul_assign(scale(self.h));
        self.gdiff_initialised = true;
    }

    pub fn initialise_sgdiff_to_first_order(&mut self) {
        let naug = self.sgdiff.len();
        for i in 0..naug {
            let sgdiff = &mut self.sgdiff[i];
            let sg = &self.sg[i];
            let dsg = &self.dsg[i];
            sgdiff.column_mut(0).copy_from(sg);
            sgdiff.column_mut(1).copy_from(dsg);
            sgdiff.column_mut(1).mul_assign(scale(self.h));
        }
        self.sgdiff_initialised = true;
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
        let nout = if let Some(out) = ode_problem.eqn.out() {
            out.nout()
        } else {
            0
        };
        if self.gdiff.nrows() != nout {
            return Err(ode_solver_error!(StateProblemMismatch));
        }
        if !self.diff_initialised {
            self.initialise_diff_to_first_order();
        }
        if !self.gdiff_initialised {
            self.initialise_gdiff_to_first_order();
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
        if !self.sgdiff_initialised {
            self.initialise_sgdiff_to_first_order();
        }
        Ok(())
    }

    fn new_internal_state(
        y: V,
        dy: V,
        g: V,
        dg: V,
        s: Vec<V>,
        ds: Vec<V>,
        sg: Vec<V>,
        dsg: Vec<V>,
        t: <V>::T,
        h: <V>::T,
        naug: usize,
    ) -> Self {
        let nstates = y.len();
        let nout = g.len();
        let diff = M::zeros(nstates, Self::MAX_ORDER + 3);
        let sdiff = vec![M::zeros(nstates, Self::MAX_ORDER + 3); naug];
        let gdiff = M::zeros(nout, Self::MAX_ORDER + 3);
        let sgdiff = vec![M::zeros(nout, Self::MAX_ORDER + 3); naug];
        Self {
            order: 1,
            diff,
            sdiff,
            gdiff,
            sgdiff,
            y,
            dy,
            g,
            dg,
            s,
            ds,
            sg,
            dsg,
            t,
            h,
            diff_initialised: false,
            sdiff_initialised: false,
            gdiff_initialised: false,
            sgdiff_initialised: false,
        }
    }

    fn g_mut(&mut self) -> &mut V {
        &mut self.g
    }
    fn dg_mut(&mut self) -> &mut V {
        &mut self.dg
    }

    fn sg(&self) -> &[V] {
        self.sg.as_slice()
    }
    fn sg_mut(&mut self) -> &mut [V] {
        &mut self.sg
    }
    fn dsg_mut(&mut self) -> &mut [V] {
        &mut self.dsg
    }
    fn dsg(&self) -> &[V] {
        self.dsg.as_slice()
    }

    fn y_g_mut(&mut self) -> (&mut V, &mut V) {
        (&mut self.y, &mut self.g)
    }
    fn g(&self) -> &V {
        &self.g
    }
    fn dg(&self) -> &V {
        &self.dg
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
