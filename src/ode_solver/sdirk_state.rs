use crate::{error::DiffsolError, OdeEquations, OdeSolverProblem, OdeSolverState, Vector};

#[derive(Clone)]
pub struct SdirkState<V: Vector> {
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
}

impl<V> SdirkState<V> where V: Vector {}

impl<V> OdeSolverState<V> for SdirkState<V>
where
    V: Vector,
{
    fn set_problem<Eqn: OdeEquations>(
        &mut self,
        _ode_problem: &OdeSolverProblem<Eqn>,
    ) -> Result<(), DiffsolError> {
        Ok(())
    }

    fn set_augmented_problem<Eqn: OdeEquations, AugmentedEqn: crate::AugmentedOdeEquations<Eqn>>(
        &mut self,
        _ode_problem: &OdeSolverProblem<Eqn>,
        _augmented_eqn: &AugmentedEqn,
    ) -> Result<(), DiffsolError> {
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
        _naug: usize,
    ) -> Self {
        Self {
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
        }
    }

    fn sg(&self) -> &[V] {
        self.sg.as_slice()
    }
    fn sg_mut(&mut self) -> &mut [V] {
        &mut self.sg
    }
    fn dsg(&self) -> &[V] {
        self.dsg.as_slice()
    }
    fn dsg_mut(&mut self) -> &mut [V] {
        &mut self.dsg
    }

    fn g(&self) -> &V {
        &self.g
    }
    fn g_mut(&mut self) -> &mut V {
        &mut self.g
    }
    fn dg(&self) -> &V {
        &self.dg
    }
    fn dg_mut(&mut self) -> &mut V {
        &mut self.dg
    }
    fn y_g_mut(&mut self) -> (&mut V, &mut V) {
        (&mut self.y, &mut self.g)
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
