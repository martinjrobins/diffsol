use anyhow::{Context, Result};
use std::rc::Rc;

use crate::{vector::Vector, NonLinearOp, OdeEquations, SensEquations};

pub struct OdeSolverProblem<Eqn: OdeEquations> {
    pub eqn: Rc<Eqn>,
    pub rtol: Eqn::T,
    pub atol: Rc<Eqn::V>,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
    pub eqn_sens: Option<Rc<SensEquations<Eqn>>>,
}

// impl clone
impl<Eqn: OdeEquations> Clone for OdeSolverProblem<Eqn> {
    fn clone(&self) -> Self {
        Self {
            eqn: self.eqn.clone(),
            rtol: self.rtol,
            atol: self.atol.clone(),
            t0: self.t0,
            h0: self.h0,
            eqn_sens: self.eqn_sens.clone(),
        }
    }
}

impl<Eqn: OdeEquations> OdeSolverProblem<Eqn> {
    pub fn default_rtol() -> Eqn::T {
        Eqn::T::from(1e-6)
    }
    pub fn default_atol(nstates: usize) -> Eqn::V {
        Eqn::V::from_element(nstates, Eqn::T::from(1e-6))
    }
    pub fn new(
        eqn: Eqn,
        rtol: Eqn::T,
        atol: Eqn::V,
        t0: Eqn::T,
        h0: Eqn::T,
        with_sensitivity: bool,
    ) -> Result<Self> {
        let eqn = Rc::new(eqn);
        let atol = Rc::new(atol);
        if with_sensitivity && !eqn.rhs().has_sens() {
            anyhow::bail!("Sensitivity requested but equations do not support it");
        }
        let eqn_sens = if with_sensitivity {
            Some(Rc::new(SensEquations::new(&eqn)))
        } else {
            None
        };
        Ok(Self {
            eqn,
            rtol,
            atol,
            t0,
            h0,
            eqn_sens,
        })
    }

    pub fn set_params(&mut self, p: Eqn::V) -> Result<()> {
        let eqn = Rc::get_mut(&mut self.eqn).context("Failed to get mutable reference to equations, is there a solver created with this problem?")?;
        eqn.set_params(p);
        Ok(())
    }
}

pub struct OdeSolverSolutionPoint<V: Vector> {
    pub state: V,
    pub t: V::T,
}

pub struct OdeSolverSolution<V: Vector> {
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
}

impl<V: Vector> OdeSolverSolution<V> {
    pub fn push(&mut self, state: V, t: V::T) {
        // find the index to insert the new point keeping the times sorted
        let index = self
            .solution_points
            .iter()
            .position(|x| x.t > t)
            .unwrap_or(self.solution_points.len());
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
    }
}

impl<V: Vector> Default for OdeSolverSolution<V> {
    fn default() -> Self {
        Self {
            solution_points: Vec::new(),
        }
    }
}
