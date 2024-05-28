use anyhow::{Context, Result};
use std::rc::Rc;

use crate::{vector::Vector, ConstantOp, LinearOp, NonLinearOp, OdeEquations, SensEquations};

pub struct OdeSolverProblem<Eqn: OdeEquations> {
    pub eqn: Rc<Eqn>,
    pub rtol: Eqn::T,
    pub atol: Rc<Eqn::V>,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
    pub eqn_sens: Option<Rc<SensEquations<Eqn>>>,
    pub sens_error_control: bool,
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
            sens_error_control: self.sens_error_control,
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
        sens_error_control: bool,
    ) -> Result<Self> {
        let eqn = Rc::new(eqn);
        let atol = Rc::new(atol);
        let mass_has_sens = if let Some(mass) = eqn.mass() {
            mass.has_sens()
        } else {
            true
        };
        let eqn_has_sens = eqn.rhs().has_sens() && eqn.init().has_sens() && mass_has_sens;
        if with_sensitivity && !eqn_has_sens {
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
            sens_error_control,
        })
    }

    pub fn set_params(&mut self, p: Eqn::V) -> Result<()> {
        let eqn = Rc::get_mut(&mut self.eqn).context("Failed to get mutable reference to equations, is there a solver created with this problem?")?;
        eqn.set_params(p);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OdeSolverSolutionPoint<V: Vector> {
    pub state: V,
    pub t: V::T,
}

pub struct OdeSolverSolution<V: Vector> {
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
    pub sens_solution_points: Option<Vec<Vec<OdeSolverSolutionPoint<V>>>>,
}

impl<V: Vector> OdeSolverSolution<V> {
    pub fn push(&mut self, state: V, t: V::T) {
        // find the index to insert the new point keeping the times sorted
        let index = self.get_index(t);
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
    }
    fn get_index(&self, t: V::T) -> usize {
        self.solution_points
            .iter()
            .position(|x| x.t > t)
            .unwrap_or(self.solution_points.len())
    }
    pub fn push_sens(&mut self, state: V, t: V::T, sens: &[V]) {
        // find the index to insert the new point keeping the times sorted
        let index = self.get_index(t);
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
        // if the sensitivity solution is not initialized, initialize it
        if self.sens_solution_points.is_none() {
            self.sens_solution_points = Some(vec![vec![]; sens.len()]);
        }
        // insert the new sensitivity point at that index
        for (i, s) in sens.iter().enumerate() {
            self.sens_solution_points.as_mut().unwrap()[i].insert(
                index,
                OdeSolverSolutionPoint {
                    state: s.clone(),
                    t,
                },
            );
        }
    }
}

impl<V: Vector> Default for OdeSolverSolution<V> {
    fn default() -> Self {
        Self {
            solution_points: Vec::new(),
            sens_solution_points: None,
        }
    }
}
