use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    vector::Vector,
    OdeEquations,
};

pub struct OdeSolverProblem<Eqn: OdeEquations> {
    pub eqn: Rc<Eqn>,
    pub rtol: Eqn::T,
    pub atol: Rc<Eqn::V>,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
    pub integrate_out: bool,
    pub sens_rtol: Option<Eqn::T>,
    pub sens_atol: Option<Rc<Eqn::V>>,
    pub out_rtol: Option<Eqn::T>,
    pub out_atol: Option<Rc<Eqn::V>>,
    pub param_rtol: Option<Eqn::T>,
    pub param_atol: Option<Rc<Eqn::V>>,
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
            integrate_out: self.integrate_out,
            out_atol: self.out_atol.clone(),
            out_rtol: self.out_rtol,
            param_atol: self.param_atol.clone(),
            param_rtol: self.param_rtol,
            sens_atol: self.sens_atol.clone(),
            sens_rtol: self.sens_rtol,
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
    pub fn output_in_error_control(&self) -> bool {
        self.integrate_out
            && self.eqn.out().is_some()
            && self.out_rtol.is_some()
            && self.out_atol.is_some()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        eqn: Rc<Eqn>,
        rtol: Eqn::T,
        atol: Rc<Eqn::V>,
        sens_rtol: Option<Eqn::T>,
        sens_atol: Option<Rc<Eqn::V>>,
        out_rtol: Option<Eqn::T>,
        out_atol: Option<Rc<Eqn::V>>,
        param_rtol: Option<Eqn::T>,
        param_atol: Option<Rc<Eqn::V>>,
        t0: Eqn::T,
        h0: Eqn::T,
        integrate_out: bool,
    ) -> Result<Self, DiffsolError> {
        Ok(Self {
            eqn,
            rtol,
            atol,
            out_atol,
            out_rtol,
            param_atol,
            param_rtol,
            sens_atol,
            sens_rtol,
            t0,
            h0,
            integrate_out,
        })
    }

    pub fn set_params(&mut self, p: Eqn::V) -> Result<(), DiffsolError> {
        let eqn =
            Rc::get_mut(&mut self.eqn).ok_or(ode_solver_error!(FailedToGetMutableReference))?;
        eqn.set_params(Rc::new(p));
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
    pub rtol: V::T,
    pub atol: V,
    pub negative_time: bool,
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
        if self.negative_time {
            self.solution_points
                .iter()
                .position(|x| x.t < t)
                .unwrap_or(self.solution_points.len())
        } else {
            self.solution_points
                .iter()
                .position(|x| x.t > t)
                .unwrap_or(self.solution_points.len())
        }
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
            rtol: V::T::from(1e-6),
            atol: V::from_element(1, V::T::from(1e-6)),
            negative_time: false,
        }
    }
}
