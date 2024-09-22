use crate::{error::DiffsolError, other_error, scale, vector::VectorRef, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Vector};

pub struct HermiteInterpolator<V> 
where 
    V: Vector, 
{
    ys: Vec<V>,
    ydots: Vec<V>,
    ts: Vec<V::T>,
}

impl<V> Default for HermiteInterpolator<V> 
where 
    V: Vector,
{
    fn default() -> Self {
        HermiteInterpolator {
            ys: Vec::new(),
            ydots: Vec::new(),
            ts: Vec::new(),
        }
    }
}

impl<V> HermiteInterpolator<V>
where 
    V: Vector,
    for<'a> &'a V: VectorRef<V>,
{
    pub fn reset<Eqn, Method, State>(&mut self, problem: &OdeSolverProblem<Eqn>, solver: &mut Method, state0: &State, state1: &State) -> Result<(), DiffsolError> 
    where 
        Eqn: OdeEquations<V = V, T = V::T>,
        Method: OdeSolverMethod<Eqn, State = State>,
        State: OdeSolverState<V>,
    {

        self.ys.clear();
        self.ydots.clear();
        self.ts.clear();
        self.ys.push(state0.y().clone());
        self.ydots.push(state0.dy().clone());
        self.ts.push(state0.t());

        solver.set_problem(state0.clone(), problem)?;
        while solver.state().as_ref().unwrap().t() < state1.t() {
            solver.step()?;
            self.ys.push(solver.state().as_ref().unwrap().y().clone());
            self.ydots.push(solver.state().as_ref().unwrap().dy().clone());
            self.ts.push(solver.state().as_ref().unwrap().t());
        }
        Ok(())

    }
        
    pub fn interpolate(&self, t: V::T) -> V {
        let idx = self.ts.iter().position(|&t0| t0 > t).unwrap_or(self.ts.len() - 1);
        let t0 = self.ts[idx - 1];
        let t1 = self.ts[idx];
        let h = t1 - t0;
        let theta = (t - t0) / h;
        let u0 = &self.ys[idx - 1];
        let u1 = &self.ys[idx];
        let f0 = &self.ydots[idx - 1];
        let f1 = &self.ydots[idx];

        u0 * scale(V::T::from(1.0) - theta)
            + u1 * scale(theta)
            + ((u1 - u0) * scale(V::T::from(1.0) - V::T::from(2.0) * theta)
                + f0 * scale(h * (theta - V::T::from(1.0)))
                + f1 * scale(h * theta))
                * scale(theta * (theta - V::T::from(1.0)))

    }
}

pub struct Checkpointing<Eqn, Method> 
where 
    Method: OdeSolverMethod<Eqn>,
    Eqn: OdeEquations,
{
    checkpoints: Vec<Method::State>,
    segment_start_idx: usize,
    segment: HermiteInterpolator<Eqn::V>,
    solver: Method,
    problem: OdeSolverProblem<Eqn>,
}

impl<Eqn, Method> Checkpointing<Eqn, Method> 
where 
    Method: OdeSolverMethod<Eqn>,
    Eqn: OdeEquations,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
{
    pub fn new(problem: &OdeSolverProblem<Eqn>, solver: Method, checkpoints: Vec<Method::State>) -> Self {
        Checkpointing {
            segment_start_idx: checkpoints.len() - 1,
            checkpoints,
            segment: HermiteInterpolator::default(),
            solver,
            problem: problem.clone(),
        }
    }
    
    pub fn interpolate(&mut self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        // if t is in current segment, interpolate
        if t >= self.checkpoints[self.segment_start_idx].t() && t <= self.checkpoints[self.segment_start_idx + 1].t() {
            return Ok(self.segment.interpolate(t));
        }

        // if t is before first segment or after last segment, return error
        if t < self.checkpoints[0].t() || t > self.checkpoints[self.checkpoints.len() - 1].t() {
            return Err(other_error!("t is outside of the checkpoints"));
        }

        // else find idx of segment
        let idx = self.checkpoints.iter().skip(1).position(|state| state.t() > t).expect("t is not in checkpoints");
        self.segment.reset(&self.problem, &mut self.solver, &self.checkpoints[idx], &self.checkpoints[idx + 1])?;
        Ok(self.segment.interpolate(t))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::{ode_solver::test_models::robertson::robertson, Bdf, BdfState, OdeSolverMethod, OdeSolverState, Vector};

    use super::Checkpointing;

    #[test]
    fn test_checkpointing() {
        let mut solver = Bdf::default();
        let (problem, soln) = robertson::<DMatrix<f64>>(false);
        let t_final = soln.solution_points.last().unwrap().t;
        let n_steps = 30;
        let state0: BdfState<_, _> = OdeSolverState::new(&problem, &solver).unwrap();
        solver.set_problem(state0.clone(), &problem).unwrap();
        let mut checkpoints = vec![state0];
        let mut i = 0;
        while solver.state().as_ref().unwrap().t() < t_final {
            solver.step().unwrap();
            i += 1;
            if i % n_steps == 0 {
                checkpoints.push(solver.checkpoint().unwrap());
            }
        }
        if i % n_steps != 0 {
            checkpoints.push(solver.checkpoint().unwrap());
        }
        let mut checkpointer = Checkpointing::new(&problem, solver, checkpoints);
        for point in soln.solution_points.iter().rev() {
            let interp = checkpointer.interpolate(point.t).unwrap();
            interp.assert_eq_norm(&point.state, &problem.atol, problem.rtol, 10.0);
        }
    }
}
