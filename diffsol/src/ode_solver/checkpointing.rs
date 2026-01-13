use num_traits::FromPrimitive;
use std::cell::RefCell;

use crate::{
    error::DiffsolError, other_error, OdeEquations, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, Scalar, Vector,
};
use num_traits::{abs, One};

/// Hermite interpolator for ODE solution trajectories.
///
/// This interpolator uses cubic Hermite interpolation based on solution values and
/// derivatives at discrete time points. It provides smooth interpolation between
/// checkpoints and is used internally by the checkpointing system for adjoint
/// sensitivity analysis.
#[derive(Clone)]
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
{
    /// Create a new Hermite interpolator with the given solution data.
    ///
    /// # Arguments
    /// - `ys`: Vector of solution values at each time point
    /// - `ydots`: Vector of solution derivatives (dy/dt) at each time point
    /// - `ts`: Vector of time points (must be sorted)
    ///
    /// # Notes
    /// All three vectors must have the same length. The time points should be
    /// sorted in increasing or decreasing order for proper interpolation.
    pub fn new(ys: Vec<V>, ydots: Vec<V>, ts: Vec<V::T>) -> Self {
        HermiteInterpolator { ys, ydots, ts }
    }

    /// Get the last time point in the interpolator.
    ///
    /// # Returns
    /// The last time value, or `None` if the interpolator is empty.
    pub fn last_t(&self) -> Option<V::T> {
        if self.ts.is_empty() {
            return None;
        }
        Some(self.ts[self.ts.len() - 1])
    }

    /// Get the last time step size in the interpolator.
    ///
    /// # Returns
    /// The difference between the last two time points, or `None` if there are
    /// fewer than two time points.
    pub fn last_h(&self) -> Option<V::T> {
        if self.ts.len() < 2 {
            return None;
        }
        Some(self.ts[self.ts.len() - 1] - self.ts[self.ts.len() - 2])
    }

    /// Reset the interpolator by solving the ODE between two checkpointed states.
    ///
    /// This method clears the current interpolation data and re-solves the ODE
    /// from `state0` to `state1`, storing all intermediate solution points.
    ///
    /// # Arguments
    /// - `solver`: The ODE solver to use for integration
    /// - `state0`: The initial state (starting point)
    /// - `state1`: The final state (target point)
    ///
    /// # Returns
    /// Ok(()) on success, or an error if the solver fails.
    pub fn reset<'a, Eqn, Method, State>(
        &mut self,
        solver: &mut Method,
        state0: &State,
        state1: &State,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<V = V, T = V::T> + 'a,
        Method: OdeSolverMethod<'a, Eqn, State = State>,
        State: OdeSolverState<V>,
    {
        let state0_ref = state0.as_ref();
        let state1_ref = state1.as_ref();
        self.ys.clear();
        self.ydots.clear();
        self.ts.clear();
        self.ys.push(state0_ref.y.clone());
        self.ydots.push(state0_ref.dy.clone());
        self.ts.push(state0_ref.t);

        solver.set_state(state0.clone());
        while solver.state().t < state1_ref.t {
            solver.step()?;
            self.ys.push(solver.state().y.clone());
            self.ydots.push(solver.state().dy.clone());
            self.ts.push(solver.state().t);
        }
        Ok(())
    }

    /// Interpolate the solution at a given time point.
    ///
    /// Uses cubic Hermite interpolation to compute the solution value at time `t`.
    ///
    /// # Arguments
    /// - `t`: The time at which to interpolate
    /// - `y`: Output vector to store the interpolated solution
    ///
    /// # Returns
    /// `Some(())` if the interpolation succeeded (t is within range), `None` if
    /// t is outside the range of stored time points.
    pub fn interpolate(&self, t: V::T, y: &mut V) -> Option<()> {
        if t < self.ts[0] || t > self.ts[self.ts.len() - 1] {
            return None;
        }
        if t == self.ts[0] {
            y.copy_from(&self.ys[0]);
            return Some(());
        }
        let idx = self
            .ts
            .iter()
            .position(|&t0| t0 > t)
            .unwrap_or(self.ts.len() - 1);
        let t0 = self.ts[idx - 1];
        let t1 = self.ts[idx];
        let h = t1 - t0;
        let theta = (t - t0) / h;
        let u0 = &self.ys[idx - 1];
        let u1 = &self.ys[idx];
        let f0 = &self.ydots[idx - 1];
        let f1 = &self.ydots[idx];

        y.copy_from(u0);
        y.axpy(V::T::one(), u1, -V::T::one());
        y.axpy(
            h * (theta - V::T::from_f64(1.0).unwrap()),
            f0,
            V::T::one() - V::T::from_f64(2.0).unwrap() * theta,
        );
        y.axpy(h * theta, f1, V::T::one());
        y.axpy(
            V::T::from_f64(1.0).unwrap() - theta,
            u0,
            theta * (theta - V::T::from_f64(1.0).unwrap()),
        );
        y.axpy(theta, u1, V::T::one());
        Some(())
    }
}

/// Checkpointing system for adjoint sensitivity analysis.
///
/// This struct manages checkpoints of an ODE solution for use in adjoint sensitivity
/// computation. It stores solution states at discrete checkpoints and uses Hermite
/// interpolation between them to provide solution values at arbitrary time points
/// during the backward adjoint solve.
///
/// The checkpointing system uses a two-level interpolation strategy:
/// 1. Checkpoints: Discrete saved states of the ODE solution
/// 2. Segments: Hermite interpolators that densely sample between adjacent checkpoints
///
/// When interpolating at a time point, the system first checks if it's in the current
/// segment. If not, it re-solves the ODE between the appropriate checkpoints to create
/// a new segment, then interpolates within that segment.
///
/// # Type Parameters
/// - `Eqn`: The ODE equations type (inferred from the solver)
/// - `Method`: The forward solver method type (inferred from the solver)
pub struct Checkpointing<'a, Eqn, Method>
where
    Method: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations,
{
    checkpoints: Vec<Method::State>,
    segment: RefCell<HermiteInterpolator<Eqn::V>>,
    previous_segment: RefCell<Option<HermiteInterpolator<Eqn::V>>>,
    solver: RefCell<Method>,
}

impl<'a, Eqn, Method> Clone for Checkpointing<'a, Eqn, Method>
where
    Method: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations,
{
    fn clone(&self) -> Self {
        Checkpointing {
            checkpoints: self.checkpoints.clone(),
            segment: RefCell::new(self.segment.borrow().clone()),
            previous_segment: RefCell::new(self.previous_segment.borrow().clone()),
            solver: RefCell::new(self.solver.borrow().clone()),
        }
    }
}

impl<'a, Eqn, Method> Checkpointing<'a, Eqn, Method>
where
    Method: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations,
{
    /// Create a new checkpointing system.
    ///
    /// # Arguments
    /// - `solver`: The forward solver to use for re-solving segments
    /// - `start_idx`: Index of the checkpoint to start from (must be < checkpoints.len() - 1)
    /// - `checkpoints`: Vector of saved solution states (must have at least 2 elements)
    /// - `segment`: Optional pre-computed Hermite interpolator for the initial segment given by `start_idx`.
    ///   If `None`, the segment between `checkpoints[start_idx]` and `checkpoints[start_idx+1]`
    ///   will be computed automatically.
    ///
    /// # Panics
    /// Panics if `checkpoints.len() < 2` or if `start_idx >= checkpoints.len() - 1`.

    pub fn new(
        mut solver: Method,
        start_idx: usize,
        checkpoints: Vec<Method::State>,
        segment: Option<HermiteInterpolator<Eqn::V>>,
    ) -> Self {
        if checkpoints.len() < 2 {
            panic!("Checkpoints must have at least 2 elements");
        }
        if start_idx >= checkpoints.len() - 1 {
            panic!("start_idx must be less than checkpoints.len() - 1");
        }
        let segment = segment.unwrap_or_else(|| {
            let mut segment = HermiteInterpolator::default();
            segment
                .reset(
                    &mut solver,
                    &checkpoints[start_idx],
                    &checkpoints[start_idx + 1],
                )
                .unwrap();
            segment
        });
        let segment = RefCell::new(segment);
        let previous_segment = RefCell::new(None);
        let solver = RefCell::new(solver);
        Checkpointing {
            checkpoints,
            segment,
            previous_segment,
            solver,
        }
    }

    /// Get the last (most recent) time point in the current segment.
    ///
    /// # Returns
    /// The last time value in the current interpolation segment.
    ///
    /// # Panics
    /// Panics if the segment is empty (should never happen with valid construction).
    pub fn last_t(&self) -> Eqn::T {
        self.segment
            .borrow()
            .last_t()
            .expect("segment should not be empty")
    }

    /// Get the last time step size in the current segment.
    ///
    /// # Returns
    /// The difference between the last two time points in the current segment,
    /// or `None` if the segment has fewer than two points.
    pub fn last_h(&self) -> Option<Eqn::T> {
        self.segment.borrow().last_h()
    }

    /// Get a reference to the ODE problem associated with this checkpointing system.
    ///
    /// # Returns
    /// A reference to the `OdeSolverProblem` that defines the ODE equations,
    /// tolerances, and other solver parameters.
    pub fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.solver.borrow().problem()
    }

    /// Interpolate the solution at a given time point.
    ///
    /// This method provides the forward solution value at time `t` for use during
    /// the backward adjoint solve. It first checks if `t` is within the current
    /// or previous segment. If not, it identifies the appropriate checkpoint interval,
    /// re-solves the ODE between those checkpoints to create a new segment, and then
    /// interpolates within that segment.
    ///
    /// # Arguments
    /// - `t`: The time at which to interpolate the solution
    /// - `y`: Output vector to store the interpolated solution
    ///
    /// # Returns
    /// - `Ok(())` if interpolation succeeded
    /// - `Err(...)` if `t` is outside the range of checkpoints (beyond roundoff tolerance)
    ///   or if re-solving the ODE fails
    ///
    /// # Notes
    /// Small deviations from the checkpoint range (within roundoff error) are automatically
    /// snapped to the nearest checkpoint boundary.
    pub fn interpolate(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        {
            let segment = self.segment.borrow();
            if segment.interpolate(t, y).is_some() {
                return Ok(());
            }
        }

        {
            let previous_segment = self.previous_segment.borrow();
            if let Some(previous_segment) = previous_segment.as_ref() {
                if previous_segment.interpolate(t, y).is_some() {
                    return Ok(());
                }
            }
        }

        // if t is before first segment or after last segment, return error
        let h = self.last_h().unwrap_or(Eqn::T::one());
        let troundoff = Eqn::T::from_f64(100.0).unwrap() * Eqn::T::EPSILON * (abs(t) + abs(h));
        if t < self.checkpoints[0].as_ref().t - troundoff
            || t > self.checkpoints[self.checkpoints.len() - 1].as_ref().t + troundoff
        {
            return Err(other_error!("t is outside of the checkpoints"));
        }

        // snap t to nearest checkpoint if outside of range
        let t = if t < self.checkpoints[0].as_ref().t {
            self.checkpoints[0].as_ref().t
        } else if t > self.checkpoints[self.checkpoints.len() - 1].as_ref().t {
            self.checkpoints[self.checkpoints.len() - 1].as_ref().t
        } else {
            t
        };

        // else find idx of segment
        let idx = self
            .checkpoints
            .iter()
            .skip(1)
            .position(|state| state.as_ref().t > t)
            .expect("t is not in checkpoints");
        if self.previous_segment.borrow().is_none() {
            self.previous_segment
                .replace(Some(HermiteInterpolator::default()));
        }
        let mut solver = self.solver.borrow_mut();
        let mut previous_segment = self.previous_segment.borrow_mut();
        let mut segment = self.segment.borrow_mut();
        previous_segment.as_mut().unwrap().reset(
            &mut *solver,
            &self.checkpoints[idx],
            &self.checkpoints[idx + 1],
        )?;
        std::mem::swap(&mut *segment, previous_segment.as_mut().unwrap());
        segment.interpolate(t, y).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::robertson::robertson, Context, NalgebraLU, OdeEquations,
        OdeSolverMethod, Op, Vector,
    };

    use super::{Checkpointing, HermiteInterpolator};

    #[test]
    fn test_checkpointing() {
        type M = NalgebraMat<f64>;
        type LS = NalgebraLU<f64>;
        let (problem, soln) = robertson::<M>(false);
        let t_final = soln.solution_points.last().unwrap().t;
        let n_steps = 30;
        let mut solver = problem.bdf::<LS>().unwrap();
        let mut checkpoints = vec![solver.checkpoint()];
        let mut i = 0;
        let mut ys = Vec::new();
        let mut ts = Vec::new();
        let mut ydots = Vec::new();
        while solver.state().t < t_final {
            ts.push(solver.state().t);
            ys.push(solver.state().y.clone());
            ydots.push(solver.state().dy.clone());
            solver.step().unwrap();
            i += 1;
            if i % n_steps == 0 && solver.state().t < t_final {
                checkpoints.push(solver.checkpoint());
                ts.clear();
                ys.clear();
                ydots.clear();
            }
        }
        checkpoints.push(solver.checkpoint());
        let segment = HermiteInterpolator::new(ys, ydots, ts);
        let checkpointer =
            Checkpointing::new(solver, checkpoints.len() - 2, checkpoints, Some(segment));
        let mut y = problem.context().vector_zeros(problem.eqn.rhs().nstates());
        for point in soln.solution_points.iter().rev() {
            checkpointer.interpolate(point.t, &mut y).unwrap();
            y.assert_eq_norm(&point.state, &problem.atol, problem.rtol, 10.0);
        }
    }
}
