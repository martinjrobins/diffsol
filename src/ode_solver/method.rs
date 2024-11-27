use std::cell::RefCell;
use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError},
    matrix::default_solver::DefaultSolver,
    ode_solver_error,
    scalar::Scalar,
    AdjointContext, AdjointEquations, AugmentedOdeEquations, Checkpointing, DefaultDenseMatrix,
    DenseMatrix, Matrix, NonLinearOp, OdeEquations, OdeEquationsAdjoint, OdeEquationsSens,
    OdeSolverProblem, OdeSolverState, Op, SensEquations, StateRef, StateRefMut, Vector,
    VectorViewMut, LinearSolver, HermiteInterpolator,
};

#[derive(Debug, PartialEq)]
pub enum OdeSolverStopReason<T: Scalar> {
    InternalTimestep,
    RootFound(T),
    TstopReached,
}

/// Trait for ODE solver methods. This is the main user interface for the ODE solvers.
///
/// The solver is responsible for stepping the solution (given in the `OdeSolverState`), and interpolating the solution at a given time.
/// However, the solver does not own the state, so the user is responsible for creating and managing the state. If the user
/// wants to change the state, they should call `set_problem` again.
///
/// # Example
///
/// ```
/// use diffsol::{ OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeEquationsImplicit, DefaultSolver };
///
/// fn solve_ode<Eqn>(solver: &mut impl OdeSolverMethod<Eqn>, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Eqn::V
/// where
///    Eqn: OdeEquationsImplicit,
///    Eqn::M: DefaultSolver,
/// {
///     let state = OdeSolverState::new(problem, solver).unwrap();
///     solver.set_problem(state, problem);
///     while solver.state().unwrap().t <= t {
///         solver.step().unwrap();
///     }
///     solver.interpolate(t).unwrap()
/// }
/// ```
pub trait OdeSolverMethod<'a, Eqn: OdeEquations>: Clone
where
    Self: Sized,
    Eqn: 'a,
{
    type State: OdeSolverState<Eqn::V>;

    /// Get the current problem
    fn problem(&self) -> &'a OdeSolverProblem<Eqn>;

    /// Take a checkpoint of the current state of the solver, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem but want to keep this solver active. If you don't need to use this solver again, you can use `take_state` instead.
    /// Note that this will force a reinitialisation of the internal Jacobian for the solver, if it has one.
    fn checkpoint(&mut self) -> Self::State;

    /// Replace the current state of the solver with a new state.
    fn set_state(&mut self, state: Self::State);

    /// Take the current state of the solver, if it exists, returning it to the user. This is useful if you want to use this
    /// state in another solver or problem. Note that this will unset the current problem and solver state, so you will need to call
    /// `set_problem` again before calling `step` or `solve`.
    fn into_state(self) -> Self::State;

    /// Get the current state of the solver
    fn state(&self) -> StateRef<Eqn::V>;

    /// Get a mutable reference to the current state of the solver
    /// Note that calling this will cause the next call to `step` to perform some reinitialisation to take into
    /// account the mutated state, this could be expensive for multi-step methods.
    fn state_mut(&mut self) -> StateRefMut<Eqn::V>;

    /// Step the solution forward by one step, altering the internal state of the solver.
    /// The return value is a `Result` containing the reason for stopping the solver, possible reasons are:
    /// - `InternalTimestep`: The solver has taken a step forward in time, the internal state of the solver is at time self.state().t
    /// - `RootFound(t_root)`: The solver has found a root at time `t_root`. Note that the internal state of the solver is at the internal time step `self.state().t`, *not* at time `t_root`.
    /// - `TstopReached`: The solver has reached the stop time set by [Self::set_stop_time], the internal state of the solver is at time `tstop`, which is the same as `self.state().t`
    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError>;

    /// Set a stop time for the solver. The solver will stop when the internal time reaches this time.
    /// Once it stops, the stop time is unset. If `tstop` is at or before the current internal time, an error is returned.
    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError>;

    /// Interpolate the solution at a given time. This time should be between the current time and the last solver time step
    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError>;

    /// Interpolate the integral of the output function at a given time. This time should be between the current time and the last solver time step
    fn interpolate_out(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError>;

    /// Interpolate the sensitivity vectors at a given time. This time should be between the current time and the last solver time step
    fn interpolate_sens(&self, t: Eqn::T) -> Result<Vec<Eqn::V>, DiffsolError>;

    /// Get the current order of accuracy of the solver (e.g. explict euler method is first-order)
    fn order(&self) -> usize;

    /// Using the provided state, solve the problem up to time `final_time`
    /// Returns a Vec of solution values at timepoints chosen by the solver.
    /// After the solver has finished, the internal state of the solver is at time `final_time`.
    #[allow(clippy::type_complexity)]
    fn solve(
        &mut self,
        final_time: Eqn::T,
    ) -> Result<(<Eqn::V as DefaultDenseMatrix>::M, Vec<Eqn::T>), DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let mut ret_t = Vec::new();
        let mut ret_y = Vec::new();
        fn write_out<Eqn: OdeEquations>(
            p: &OdeSolverProblem<Eqn>,
            ret_y: &mut Vec<Eqn::V>,
            ret_t: &mut Vec<Eqn::T>,
            t: Eqn::T,
            y: &Eqn::V,
            g: &Eqn::V,
        ) {
            ret_t.push(t);
            match p.eqn.out() {
                Some(out) => {
                    if p.integrate_out {
                        ret_y.push(g.clone());
                    } else {
                        ret_y.push(out.call(y, t));
                    }
                }
                None => ret_y.push(y.clone()),
            }
        }

        // do the main loop
        write_out(
            self.problem(),
            &mut ret_y,
            &mut ret_t,
            self.state().t,
            self.state().y,
            self.state().g,
        );
        self.set_stop_time(final_time)?;
        while self.step()? != OdeSolverStopReason::TstopReached {
            write_out(
                self.problem(),
                &mut ret_y,
                &mut ret_t,
                self.state().t,
                self.state().y,
                self.state().g,
            );
        }

        // store the final step
        write_out(
            self.problem(),
            &mut ret_y,
            &mut ret_t,
            self.state().t,
            self.state().y,
            self.state().g,
        );
        let ntimes = ret_t.len();
        let nrows = ret_y[0].len();
        let mut ret_y_matrix = <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nrows, ntimes);
        for (i, y) in ret_y.iter().enumerate() {
            ret_y_matrix.column_mut(i).copy_from(y);
        }
        Ok((ret_y_matrix, ret_t))
    }

    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a Vec of solution values at timepoints given by `t_eval`.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    fn solve_dense(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<<Eqn::V as DefaultDenseMatrix>::M, DiffsolError>
    where
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        let nrows = if self.problem().eqn.out().is_some() {
            self.problem().eqn.out().unwrap().nout()
        } else {
            self.problem().eqn.rhs().nstates()
        };
        let mut ret = <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nrows, t_eval.len());

        // check t_eval is increasing and all values are greater than or equal to the current time
        let t0 = self.state().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        fn write_out<Eqn: OdeEquations>(
            p: &OdeSolverProblem<Eqn>,
            y_out: &mut <Eqn::V as DefaultDenseMatrix>::M,
            t_eval: &[Eqn::T],
            i: usize,
            y: Option<&Eqn::V>,
            g: Option<&Eqn::V>,
        ) where
            Eqn::V: DefaultDenseMatrix,
        {
            let mut y_out = y_out.column_mut(i);
            if let Some(g) = g {
                y_out.copy_from(g);
            } else if let Some(y) = y {
                match p.eqn.out() {
                    Some(out) => y_out.copy_from(&out.call(y, t_eval[i])),
                    None => y_out.copy_from(y),
                }
            }
        }

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().take(t_eval.len() - 1).enumerate() {
            while self.state().t < *t {
                step_reason = self.step()?;
            }
            if self.problem().integrate_out {
                let g = self.interpolate_out(*t)?;

                write_out(self.problem(), &mut ret, t_eval, i, None, Some(&g));
            } else {
                let y = self.interpolate(*t)?;
                write_out(self.problem(), &mut ret, t_eval, i, Some(&y), None);
            }
        }

        // do final step
        while step_reason != OdeSolverStopReason::TstopReached {
            step_reason = self.step()?;
        }
        if self.problem().integrate_out {
            write_out(
                self.problem(),
                &mut ret,
                t_eval,
                t_eval.len() - 1,
                None,
                Some(self.state().g),
            );
        } else {
            write_out(
                self.problem(),
                &mut ret,
                t_eval,
                t_eval.len() - 1,
                Some(self.state().y),
                None,
            );
        }
        Ok(ret)
    }

    /// Using the provided state, solve the forwards and adjoint problem from the current time up to `final_time`.
    /// An output function must be provided and the problem must be setup to integrate this output
    /// function over time. Returns a tuple of `(g, sgs)`, where `g` is the vector of the integral
    /// of the output function from the current time to `final_time`, and `sgs` is a `Vec` where
    /// the ith element is the sensitivities of the ith element of `g` with respect to the
    /// parameters.
    #[allow(clippy::type_complexity)]
    fn solve_adjoint<LS: LinearSolver<Eqn::M>>(
        mut self,
        final_time: Eqn::T,
        max_steps_between_checkpoints: Option<usize>,
    ) -> Result<(Eqn::V, Vec<Eqn::V>), DiffsolError>
    where
        Self: AdjointOdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsAdjoint,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        if self.problem().eqn.out().is_none() {
            return Err(ode_solver_error!(
                Other,
                "Cannot solve adjoint without output function"
            ));
        }
        if !self.problem().integrate_out {
            return Err(ode_solver_error!(
                Other,
                "Cannot solve adjoint without integrating out"
            ));
        }
        let max_steps_between_checkpoints = max_steps_between_checkpoints.unwrap_or(500);
        let t0 = self.state().t;
        let mut ts = vec![t0];
        let mut ys = vec![self.state().y.clone()];
        let mut ydots = vec![self.state().dy.clone()];

        // do the main forward solve, saving checkpoints
        self.set_stop_time(final_time)?;
        let mut nsteps = 0;
        let mut checkpoints = vec![self.checkpoint()];
        while self.step()? != OdeSolverStopReason::TstopReached {
            ts.push(self.state().t);
            ys.push(self.state().y.clone());
            ydots.push(self.state().dy.clone());
            nsteps += 1;
            if nsteps > max_steps_between_checkpoints {
                checkpoints.push(self.checkpoint());
                nsteps = 0;
                ts.clear();
                ys.clear();
                ydots.clear();
            }
        }
        ts.push(self.state().t);
        ys.push(self.state().y.clone());
        ydots.push(self.state().dy.clone());
        checkpoints.push(self.checkpoint());

        // save integrateed out function
        let g = self.state().g.clone();

        // construct the adjoint solver
        let last_segment = HermiteInterpolator::new(ys, ydots, ts);
        let adjoint_aug_eqn = self.adjoint_equations(checkpoints, last_segment)?;
        let mut adjoint_solver = self.default_adjoint_solver::<LS>(adjoint_aug_eqn)?;

        // solve the adjoint problem
        adjoint_solver.set_stop_time(t0).unwrap();
        while adjoint_solver.step()? != OdeSolverStopReason::TstopReached {}

        // correct the adjoint solution for the initial conditions
        let (mut state, aug_eqn) = adjoint_solver.into_state_and_eqn();
        let aug_eqn = aug_eqn.unwrap();
        let state_mut = state.as_mut();
        aug_eqn.correct_sg_for_init(t0, state_mut.s, state_mut.sg);

        // return the solution
        Ok((g, state_mut.sg.to_owned()))
    }

    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a tuple `(y, sens)`, where `y` is a dense matrix of solution values at timepoints given by `t_eval`,
    /// and `sens` is a Vec of dense matrices, the ith element of the Vec are the the sensitivities with respect to the ith parameter.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    #[allow(clippy::type_complexity)]
    fn solve_dense_sensitivities(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<
        (
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<<Eqn::V as DefaultDenseMatrix>::M>,
        ),
        DiffsolError,
    >
    where
        Self: SensitivitiesOdeSolverMethod<'a, Eqn>,
        Eqn: OdeEquationsSens,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        if self.problem().integrate_out {
            return Err(ode_solver_error!(
                Other,
                "Cannot integrate out when solving for sensitivities"
            ));
        }
        let nrows = self.problem().eqn.rhs().nstates();
        let mut ret = <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nrows, t_eval.len());
        let mut ret_sens =
            vec![
                <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nrows, t_eval.len());
                self.problem().eqn.rhs().nparams()
            ];

        // check t_eval is increasing and all values are greater than or equal to the current time
        let t0 = self.state().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().take(t_eval.len() - 1).enumerate() {
            while self.state().t < *t {
                step_reason = self.step()?;
            }
            let y = self.interpolate(*t)?;
            ret.column_mut(i).copy_from(&y);
            let s = self.interpolate_sens(*t)?;
            for (j, s_j) in s.iter().enumerate() {
                ret_sens[j].column_mut(i).copy_from(s_j);
            }
        }

        // do final step
        while step_reason != OdeSolverStopReason::TstopReached {
            step_reason = self.step()?;
        }
        let y = self.state().y;
        ret.column_mut(t_eval.len() - 1).copy_from(y);
        let s = self.state().s;
        for (j, s_j) in s.iter().enumerate() {
            ret_sens[j].column_mut(t_eval.len() - 1).copy_from(s_j);
        }
        Ok((ret, ret_sens))
    }
}

pub trait AugmentedOdeSolverMethod<'a, Eqn, AugmentedEqn>: OdeSolverMethod<'a, Eqn>
where
    Eqn: OdeEquations + 'a,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugmentedEqn>);
}

pub trait SensitivitiesOdeSolverMethod<'a, Eqn>:
    AugmentedOdeSolverMethod<'a, Eqn, SensEquations<'a, Eqn>>
where
    Eqn: OdeEquationsSens + 'a,
{
}

pub trait AdjointOdeSolverMethod<'a, Eqn>: OdeSolverMethod<'a, Eqn>
where
    Eqn: OdeEquationsAdjoint + 'a,
    Self: 'a,
{
    type DefaultAdjointSolver: AugmentedOdeSolverMethod<
        'a,
        Eqn,
        AdjointEquations<'a, Eqn, Self>,
        State = Self::State,
    >;


    fn default_adjoint_solver<LS: LinearSolver<Eqn::M>>(
        self,
        aug_eqn: AdjointEquations<'a, Eqn, Self>,
    ) -> Result<Self::DefaultAdjointSolver, DiffsolError>;

    fn adjoint_equations(
        &self,
        checkpoints: Vec<Self::State>,
        last_segment: HermiteInterpolator<Eqn::V>,
    ) -> Result<
        AdjointEquations<'a, Eqn, Self>,
        DiffsolError,
    >
    where
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
    {
        let problem = self.problem();
        let checkpointer_solver = self.clone();

        // construct checkpointing
        let checkpointer = Checkpointing::new(
            checkpointer_solver,
            checkpoints.len() - 2,
            checkpoints,
            Some(last_segment),
        );

        // construct adjoint equations and problem
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer)));
        let new_augmented_eqn = AdjointEquations::new(problem, context, true);

        Ok(new_augmented_eqn)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::test_models::exponential_decay::{
            exponential_decay_problem, exponential_decay_problem_adjoint,
            exponential_decay_problem_sens,
        },
        scale, NalgebraLU, OdeSolverMethod, Vector,
    };

    #[test]
    fn test_solve() {
        let (problem, _soln) = exponential_decay_problem::<nalgebra::DMatrix<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let k = 0.1;
        let y0 = nalgebra::DVector::from_vec(vec![1.0, 1.0]);
        let expect = |t: f64| &y0 * scale(f64::exp(-k * t));
        let (y, t) = s.solve(10.0).unwrap();
        assert!((t[0] - 0.0).abs() < 1e-10);
        assert!((t[t.len() - 1] - 10.0).abs() < 1e-10);
        for (i, t_i) in t.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&expect(*t_i), &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_solve_integrate_out() {
        let (problem, _soln) = exponential_decay_problem_adjoint::<nalgebra::DMatrix<f64>>();
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let k = 0.1;
        let y0 = nalgebra::DVector::from_vec(vec![1.0, 1.0]);
        let t0 = 0.0;
        let expect = |t: f64| {
            let g = &y0 * scale((f64::exp(-k * t0) - f64::exp(-k * t)) / k);
            nalgebra::DVector::<f64>::from_vec(vec![
                1.0 * g[0] + 2.0 * g[1],
                3.0 * g[0] + 4.0 * g[1],
            ])
        };
        let (y, t) = s.solve(10.0).unwrap();
        for (i, t_i) in t.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&expect(*t_i), &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_dense_solve() {
        let (problem, soln) = exponential_decay_problem::<nalgebra::DMatrix<f64>>(false);
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let y = s.solve_dense(t_eval.as_slice()).unwrap();
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_dense_solve_integrate_out() {
        let (problem, soln) = exponential_decay_problem_adjoint::<nalgebra::DMatrix<f64>>();
        let mut s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let y = s.solve_dense(t_eval.as_slice()).unwrap();
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
    }

    #[test]
    fn test_dense_solve_sensitivities() {
        let (problem, soln) = exponential_decay_problem_sens::<nalgebra::DMatrix<f64>>(false);
        let mut s = problem.bdf_sens::<NalgebraLU<f64>>().unwrap();

        let t_eval = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (y, sens) = s.solve_dense_sensitivities(t_eval.as_slice()).unwrap();
        for (i, soln_pt) in soln.solution_points.iter().enumerate() {
            let y_i = y.column(i).into_owned();
            y_i.assert_eq_norm(&soln_pt.state, &problem.atol, problem.rtol, 15.0);
        }
        for (j, soln_pts) in soln.sens_solution_points.unwrap().iter().enumerate() {
            for (i, soln_pt) in soln_pts.iter().enumerate() {
                let sens_i = sens[j].column(i).into_owned();
                sens_i.assert_eq_norm(
                    &soln_pt.state,
                    problem.sens_atol.as_ref().unwrap(),
                    problem.sens_rtol.unwrap(),
                    15.0,
                );
            }
        }
    }

    #[test]
    fn test_solve_adjoint() {
        let (problem, soln) = exponential_decay_problem_adjoint::<nalgebra::DMatrix<f64>>();
        let s = problem.bdf::<NalgebraLU<f64>>().unwrap();

        let final_time = soln.solution_points[soln.solution_points.len() - 1].t;
        let (g, gs_adj) = s.solve_adjoint::<NalgebraLU<f64>>(final_time, None).unwrap();
        g.assert_eq_norm(
            &soln.solution_points[soln.solution_points.len() - 1].state,
            problem.out_atol.as_ref().unwrap(),
            problem.out_rtol.unwrap(),
            15.0,
        );
        for (j, soln_pts) in soln.sens_solution_points.unwrap().iter().enumerate() {
            gs_adj[j].assert_eq_norm(
                &soln_pts[0].state,
                problem.out_atol.as_ref().unwrap(),
                problem.out_rtol.unwrap(),
                15.0,
            );
        }
    }
}
