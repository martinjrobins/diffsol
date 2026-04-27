use std::{cell::Ref, marker::PhantomData};

use crate::{
    error::DiffsolError, OdeEquations, OdeSolverConfig, OdeSolverConfigMut, OdeSolverConfigRef,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason, StateRef, StateRefMut,
};

pub struct NoCheckpointingSolverConfig<T>(PhantomData<T>);

impl<T> Clone for NoCheckpointingSolverConfig<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for NoCheckpointingSolverConfig<T> {}

impl<T> Default for NoCheckpointingSolverConfig<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T> OdeSolverConfig<T> for NoCheckpointingSolverConfig<T> {
    fn as_base_ref(&self) -> OdeSolverConfigRef<'_, T> {
        panic!("checkpoint replay solver config is unavailable")
    }

    fn as_base_mut(&mut self) -> OdeSolverConfigMut<'_, T> {
        panic!("checkpoint replay solver config is unavailable")
    }
}

pub struct NoCheckpointingSolver<Eqn, State>(PhantomData<(Eqn, State)>)
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>;

impl<Eqn, State> Clone for NoCheckpointingSolver<Eqn, State>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
{
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<Eqn, State> Default for NoCheckpointingSolver<Eqn, State>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'a, Eqn, State> OdeSolverMethod<'a, Eqn> for NoCheckpointingSolver<Eqn, State>
where
    Eqn: OdeEquations + 'a,
    State: OdeSolverState<Eqn::V>,
{
    type State = State;
    type Config = NoCheckpointingSolverConfig<Eqn::T>;

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn checkpoint(&mut self) -> Self::State {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state_clone(&self) -> Self::State {
        panic!("checkpoint replay solver is unavailable")
    }

    fn set_state(&mut self, _state: Self::State) {
        panic!("checkpoint replay solver is unavailable")
    }

    fn into_state(self) -> Self::State {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state(&self) -> StateRef<'_, Eqn::V> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn config(&self) -> &Self::Config {
        panic!("checkpoint replay solver config is unavailable")
    }

    fn config_mut(&mut self) -> &mut Self::Config {
        panic!("checkpoint replay solver config is unavailable")
    }

    fn jacobian(&self) -> Option<Ref<'_, Eqn::M>> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn mass(&self) -> Option<Ref<'_, Eqn::M>> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn set_stop_time(&mut self, _tstop: Eqn::T) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_inplace(&self, _t: Eqn::T, _y: &mut Eqn::V) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_dy_inplace(&self, _t: Eqn::T, _dy: &mut Eqn::V) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_out_inplace(&self, _t: Eqn::T, _g: &mut Eqn::V) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn interpolate_sens_inplace(
        &self,
        _t: Eqn::T,
        _sens: &mut [Eqn::V],
    ) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn state_mut_back(&mut self, _t: Eqn::T) -> Result<(), DiffsolError> {
        panic!("checkpoint replay solver is unavailable")
    }

    fn order(&self) -> usize {
        panic!("checkpoint replay solver is unavailable")
    }
}

#[cfg(test)]
mod tests {
    use std::{any::Any, panic};

    use super::{NoCheckpointingSolver, NoCheckpointingSolverConfig};
    use crate::{
        ode_equations::test_models::exponential_decay::exponential_decay_problem, BdfState,
        ConstantOp, DefaultDenseMatrix, NalgebraLU, NalgebraMat, OdeEquations, OdeSolverConfig,
        OdeSolverMethod, OdeSolverProblem,
    };

    fn make_solver<Eqn>(
        _problem: &OdeSolverProblem<Eqn>,
    ) -> NoCheckpointingSolver<Eqn, BdfState<Eqn::V>>
    where
        Eqn: OdeEquations,
        Eqn::V: DefaultDenseMatrix,
    {
        NoCheckpointingSolver::default()
    }

    fn panic_message(panic: Box<dyn Any + Send>) -> String {
        if let Some(message) = panic.downcast_ref::<&'static str>() {
            (*message).to_string()
        } else if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else {
            String::new()
        }
    }

    macro_rules! assert_panics_with {
        ($expr:expr, $needle:literal) => {{
            let err = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                $expr;
            }))
            .expect_err("expected checkpoint replay placeholder to panic");
            let msg = panic_message(err);
            assert!(
                msg.contains($needle),
                "expected panic containing {:?}, got {:?}",
                $needle,
                msg
            );
        }};
    }

    #[test]
    fn config_default_clone_copy_and_base_accessors_panic() {
        let config = NoCheckpointingSolverConfig::<f64>::default();
        let mut config_mut = config;
        let _copy = config;
        let _clone = <NoCheckpointingSolverConfig<f64> as Clone>::clone(&config);

        assert_panics_with!(
            {
                let _ = config.as_base_ref();
            },
            "checkpoint replay solver config is unavailable"
        );
        assert_panics_with!(
            {
                let _ = config_mut.as_base_mut();
            },
            "checkpoint replay solver config is unavailable"
        );
    }

    #[test]
    fn solver_default_clone_and_config_accessors_panic() {
        type M = NalgebraMat<f64>;
        let (problem, _) = exponential_decay_problem::<M>(false);
        let mut solver = make_solver(&problem);
        let _clone = solver.clone();

        assert_panics_with!(
            {
                let _ = solver.config();
            },
            "checkpoint replay solver config is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.config_mut();
            },
            "checkpoint replay solver config is unavailable"
        );
    }

    #[test]
    fn solver_state_and_problem_methods_panic() {
        type M = NalgebraMat<f64>;
        type LS = NalgebraLU<f64>;
        let (problem, _) = exponential_decay_problem::<M>(false);
        let mut solver = make_solver(&problem);

        assert_panics_with!(
            {
                let _ = solver.problem();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.checkpoint();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.state_clone();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let state = problem.bdf_state::<LS>().unwrap();
                solver.set_state(state);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let solver = make_solver(&problem);
                let _ = solver.into_state();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.state();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.state_mut();
            },
            "checkpoint replay solver is unavailable"
        );
    }

    #[test]
    fn solver_step_interpolation_and_matrix_methods_panic() {
        type M = NalgebraMat<f64>;
        let (problem, _) = exponential_decay_problem::<M>(false);
        let mut solver = make_solver(&problem);
        let mut y = problem.eqn.init().call(problem.t0);
        let mut dy = y.clone();
        let mut g = y.clone();
        let mut sens = vec![y.clone(), y.clone()];

        assert_panics_with!(
            {
                let _ = solver.jacobian();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.mass();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.step();
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.set_stop_time(problem.t0);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.interpolate_inplace(problem.t0, &mut y);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.interpolate_dy_inplace(problem.t0, &mut dy);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.interpolate_out_inplace(problem.t0, &mut g);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.interpolate_sens_inplace(problem.t0, &mut sens);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.state_mut_back(problem.t0);
            },
            "checkpoint replay solver is unavailable"
        );
        assert_panics_with!(
            {
                let _ = solver.order();
            },
            "checkpoint replay solver is unavailable"
        );
    }
}
