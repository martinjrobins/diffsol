use crate::{OdeSolverOptions, Scalar};

#[derive(Clone, Copy)]
pub enum SolverState {
    StepSuccess,
    FirstConvergenceFail,
    SecondConvergenceFail,
    ErrorTestFail,
    Checkpoint,
}

/// Run `solve` and, after each failure, the first of the remaining Jacobian updates `retries`
/// that changes the Newton matrix (`update` makes one and returns whether it did), then `solve`
/// again. Once no update left changes the matrix, the last failure is returned.
pub(crate) fn solve_with_jacobian_updates<S, E>(
    solver: &mut S,
    retries: &[SolverState],
    mut solve: impl FnMut(&mut S) -> Result<(), E>,
    mut update: impl FnMut(&mut S, SolverState) -> bool,
) -> Result<(), E> {
    let mut retries = retries.iter().copied();
    loop {
        match solve(solver) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if !retries.any(|state| update(solver, state)) {
                    return Err(e);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct JacobianUpdate<T: Scalar> {
    steps_since_jacobian_eval: usize,
    steps_since_rhs_jacobian_eval: usize,
    h_at_last_jacobian_update: T,
    threshold_to_update_jacobian: T,
    threshold_to_update_rhs_jacobian: T,
    update_jacobian_after_steps: usize,
    update_rhs_jacobian_after_steps: usize,
}

impl<T: Scalar> JacobianUpdate<T> {
    pub fn new(options: &OdeSolverOptions<T>) -> Self {
        Self {
            steps_since_jacobian_eval: 0,
            steps_since_rhs_jacobian_eval: 0,
            h_at_last_jacobian_update: T::one(),
            threshold_to_update_jacobian: options.threshold_to_update_jacobian,
            threshold_to_update_rhs_jacobian: options.threshold_to_update_rhs_jacobian,
            update_jacobian_after_steps: options.update_jacobian_after_steps,
            update_rhs_jacobian_after_steps: options.update_rhs_jacobian_after_steps,
        }
    }

    pub fn update_jacobian(&mut self, h: T) {
        self.steps_since_jacobian_eval = 0;
        self.h_at_last_jacobian_update = h;
    }

    pub fn update_rhs_jacobian(&mut self, h: T) {
        self.steps_since_rhs_jacobian_eval = 0;
        self.steps_since_jacobian_eval = 0;
        self.h_at_last_jacobian_update = h;
    }

    pub fn step(&mut self) {
        self.steps_since_jacobian_eval += 1;
        self.steps_since_rhs_jacobian_eval += 1;
    }

    pub fn check_jacobian_update(&mut self, h: T, state: &SolverState) -> bool {
        match state {
            SolverState::StepSuccess => {
                self.steps_since_jacobian_eval >= self.update_jacobian_after_steps
                    || (h / self.h_at_last_jacobian_update - T::one()).abs()
                        > self.threshold_to_update_jacobian
            }
            // after a failed solve, a Newton matrix with the same Jacobian and step size would
            // fail the same way
            SolverState::FirstConvergenceFail | SolverState::SecondConvergenceFail => {
                h != self.h_at_last_jacobian_update
            }
            SolverState::ErrorTestFail => true,
            SolverState::Checkpoint => true,
        }
    }

    pub fn check_rhs_jacobian_update(&mut self, h: T, state: &SolverState) -> bool {
        match state {
            SolverState::StepSuccess => {
                self.steps_since_rhs_jacobian_eval >= self.update_rhs_jacobian_after_steps
            }
            SolverState::FirstConvergenceFail => {
                (h / self.h_at_last_jacobian_update - T::one()).abs()
                    < self.threshold_to_update_rhs_jacobian
            }
            SolverState::SecondConvergenceFail => self.steps_since_rhs_jacobian_eval > 0,
            SolverState::ErrorTestFail => false,
            SolverState::Checkpoint => true,
        }
    }
}
