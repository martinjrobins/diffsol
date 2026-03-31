use std::sync::{Arc, Mutex, MutexGuard};

use diffsol::DiffsolError;

use crate::{error::DiffsolJsError, host_array::HostArray, solution::Solution};

#[derive(Clone)]
pub struct SolutionWrapper(Arc<Mutex<Option<Box<dyn Solution>>>>);

impl SolutionWrapper {
    pub(crate) fn new(solution: Box<dyn Solution>) -> Self {
        Self(Arc::new(Mutex::new(Some(solution))))
    }

    fn guard(&self) -> Result<MutexGuard<'_, Option<Box<dyn Solution>>>, DiffsolJsError> {
        self.0
            .lock()
            .map_err(|_| DiffsolError::Other("Solution mutex poisoned".to_string()).into())
    }

    pub(crate) fn take_solution(&self) -> Result<Box<dyn Solution>, DiffsolJsError> {
        let mut guard = self.guard()?;
        guard
            .take()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()).into())
    }

    pub(crate) fn replace_solution(
        &self,
        solution: Box<dyn Solution>,
    ) -> Result<(), DiffsolJsError> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| DiffsolError::Other("Solution mutex poisoned".to_string()))?;
        *guard = Some(solution);
        Ok(())
    }

    pub(crate) fn get_ys(&self) -> Result<HostArray, DiffsolJsError> {
        let guard = self.guard()?;
        let solution = guard
            .as_ref()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        Ok(solution.get_ys())
    }

    pub(crate) fn get_ts(&self) -> Result<HostArray, DiffsolJsError> {
        let guard = self.guard()?;
        let solution = guard
            .as_ref()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        Ok(solution.get_ts())
    }

    pub(crate) fn get_sens(&self) -> Result<Vec<HostArray>, DiffsolJsError> {
        let guard = self.guard()?;
        let solution = guard
            .as_ref()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        Ok(solution.get_sens())
    }

    pub(crate) fn set_current_state(&self, y: &[f64]) -> Result<(), DiffsolJsError> {
        let mut guard = self.guard()?;
        let solution = guard
            .as_mut()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        solution.set_state_y(y)?;
        Ok(())
    }

    pub(crate) fn get_current_state(&self) -> Result<HostArray, DiffsolJsError> {
        let guard = self.guard()?;
        let solution = guard
            .as_ref()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        Ok(solution.get_state_y())
    }
}
