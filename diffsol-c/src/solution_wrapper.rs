use std::sync::{Arc, Mutex, MutexGuard};

use diffsol::DiffsolError;
use serde::{Serialize, Serializer, ser::SerializeStruct};

use crate::{
    error::DiffsolJsError,
    host_array::{FromHostArray, HostArray},
    solution::Solution,
};

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

    pub fn set_current_state(&self, y: &[f64]) -> Result<(), DiffsolJsError> {
        let mut guard = self.guard()?;
        let solution = guard
            .as_mut()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        solution.set_state_y(y)?;
        Ok(())
    }

    pub fn get_current_state(&self) -> Result<HostArray, DiffsolJsError> {
        let guard = self.guard()?;
        let solution = guard
            .as_ref()
            .ok_or_else(|| DiffsolError::Other("Solution payload missing".to_string()))?;
        Ok(solution.get_state_y())
    }
}

impl Serialize for SolutionWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let ts_host = self.get_ts().map_err(serde::ser::Error::custom)?;
        let ys_host = self.get_ys().map_err(serde::ser::Error::custom)?;
        let current_state_host = self
            .get_current_state()
            .map_err(serde::ser::Error::custom)?;

        let ts = Vec::<f64>::from_host_array(ts_host).map_err(serde::ser::Error::custom)?;
        let ys = Vec::<Vec<f64>>::from_host_array(ys_host).map_err(serde::ser::Error::custom)?;
        let current_state =
            Vec::<f64>::from_host_array(current_state_host).map_err(serde::ser::Error::custom)?;
        let sensitivities = self
            .get_sens()
            .map_err(serde::ser::Error::custom)?
            .into_iter()
            .map(Vec::<Vec<f64>>::from_host_array)
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::ser::Error::custom)?;

        let mut state = serializer.serialize_struct("SolutionWrapper", 4)?;
        state.serialize_field("ts", &ts)?;
        state.serialize_field("ys", &ys)?;
        state.serialize_field("current_state", &current_state)?;
        state.serialize_field("sensitivities", &sensitivities)?;
        state.end()
    }
}
