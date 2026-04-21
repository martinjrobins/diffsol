use std::sync::{Arc, Mutex, MutexGuard};

use diffsol::DiffsolError;
use serde::{ser::SerializeStruct, Serialize, Serializer};

use crate::{
    error::DiffsolRtError,
    host_array::{FromHostArray, HostArray},
    solution::Solution,
};

#[derive(Clone)]
pub struct SolutionWrapper(Arc<Mutex<Box<dyn Solution>>>);

impl SolutionWrapper {
    pub(crate) fn new(solution: Box<dyn Solution>) -> Self {
        Self(Arc::new(Mutex::new(solution)))
    }

    fn guard(&self) -> Result<MutexGuard<'_, Box<dyn Solution>>, DiffsolRtError> {
        self.0
            .lock()
            .map_err(|_| DiffsolError::Other("Solution mutex poisoned".to_string()).into())
    }

    pub fn get_ys(&self) -> Result<HostArray, DiffsolRtError> {
        let guard = self.guard()?;
        Ok(guard.get_ys())
    }

    pub fn get_ts(&self) -> Result<HostArray, DiffsolRtError> {
        let guard = self.guard()?;
        Ok(guard.get_ts())
    }

    pub fn get_sens(&self) -> Result<Vec<HostArray>, DiffsolRtError> {
        let guard = self.guard()?;
        Ok(guard.get_sens())
    }
}

impl Serialize for SolutionWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let ts_host = self.get_ts().map_err(serde::ser::Error::custom)?;
        let ys_host = self.get_ys().map_err(serde::ser::Error::custom)?;

        let ts = Vec::<f64>::from_host_array(ts_host).map_err(serde::ser::Error::custom)?;
        let ys = Vec::<Vec<f64>>::from_host_array(ys_host).map_err(serde::ser::Error::custom)?;
        let sensitivities = self
            .get_sens()
            .map_err(serde::ser::Error::custom)?
            .into_iter()
            .map(Vec::<Vec<f64>>::from_host_array)
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::ser::Error::custom)?;

        let mut state = serializer.serialize_struct("SolutionWrapper", 3)?;
        state.serialize_field("ts", &ts)?;
        state.serialize_field("ys", &ys)?;
        state.serialize_field("sensitivities", &sensitivities)?;
        state.end()
    }
}
