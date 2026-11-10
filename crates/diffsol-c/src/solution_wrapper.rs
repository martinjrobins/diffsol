use std::sync::{Arc, Mutex, MutexGuard};

use diffsol::DiffsolError;
use serde::{ser::SerializeStruct, Serialize, Serializer};

use crate::{
    error::DiffsolRtError, host_array::HostArray, scalar_type::ScalarType, solution::Solution,
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

/// Read the values at `indices` as `f64` values. The array has the scalar type of the solver.
fn read_f64(
    array: &HostArray,
    indices: impl Iterator<Item = Vec<usize>>,
) -> Result<Vec<f64>, DiffsolRtError> {
    let offsets = indices.map(|index| array.offset(&index));
    Ok(match array.dtype() {
        ScalarType::F32 => {
            let data = array.as_slice::<f32>()?;
            offsets.map(|i| data[i] as f64).collect()
        }
        ScalarType::F64 => {
            let data = array.as_slice::<f64>()?;
            offsets.map(|i| data[i]).collect()
        }
    })
}

/// Read a 3D array, for example `[nout, nt, nbatch]`, as nested `f64` values in the order of its
/// axes: `values[i][j][k]` is the value at the index `(i, j, k)`.
fn nested(array: &HostArray) -> Result<Vec<Vec<Vec<f64>>>, DiffsolRtError> {
    let (n0, n1, n2) = (array.shape()[0], array.shape()[1], array.shape()[2]);
    (0..n0)
        .map(|i| {
            (0..n1)
                .map(|j| read_f64(array, (0..n2).map(|k| vec![i, j, k])))
                .collect()
        })
        .collect()
}

impl Serialize for SolutionWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let ts_host = self.get_ts().map_err(serde::ser::Error::custom)?;
        let ys_host = self.get_ys().map_err(serde::ser::Error::custom)?;

        // the values have the scalar type of the solver, and are written as `f64`
        let ts = read_f64(&ts_host, (0..ts_host.shape()[0]).map(|k| vec![k]))
            .map_err(serde::ser::Error::custom)?;
        let ys = nested(&ys_host).map_err(serde::ser::Error::custom)?;
        let sensitivities = self
            .get_sens()
            .map_err(serde::ser::Error::custom)?
            .iter()
            .map(nested)
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::ser::Error::custom)?;

        let mut state = serializer.serialize_struct("SolutionWrapper", 3)?;
        state.serialize_field("ts", &ts)?;
        state.serialize_field("ys", &ys)?;
        state.serialize_field("sensitivities", &sensitivities)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::nested;
    use crate::host_array::{HostArray, ToHostArray};

    #[test]
    fn nested_keep_zero_length_axes() {
        let no_rows = HostArray::from_vec(Vec::<f64>::new(), vec![2, 0, 4]);
        assert_eq!(nested(&no_rows).unwrap(), vec![Vec::<Vec<f64>>::new(); 2]);
        let no_cols = HostArray::from_vec(Vec::<f64>::new(), vec![3, 1, 0]);
        assert_eq!(nested(&no_cols).unwrap(), vec![vec![Vec::<f64>::new()]; 3]);
    }

    #[test]
    fn nested_read_f32_values() {
        // a `[2, 2, 1]` array of `f32` values: a 2x2 column-major matrix with one lane
        let m = <diffsol::NalgebraMat<f32> as diffsol::DenseMatrix>::from_vec(
            2,
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            Default::default(),
        );
        let array = m.to_host_array();
        assert_eq!(
            nested(&array).unwrap(),
            vec![vec![vec![1.0], vec![3.0]], vec![vec![2.0], vec![4.0]]]
        );
    }
}
