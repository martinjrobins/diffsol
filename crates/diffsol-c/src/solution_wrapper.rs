use std::sync::Arc;

use serde::{ser::SerializeStruct, Serialize, Serializer};

use crate::{error::DiffsolRtError, scalar_type::ScalarType, solution::Solution, tensor::Tensor};

/// A handle to the solution of a solve. Clones share the same solution.
///
/// The solution does not change after the solve, so the tensors that the methods give share its
/// data, with no copy. A tensor stays valid after the handle is dropped.
#[derive(Clone)]
pub struct SolutionWrapper(Arc<dyn Solution>);

impl SolutionWrapper {
    pub(crate) fn new(solution: Arc<dyn Solution>) -> Self {
        Self(solution)
    }

    /// The solution values, as a `[nout, nt, nbatch]` array.
    pub fn get_ys(&self) -> Result<Tensor, DiffsolRtError> {
        Ok(self.0.clone().get_ys())
    }

    /// The solution times, as a `[nt]` array. All the lanes use the same times.
    pub fn get_ts(&self) -> Result<Tensor, DiffsolRtError> {
        Ok(self.0.clone().get_ts())
    }

    /// The sensitivities, as `nparams` arrays, each `[nout, nt, nbatch]`.
    pub fn get_sens(&self) -> Result<Vec<Tensor>, DiffsolRtError> {
        Ok(self.0.clone().get_sens())
    }
}

/// Read the values at `indices` as `f64` values. The array has the scalar type of the solver.
fn read_f64(
    tensor: &Tensor,
    indices: impl Iterator<Item = Vec<usize>>,
) -> Result<Vec<f64>, DiffsolRtError> {
    let offsets = indices.map(|index| tensor.offset(&index));
    Ok(match tensor.dtype() {
        ScalarType::F32 => {
            let data = tensor.as_slice::<f32>()?;
            offsets.map(|i| data[i] as f64).collect()
        }
        ScalarType::F64 => {
            let data = tensor.as_slice::<f64>()?;
            offsets.map(|i| data[i]).collect()
        }
    })
}

/// Read a 3D array, for example `[nout, nt, nbatch]`, as nested `f64` values in the order of its
/// axes: `values[i][j][k]` is the value at the index `(i, j, k)`.
fn nested(tensor: &Tensor) -> Result<Vec<Vec<Vec<f64>>>, DiffsolRtError> {
    let (n0, n1, n2) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
    (0..n0)
        .map(|i| {
            (0..n1)
                .map(|j| read_f64(tensor, (0..n2).map(|k| vec![i, j, k])))
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
    use crate::tensor::{Tensor, ToTensor};

    #[test]
    fn nested_keep_zero_length_axes() {
        let no_rows = Tensor::from_vec(Vec::<f64>::new(), vec![2, 0, 4]);
        assert_eq!(nested(&no_rows).unwrap(), vec![Vec::<Vec<f64>>::new(); 2]);
        let no_cols = Tensor::from_vec(Vec::<f64>::new(), vec![3, 1, 0]);
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
        let tensor = m.to_tensor();
        assert_eq!(
            nested(&tensor).unwrap(),
            vec![vec![vec![1.0], vec![3.0]], vec![vec![2.0], vec![4.0]]]
        );
    }
}
