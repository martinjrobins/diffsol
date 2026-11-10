use std::sync::Arc;

use serde::{ser::SerializeStruct, Serialize, Serializer};

use crate::{
    error::DiffsolRtError,
    scalar_type::{Scalar, ScalarType},
    solution::Solution,
    tensor::Tensor,
};

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

/// Read a 3D array, for example `[nout, nt, nbatch]`, as nested values of the scalar type `T` in
/// the order of its axes: `values[i][j][k]` is the value at the index `(i, j, k)`.
fn nested<T: Scalar>(tensor: &Tensor) -> Result<Vec<Vec<Vec<T>>>, DiffsolRtError> {
    let (n0, n1, n2) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
    // `columns[j][k]` is the column `[:, j, k]`
    let columns = (0..n1)
        .map(|j| {
            (0..n2)
                .map(|k| tensor.column::<T>(&[j, k]))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..n0)
        .map(|i| {
            (0..n1)
                .map(|j| (0..n2).map(|k| columns[j][k][i]).collect())
                .collect()
        })
        .collect())
}

impl SolutionWrapper {
    /// Serialize the solution with the values as `T`, the scalar type of the solver.
    fn serialize_as<T: Scalar + Serialize, S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let error = serde::ser::Error::custom;
        let ts_tensor = self.get_ts().map_err(error)?;
        let ys_tensor = self.get_ys().map_err(error)?;
        let ts = ts_tensor.column::<T>(&[]).map_err(error)?;
        let ys = nested::<T>(&ys_tensor).map_err(error)?;
        let sensitivities = self
            .get_sens()
            .map_err(error)?
            .iter()
            .map(nested::<T>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(error)?;

        let mut state = serializer.serialize_struct("SolutionWrapper", 4)?;
        state.serialize_field("dtype", &T::scalar_type())?;
        state.serialize_field("ts", ts)?;
        state.serialize_field("ys", &ys)?;
        state.serialize_field("sensitivities", &sensitivities)?;
        state.end()
    }
}

/// The solution serializes as `{dtype, ts, ys, sensitivities}`. The values have the scalar type
/// of the solver, `dtype`, so an `f32` solution keeps its values exactly as `f32` (in JSON, `0.1`
/// and not `0.10000000149011612`).
impl Serialize for SolutionWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let dtype = self.get_ts().map_err(serde::ser::Error::custom)?.dtype();
        match dtype {
            ScalarType::F32 => self.serialize_as::<f32, S>(serializer),
            ScalarType::F64 => self.serialize_as::<f64, S>(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{nested, SolutionWrapper};
    use crate::solution::Solution;
    use crate::tensor::{Tensor, ToTensor};

    #[test]
    fn nested_keep_zero_length_axes() {
        let no_rows = Tensor::from_vec(Vec::<f64>::new(), &[2, 0, 4]);
        assert_eq!(
            nested::<f64>(&no_rows).unwrap(),
            vec![Vec::<Vec<f64>>::new(); 2]
        );
        let no_cols = Tensor::from_vec(Vec::<f64>::new(), &[3, 1, 0]);
        assert_eq!(
            nested::<f64>(&no_cols).unwrap(),
            vec![vec![Vec::<f64>::new()]; 3]
        );
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
            nested::<f32>(&tensor).unwrap(),
            vec![vec![vec![1.0], vec![3.0]], vec![vec![2.0], vec![4.0]]]
        );
        assert!(nested::<f64>(&tensor).is_err());
    }

    /// A solution of `f32` values with 2 times, 1 output and 1 lane, and no sensitivities.
    struct F32Solution;

    impl Solution for F32Solution {
        fn get_ys(self: Arc<Self>) -> Tensor {
            Tensor::from_vec(vec![0.1f32, 0.2], &[1, 2, 1])
        }
        fn get_ts(self: Arc<Self>) -> Tensor {
            Tensor::from_vec(vec![0.5f32, 1.0], &[2])
        }
        fn get_sens(self: Arc<Self>) -> Vec<Tensor> {
            Vec::new()
        }
    }

    #[test]
    fn solutions_serialize_with_the_scalar_type_of_the_solver() {
        let solution = SolutionWrapper::new(Arc::new(F32Solution));
        let json = serde_json::to_string(&solution).unwrap();
        assert_eq!(
            json,
            r#"{"dtype":"f32","ts":[0.5,1.0],"ys":[[[0.1],[0.2]]],"sensitivities":[]}"#
        );
    }
}
