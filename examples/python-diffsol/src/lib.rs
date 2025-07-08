use std::sync::{Arc, Mutex};

use diffsol::{
    error::DiffsolError, NonLinearOp, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem,
    Op, Vector, VectorHost,
};
use numpy::{
    ndarray::{s, Array2},
    IntoPyArray, PyArray2, PyReadonlyArray1,
};
use pyo3::{exceptions::PyValueError, prelude::*};

type M = diffsol::NalgebraMat<f64>;
type V = diffsol::NalgebraVec<f64>;
type C = diffsol::NalgebraContext;
type LS = diffsol::NalgebraLU<f64>;
#[cfg(not(feature = "diffsol-llvm"))]
type CG = diffsol::CraneliftJitModule;
#[cfg(feature = "diffsol-llvm")]
type CG = diffsol::LlvmModule;
type Eqn = diffsol::DiffSl<M, CG>;

#[pyclass]
struct PyDiffsol {
    problem: Arc<Mutex<OdeSolverProblem<Eqn>>>,
}

#[pymethods]
impl PyDiffsol {
    #[new]
    fn new(code: &str) -> Result<Self, PyDiffsolError> {
        let problem = OdeBuilder::<M>::new().build_from_diffsl(code)?;
        Ok(Self {
            problem: Arc::new(Mutex::new(problem)),
        })
    }

    #[pyo3(signature = (params, times))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        let mut problem = self
            .problem
            .lock()
            .map_err(|e| PyDiffsolError(DiffsolError::Other(e.to_string())))?;
        let times = times.as_array();
        let params = V::from_slice(params.as_array().as_slice().unwrap(), C::default());
        problem.eqn.set_params(&params);
        let mut solver = problem.bdf::<LS>()?;
        let nout = if let Some(_out) = problem.eqn.out() {
            problem.eqn.nout()
        } else {
            problem.eqn.nstates()
        };
        let mut sol = Array2::zeros((nout, times.len()));
        for (i, &t) in times.iter().enumerate() {
            while solver.state().t < t {
                solver.step()?;
            }
            let y = solver.interpolate(t)?;
            let out = if let Some(out) = problem.eqn.out() {
                out.call(&y, t)
            } else {
                y
            };
            sol.slice_mut(s![.., i])
                .iter_mut()
                .zip(out.as_slice().iter())
                .for_each(|(a, b)| *a = *b);
        }
        Ok(sol.into_pyarray(py))
    }
}

struct PyDiffsolError(DiffsolError);

impl From<PyDiffsolError> for PyErr {
    fn from(error: PyDiffsolError) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

impl From<DiffsolError> for PyDiffsolError {
    fn from(other: DiffsolError) -> Self {
        Self(other)
    }
}

#[pymodule]
fn python_diffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiffsol>()?;
    Ok(())
}
