use diffsol::{
    BdfState, DefaultDenseMatrix, DenseMatrix, DiffsolError, MatrixCommon, OdeSolverState, RkState,
    Vector, VectorCommon, VectorHost, VectorViewMut,
};
use num_traits::FromPrimitive;
use std::any::Any;

use crate::error::DiffsolJsError;
use crate::host_array::{HostArray, ToHostArray};

pub(crate) trait Solution: Any + Send + Sync {
    fn get_ys<'py>(&self) -> HostArray;
    fn get_ts<'py>(&self) -> HostArray;
    fn get_sens<'py>(&self) -> Vec<HostArray>;
    fn set_state_y(&mut self, y: &[f64]) -> Result<(), DiffsolJsError>;
    fn get_state_y<'py>(&self) -> HostArray;
}

impl dyn Solution + '_ {
    pub(crate) fn downcast_typed_solution<V>(&self) -> Result<&GenericSolution<V>, DiffsolJsError>
    where
        V: Vector + DefaultDenseMatrix + 'static,
    {
        (self as &dyn Any)
            .downcast_ref::<GenericSolution<V>>()
            .ok_or_else(|| {
                DiffsolError::Other(
                    "Provided Solution type is incompatible with this Ode instance".to_string(),
                )
                .into()
            })
    }

    pub(crate) fn downcast_typed_solution_mut<V>(
        &mut self,
    ) -> Result<&mut GenericSolution<V>, DiffsolJsError>
    where
        V: Vector + DefaultDenseMatrix + 'static,
    {
        (self as &mut dyn Any)
            .downcast_mut::<GenericSolution<V>>()
            .ok_or_else(|| {
                DiffsolError::Other(
                    "Provided Solution type is incompatible with this Ode instance".to_string(),
                )
                .into()
            })
    }
}

#[derive(Clone)]
pub(crate) enum GenericState<V: Vector + DefaultDenseMatrix> {
    Bdf(BdfState<V>),
    Rk(RkState<V>),
}

pub(crate) struct GenericSolution<V: Vector + DefaultDenseMatrix> {
    state: Option<GenericState<V>>,
    ys: <V as DefaultDenseMatrix>::M,
    ts: Vec<V::T>,
    sens: Vec<<V as DefaultDenseMatrix>::M>,
}

impl<V: Vector + DefaultDenseMatrix> GenericSolution<V> {
    pub(crate) fn new(
        state: GenericState<V>,
        ys: <V as DefaultDenseMatrix>::M,
        ts: Vec<V::T>,
        sens: Vec<<V as DefaultDenseMatrix>::M>,
    ) -> Self {
        Self {
            state: Some(state),
            ys,
            ts,
            sens,
        }
    }

    fn current_state(&self) -> &GenericState<V> {
        self.state
            .as_ref()
            .expect("solution current state missing unexpectedly")
    }

    fn current_state_mut(&mut self) -> &mut GenericState<V> {
        self.state
            .as_mut()
            .expect("solution current state missing unexpectedly")
    }

    pub(crate) fn state_clone(&self) -> Result<GenericState<V>, DiffsolJsError> {
        self.state
            .as_ref()
            .cloned()
            .ok_or_else(|| DiffsolError::Other("Solution current state missing".to_string()).into())
    }

    pub(crate) fn append(
        &mut self,
        state: GenericState<V>,
        ys: <V as DefaultDenseMatrix>::M,
        ts: Vec<V::T>,
        sens: Vec<<V as DefaultDenseMatrix>::M>,
    ) -> Result<(), String> {
        if self.ys.nrows() != ys.nrows() {
            return Err(format!(
                "Cannot append ys with mismatched rows ({} vs {})",
                self.ys.nrows(),
                ys.nrows()
            ));
        }

        let self_has_sens = !self.sens.is_empty();
        let new_has_sens = !sens.is_empty();
        if self_has_sens != new_has_sens {
            return Err(format!(
                "Cannot append solution with sensitivities={} to solution with sensitivities={}",
                new_has_sens, self_has_sens
            ));
        }

        // Validate sensitivity dimensions before mutating any buffers.
        if self_has_sens {
            if self.sens.len() != sens.len() {
                return Err(format!(
                    "Cannot append sens with mismatched length ({} vs {})",
                    self.sens.len(),
                    sens.len()
                ));
            }
            for (dst, src) in self.sens.iter().zip(sens.iter()) {
                if dst.nrows() != src.nrows() {
                    return Err(format!(
                        "Cannot append sens with mismatched rows ({} vs {})",
                        dst.nrows(),
                        src.nrows()
                    ));
                }
            }
        }

        append_matrix_columns(&mut self.ys, &ys);
        if self_has_sens {
            for (dst, src) in self.sens.iter_mut().zip(sens.iter()) {
                append_matrix_columns(dst, src);
            }
        }
        self.ts.extend(ts);
        self.state = Some(state);
        Ok(())
    }
}

fn append_matrix_columns<M: DenseMatrix>(dst: &mut M, src: &M) {
    let old_cols = dst.ncols();
    let add_cols = src.ncols();
    dst.resize_cols(old_cols + add_cols);
    for j in 0..add_cols {
        dst.column_mut(old_cols + j).copy_from_view(&src.column(j));
    }
}

fn copy_slice_to_vec<V: VectorHost>(state: &mut V, y: &[f64]) -> Result<(), DiffsolJsError> {
    if state.len() != y.len() {
        return Err(DiffsolError::Other(format!(
            "Expected current_state length {} but got {}",
            state.len(),
            y.len()
        ))
        .into());
    }
    for (yi, &y_val) in state.as_mut_slice().iter_mut().zip(y.iter()) {
        *yi = V::T::from_f64(y_val).unwrap();
    }
    Ok(())
}

impl<V: VectorHost + DefaultDenseMatrix + Send + Sync + 'static> Solution for GenericSolution<V>
where
    <V as DefaultDenseMatrix>::M: Send + Sync,
    <V as VectorCommon>::Inner: ToHostArray<V::T> + Clone,
    <<V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: ToHostArray<V::T> + Clone,
{
    fn set_state_y(&mut self, y: &[f64]) -> Result<(), DiffsolJsError> {
        match self.current_state_mut() {
            GenericState::Bdf(state) => copy_slice_to_vec(state.as_mut().y, y),
            GenericState::Rk(state) => copy_slice_to_vec(state.as_mut().y, y),
        }
    }

    fn get_state_y<'py>(&self) -> HostArray {
        match self.current_state() {
            GenericState::Bdf(state) => (*state.as_ref().y.inner()).clone().to_host_array(),
            GenericState::Rk(state) => (*state.as_ref().y.inner()).clone().to_host_array(),
        }
    }

    fn get_sens<'py>(&self) -> Vec<HostArray> {
        self.sens
            .iter()
            .map(|s| (*s.inner()).clone().to_host_array())
            .collect()
    }

    fn get_ts<'py>(&self) -> HostArray {
        let ctx = match self.current_state() {
            GenericState::Bdf(state) => state.as_ref().y.context().clone(),
            GenericState::Rk(state) => state.as_ref().y.context().clone(),
        };
        (*V::from_slice(&self.ts, ctx).inner())
            .clone()
            .to_host_array()
    }

    fn get_ys<'py>(&self) -> HostArray {
        (*self.ys.inner()).clone().to_host_array()
    }
}
