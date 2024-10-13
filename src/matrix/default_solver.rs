use crate::{LinearSolver, NonLinearOpJacobian};

use super::Matrix;

pub trait DefaultSolver: Matrix {
    type LS<C: NonLinearOpJacobian<M = Self, V = Self::V, T = Self::T>>: LinearSolver<C> + Default;
    fn default_solver<C: NonLinearOpJacobian<M = Self, V = Self::V, T = Self::T>>() -> Self::LS<C> {
        Self::LS::default()
    }
}
