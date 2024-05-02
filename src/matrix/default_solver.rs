use crate::{LinearSolver, NonLinearOp};

use super::Matrix;

pub trait DefaultSolver: Matrix {
    type LS<C: NonLinearOp<M = Self, V = Self::V, T = Self::T>>: LinearSolver<C> + Default;
    fn default_solver<C: NonLinearOp<M = Self, V = Self::V, T = Self::T>>() -> Self::LS<C> {
        Self::LS::default()
    }
}
