use crate::{linear_solver::LinearSolver, op::LinearOp};

use super::Matrix;

pub trait DefaultSolver: Matrix {
    type LS<C: LinearOp<M = Self, V = Self::V, T = Self::T>>: LinearSolver<C> + Default;
    fn default_solver<C: LinearOp<M = Self, V = Self::V, T = Self::T>>() -> Self::LS<C> {
        Self::LS::default()
    }
}
