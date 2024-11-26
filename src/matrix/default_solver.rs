use crate::LinearSolver;

use super::Matrix;

pub trait DefaultSolver: Matrix {
    type LS<'a>: LinearSolver<'a, Self> + Default;
    fn default_solver<'a>() -> Self::LS<'a> {
        Self::LS::default()
    }
}
