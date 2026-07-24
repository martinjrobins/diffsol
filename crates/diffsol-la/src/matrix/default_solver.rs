use crate::LinearSolver;

use super::Matrix;

pub trait DefaultSolver: Matrix {
    type LS: LinearSolver<Self> + Default;
    fn default_solver() -> Self::LS {
        Self::LS::default()
    }
}
