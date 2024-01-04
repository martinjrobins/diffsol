
pub trait Scalar: nalgebra::Scalar + Display + SimdRealField + Copy + ClosedSub + From<f64> + ClosedMul + ClosedDiv + ClosedAdd + Signed + PartialOrd + Pow<Self, Output=Self> + Pow<i32, Output=Self> {
    const EPSILON: Self;
}
impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
}

type IndexType = usize;

pub mod vector;
pub mod matrix;
pub mod linear_solver;
pub mod callable;
pub mod nonlinear_solver;
pub mod ode_solver;

use std::fmt::Display;

use nalgebra::{ClosedSub, ClosedMul, ClosedDiv, ClosedAdd, SimdRealField};
use num_traits::{Signed, Pow};
use vector::Vector;
use callable::Callable;
use matrix::Matrix;
use linear_solver::LinearSolver;
use nonlinear_solver::NonLinearSolver;
