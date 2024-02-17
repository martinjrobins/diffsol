
pub trait Scalar: nalgebra::Scalar + From<f64> + Display + SimdRealField + ComplexField + Copy + ClosedSub + From<f64> + ClosedMul + ClosedDiv + ClosedAdd + Signed + PartialOrd + Pow<Self, Output=Self> + Pow<i32, Output=Self> {
    const EPSILON: Self;
    const INFINITY: Self;
}

type IndexType = usize;


impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
}


pub mod vector;
pub mod matrix;
pub mod linear_solver;
pub mod callable;
pub mod nonlinear_solver;
pub mod ode_solver;
pub mod solver;

use std::fmt::Display;

use nalgebra::{ClosedSub, ClosedMul, ClosedDiv, ClosedAdd, SimdRealField, ComplexField};
use num_traits::{Signed, Pow};
use vector::{Vector, VectorView, VectorViewMut, VectorIndex, VectorRef};
use nonlinear_solver::newton::NewtonNonlinearSolver;
use callable::{NonLinearOp, LinearOp, ConstantOp};
use matrix::{Matrix, MatrixViewMut, MatrixCommon};
use solver::{SolverProblem, NonLinearSolver, LinearSolver};
use linear_solver::lu::LU;
