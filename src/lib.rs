
pub trait Scalar: nalgebra::Scalar + Display + SimdRealField + ComplexField + Copy + ClosedSub + From<f64> + ClosedMul + ClosedDiv + ClosedAdd + Signed + PartialOrd + Pow<Self, Output=Self> + Pow<i32, Output=Self> {
    const EPSILON: Self;
    const INFINITY: Self;
}


impl Scalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
}

type IndexType = usize;

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
use vector::{Vector, VectorView, VectorViewMut, VectorIndex};
use nonlinear_solver::newton::NewtonNonlinearSolver;
use callable::{Callable, Jacobian, Diagonal, GatherCallable};
use matrix::{Matrix, MatrixView, MatrixViewMut};
use solver::{Solver, SolverStatistics, SolverOptions, SolverProblem};
use linear_solver::lu::LU;
