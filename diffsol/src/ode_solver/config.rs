use crate::{OdeSolverOptions, Scalar};

pub trait OdeSolverConfig<T> {
    fn as_base_ref(&self) -> OdeSolverConfigRef<'_, T>;
    fn as_base_mut(&mut self) -> OdeSolverConfigMut<'_, T>;
}
pub struct OdeSolverConfigRef<'a, T> {
    pub minimum_timestep: &'a T,
    pub maximum_error_test_failures: &'a usize,
    pub maximum_timestep_growth: &'a T,
    pub minimum_timestep_shrink: &'a T,
}

pub struct OdeSolverConfigMut<'a, T> {
    pub minimum_timestep: &'a mut T,
    pub maximum_error_test_failures: &'a mut usize,
    pub maximum_timestep_growth: &'a mut T,
    pub minimum_timestep_shrink: &'a mut T,
}

#[derive(Debug, Clone)]
pub struct BdfConfig<T> {
    pub minimum_timestep: T,
    pub maximum_error_test_failures: usize,
    pub maximum_timestep_growth: T,
    pub minimum_timestep_growth: T,
    pub maximum_timestep_shrink: T,
    pub minimum_timestep_shrink: T,
    pub maximum_newton_iterations: usize,
}

impl<T: Scalar> OdeSolverConfig<T> for BdfConfig<T> {
    fn as_base_ref(&self) -> OdeSolverConfigRef<'_, T> {
        OdeSolverConfigRef {
            minimum_timestep: &self.minimum_timestep,
            maximum_error_test_failures: &self.maximum_error_test_failures,
            maximum_timestep_growth: &self.maximum_timestep_growth,
            minimum_timestep_shrink: &self.minimum_timestep_shrink,
        }
    }

    fn as_base_mut(&mut self) -> OdeSolverConfigMut<'_, T> {
        OdeSolverConfigMut {
            minimum_timestep: &mut self.minimum_timestep,
            maximum_error_test_failures: &mut self.maximum_error_test_failures,
            maximum_timestep_growth: &mut self.maximum_timestep_growth,
            minimum_timestep_shrink: &mut self.minimum_timestep_shrink,
        }
    }
}

impl<T: Scalar> BdfConfig<T> {
    pub fn new(ode_options: &OdeSolverOptions<T>) -> Self {
        Self {
            minimum_timestep: ode_options.min_timestep,
            maximum_error_test_failures: ode_options.max_error_test_failures,
            maximum_timestep_growth: T::from_f64(2.1).unwrap(),
            minimum_timestep_growth: T::from_f64(2.0).unwrap(),
            maximum_timestep_shrink: T::from_f64(0.9).unwrap(),
            minimum_timestep_shrink: T::from_f64(0.5).unwrap(),
            maximum_newton_iterations: ode_options.max_nonlinear_solver_iterations,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SdirkConfig<T> {
    pub minimum_timestep: T,
    pub maximum_error_test_failures: usize,
    pub maximum_timestep_growth: T,
    pub minimum_timestep_shrink: T,
    pub maximum_newton_iterations: usize,
}

impl<T: Scalar> SdirkConfig<T> {
    pub fn new(ode_options: &OdeSolverOptions<T>) -> Self {
        Self {
            minimum_timestep: ode_options.min_timestep,
            maximum_error_test_failures: ode_options.max_error_test_failures,
            maximum_timestep_growth: T::from_f64(10.0).unwrap(),
            minimum_timestep_shrink: T::from_f64(0.2).unwrap(),
            maximum_newton_iterations: ode_options.max_nonlinear_solver_iterations,
        }
    }
}

impl<T: Scalar> OdeSolverConfig<T> for SdirkConfig<T> {
    fn as_base_ref(&self) -> OdeSolverConfigRef<'_, T> {
        OdeSolverConfigRef {
            minimum_timestep: &self.minimum_timestep,
            maximum_error_test_failures: &self.maximum_error_test_failures,
            maximum_timestep_growth: &self.maximum_timestep_growth,
            minimum_timestep_shrink: &self.minimum_timestep_shrink,
        }
    }

    fn as_base_mut(&mut self) -> OdeSolverConfigMut<'_, T> {
        OdeSolverConfigMut {
            minimum_timestep: &mut self.minimum_timestep,
            maximum_error_test_failures: &mut self.maximum_error_test_failures,
            maximum_timestep_growth: &mut self.maximum_timestep_growth,
            minimum_timestep_shrink: &mut self.minimum_timestep_shrink,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExplicitRkConfig<T> {
    pub minimum_timestep: T,
    pub maximum_error_test_failures: usize,
    pub maximum_timestep_growth: T,
    pub minimum_timestep_shrink: T,
}

impl<T: Scalar> ExplicitRkConfig<T> {
    pub fn new(ode_options: &OdeSolverOptions<T>) -> Self {
        Self {
            minimum_timestep: ode_options.min_timestep,
            maximum_error_test_failures: ode_options.max_error_test_failures,
            maximum_timestep_growth: T::from_f64(10.0).unwrap(),
            minimum_timestep_shrink: T::from_f64(0.2).unwrap(),
        }
    }
}

impl<T: Scalar> OdeSolverConfig<T> for ExplicitRkConfig<T> {
    fn as_base_ref(&self) -> OdeSolverConfigRef<'_, T> {
        OdeSolverConfigRef {
            minimum_timestep: &self.minimum_timestep,
            maximum_error_test_failures: &self.maximum_error_test_failures,
            maximum_timestep_growth: &self.maximum_timestep_growth,
            minimum_timestep_shrink: &self.minimum_timestep_shrink,
        }
    }

    fn as_base_mut(&mut self) -> OdeSolverConfigMut<'_, T> {
        OdeSolverConfigMut {
            minimum_timestep: &mut self.minimum_timestep,
            maximum_error_test_failures: &mut self.maximum_error_test_failures,
            maximum_timestep_growth: &mut self.maximum_timestep_growth,
            minimum_timestep_shrink: &mut self.minimum_timestep_shrink,
        }
    }
}
