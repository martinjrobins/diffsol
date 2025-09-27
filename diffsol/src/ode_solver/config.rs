use crate::Scalar;

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

impl<T: Scalar> Default for BdfConfig<T> {
    fn default() -> Self {
        Self {
            minimum_timestep: T::from(1e-32),
            maximum_error_test_failures: 40,
            maximum_timestep_growth: T::from(2.1),
            minimum_timestep_growth: T::from(2.0),
            maximum_timestep_shrink: T::from(0.9),
            minimum_timestep_shrink: T::from(0.5),
            maximum_newton_iterations: 4,
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

impl<T: Scalar> Default for SdirkConfig<T> {
    fn default() -> Self {
        Self {
            minimum_timestep: T::from(1e-13),
            maximum_error_test_failures: 40,
            maximum_timestep_growth: T::from(10.0),
            minimum_timestep_shrink: T::from(0.2),
            maximum_newton_iterations: 10,
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

impl<T: Scalar> Default for ExplicitRkConfig<T> {
    fn default() -> Self {
        Self {
            minimum_timestep: T::from(1e-13),
            maximum_error_test_failures: 40,
            maximum_timestep_growth: T::from(10.0),
            minimum_timestep_shrink: T::from(0.2),
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
