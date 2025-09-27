use crate::Scalar;

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
