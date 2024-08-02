use crate::Scalar;

pub struct JacobianUpdate<T: Scalar> {
    steps_since_jacobian_eval: usize,
    steps_since_rhs_jacobian_eval: usize,
    h_at_last_jacobian_update: T,
    threshold_to_update_jacobian: T,
    update_jacobian_after_steps: usize,
    update_rhs_jacobian_after_steps: usize,
}

impl<T: Scalar> JacobianUpdate<T> {
    pub fn new() -> Self {
        Self {
            steps_since_jacobian_eval: 0,
            steps_since_rhs_jacobian_eval: 0,
            h_at_last_jacobian_update: T::one(),
            threshold_to_update_jacobian: T::from(0.3),
            update_jacobian_after_steps: 20,
            update_rhs_jacobian_after_steps: 50,
        }
    }

    pub fn update_jacobian(&mut self, h: T) {
        self.steps_since_jacobian_eval = 0;
        self.h_at_last_jacobian_update = h;
    }

    pub fn update_rhs_jacobian(&mut self) {
        self.steps_since_rhs_jacobian_eval = 0;
    }

    pub fn step(&mut self) {
        self.steps_since_jacobian_eval += 1;
        self.steps_since_rhs_jacobian_eval += 1;
    }

    pub fn check_jacobian_update(&mut self, h: T) -> bool {
        if self.steps_since_jacobian_eval >= self.update_jacobian_after_steps 
            || (h / self.h_at_last_jacobian_update - T::one()).abs() > self.threshold_to_update_jacobian {
            return true;
        }
        false
    }
    
    pub fn check_rhs_jacobian_update(&mut self) -> bool {
        if self.steps_since_rhs_jacobian_eval >= self.update_rhs_jacobian_after_steps {
            return true;
        }
        false
    }
}

impl<T: Scalar> Default for JacobianUpdate<T> {
    fn default() -> Self {
        Self::new()
    }
}


    