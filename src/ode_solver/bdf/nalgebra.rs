#[cfg(test)]
mod test {
    use crate::{
        ode_solver::tests::{test_interpolate, test_no_set_problem, test_take_state},
        Bdf,
    };

    type M = nalgebra::DMatrix<f64>;
    #[test]
    fn bdf_no_set_problem() {
        test_no_set_problem::<M, _>(Bdf::default())
    }
    #[test]
    fn bdf_take_state() {
        test_take_state::<M, _>(Bdf::default())
    }
    #[test]
    fn bdf_test_interpolate() {
        test_interpolate::<M, _>(Bdf::default())
    }
}
