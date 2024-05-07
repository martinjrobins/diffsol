use std::rc::Rc;

use crate::{op::unit::UnitCallable, scalar::Scalar, LinearOp, Matrix, NonLinearOp, Vector};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct OdeEquationsStatistics {
    pub number_of_rhs_evals: usize,
    pub number_of_jac_mul_evals: usize,
    pub number_of_mass_evals: usize,
    pub number_of_mass_matrix_evals: usize,
    pub number_of_jacobian_matrix_evals: usize,
}

impl Default for OdeEquationsStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl OdeEquationsStatistics {
    pub fn new() -> Self {
        Self {
            number_of_rhs_evals: 0,
            number_of_jac_mul_evals: 0,
            number_of_mass_evals: 0,
            number_of_mass_matrix_evals: 0,
            number_of_jacobian_matrix_evals: 0,
        }
    }
}

/// this is the trait that defines the ODE equations of the form
///
/// $$
///  M \frac{dy}{dt} = F(t, y)
///  y(t_0) = y_0(t_0)
/// $$
///
/// The ODE equations are defined by:
/// - the right-hand side function `F(t, y)`, which is given as a [NonLinearOp] using the `Rhs` associated type and [Self::rhs] function,
/// - the mass matrix `M` which is given as a [LinearOp] using the `Mass` associated type and the [Self::mass] function,
/// - the initial condition `y_0(t_0)`, which is given using the [Self::init] function.
pub trait OdeEquations {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    type M: Matrix<T = Self::T, V = Self::V>;
    type Mass: LinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Rhs: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Root: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;

    /// The parameters of the ODE equations are assumed to be constant. This function sets the parameters to the given value before solving the ODE.
    /// Note that `set_params` must always be called before calling any of the other functions in this trait.
    fn set_params(&mut self, p: Self::V);

    /// returns the right-hand side function `F(t, y)` as a [NonLinearOp]
    fn rhs(&self) -> &Rc<Self::Rhs>;

    /// returns the mass matrix `M` as a [LinearOp]
    fn mass(&self) -> &Rc<Self::Mass>;

    fn root(&self) -> Option<&Rc<Self::Root>> {
        None
    }

    /// returns the initial condition, i.e. `y(t)`, where `t` is the initial time
    fn init(&self, t: Self::T) -> Self::V;

    /// returns true if the mass matrix is constant over time
    fn is_mass_constant(&self) -> bool {
        true
    }
}

/// This struct implements the ODE equation trait [OdeEquations] for a given right-hand side op, mass op, optional root op, and initial condition function.
pub struct OdeSolverEquations<M, Rhs, I, Mass = UnitCallable<M>, Root = UnitCallable<M>>
where
    M: Matrix,
    Rhs: NonLinearOp<M = M, V = M::V, T = M::T>,
    Mass: LinearOp<M = M, V = M::V, T = M::T>,
    Root: NonLinearOp<M = M, V = M::V, T = M::T>,
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: Rc<Rhs>,
    mass: Rc<Mass>,
    root: Option<Rc<Root>>,
    init: I,
    p: Rc<M::V>,
    mass_is_constant: bool,
}

impl<M, Rhs, Mass, Root, I> OdeSolverEquations<M, Rhs, I, Mass, Root>
where
    M: Matrix,
    Rhs: NonLinearOp<M = M, V = M::V, T = M::T>,
    Mass: LinearOp<M = M, V = M::V, T = M::T>,
    Root: NonLinearOp<M = M, V = M::V, T = M::T>,
    I: Fn(&M::V, M::T) -> M::V,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rhs: Rc<Rhs>,
        mass: Rc<Mass>,
        root: Option<Rc<Root>>,
        init: I,
        p: Rc<M::V>,
        mass_is_constant: bool,
    ) -> Self {
        Self {
            rhs,
            mass,
            root,
            init,
            p,
            mass_is_constant,
        }
    }
}

impl<M, Rhs, Mass, Root, I> OdeEquations for OdeSolverEquations<M, Rhs, I, Mass, Root>
where
    M: Matrix,
    Rhs: NonLinearOp<M = M, V = M::V, T = M::T>,
    Mass: LinearOp<M = M, V = M::V, T = M::T>,
    Root: NonLinearOp<M = M, V = M::V, T = M::T>,
    I: Fn(&M::V, M::T) -> M::V,
{
    type T = M::T;
    type V = M::V;
    type M = M;
    type Rhs = Rhs;
    type Mass = Mass;
    type Root = Root;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn mass(&self) -> &Rc<Self::Mass> {
        &self.mass
    }
    fn root(&self) -> Option<&Rc<Self::Root>> {
        self.root.as_ref()
    }
    fn is_mass_constant(&self) -> bool {
        self.mass_is_constant
    }
    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
        Rc::<Rhs>::get_mut(&mut self.rhs)
            .unwrap()
            .set_params(self.p.clone());
        Rc::<Mass>::get_mut(&mut self.mass)
            .unwrap()
            .set_params(self.p.clone());
        if let Some(r) = self.root.as_mut() {
            Rc::<Root>::get_mut(r).unwrap().set_params(self.p.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::ode_solver::equations::OdeEquations;
    use crate::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use crate::ode_solver::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::vector::Vector;
    use crate::LinearOp;
    use crate::NonLinearOp;

    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn ode_equation_test() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let y = DVector::from_vec(vec![1.0, 1.0]);
        let rhs_y = problem.eqn.rhs().call(&y, 0.0);
        let expect_rhs_y = DVector::from_vec(vec![-0.1, -0.1]);
        rhs_y.assert_eq_st(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.rhs().jac_mul(&y, 0.0, &y);
        let expect_jac_rhs_y = Vcpu::from_vec(vec![-0.1, -0.1]);
        jac_rhs_y.assert_eq_st(&expect_jac_rhs_y, 1e-10);
        let mass = problem.eqn.mass().matrix(0.0);
        assert_eq!(mass[(0, 0)], 1.0);
        assert_eq!(mass[(1, 1)], 1.0);
        assert_eq!(mass[(0, 1)], 0.);
        assert_eq!(mass[(1, 0)], 0.);
        let jac = problem.eqn.rhs().jacobian(&y, 0.0);
        assert_eq!(jac[(0, 0)], -0.1);
        assert_eq!(jac[(1, 1)], -0.1);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
    }

    #[test]
    fn ode_with_mass_test() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let rhs_y = problem.eqn.rhs().call(&y, 0.0);
        let expect_rhs_y = DVector::from_vec(vec![-0.1, -0.1, 0.0]);
        rhs_y.assert_eq_st(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.rhs().jac_mul(&y, 0.0, &y);
        let expect_jac_rhs_y = Vcpu::from_vec(vec![-0.1, -0.1, 0.0]);
        jac_rhs_y.assert_eq_st(&expect_jac_rhs_y, 1e-10);
        let mass = problem.eqn.mass().matrix(0.0);
        assert_eq!(mass[(0, 0)], 1.);
        assert_eq!(mass[(1, 1)], 1.);
        assert_eq!(mass[(2, 2)], 0.);
        assert_eq!(mass[(0, 1)], 0.);
        assert_eq!(mass[(1, 0)], 0.);
        assert_eq!(mass[(0, 2)], 0.);
        assert_eq!(mass[(2, 0)], 0.);
        assert_eq!(mass[(1, 2)], 0.);
        assert_eq!(mass[(2, 1)], 0.);
        let jac = problem.eqn.rhs().jacobian(&y, 0.0);
        assert_eq!(jac[(0, 0)], -0.1);
        assert_eq!(jac[(1, 1)], -0.1);
        assert_eq!(jac[(2, 2)], 1.0);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(0, 2)], 0.0);
        assert_eq!(jac[(2, 0)], 0.0);
        assert_eq!(jac[(1, 2)], 0.0);
        assert_eq!(jac[(2, 1)], -1.0);
    }
}
