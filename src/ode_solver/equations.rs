use num_traits::Zero;
use std::{cell::RefCell, rc::Rc};

use crate::{
    op::unit::UnitCallable, scalar::Scalar, Closure, LinearClosure, LinearOp, Matrix, NonLinearOp, Vector
};
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
///  M \frac{dy}{dt} = F(t, y, p)
///  y(t_0) = y_0(t_0, p)
/// $$
///
/// The ODE equations are defined by the right-hand side function $F(t, y, p)$, the initial condition $y_0(t_0, p)$, and the mass matrix $M$.
pub trait OdeEquations {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    type M: Matrix<T = Self::T, V = Self::V>;
    type Mass: LinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Rhs: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;

    /// The parameters of the ODE equations are assumed to be constant. This function sets the parameters to the given value before solving the ODE.
    /// Note that `set_params` must always be called before calling any of the other functions in this trait.
    fn set_params(&mut self, p: Self::V);

    fn rhs(&self) -> &Rc<Self::Rhs>;
    fn mass(&self) -> &Rc<Self::Mass>;

    /// returns the initial condition, i.e. $y(t_0, p)$
    fn init(&self, t: Self::T) -> Self::V;

    fn is_mass_constant(&self) -> bool {
        true
    }

    /// calculate and return the statistics of the ODE equation object (i.e. how many times the right-hand side function was evaluated, how many times the jacobian was multiplied, etc.)
    /// The default implementation returns an empty statistics object.
    fn get_statistics(&self) -> OdeEquationsStatistics {
        OdeEquationsStatistics::new()
    }
}

/// This struct implements the ODE equation trait [OdeEquations] for a given right-hand side function, jacobian function, mass matrix function, and initial condition function.
/// These functions are provided as closures, and the parameters are assumed to be constant.
pub struct OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: Rc<Closure<M, F, G>>,
    mass: Rc<LinearClosure<M, H>>,
    init: I,
    p: Rc<M::V>,
    statistics: RefCell<OdeEquationsStatistics>,
    mass_is_constant: bool,
}

impl<M, F, G, H, I> OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_ode_with_mass(
        rhs: F,
        rhs_jac: G,
        mass: H,
        init: I,
        p: M::V,
        t0: M::T,
        calculate_sparsity: bool,
        mass_is_constant: bool,
    ) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);
        let statistics = RefCell::default();
        let mut rhs = Closure::<M, _, _>::new(rhs, rhs_jac, nstates, nstates, p.clone());
        let mut mass = LinearClosure::<M, _>::new(mass, nstates, nstates, p.clone());
        if calculate_sparsity {
            rhs.calculate_sparsity(&y0, t0);
            mass.calculate_sparsity(t0);
        }
        let rhs = Rc::new(rhs);
        let mass = Rc::new(mass);
        Self {
            rhs,
            mass,
            init,
            p: p.clone(),
            statistics,
            mass_is_constant,
        }
    }
}


impl<M, F, G, H, I> OdeEquations for OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{

    type T = M::T;
    type V = M::V;
    type M = M;
    type Rhs = Closure<M, F, G>;
    type Mass = LinearClosure<M, H>;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn mass(&self) -> &Rc<Self::Mass> {
        &self.mass
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
    }

    fn get_statistics(&self) -> OdeEquationsStatistics {
        self.statistics.borrow().clone()
    }
}

/// This struct implements the ODE equation trait [OdeEquations] for a given right-hand side function, jacobian function, and initial condition function.
/// These functions are provided as closures, and the parameters are assumed to be constant.
/// The mass matrix is assumed to be the identity matrix.
pub struct OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{

    rhs: Rc<Closure<M, F, G>>,
    mass: Rc<UnitCallable<M>>,
    init: I,
    p: Rc<M::V>,
    statistics: RefCell<OdeEquationsStatistics>,
}

impl<M, F, G, I> OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode(rhs: F, rhs_jac: G, init: I, p: M::V, t0: M::T, calculate_sparsity: bool) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);

        let statistics = RefCell::default();
        let mut rhs = Closure::<M, _, _>::new(rhs, rhs_jac, nstates, nstates, p.clone());
        if calculate_sparsity {
            rhs.calculate_sparsity(&y0, t0);
        }
        let mass = UnitCallable::<M>::new(nstates);
        let rhs = Rc::new(rhs);
        let mass = Rc::new(mass);
        Self {
            rhs,
            mass,
            init,
            p: p.clone(),
            statistics,
        }
    }
}


impl<M, F, G, I> OdeEquations for OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    type T = M::T;
    type V = M::V;
    type M = M;
    type Rhs = Closure<M, F, G>;
    type Mass = UnitCallable<M>;

    fn mass(&self) -> &Rc<Self::Mass> {
        &self.mass
    }

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }

    fn is_mass_constant(&self) -> bool {
        true
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }

    
    fn get_statistics(&self) -> OdeEquationsStatistics {
        self.statistics.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::ode_solver::equations::OdeEquations;
    use crate::NonLinearOp;
    use crate::LinearOp;
    use crate::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use crate::ode_solver::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::vector::Vector;

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
