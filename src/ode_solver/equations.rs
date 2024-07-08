use std::rc::Rc;

use crate::{
    op::{unit::UnitCallable, ConstantOp},
    scalar::Scalar,
    LinearOp, Matrix, NonLinearOp, Vector,
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
///  M \frac{dy}{dt} = F(t, y)
///  y(t_0) = y_0(t_0)
/// $$
///
/// The ODE equations are defined by:
/// - the right-hand side function `F(t, y)`, which is given as a [NonLinearOp] using the `Rhs` associated type and [Self::rhs] function,
/// - the initial condition `y_0(t_0)`, which is given using the [Self::init] function.
/// 
/// Optionally, the ODE equations can also include:
/// - the mass matrix `M` which is given as a [LinearOp] using the `Mass` associated type and the [Self::mass] function,
/// - the root function `G(t, y)` which is given as a [NonLinearOp] using the `Root` associated type and the [Self::root] function
/// - the output function `H(t, y)` which is given as a [NonLinearOp] using the `Out` associated type and the [Self::out] function
pub trait OdeEquations {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    type M: Matrix<T = Self::T, V = Self::V>;
    type Mass: LinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Rhs: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Root: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Init: ConstantOp<M = Self::M, V = Self::V, T = Self::T>;
    type Out: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;

    /// The parameters of the ODE equations are assumed to be constant. This function sets the parameters to the given value before solving the ODE.
    /// Note that `set_params` must always be called before calling any of the other functions in this trait.
    fn set_params(&mut self, p: Self::V);

    /// returns the right-hand side function `F(t, y)` as a [NonLinearOp]
    fn rhs(&self) -> &Rc<Self::Rhs>;

    /// returns the mass matrix `M` as a [LinearOp]
    fn mass(&self) -> Option<&Rc<Self::Mass>>;

    /// returns the root function `G(t, y)` as a [NonLinearOp]
    fn root(&self) -> Option<&Rc<Self::Root>> {
        None
    }

    /// returns the output function `H(t, y)` as a [NonLinearOp]
    fn out(&self) -> Option<&Rc<Self::Out>> {
        None
    }

    /// returns the initial condition, i.e. `y(t)`, where `t` is the initial time
    fn init(&self) -> &Rc<Self::Init>;
}

/// This struct implements the ODE equation trait [OdeEquations] for a given right-hand side op, mass op, optional root op, and initial condition function.
/// While the [crate::OdeBuilder] struct is the easiest way to define an ODE problem, occasionally a user might want to use their own structs that define the equations instead of closures or the DiffSL languave, and this can be done using [OdeSolverEquations].
///
/// The main traits that you need to implement are the [crate::Op] and [NonLinearOp] trait, which define a nonlinear operator or function `F` that maps an input vector `x` to an output vector `y`, (i.e. `y = F(x)`).
/// Once you have implemented this trait, you can then pass an instance of your struct to the `rhs` argument of the [Self::new] method. Once you have created an instance of [OdeSolverEquations], you can then use [crate::OdeSolverProblem::new] to create a problem.
///
/// For example:
///
/// ```rust
/// use std::rc::Rc;
/// use diffsol::{Bdf, OdeSolverState, OdeSolverMethod, NonLinearOp, OdeSolverEquations, OdeSolverProblem, Op, UnitCallable, ConstantClosure};
/// type M = nalgebra::DMatrix<f64>;
/// type V = nalgebra::DVector<f64>;
///
/// struct MyProblem;
/// impl Op for MyProblem {
///    type V = V;
///    type T = f64;
///    type M = M;
///    fn nstates(&self) -> usize {
///       1
///    }
///    fn nout(&self) -> usize {
///       1
///    }
/// }
///   
/// // implement rhs equations for the problem
/// impl NonLinearOp for MyProblem {
///    fn call_inplace(&self, x: &V, _t: f64, y: &mut V) {
///       y[0] = -0.1 * x[0];
///   }
///    fn jac_mul_inplace(&self, x: &V, _t: f64, v: &V, y: &mut V) {
///      y[0] = -0.1 * v[0];
///   }
/// }
///
///
/// let rhs = Rc::new(MyProblem);
///
/// // use the provided constant closure to define the initial condition
/// let init_fn = |p: &V, _t: f64| V::from_vec(vec![1.0]);
/// let init = Rc::new(ConstantClosure::new(init_fn, Rc::new(V::from_vec(vec![]))));
///
/// // we don't have a mass matrix, root or output functions, so we can set to None
/// // we still need to give a placeholder type for these, so we use the diffsol::UnitCallable type
/// let mass: Option<Rc<UnitCallable<M>>> = None;
/// let root: Option<Rc<UnitCallable<M>>> = None;
/// let out: Option<Rc<UnitCallable<M>>> = None;
///
/// let p = Rc::new(V::from_vec(vec![]));
/// let eqn = OdeSolverEquations::new(rhs, mass, root, init, out, p);
///
/// let rtol = 1e-6;
/// let atol = V::from_vec(vec![1e-6]);
/// let t0 = 0.0;
/// let h0 = 0.1;
/// let with_sensitivity = false;
/// let sensitivity_error_control = false;
/// let problem = OdeSolverProblem::new(eqn, rtol, atol, t0, h0, with_sensitivity, sensitivity_error_control).unwrap();
///
/// let mut solver = Bdf::default();
/// let t = 0.4;
/// let state = OdeSolverState::new(&problem, &solver).unwrap();
/// solver.set_problem(state, &problem);
/// while solver.state().unwrap().t <= t {
///    solver.step().unwrap();
/// }
/// let y = solver.interpolate(t);
/// ```
///
pub struct OdeSolverEquations<
    M,
    Rhs,
    Init,
    Mass = UnitCallable<M>,
    Root = UnitCallable<M>,
    Out = UnitCallable<M>,
> where
    M: Matrix,
    Rhs: NonLinearOp<M = M, V = M::V, T = M::T>,
    Mass: LinearOp<M = M, V = M::V, T = M::T>,
    Root: NonLinearOp<M = M, V = M::V, T = M::T>,
    Init: ConstantOp<M = M, V = M::V, T = M::T>,
    Out: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    rhs: Rc<Rhs>,
    mass: Option<Rc<Mass>>,
    root: Option<Rc<Root>>,
    init: Rc<Init>,
    out: Option<Rc<Out>>,
    p: Rc<M::V>,
}

impl<M, Rhs, Init, Mass, Root, Out> OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
    Rhs: NonLinearOp<M = M, V = M::V, T = M::T>,
    Mass: LinearOp<M = M, V = M::V, T = M::T>,
    Root: NonLinearOp<M = M, V = M::V, T = M::T>,
    Init: ConstantOp<M = M, V = M::V, T = M::T>,
    Out: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rhs: Rc<Rhs>,
        mass: Option<Rc<Mass>>,
        root: Option<Rc<Root>>,
        init: Rc<Init>,
        out: Option<Rc<Out>>,
        p: Rc<M::V>,
    ) -> Self {
        Self {
            rhs,
            mass,
            root,
            init,
            out,
            p,
        }
    }
}

impl<M, Rhs, Init, Mass, Root, Out> OdeEquations
    for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
    Rhs: NonLinearOp<M = M, V = M::V, T = M::T>,
    Mass: LinearOp<M = M, V = M::V, T = M::T>,
    Root: NonLinearOp<M = M, V = M::V, T = M::T>,
    Init: ConstantOp<M = M, V = M::V, T = M::T>,
    Out: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    type T = M::T;
    type V = M::V;
    type M = M;
    type Rhs = Rhs;
    type Mass = Mass;
    type Root = Root;
    type Init = Init;
    type Out = Out;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn mass(&self) -> Option<&Rc<Self::Mass>> {
        self.mass.as_ref()
    }
    fn root(&self) -> Option<&Rc<Self::Root>> {
        self.root.as_ref()
    }
    fn init(&self) -> &Rc<Self::Init> {
        &self.init
    }

    fn out(&self) -> Option<&Rc<Self::Out>> {
        self.out.as_ref()
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
        Rc::<Rhs>::get_mut(&mut self.rhs)
            .unwrap()
            .set_params(self.p.clone());
        if let Some(m) = self.mass.as_mut() {
            Rc::<Mass>::get_mut(m).unwrap().set_params(self.p.clone());
        }
        if let Some(r) = self.root.as_mut() {
            Rc::<Root>::get_mut(r).unwrap().set_params(self.p.clone())
        }
        if let Some(o) = self.out.as_mut() {
            Rc::<Out>::get_mut(o).unwrap().set_params(self.p.clone())
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
        assert!(problem.eqn.mass().is_none());
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
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
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
