use crate::{
    op::{constant_op::ConstantOpSensAdjoint, linear_op::LinearOpTranspose, ParameterisedOp},
    ConstantOp, ConstantOpSens, LinearOp, Matrix, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, Op, StochOp, UnitCallable,
    Vector,
};
use serde::Serialize;

pub mod adjoint_equations;
#[cfg(feature = "diffsl")]
pub mod diffsl;
pub mod sens_equations;
pub mod test_models;

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

pub trait AugmentedOdeEquations<Eqn: OdeEquations>:
    OdeEquations<T = Eqn::T, V = Eqn::V, M = Eqn::M, C = Eqn::C> + Clone
{
    fn update_rhs_out_state(&mut self, y: &Eqn::V, dy: &Eqn::V, t: Eqn::T);
    fn set_index(&mut self, index: usize);
    fn max_index(&self) -> usize;
    fn include_in_error_control(&self) -> bool;
    fn include_out_in_error_control(&self) -> bool;
    fn rtol(&self) -> Option<Eqn::T>;
    fn atol(&self) -> Option<&Eqn::V>;
    fn out_rtol(&self) -> Option<Eqn::T>;
    fn out_atol(&self) -> Option<&Eqn::V>;
    fn integrate_main_eqn(&self) -> bool;
}

pub trait AugmentedOdeEquationsImplicit<Eqn: OdeEquationsImplicit>:
    AugmentedOdeEquations<Eqn> + OdeEquationsImplicit<T = Eqn::T, V = Eqn::V, M = Eqn::M, C = Eqn::C>
{
}

impl<Aug, Eqn> AugmentedOdeEquationsImplicit<Eqn> for Aug
where
    Aug: AugmentedOdeEquations<Eqn>
        + OdeEquationsImplicit<T = Eqn::T, V = Eqn::V, M = Eqn::M, C = Eqn::C>,
    Eqn: OdeEquationsImplicit,
{
}

pub struct NoAug<Eqn: OdeEquations> {
    _phantom: std::marker::PhantomData<Eqn>,
}

impl<Eqn: OdeEquations> Clone for NoAug<Eqn> {
    fn clone(&self) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Eqn> Op for NoAug<Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type C = Eqn::C;

    fn nout(&self) -> usize {
        panic!("This should never be called")
    }
    fn nparams(&self) -> usize {
        panic!("This should never be called")
    }
    fn nstates(&self) -> usize {
        panic!("This should never be called")
    }
    fn statistics(&self) -> crate::op::OpStatistics {
        panic!("This should never be called")
    }
    fn context(&self) -> &Self::C {
        panic!("This should never be called")
    }
}

impl<'a, Eqn: OdeEquations> OdeEquationsRef<'a> for NoAug<Eqn> {
    type Mass = <Eqn as OdeEquationsRef<'a>>::Mass;
    type Rhs = <Eqn as OdeEquationsRef<'a>>::Rhs;
    type Root = <Eqn as OdeEquationsRef<'a>>::Root;
    type Init = <Eqn as OdeEquationsRef<'a>>::Init;
    type Out = <Eqn as OdeEquationsRef<'a>>::Out;
    type Reset = <Eqn as OdeEquationsRef<'a>>::Reset;
}

impl<Eqn: OdeEquations> OdeEquations for NoAug<Eqn> {
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        panic!("This should never be called")
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        panic!("This should never be called")
    }

    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        panic!("This should never be called")
    }

    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        panic!("This should never be called")
    }

    fn reset(&self) -> Option<<Self as OdeEquationsRef<'_>>::Reset> {
        panic!("This should never be called")
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        panic!("This should never be called")
    }

    fn set_params(&mut self, _p: &Self::V) {
        panic!("This should never be called")
    }

    fn set_model_index(&mut self, _m: usize) {
        panic!("This should never be called")
    }

    fn get_params(&self, _p: &mut Self::V) {
        panic!("This should never be called")
    }
}

impl<Eqn: OdeEquations> AugmentedOdeEquations<Eqn> for NoAug<Eqn> {
    fn update_rhs_out_state(&mut self, _y: &Eqn::V, _dy: &Eqn::V, _t: Eqn::T) {
        panic!("This should never be called")
    }
    fn set_index(&mut self, _index: usize) {
        panic!("This should never be called")
    }
    fn atol(&self) -> Option<&<Eqn as Op>::V> {
        panic!("This should never be called")
    }
    fn include_out_in_error_control(&self) -> bool {
        panic!("This should never be called")
    }
    fn out_atol(&self) -> Option<&<Eqn as Op>::V> {
        panic!("This should never be called")
    }
    fn out_rtol(&self) -> Option<<Eqn as Op>::T> {
        panic!("This should never be called")
    }
    fn rtol(&self) -> Option<<Eqn as Op>::T> {
        panic!("This should never be called")
    }
    fn max_index(&self) -> usize {
        panic!("This should never be called")
    }
    fn include_in_error_control(&self) -> bool {
        panic!("This should never be called")
    }
    fn integrate_main_eqn(&self) -> bool {
        panic!("This should never be called")
    }
}

/// this is the reference trait that defines the ODE equations of the form, this is used to define the ODE equations for a given lifetime.
/// See [OdeEquations] for the main trait that defines the ODE equations.
///
/// $$
///  M \frac{dy}{dt} = F(t, y)
///  y(t_0) = y_0(t_0)
/// $$
///
/// The ODE equations are defined by:
/// - the right-hand side function `F(t, y)`, which is given as a [NonLinearOp] using the `Rhs` associated type and [OdeEquations::rhs] function,
/// - the initial condition `y_0(t_0)`, which is given as a [ConstantOp] using the `Init` associated type and [OdeEquations::init] function.
///
/// Optionally, the ODE equations can also include:
/// - the mass matrix `M` which is given as a [LinearOp] using the `Mass` associated type and the [OdeEquations::mass] function,
/// - the root function `G(t, y)` which is given as a [NonLinearOp] using the `Root` associated type and the [OdeEquations::root] function
/// - the output function `H(t, y)` which is given as a [NonLinearOp] using the `Out` associated type and the [OdeEquations::out] function
pub trait OdeEquationsRef<'a, ImplicitBounds: Sealed = Bounds<&'a Self>>: Op {
    type Mass: LinearOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>;
    type Rhs: NonLinearOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>;
    type Root: NonLinearOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>;
    type Init: ConstantOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>;
    type Out: NonLinearOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>;
    type Reset: NonLinearOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>;
}

impl<'a, T: OdeEquationsRef<'a>> OdeEquationsRef<'a> for &T {
    type Mass = <T as OdeEquationsRef<'a>>::Mass;
    type Rhs = <T as OdeEquationsRef<'a>>::Rhs;
    type Root = <T as OdeEquationsRef<'a>>::Root;
    type Init = <T as OdeEquationsRef<'a>>::Init;
    type Out = <T as OdeEquationsRef<'a>>::Out;
    type Reset = <T as OdeEquationsRef<'a>>::Reset;
}

// seal the trait so that users must use the provided default type for ImplicitBounds
mod sealed {
    pub trait Sealed: Sized {}
    pub struct Bounds<T>(T);
    impl<T> Sealed for Bounds<T> {}
}
use sealed::{Bounds, Sealed};

/// this is the trait that defines the ODE equations of the form
///
/// $$
///  M \frac{dy}{dt} = F(t, y)
///  y(t_0) = y_0(t_0)
/// $$
///
/// The ODE equations are defined by:
/// - the right-hand side function `F(t, y)`, which is given as a [NonLinearOp] using the `Rhs` associated type and [OdeEquations::rhs] function,
/// - the initial condition `y_0(t_0)`, which is given as a [ConstantOp] using the `Init` associated type and [OdeEquations::init] function.
///
/// Optionally, the ODE equations can also include:
/// - the mass matrix `M` which is given as a [LinearOp] using the `Mass` associated type and the [OdeEquations::mass] function,
/// - the root function `G(t, y)` which is given as a [NonLinearOp] using the `Root` associated type and the [OdeEquations::root] function
/// - the output function `H(t, y)` which is given as a [NonLinearOp] using the `Out` associated type and the [OdeEquations::out] function
pub trait OdeEquations: for<'a> OdeEquationsRef<'a> {
    /// returns the right-hand side function `F(t, y)` as a [NonLinearOp]
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs;

    /// returns the mass matrix `M` as a [LinearOp]
    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass>;

    /// returns the root function `G(t, y)` as a [NonLinearOp]
    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        None
    }

    /// returns the output function `H(t, y)` as a [NonLinearOp]
    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        None
    }

    /// returns the optional reset function `R(y)` as a [NonLinearOp].
    /// When provided, this is called after a root event at index 0 to reset the state.
    fn reset(&self) -> Option<<Self as OdeEquationsRef<'_>>::Reset> {
        None
    }

    /// returns the initial condition, i.e. `y(t)`, where `t` is the initial time
    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init;

    /// sets the current parameters of the equations
    fn set_params(&mut self, p: &Self::V);

    /// sets the current model index of the equations
    ///
    /// Implementations that do not support multiple models may ignore `m`.
    fn set_model_index(&mut self, _m: usize) {}

    /// gets the current parameters of the equations
    fn get_params(&self, p: &mut Self::V);
}

impl<T: OdeEquations> OdeEquations for &'_ T {
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        (*self).rhs()
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        (*self).mass()
    }

    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        (*self).root()
    }

    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        (*self).out()
    }

    fn reset(&self) -> Option<<Self as OdeEquationsRef<'_>>::Reset> {
        (*self).reset()
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        (*self).init()
    }

    fn set_params(&mut self, _p: &Self::V) {
        unimplemented!()
    }

    fn set_model_index(&mut self, _m: usize) {
        unimplemented!()
    }

    fn get_params(&self, p: &mut Self::V) {
        (*self).get_params(p)
    }
}

pub trait OdeEquationsImplicit:
    OdeEquations<Rhs: NonLinearOpJacobian<M = Self::M, V = Self::V, T = Self::T, C = Self::C>>
{
}

impl<T> OdeEquationsImplicit for T where
    T: OdeEquations<Rhs: NonLinearOpJacobian<M = T::M, V = T::V, T = T::T, C = T::C>>
{
}

pub trait OdeEquationsStoch:
    OdeEquations<
    Rhs: NonLinearOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>
             + StochOp<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
>
{
}

impl<T> OdeEquationsStoch for T where
    T: OdeEquations<
        Rhs: NonLinearOp<M = T::M, V = T::V, T = T::T, C = T::C>
                 + StochOp<M = T::M, V = T::V, T = T::T, C = T::C>,
    >
{
}

pub trait OdeEquationsImplicitSens:
    OdeEquationsImplicit<
    Rhs: NonLinearOpSens<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Out: NonLinearOpSens<M = Self::M, V = Self::V, T = Self::T, C = Self::C>
             + NonLinearOpJacobian<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Init: ConstantOpSens<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Reset: NonLinearOpJacobian<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
>
{
}

impl<T> OdeEquationsImplicitSens for T where
    T: OdeEquationsImplicit<
        Rhs: NonLinearOpSens<M = T::M, V = T::V, T = T::T, C = T::C>,
        Out: NonLinearOpSens<M = T::M, V = T::V, T = T::T, C = T::C>
                 + NonLinearOpJacobian<M = T::M, V = T::V, T = T::T, C = T::C>,
        Init: ConstantOpSens<M = T::M, V = T::V, T = T::T, C = T::C>,
        Reset: NonLinearOpJacobian<M = T::M, V = T::V, T = T::T, C = T::C>,
    >
{
}

pub trait OdeEquationsImplicitAdjoint:
    OdeEquationsImplicit<
    Rhs: NonLinearOpAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>
             + NonLinearOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Init: ConstantOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Out: NonLinearOpAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>
             + NonLinearOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Mass: LinearOpTranspose<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
>
{
}

impl<T> OdeEquationsImplicitAdjoint for T where
    T: OdeEquationsImplicit<
        Rhs: NonLinearOpAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>
                 + NonLinearOpSensAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>,
        Init: ConstantOpSensAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>,
        Out: NonLinearOpAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>
                 + NonLinearOpSensAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>,
        Mass: LinearOpTranspose<M = T::M, V = T::V, T = T::T, C = T::C>,
    >
{
}

pub trait OdeEquationsAdjoint:
    OdeEquations<
    Rhs: NonLinearOpAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>
             + NonLinearOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Init: ConstantOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Out: NonLinearOpAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>
             + NonLinearOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
    Mass: LinearOpTranspose<M = Self::M, V = Self::V, T = Self::T, C = Self::C>,
>
{
}

impl<T> OdeEquationsAdjoint for T where
    T: OdeEquations<
        Rhs: NonLinearOpAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>
                 + NonLinearOpSensAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>,
        Init: ConstantOpSensAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>,
        Out: NonLinearOpAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>
                 + NonLinearOpSensAdjoint<M = T::M, V = T::V, T = T::T, C = T::C>,
        Mass: LinearOpTranspose<M = T::M, V = T::V, T = T::T, C = T::C>,
    >
{
}

/// This struct implements the ODE equation trait [OdeEquations] for a given right-hand side op, mass op, optional root op, and initial condition function.
///
/// While the [crate::OdeBuilder] struct is the easiest way to define an ODE problem,
/// occasionally a user might want to use their own structs that define the equations instead of closures or the DiffSL languave,
/// and this can be done using [OdeSolverEquations].
///
/// The main traits that you need to implement are the [crate::Op] and [NonLinearOp] trait,
/// which define a nonlinear operator or function `F` that maps an input vector `x` to an output vector `y`, (i.e. `y = F(x)`).
/// Once you have implemented this trait, you can then pass an instance of your struct to the `rhs` argument of the [Self::new] method.
/// Once you have created an instance of [OdeSolverEquations], you can then use [crate::OdeBuilder::build_from_eqn] to create a problem.
pub struct OdeSolverEquations<M, Rhs, Init, Mass, Root, Out, Reset = UnitCallable<M>>
where
    M: Matrix,
{
    rhs: Rhs,
    mass: Option<Mass>,
    root: Option<Root>,
    init: Init,
    out: Option<Out>,
    reset: Option<Reset>,
    p: M::V,
}

impl<M, Rhs, Init, Mass, Root, Out, Reset> OdeSolverEquations<M, Rhs, Init, Mass, Root, Out, Reset>
where
    M: Matrix,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rhs: Rhs,
        init: Init,
        mass: Option<Mass>,
        root: Option<Root>,
        out: Option<Out>,
        reset: Option<Reset>,
        p: M::V,
    ) -> Self {
        Self {
            rhs,
            mass,
            root,
            init,
            out,
            reset,
            p,
        }
    }
    fn params_mut(&mut self) -> &mut M::V {
        &mut self.p
    }
    fn params(&self) -> &M::V {
        &self.p
    }
}

impl<M, Rhs, Init, Mass, Root, Out, Reset> Op
    for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out, Reset>
where
    M: Matrix,
    Init: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Rhs: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Mass: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Root: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Out: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Reset: Op<M = M, V = M::V, T = M::T, C = M::C>,
{
    type T = M::T;
    type V = M::V;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        self.rhs.nstates()
    }
    fn nout(&self) -> usize {
        self.out
            .as_ref()
            .map(|out| out.nout())
            .unwrap_or(self.rhs.nout())
    }
    fn nparams(&self) -> usize {
        self.rhs.nparams()
    }
    fn statistics(&self) -> crate::op::OpStatistics {
        self.rhs.statistics()
    }
    fn context(&self) -> &Self::C {
        self.rhs.context()
    }
}

impl<'a, M, Rhs, Init, Mass, Root, Out, Reset> OdeEquationsRef<'a>
    for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out, Reset>
where
    M: Matrix,
    Rhs: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Init: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Mass: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Root: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Out: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Reset: Op<M = M, V = M::V, T = M::T, C = M::C>,
    ParameterisedOp<'a, Rhs>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    ParameterisedOp<'a, Init>: ConstantOp<M = M, V = M::V, T = M::T, C = M::C>,
    ParameterisedOp<'a, Mass>: LinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    ParameterisedOp<'a, Root>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    ParameterisedOp<'a, Out>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    ParameterisedOp<'a, Reset>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
{
    type Rhs = ParameterisedOp<'a, Rhs>;
    type Mass = ParameterisedOp<'a, Mass>;
    type Root = ParameterisedOp<'a, Root>;
    type Init = ParameterisedOp<'a, Init>;
    type Out = ParameterisedOp<'a, Out>;
    type Reset = ParameterisedOp<'a, Reset>;
}

impl<M, Rhs, Init, Mass, Root, Out, Reset> OdeEquations
    for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out, Reset>
where
    M: Matrix,
    Rhs: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Init: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Mass: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Root: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Out: Op<M = M, V = M::V, T = M::T, C = M::C>,
    Reset: Op<M = M, V = M::V, T = M::T, C = M::C>,
    for<'a> ParameterisedOp<'a, Rhs>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    for<'a> ParameterisedOp<'a, Init>: ConstantOp<M = M, V = M::V, T = M::T, C = M::C>,
    for<'a> ParameterisedOp<'a, Mass>: LinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    for<'a> ParameterisedOp<'a, Root>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    for<'a> ParameterisedOp<'a, Out>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    for<'a> ParameterisedOp<'a, Reset>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
{
    fn rhs(&self) -> ParameterisedOp<'_, Rhs> {
        ParameterisedOp::new(&self.rhs, self.params())
    }
    fn mass(&self) -> Option<ParameterisedOp<'_, Mass>> {
        self.mass
            .as_ref()
            .map(|mass| ParameterisedOp::new(mass, self.params()))
    }
    fn root(&self) -> Option<ParameterisedOp<'_, Root>> {
        self.root
            .as_ref()
            .map(|root| ParameterisedOp::new(root, self.params()))
    }
    fn reset(&self) -> Option<ParameterisedOp<'_, Reset>> {
        self.reset
            .as_ref()
            .map(|reset| ParameterisedOp::new(reset, self.params()))
    }
    fn init(&self) -> ParameterisedOp<'_, Init> {
        ParameterisedOp::new(&self.init, self.params())
    }
    fn out(&self) -> Option<ParameterisedOp<'_, Out>> {
        self.out
            .as_ref()
            .map(|out| ParameterisedOp::new(out, self.params()))
    }
    fn set_params(&mut self, p: &Self::V) {
        self.params_mut().copy_from(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        p.copy_from(self.params());
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::context::nalgebra::NalgebraContext;
    use crate::matrix::dense_nalgebra_serial::NalgebraMat;
    use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem;
    use crate::ode_equations::test_models::exponential_decay::{
        exponential_decay_problem_with_root, exponential_decay_with_reset_problem,
    };
    use crate::ode_equations::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::vector::Vector;
    use crate::OdeEquations;
    use crate::{
        ConstantOp, Context, DenseMatrix, LinearOp, NonLinearOp, NonLinearOpJacobian, Op,
        UnitCallable,
    };

    use super::{AugmentedOdeEquations, NoAug, OdeEquationsStatistics};

    type Mcpu = NalgebraMat<f64>;

    #[derive(Clone)]
    struct FakeEqn {
        ctx: NalgebraContext,
    }

    #[derive(Clone)]
    struct FakeInit {
        ctx: NalgebraContext,
    }

    impl Op for FakeEqn {
        type T = f64;
        type V = crate::NalgebraVec<f64>;
        type M = Mcpu;
        type C = NalgebraContext;

        fn context(&self) -> &Self::C {
            &self.ctx
        }
        fn nstates(&self) -> usize {
            1
        }
        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            0
        }
    }

    impl<'a> super::OdeEquationsRef<'a> for FakeEqn {
        type Mass = UnitCallable<Mcpu>;
        type Rhs = UnitCallable<Mcpu>;
        type Root = UnitCallable<Mcpu>;
        type Init = FakeInit;
        type Out = UnitCallable<Mcpu>;
        type Reset = UnitCallable<Mcpu>;
    }

    impl Op for FakeInit {
        type T = f64;
        type V = crate::NalgebraVec<f64>;
        type M = Mcpu;
        type C = NalgebraContext;

        fn context(&self) -> &Self::C {
            &self.ctx
        }
        fn nstates(&self) -> usize {
            1
        }
        fn nout(&self) -> usize {
            1
        }
        fn nparams(&self) -> usize {
            0
        }
    }

    impl ConstantOp for FakeInit {
        fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
            y.fill(0.0);
        }
    }

    impl OdeEquations for FakeEqn {
        fn rhs(&self) -> <Self as super::OdeEquationsRef<'_>>::Rhs {
            UnitCallable::new(1, self.ctx)
        }

        fn mass(&self) -> Option<<Self as super::OdeEquationsRef<'_>>::Mass> {
            None
        }

        fn init(&self) -> <Self as super::OdeEquationsRef<'_>>::Init {
            FakeInit { ctx: self.ctx }
        }

        fn set_params(&mut self, _p: &Self::V) {}

        fn get_params(&self, _p: &mut Self::V) {}
    }

    #[test]
    fn ode_equation_test() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let y = problem.context().vector_from_vec(vec![1.0, 1.0]);
        let rhs_y = problem.eqn.rhs().call(&y, 0.0);
        let expect_rhs_y = problem.context().vector_from_vec(vec![-0.1, -0.1]);
        rhs_y.assert_eq_st(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.rhs().jac_mul(&y, 0.0, &y);
        let expect_jac_rhs_y = problem.context().vector_from_vec(vec![-0.1, -0.1]);
        jac_rhs_y.assert_eq_st(&expect_jac_rhs_y, 1e-10);
        assert!(problem.eqn.mass().is_none());
        let jac = problem.eqn.rhs().jacobian(&y, 0.0);
        assert_eq!(jac.get_index(0, 0), -0.1);
        assert_eq!(jac.get_index(1, 1), -0.1);
        assert_eq!(jac.get_index(0, 1), 0.0);
        assert_eq!(jac.get_index(1, 0), 0.0);
    }

    #[test]
    fn ode_with_mass_test() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y = problem.context().vector_from_vec(vec![1.0, 1.0, 1.0]);
        let rhs_y = problem.eqn.rhs().call(&y, 0.0);
        let expect_rhs_y = problem.context().vector_from_vec(vec![-0.1, -0.1, 0.0]);
        rhs_y.assert_eq_st(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.rhs().jac_mul(&y, 0.0, &y);
        let expect_jac_rhs_y = problem.context().vector_from_vec(vec![-0.1, -0.1, 0.0]);
        jac_rhs_y.assert_eq_st(&expect_jac_rhs_y, 1e-10);
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        assert_eq!(mass.get_index(0, 0), 1.);
        assert_eq!(mass.get_index(1, 1), 1.);
        assert_eq!(mass.get_index(2, 2), 0.);
        assert_eq!(mass.get_index(0, 1), 0.);
        assert_eq!(mass.get_index(1, 0), 0.);
        assert_eq!(mass.get_index(0, 2), 0.);
        assert_eq!(mass.get_index(2, 0), 0.);
        assert_eq!(mass.get_index(1, 2), 0.);
        assert_eq!(mass.get_index(2, 1), 0.);
        let jac = problem.eqn.rhs().jacobian(&y, 0.0);
        assert_eq!(jac.get_index(0, 0), -0.1);
        assert_eq!(jac.get_index(1, 1), -0.1);
        assert_eq!(jac.get_index(2, 2), 1.0);
        assert_eq!(jac.get_index(0, 1), 0.0);
        assert_eq!(jac.get_index(1, 0), 0.0);
        assert_eq!(jac.get_index(0, 2), 0.0);
        assert_eq!(jac.get_index(2, 0), 0.0);
        assert_eq!(jac.get_index(1, 2), 0.0);
        assert_eq!(jac.get_index(2, 1), -1.0);
    }

    #[test]
    fn ode_equations_statistics_new_matches_default() {
        let stats = OdeEquationsStatistics::new();
        let default_stats = OdeEquationsStatistics::default();
        assert_eq!(stats.number_of_rhs_evals, default_stats.number_of_rhs_evals);
        assert_eq!(
            stats.number_of_jac_mul_evals,
            default_stats.number_of_jac_mul_evals
        );
        assert_eq!(stats.number_of_mass_evals, default_stats.number_of_mass_evals);
        assert_eq!(
            stats.number_of_mass_matrix_evals,
            default_stats.number_of_mass_matrix_evals
        );
        assert_eq!(
            stats.number_of_jacobian_matrix_evals,
            default_stats.number_of_jacobian_matrix_evals
        );
    }

    #[test]
    fn ode_solver_equations_optional_members_and_param_roundtrip_work() {
        let (mut problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        assert_eq!(problem.eqn.nout(), problem.eqn.rhs().nout());
        assert!(problem.eqn.mass().is_none());
        assert!(problem.eqn.root().is_none());
        assert!(problem.eqn.out().is_none());
        assert!(problem.eqn.reset().is_none());

        let p = problem.context().vector_from_vec(vec![0.2, 3.0]);
        problem.eqn.set_params(&p);
        let mut out = problem.context().vector_zeros(2);
        problem.eqn.get_params(&mut out);
        out.assert_eq_st(&p, 1e-12);
    }

    #[test]
    fn ode_solver_equations_expose_optional_root_and_reset_members() {
        let (problem_with_root, _) = exponential_decay_problem_with_root::<Mcpu>(false, false);
        assert!(problem_with_root.eqn.root().is_some());
        assert!(problem_with_root.eqn.reset().is_none());

        let (problem_with_reset, _) = exponential_decay_with_reset_problem::<Mcpu>();
        assert!(problem_with_reset.eqn.root().is_some());
        assert!(problem_with_reset.eqn.reset().is_some());
    }

    #[test]
    fn ode_equations_reference_impl_forwards_reads() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let eqn_ref = &problem.eqn;
        let y = problem.context().vector_from_vec(vec![1.0, 1.0]);
        eqn_ref
            .rhs()
            .call(&y, 0.0)
            .assert_eq_st(&problem.context().vector_from_vec(vec![-0.1, -0.1]), 1e-12);
        assert!(eqn_ref.mass().is_none());
        assert!(eqn_ref.root().is_none());
        assert!(eqn_ref.out().is_none());
        assert!(eqn_ref.reset().is_none());
        eqn_ref
            .init()
            .call(0.0)
            .assert_eq_st(&problem.context().vector_from_vec(vec![1.0, 1.0]), 1e-12);

        let mut p = problem.context().vector_zeros(2);
        eqn_ref.get_params(&mut p);
        p.assert_eq_st(&problem.context().vector_from_vec(vec![0.1, 1.0]), 1e-12);
    }

    #[test]
    fn no_aug_panics_for_all_runtime_methods() {
        let no_aug = NoAug::<FakeEqn> {
            _phantom: std::marker::PhantomData,
        };
        let mut v = crate::NalgebraVec::zeros(0, NalgebraContext);
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.nout())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.nparams())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.nstates())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.statistics())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.context())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.rhs())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.mass())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.root())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.out())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.reset())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.init())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut no_aug = no_aug.clone();
            no_aug.set_params(&v)
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut no_aug = no_aug.clone();
            no_aug.set_model_index(0)
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.get_params(&mut v))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut no_aug = no_aug.clone();
            no_aug.update_rhs_out_state(&v, &v, 0.0)
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut no_aug = no_aug.clone();
            no_aug.set_index(0)
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.atol())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.include_out_in_error_control())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.out_atol())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.out_rtol())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.rtol())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.max_index())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.include_in_error_control())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| no_aug.integrate_main_eqn())).is_err());
    }
}
