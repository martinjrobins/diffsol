use crate::{
    error::{DiffsolError, OdeSolverError},
    matrix::dense_nalgebra_serial::NalgebraMat,
    ode_solver_error,
    op::{linear_closure_with_adjoint::LinearClosureWithAdjoint, BuilderOp},
    Closure, ClosureNoJac, ClosureWithAdjoint, ClosureWithSens, ConstantClosure,
    ConstantClosureWithAdjoint, ConstantClosureWithSens, ConstantOp, LinearClosure, LinearOp,
    Matrix, NonLinearOp, OdeEquations, OdeSolverProblem, Op, ParameterisedOp, UnitCallable, Vector,
};

use crate::OdeSolverEquations;
use num_traits::{FromPrimitive, One, Zero};

/// Builder for ODE problems. Use methods to set parameters and then call one of the build methods when done.
pub struct OdeBuilder<
    M: Matrix = NalgebraMat<f64>,
    Rhs = UnitCallable<M>,
    Init = UnitCallable<M>,
    Mass = UnitCallable<M>,
    Root = UnitCallable<M>,
    Out = UnitCallable<M>,
> {
    t0: M::T,
    h0: M::T,
    rtol: M::T,
    atol: Vec<M::T>,
    sens_atol: Option<Vec<M::T>>,
    sens_rtol: Option<M::T>,
    out_rtol: Option<M::T>,
    out_atol: Option<Vec<M::T>>,
    param_rtol: Option<M::T>,
    param_atol: Option<Vec<M::T>>,
    p: Vec<M::T>,
    use_coloring: bool,
    integrate_out: bool,
    rhs: Option<Rhs>,
    init: Option<Init>,
    mass: Option<Mass>,
    root: Option<Root>,
    out: Option<Out>,
    ctx: M::C,
}

impl Default for OdeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for ODE problems. Use methods to set parameters and then call one of the build methods when done.
///
/// # Example
///  
/// ```rust
/// use diffsol::{OdeBuilder, NalgebraLU, Bdf, OdeSolverState, OdeSolverMethod, NalgebraMat};
/// type M = NalgebraMat<f64>;
/// type LS = NalgebraLU<f64>;
///
/// let problem = OdeBuilder::<M>::new()
///   .rtol(1e-6)
///   .p([0.1])
///   .rhs_implicit(
///     // dy/dt = -ay
///     |x, p, t, y| {
///       y[0] = -p[0] * x[0];
///     },
///     // Jv = -av
///     |x, p, t, v, y| {
///       y[0] = -p[0] * v[0];
///     },
///   )
///   .init(
///     // y(0) = 1
///    |p, t, y| y[0] = 1.0,
///    1,
///   )
///   .build()
///   .unwrap();
///
/// let mut solver = problem.bdf::<LS>().unwrap();
/// let t = 0.4;
/// while solver.state().t <= t {
///     solver.step().unwrap();
/// }
/// let y = solver.interpolate(t);
/// ```
///
impl<M, Rhs, Init, Mass, Root, Out> OdeBuilder<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
{
    /// Create a new builder with default parameters:
    /// - t0 = 0.0
    /// - h0 = 1.0
    /// - rtol = 1e-6
    /// - atol = [1e-6]
    /// - p = []
    /// - use_coloring = false
    /// - constant_mass = false
    pub fn new() -> Self {
        let default_atol = vec![M::T::from_f64(1e-6).unwrap()];
        let default_rtol = M::T::from_f64(1e-6).unwrap();
        Self {
            rhs: None,
            init: None,
            mass: None,
            root: None,
            out: None,
            t0: M::T::zero(),
            h0: M::T::one(),
            rtol: default_rtol,
            atol: default_atol.clone(),
            p: vec![],
            use_coloring: false,
            integrate_out: false,
            out_rtol: Some(default_rtol),
            out_atol: Some(default_atol.clone()),
            param_rtol: Some(default_rtol),
            param_atol: Some(default_atol.clone()),
            sens_atol: Some(default_atol),
            sens_rtol: Some(default_rtol),
            ctx: M::C::default(),
        }
    }

    /// Set the right-hand side of the ODE.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    pub fn rhs<F>(self, rhs: F) -> OdeBuilder<M, ClosureNoJac<M, F>, Init, Mass, Root, Out>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, ClosureNoJac<M, F>, Init, Mass, Root, Out> {
            rhs: Some(ClosureNoJac::new(
                rhs,
                nstates,
                nstates,
                nstates,
                self.ctx.clone(),
            )),
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the right-hand side of the ODE.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    pub fn rhs_implicit<F, G>(
        self,
        rhs: F,
        rhs_jac: G,
    ) -> OdeBuilder<M, Closure<M, F, G>, Init, Mass, Root, Out>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Closure<M, F, G>, Init, Mass, Root, Out> {
            rhs: Some(Closure::new(
                rhs,
                rhs_jac,
                nstates,
                nstates,
                nstates,
                self.ctx.clone(),
            )),
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the right-hand side of the ODE for forward sensitivity analysis.
    ///
    /// # Arguments
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    /// - `rhs_sens`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the partial derivative of the rhs wrt the parameters, with the vector v.
    pub fn rhs_sens_implicit<F, G, H>(
        self,
        rhs: F,
        rhs_jac: G,
        rhs_sens: H,
    ) -> OdeBuilder<M, ClosureWithSens<M, F, G, H>, Init, Mass, Root, Out>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, ClosureWithSens<M, F, G, H>, Init, Mass, Root, Out> {
            rhs: Some(ClosureWithSens::new(
                rhs,
                rhs_jac,
                rhs_sens,
                nstates,
                nstates,
                nstates,
                self.ctx.clone(),
            )),
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn rhs_adjoint_implicit<F, G, H, I>(
        self,
        rhs: F,
        rhs_jac: G,
        rhs_adjoint: H,
        rhs_sens_adjoint: I,
    ) -> OdeBuilder<M, ClosureWithAdjoint<M, F, G, H, I>, Init, Mass, Root, Out>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, ClosureWithAdjoint<M, F, G, H, I>, Init, Mass, Root, Out> {
            rhs: Some(ClosureWithAdjoint::new(
                rhs,
                rhs_jac,
                rhs_adjoint,
                rhs_sens_adjoint,
                nstates,
                nstates,
                nstates,
                self.ctx.clone(),
            )),
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the initial condition of the ODE.
    ///
    /// # Arguments
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    pub fn init<F>(
        self,
        init: F,
        nstates: usize,
    ) -> OdeBuilder<M, Rhs, ConstantClosure<M, F>, Mass, Root, Out>
    where
        F: Fn(&M::V, M::T, &mut M::V),
    {
        OdeBuilder::<M, Rhs, ConstantClosure<M, F>, Mass, Root, Out> {
            rhs: self.rhs,
            init: Some(ConstantClosure::new(init, nstates, 0, self.ctx.clone())),
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the initial condition of the ODE for forward sensitivity analysis.
    ///
    /// # Arguments
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    /// - `init_sens`: Function of type Fn(p: &V, t: S, y: &mut V) that computes the multiplication of the partial derivative of the initial state wrt the parameters, with the vector v.
    pub fn init_sens<F, G>(
        self,
        init: F,
        init_sens: G,
        nstates: usize,
    ) -> OdeBuilder<M, Rhs, ConstantClosureWithSens<M, F, G>, Mass, Root, Out>
    where
        F: Fn(&M::V, M::T, &mut M::V),
        G: Fn(&M::V, M::T, &M::V, &mut M::V),
    {
        OdeBuilder::<M, Rhs, ConstantClosureWithSens<M, F, G>, Mass, Root, Out> {
            rhs: self.rhs,
            init: Some(ConstantClosureWithSens::new(
                init,
                init_sens,
                nstates,
                0,
                self.ctx.clone(),
            )),
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the initial condition of the ODE for adjoint sensitivity analysis.
    ///
    /// # Arguments
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    /// - `init_sens_adjoint`: Function of type Fn(p: &V, t: S, y: &V, y_adj: &mut V) that computes the multiplication of the partial derivative of the initial state wrt the parameters, with the vector v.
    ///
    pub fn init_adjoint<F, G>(
        self,
        init: F,
        init_sens_adjoint: G,
        nstates: usize,
    ) -> OdeBuilder<M, Rhs, ConstantClosureWithAdjoint<M, F, G>, Mass, Root, Out>
    where
        F: Fn(&M::V, M::T, &mut M::V),
        G: Fn(&M::V, M::T, &M::V, &mut M::V),
    {
        OdeBuilder::<M, Rhs, ConstantClosureWithAdjoint<M, F, G>, Mass, Root, Out> {
            rhs: self.rhs,
            init: Some(ConstantClosureWithAdjoint::new(
                init,
                init_sens_adjoint,
                nstates,
                0,
                self.ctx.clone(),
            )),
            mass: self.mass,
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the mass matrix of the ODE.
    ///
    /// # Arguments
    /// - `mass`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the mass matrix with the vector v (i.e. y = M * v + beta * y).
    pub fn mass<F>(self, mass: F) -> OdeBuilder<M, Rhs, Init, LinearClosure<M, F>, Root, Out>
    where
        F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, Init, LinearClosure<M, F>, Root, Out> {
            rhs: self.rhs,
            init: self.init,
            mass: Some(LinearClosure::new(
                mass,
                nstates,
                nstates,
                nstates,
                self.ctx.clone(),
            )),
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the mass matrix of the ODE for adjoint sensitivity analysis.
    ///
    /// # Arguments
    ///
    /// - `mass`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the mass matrix with
    ///   the vector v (i.e. y = M * v + beta * y).
    /// - `mass_adjoint`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the transpose of the mass matrix with
    ///   the vector v (i.e. y = M^T * v + beta * y).
    pub fn mass_adjoint<F, G>(
        self,
        mass: F,
        mass_adjoint: G,
    ) -> OdeBuilder<M, Rhs, Init, LinearClosureWithAdjoint<M, F, G>, Root, Out>
    where
        F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, Init, LinearClosureWithAdjoint<M, F, G>, Root, Out> {
            rhs: self.rhs,
            init: self.init,
            mass: Some(LinearClosureWithAdjoint::new(
                mass,
                mass_adjoint,
                nstates,
                nstates,
                nstates,
                self.ctx.clone(),
            )),
            root: self.root,
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set a root equation for the ODE.
    ///
    /// # Arguments
    /// - `root`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the root function.
    /// - `nroots`: Number of roots (i.e. number of elements in the `y` arg in `root`), an event is triggered when any of the roots changes sign.
    pub fn root<F>(
        self,
        root: F,
        nroots: usize,
    ) -> OdeBuilder<M, Rhs, Init, Mass, ClosureNoJac<M, F>, Out>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, Init, Mass, ClosureNoJac<M, F>, Out> {
            rhs: self.rhs,
            init: self.init,
            mass: self.mass,
            root: Some(ClosureNoJac::new(
                root,
                nstates,
                nroots,
                nroots,
                self.ctx.clone(),
            )),
            out: self.out,

            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    pub fn out_implicit<F, G>(
        self,
        out: F,
        out_jac: G,
        nout: usize,
    ) -> OdeBuilder<M, Rhs, Init, Mass, Root, Closure<M, F, G>>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, Init, Mass, Root, Closure<M, F, G>> {
            rhs: self.rhs,
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: Some(Closure::new(
                out,
                out_jac,
                nstates,
                nout,
                nstates,
                self.ctx.clone(),
            )),
            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    pub fn out_sens_implicit<F, G, H>(
        self,
        out: F,
        out_jac: G,
        out_sens: H,
        nout: usize,
    ) -> OdeBuilder<M, Rhs, Init, Mass, Root, ClosureWithSens<M, F, G, H>>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, Init, Mass, Root, ClosureWithSens<M, F, G, H>> {
            rhs: self.rhs,
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: Some(ClosureWithSens::new(
                out,
                out_jac,
                out_sens,
                nstates,
                nstates,
                nout,
                self.ctx.clone(),
            )),
            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn out_adjoint_implicit<F, G, H, I>(
        self,
        out: F,
        out_jac: G,
        out_adjoint: H,
        out_sens_adjoint: I,
        nout: usize,
    ) -> OdeBuilder<M, Rhs, Init, Mass, Root, ClosureWithAdjoint<M, F, G, H, I>>
    where
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, Init, Mass, Root, ClosureWithAdjoint<M, F, G, H, I>> {
            rhs: self.rhs,
            init: self.init,
            mass: self.mass,
            root: self.root,
            out: Some(ClosureWithAdjoint::new(
                out,
                out_jac,
                out_adjoint,
                out_sens_adjoint,
                nstates,
                nout,
                nstates,
                self.ctx.clone(),
            )),
            t0: self.t0,
            h0: self.h0,
            rtol: self.rtol,
            atol: self.atol,
            sens_atol: self.sens_atol,
            sens_rtol: self.sens_rtol,
            out_rtol: self.out_rtol,
            out_atol: self.out_atol,
            param_rtol: self.param_rtol,
            param_atol: self.param_atol,
            p: self.p,
            use_coloring: self.use_coloring,
            integrate_out: self.integrate_out,
            ctx: self.ctx,
        }
    }

    /// Set the initial time.
    pub fn t0(mut self, t0: f64) -> Self {
        self.t0 = M::T::from_f64(t0).unwrap();
        self
    }

    pub fn sens_rtol(mut self, sens_rtol: f64) -> Self {
        self.sens_rtol = Some(M::T::from_f64(sens_rtol).unwrap());
        self
    }

    pub fn sens_atol<V>(mut self, sens_atol: V) -> Self
    where
        V: IntoIterator<Item = f64>,
    {
        self.sens_atol = Some(
            sens_atol
                .into_iter()
                .map(|x| M::T::from_f64(x).unwrap())
                .collect(),
        );
        self
    }

    pub fn turn_off_sensitivities_error_control(mut self) -> Self {
        self.sens_atol = None;
        self.sens_rtol = None;
        self
    }

    pub fn turn_off_output_error_control(mut self) -> Self {
        self.out_atol = None;
        self.out_rtol = None;
        self
    }

    pub fn turn_off_param_error_control(mut self) -> Self {
        self.param_atol = None;
        self.param_rtol = None;
        self
    }

    pub fn out_rtol(mut self, out_rtol: f64) -> Self {
        self.out_rtol = Some(M::T::from_f64(out_rtol).unwrap());
        self
    }

    pub fn out_atol<V, T>(mut self, out_atol: V) -> Self
    where
        V: IntoIterator<Item = T>,
        M::T: From<T>,
    {
        self.out_atol = Some(out_atol.into_iter().map(|x| M::T::from(x)).collect());
        self
    }

    pub fn param_rtol(mut self, param_rtol: f64) -> Self {
        self.param_rtol = Some(M::T::from_f64(param_rtol).unwrap());
        self
    }

    pub fn param_atol<V>(mut self, param_atol: V) -> Self
    where
        V: IntoIterator<Item = f64>,
    {
        self.param_atol = Some(
            param_atol
                .into_iter()
                .map(|x| M::T::from_f64(x).unwrap())
                .collect(),
        );
        self
    }

    /// Set whether to integrate the output.
    /// If true, the output will be integrated using the same method as the ODE.
    pub fn integrate_out(mut self, integrate_out: bool) -> Self {
        self.integrate_out = integrate_out;
        self
    }

    /// Set the initial step size.
    pub fn h0(mut self, h0: f64) -> Self {
        self.h0 = M::T::from_f64(h0).unwrap();
        self
    }

    /// Set the relative tolerance.
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = M::T::from_f64(rtol).unwrap();
        self
    }

    /// Set the absolute tolerance.
    pub fn atol<V>(mut self, atol: V) -> Self
    where
        V: IntoIterator<Item = f64>,
    {
        self.atol = atol
            .into_iter()
            .map(|x| M::T::from_f64(x).unwrap())
            .collect();
        self
    }

    /// Set the parameters.
    pub fn p<V>(mut self, p: V) -> Self
    where
        V: IntoIterator<Item = f64>,
    {
        self.p = p.into_iter().map(|x| M::T::from_f64(x).unwrap()).collect();
        self
    }

    /// Set whether to use coloring when computing the Jacobian.
    /// This is always true if matrix type is sparse, but can be set to true for dense matrices as well.
    /// This can speed up the computation of the Jacobian for large sparse systems.
    /// However, it relys on the sparsity of the Jacobian being constant,
    /// and for certain systems it may detect the wrong sparsity pattern.
    pub fn use_coloring(mut self, use_coloring: bool) -> Self {
        self.use_coloring = use_coloring;
        self
    }

    fn build_atol(
        atol: Vec<M::T>,
        nstates: usize,
        ty: &str,
        ctx: M::C,
    ) -> Result<M::V, DiffsolError> {
        if atol.len() == 1 {
            Ok(M::V::from_element(nstates, atol[0], ctx))
        } else if atol.len() != nstates {
            Err(ode_solver_error!(
                BuilderError,
                format!(
                    "Invalid number of {} absolute tolerances. Expected 1 or {}, got {}.",
                    ty,
                    nstates,
                    atol.len()
                )
            ))
        } else {
            Ok(M::V::from_vec(atol, ctx))
        }
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn build_atols(
        atol: Vec<M::T>,
        sens_atol: Option<Vec<M::T>>,
        out_atol: Option<Vec<M::T>>,
        param_atol: Option<Vec<M::T>>,
        nstates: usize,
        nout: Option<usize>,
        nparam: usize,
        ctx: M::C,
    ) -> Result<(M::V, Option<M::V>, Option<M::V>, Option<M::V>), DiffsolError> {
        let atol = Self::build_atol(atol, nstates, "states", ctx.clone())?;
        let out_atol = match out_atol {
            Some(out_atol) => Some(Self::build_atol(
                out_atol,
                nout.unwrap_or(nstates),
                "output",
                ctx.clone(),
            )?),
            None => None,
        };
        let param_atol = match param_atol {
            Some(param_atol) => Some(Self::build_atol(
                param_atol,
                nparam,
                "parameters",
                ctx.clone(),
            )?),
            None => None,
        };
        let sens_atol = match sens_atol {
            Some(sens_atol) => Some(Self::build_atol(
                sens_atol,
                nstates,
                "sensitivity",
                ctx.clone(),
            )?),
            None => None,
        };
        Ok((atol, sens_atol, out_atol, param_atol))
    }

    fn build_p(p: Vec<M::T>, ctx: M::C) -> M::V {
        M::V::from_vec(p, ctx)
    }

    #[allow(clippy::type_complexity)]
    pub fn build(
        self,
    ) -> Result<OdeSolverProblem<OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>>, DiffsolError>
    where
        M: Matrix,
        Rhs: BuilderOp<V = M::V, T = M::T, M = M, C = M::C>,
        Init: BuilderOp<V = M::V, T = M::T, M = M, C = M::C>,
        Mass: BuilderOp<V = M::V, T = M::T, M = M, C = M::C>,
        Root: BuilderOp<V = M::V, T = M::T, M = M, C = M::C>,
        Out: BuilderOp<V = M::V, T = M::T, M = M, C = M::C>,
        for<'a> ParameterisedOp<'a, Rhs>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
        for<'a> ParameterisedOp<'a, Init>: ConstantOp<M = M, V = M::V, T = M::T, C = M::C>,
        for<'a> ParameterisedOp<'a, Mass>: LinearOp<M = M, V = M::V, T = M::T, C = M::C>,
        for<'a> ParameterisedOp<'a, Root>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
        for<'a> ParameterisedOp<'a, Out>: NonLinearOp<M = M, V = M::V, T = M::T, C = M::C>,
    {
        let p = Self::build_p(self.p, self.ctx.clone());
        let nparams = p.len();
        let mut rhs = self
            .rhs
            .ok_or(ode_solver_error!(BuilderError, "Missing right-hand side"))?;
        let mut init = self
            .init
            .ok_or(ode_solver_error!(BuilderError, "Missing initial state"))?;
        let mut mass = self.mass;
        let mut root = self.root;
        let mut out = self.out;

        let init_op = ParameterisedOp::new(&init, &p);
        let y0 = init_op.call(self.t0);
        let nstates = y0.len();

        rhs.set_nstates(nstates);
        rhs.set_nout(nstates);
        rhs.set_nparams(nparams);

        init.set_nout(nstates);
        init.set_nparams(nparams);

        if let Some(ref mut mass) = mass {
            mass.set_nstates(nstates);
            mass.set_nparams(nparams);
            mass.set_nout(nstates);
        }

        if let Some(ref mut root) = root {
            root.set_nstates(nstates);
            root.set_nparams(nparams);
        }

        if let Some(ref mut out) = out {
            out.set_nstates(nstates);
            out.set_nparams(nparams);
        }

        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, self.t0, &p);
            if let Some(ref mut mass) = mass {
                mass.calculate_sparsity(&y0, self.t0, &p);
            }
        }
        let nout = out.as_ref().map(|out| out.nout());
        let eqn = OdeSolverEquations::new(rhs, init, mass, root, out, p);

        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            nout,
            nparams,
            self.ctx,
        )?;
        OdeSolverProblem::new(
            eqn,
            self.rtol,
            atol,
            self.sens_rtol,
            sens_atol,
            self.out_rtol,
            out_atol,
            self.param_rtol,
            param_atol,
            self.t0,
            self.h0,
            self.integrate_out,
        )
    }

    #[cfg(feature = "diffsl")]
    pub fn build_from_diffsl<CG: crate::CodegenModuleJit + crate::CodegenModuleCompile>(
        mut self,
        code: &str,
    ) -> Result<OdeSolverProblem<crate::DiffSl<M, CG>>, DiffsolError>
    where
        M: Matrix<V: crate::VectorHost, T = f64>,
    {
        #[cfg(feature = "diffsl-cranelift")]
        let include_sensitivities = M::is_sparse()
            && std::any::TypeId::of::<CG>() != std::any::TypeId::of::<diffsl::CraneliftJitModule>();
        #[cfg(not(feature = "diffsl-cranelift"))]
        let include_sensitivities = M::is_sparse();
        let eqn = crate::DiffSl::compile(code, self.ctx.clone(), include_sensitivities)?;
        // if the user hasn't set the parameters, resize them to match the number of parameters in the equations
        let nparams = eqn.rhs().nparams();
        if self.p.len() != nparams && self.p.is_empty() {
            self.p.resize(nparams, 0.0);
        }
        self.build_from_eqn(eqn)
    }

    /// Build an ODE problem from a set of equations
    pub fn build_from_eqn<Eqn>(self, mut eqn: Eqn) -> Result<OdeSolverProblem<Eqn>, DiffsolError>
    where
        Eqn: OdeEquations<M = M, V = M::V, T = M::T>,
    {
        let nparams = eqn.rhs().nparams();
        let nstates = eqn.rhs().nstates();
        let nout = eqn.out().map(|out| out.nout());
        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            nout,
            nparams,
            self.ctx.clone(),
        )?;
        if self.p.len() != nparams {
            return Err(ode_solver_error!(
                BuilderError,
                format!(
                    "Number of parameters on builder does not match number of parameters in equations. Expected {}, got {}.",
                    nparams,
                    self.p.len()
                )
            ));
        }

        let p = Self::build_p(self.p, self.ctx);
        eqn.set_params(&p);
        OdeSolverProblem::new(
            eqn,
            self.rtol,
            atol,
            self.sens_rtol,
            sens_atol,
            self.out_rtol,
            out_atol,
            self.param_rtol,
            param_atol,
            self.t0,
            self.h0,
            self.integrate_out,
        )
    }
}
