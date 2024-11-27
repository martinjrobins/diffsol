use nalgebra::DMatrix;

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    op::{linear_closure_with_adjoint::LinearClosureWithAdjoint, BuilderOp},
    Closure, ClosureNoJac, ClosureWithAdjoint, ClosureWithSens, ConstantClosure,
    ConstantClosureWithAdjoint, ConstantClosureWithSens, ConstantOp, LinearClosure, LinearOp,
    Matrix, NonLinearOp, OdeEquations, OdeSolverProblem, Op, ParametrisedOp, UnitCallable, Vector,
};

use super::equations::OdeSolverEquations;

/// Builder for ODE problems. Use methods to set parameters and then call one of the build methods when done.
pub struct OdeBuilder<
    M: Matrix = DMatrix<f64>,
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
/// use diffsol::{OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod};
/// type M = nalgebra::DMatrix<f64>;
///
/// let problem = OdeBuilder::new()
///   .rtol(1e-6)
///   .p([0.1])
///   .build_ode::<M, _, _, _>(
///     // dy/dt = -ay
///     |x, p, t, y| {
///       y[0] = -p[0] * x[0];
///     },
///     // Jv = -av
///     |x, p, t, v, y| {
///       y[0] = -p[0] * v[0];
///     },
///     // y(0) = 1
///    |p, t| {
///       nalgebra::DVector::from_vec(vec![1.0])
///    },
///   ).unwrap();
///
/// let mut solver = Bdf::default();
/// let t = 0.4;
/// let mut state = OdeSolverState::new(&problem, &solver).unwrap();
/// solver.set_problem(state, &problem);
/// while solver.state().unwrap().t <= t {
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
        let default_atol = vec![M::T::from(1e-6)];
        let default_rtol = 1e-6.into();
        Self {
            rhs: None,
            init: None,
            mass: None,
            root: None,
            out: None,
            t0: 0.0.into(),
            h0: 1.0.into(),
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
        }
    }

    /// Set the right-hand side of the ODE.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    pub fn rhs_implicit<'a, F, G>(
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
            rhs: Some(Closure::new(rhs, rhs_jac, nstates, nstates, nstates)),
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
                rhs, rhs_jac, rhs_sens, nstates, nstates, nstates,
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
        }
    }

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
        }
    }

    /// Set the initial condition of the ODE.
    ///
    /// # Arguments
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    pub fn init<F>(self, init: F) -> OdeBuilder<M, Rhs, ConstantClosure<M, F>, Mass, Root, Out>
    where
        F: Fn(&M::V, M::T) -> M::V,
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, ConstantClosure<M, F>, Mass, Root, Out> {
            rhs: self.rhs,
            init: Some(ConstantClosure::new(init, nstates, nstates)),
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
    ) -> OdeBuilder<M, Rhs, ConstantClosureWithSens<M, F, G>, Mass, Root, Out>
    where
        F: Fn(&M::V, M::T) -> M::V,
        G: Fn(&M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, ConstantClosureWithSens<M, F, G>, Mass, Root, Out> {
            rhs: self.rhs,
            init: Some(ConstantClosureWithSens::new(
                init, init_sens, nstates, nstates,
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
    ) -> OdeBuilder<M, Rhs, ConstantClosureWithAdjoint<M, F, G>, Mass, Root, Out>
    where
        F: Fn(&M::V, M::T) -> M::V,
        G: Fn(&M::V, M::T, &M::V, &mut M::V),
    {
        let nstates = 0;
        OdeBuilder::<M, Rhs, ConstantClosureWithAdjoint<M, F, G>, Mass, Root, Out> {
            rhs: self.rhs,
            init: Some(ConstantClosureWithAdjoint::new(
                init,
                init_sens_adjoint,
                nstates,
                nstates,
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
            mass: Some(LinearClosure::new(mass, nstates, nstates, nstates)),
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
        }
    }

    /// Set the mass matrix of the ODE for adjoint sensitivity analysis.
    ///
    /// # Arguments
    ///
    /// - `mass`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the mass matrix with
    /// the vector v (i.e. y = M * v + beta * y).
    /// - `mass_adjoint`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the transpose of the mass matrix with
    /// the vector v (i.e. y = M^T * v + beta * y).
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
            root: Some(ClosureNoJac::new(root, nstates, nroots, nroots)),
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
            out: Some(Closure::new(out, out_jac, nstates, nout, nstates)),
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
        }
    }

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
        }
    }

    /// Set the initial time.
    pub fn t0(mut self, t0: f64) -> Self {
        self.t0 = t0.into();
        self
    }

    pub fn sens_rtol(mut self, sens_rtol: Option<f64>) -> Self {
        self.sens_rtol = sens_rtol.map(M::T::from);
        self
    }

    pub fn sens_atol<V, T>(mut self, sens_atol: Option<V>) -> Self
    where
        V: IntoIterator<Item = T>,
        M::T: From<T>,
    {
        self.sens_atol = sens_atol.map(|atol| atol.into_iter().map(|x| M::T::from(x)).collect());
        self
    }

    pub fn out_rtol(mut self, out_rtol: Option<f64>) -> Self {
        self.out_rtol = out_rtol.map(M::T::from);
        self
    }

    pub fn out_atol<V, T>(mut self, out_atol: Option<V>) -> Self
    where
        V: IntoIterator<Item = T>,
        M::T: From<T>,
    {
        self.out_atol = out_atol.map(|atol| atol.into_iter().map(|x| M::T::from(x)).collect());
        self
    }

    pub fn param_rtol(mut self, param_rtol: Option<f64>) -> Self {
        self.param_rtol = param_rtol.map(M::T::from);
        self
    }

    pub fn param_atol<V, T>(mut self, param_atol: Option<V>) -> Self
    where
        V: IntoIterator<Item = T>,
        M::T: From<T>,
    {
        self.param_atol = param_atol.map(|atol| atol.into_iter().map(|x| M::T::from(x)).collect());
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
        self.h0 = h0.into();
        self
    }

    /// Set the relative tolerance.
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol.into();
        self
    }

    /// Set the absolute tolerance.
    pub fn atol<V, T>(mut self, atol: V) -> Self
    where
        V: IntoIterator<Item = T>,
        M::T: From<T>,
    {
        self.atol = atol.into_iter().map(|x| M::T::from(x)).collect();
        self
    }

    /// Set the parameters.
    pub fn p<V, T>(mut self, p: V) -> Self
    where
        V: IntoIterator<Item = T>,
        M::T: From<T>,
    {
        self.p = p.into_iter().map(|x| M::T::from(x)).collect();
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

    fn build_atol(atol: Vec<M::T>, nstates: usize, ty: &str) -> Result<M::V, DiffsolError> {
        if atol.len() == 1 {
            Ok(M::V::from_element(nstates, atol[0]))
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
            let mut v = M::V::zeros(nstates);
            for (i, &a) in atol.iter().enumerate() {
                v[i] = a;
            }
            Ok(v)
        }
    }

    #[allow(clippy::type_complexity)]
    fn build_atols(
        atol: Vec<M::T>,
        sens_atol: Option<Vec<M::T>>,
        out_atol: Option<Vec<M::T>>,
        param_atol: Option<Vec<M::T>>,
        nstates: usize,
        nout: Option<usize>,
        nparam: usize,
    ) -> Result<(M::V, Option<M::V>, Option<M::V>, Option<M::V>), DiffsolError> {
        let atol = Self::build_atol(atol, nstates, "states")?;
        let out_atol = match out_atol {
            Some(out_atol) => Some(Self::build_atol(out_atol, nout.unwrap_or(0), "output")?),
            None => None,
        };
        let param_atol = match param_atol {
            Some(param_atol) => Some(Self::build_atol(param_atol, nparam, "parameters")?),
            None => None,
        };
        let sens_atol = match sens_atol {
            Some(sens_atol) => Some(Self::build_atol(sens_atol, nstates, "sensitivity")?),
            None => None,
        };
        Ok((atol, sens_atol, out_atol, param_atol))
    }

    fn build_p(p: Vec<M::T>) -> M::V {
        let mut v = M::V::zeros(p.len());
        for (i, &p) in p.iter().enumerate() {
            v[i] = p;
        }
        v
    }

    pub fn build(
        self,
    ) -> Result<OdeSolverProblem<OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>>, DiffsolError>
    where
        M: Matrix,
        Rhs: BuilderOp<V = M::V, T = M::T, M = M>,
        Init: BuilderOp<V = M::V, T = M::T, M = M>,
        Mass: BuilderOp<V = M::V, T = M::T, M = M>,
        Root: BuilderOp<V = M::V, T = M::T, M = M>,
        Out: BuilderOp<V = M::V, T = M::T, M = M>,
        for<'a> ParametrisedOp<'a, Rhs>: NonLinearOp<M = M, V = M::V, T = M::T>,
        for<'a> ParametrisedOp<'a, Init>: ConstantOp<M = M, V = M::V, T = M::T>,
        for<'a> ParametrisedOp<'a, Mass>: LinearOp<M = M, V = M::V, T = M::T>,
        for<'a> ParametrisedOp<'a, Root>: NonLinearOp<M = M, V = M::V, T = M::T>,
        for<'a> ParametrisedOp<'a, Out>: NonLinearOp<M = M, V = M::V, T = M::T>,
    {
        let p = Self::build_p(self.p);
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

        let init_op = ParametrisedOp::new(&init, &p);
        let y0 = init_op.call(self.t0);
        let nstates = y0.len();

        rhs.set_nstates(nstates);
        rhs.set_nout(nstates);
        rhs.set_nparams(nparams);

        init.set_nout(nstates);
        init.set_nparams(nparams);

        mass.as_mut().map(|mass| mass.set_nstates(nstates));
        mass.as_mut().map(|mass| mass.set_nparams(nparams));
        mass.as_mut().map(|mass| mass.set_nout(nstates));

        root.as_mut().map(|root| root.set_nstates(nstates));
        root.as_mut().map(|root| root.set_nparams(nparams));

        out.as_mut().map(|out| out.set_nstates(nstates));
        out.as_mut().map(|out| out.set_nparams(nparams));

        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, self.t0, &p);
            mass.as_mut()
                .map(|mass| mass.calculate_sparsity(&y0, self.t0, &p));
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

        let p = Self::build_p(self.p);
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
