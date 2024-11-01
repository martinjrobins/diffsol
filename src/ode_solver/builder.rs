use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    vector::DefaultDenseMatrix,
    Closure, ClosureNoJac, ClosureWithSens, ConstantClosure, ConstantClosureWithSens,
    LinearClosure, Matrix, OdeEquations, OdeSolverProblem, Op, UnitCallable, Vector,
};
use std::rc::Rc;

use super::equations::OdeSolverEquations;

/// Builder for ODE problems. Use methods to set parameters and then call one of the build methods when done.
pub struct OdeBuilder {
    t0: f64,
    h0: f64,
    rtol: f64,
    atol: Vec<f64>,
    sens_atol: Option<Vec<f64>>,
    sens_rtol: Option<f64>,
    out_rtol: Option<f64>,
    out_atol: Option<Vec<f64>>,
    param_rtol: Option<f64>,
    param_atol: Option<Vec<f64>>,
    p: Vec<f64>,
    use_coloring: bool,
    integrate_out: bool,
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
impl OdeBuilder {
    /// Create a new builder with default parameters:
    /// - t0 = 0.0
    /// - h0 = 1.0
    /// - rtol = 1e-6
    /// - atol = [1e-6]
    /// - p = []
    /// - use_coloring = false
    /// - constant_mass = false
    pub fn new() -> Self {
        let default_atol = vec![1e-6];
        let default_rtol = 1e-6;
        Self {
            t0: 0.0,
            h0: 1.0,
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

    /// Set the initial time.
    pub fn t0(mut self, t0: f64) -> Self {
        self.t0 = t0;
        self
    }

    pub fn sens_rtol(mut self, sens_rtol: Option<f64>) -> Self {
        self.sens_rtol = sens_rtol;
        self
    }

    pub fn sens_atol<V, T>(mut self, sens_atol: Option<V>) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.sens_atol = sens_atol.map(|atol| atol.into_iter().map(|x| f64::from(x)).collect());
        self
    }

    pub fn out_rtol(mut self, out_rtol: Option<f64>) -> Self {
        self.out_rtol = out_rtol;
        self
    }

    pub fn out_atol<V, T>(mut self, out_atol: Option<V>) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.out_atol = out_atol.map(|atol| atol.into_iter().map(|x| f64::from(x)).collect());
        self
    }

    pub fn param_rtol(mut self, param_rtol: Option<f64>) -> Self {
        self.param_rtol = param_rtol;
        self
    }

    pub fn param_atol<V, T>(mut self, param_atol: Option<V>) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.param_atol = param_atol.map(|atol| atol.into_iter().map(|x| f64::from(x)).collect());
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
        self.h0 = h0;
        self
    }

    /// Set the relative tolerance.
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set the absolute tolerance.
    pub fn atol<V, T>(mut self, atol: V) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.atol = atol.into_iter().map(|x| f64::from(x)).collect();
        self
    }

    /// Set the parameters.
    pub fn p<V, T>(mut self, p: V) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.p = p.into_iter().map(|x| f64::from(x)).collect();
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

    fn build_atol<V: Vector>(atol: Vec<f64>, nstates: usize, ty: &str) -> Result<V, DiffsolError> {
        if atol.len() == 1 {
            Ok(V::from_element(nstates, V::T::from(atol[0])))
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
            let mut v = V::zeros(nstates);
            for (i, &a) in atol.iter().enumerate() {
                v[i] = V::T::from(a);
            }
            Ok(v)
        }
    }

    #[allow(clippy::type_complexity)]
    fn build_atols<V: Vector>(
        atol: Vec<f64>,
        sens_atol: Option<Vec<f64>>,
        out_atol: Option<Vec<f64>>,
        param_atol: Option<Vec<f64>>,
        nstates: usize,
        nout: Option<usize>,
        nparam: usize,
    ) -> Result<(Rc<V>, Option<Rc<V>>, Option<Rc<V>>, Option<Rc<V>>), DiffsolError> {
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
        Ok((
            Rc::new(atol),
            sens_atol.map(Rc::new),
            out_atol.map(Rc::new),
            param_atol.map(Rc::new),
        ))
    }

    fn build_p<V: Vector>(p: Vec<f64>) -> V {
        let mut v = V::zeros(p.len());
        for (i, &p) in p.iter().enumerate() {
            v[i] = V::T::from(p);
        }
        v
    }

    /// Build an ODE problem with a mass matrix.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    /// - `mass`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the mass matrix with the vector v (i.e. y = M * v + beta * y).
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    ///
    /// # Generic Arguments
    ///
    /// - `M`: Type that implements the `Matrix` trait. Often this must be provided explicitly (i.e. `type M = DMatrix<f64>; builder.build_ode::<M, _, _, _>`).
    ///
    /// # Example
    ///
    /// ```
    /// use diffsol::OdeBuilder;
    /// use nalgebra::DVector;
    /// type M = nalgebra::DMatrix<f64>;
    ///
    /// // dy/dt = y
    /// // 0 = z - y
    /// // y(0) = 0.1
    /// // z(0) = 0.1
    /// let problem = OdeBuilder::new()
    ///   .build_ode_with_mass::<M, _, _, _, _>(
    ///       |x, _p, _t, y| {
    ///           y[0] = x[0];
    ///           y[1] = x[1] - x[0];
    ///       },
    ///       |x, _p, _t, v, y|  {
    ///           y[0] = v[0];
    ///           y[1] = v[1] - v[0];
    ///       },
    ///       |v, _p, _t, beta, y| {
    ///           y[0] = v[0] + beta * y[0];
    ///           y[1] = beta * y[1];
    ///       },
    ///       |p, _t| DVector::from_element(2, 0.1),
    /// );
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn build_ode_with_mass<M, F, G, H, I>(
        self,
        rhs: F,
        rhs_jac: G,
        mass: H,
        init: I,
    ) -> Result<
        OdeSolverProblem<
            OdeSolverEquations<M, Closure<M, F, G>, ConstantClosure<M, I>, LinearClosure<M, H>>,
        >,
        DiffsolError,
    >
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Rc::new(Self::build_p(self.p));
        let t0 = M::T::from(self.t0);
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let mut rhs = Closure::new(rhs, rhs_jac, nstates, nstates, p.clone());
        let mut mass = LinearClosure::new(mass, nstates, nstates, p.clone());
        let init = ConstantClosure::new(init, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
            mass.calculate_sparsity(t0);
        }
        let mass = Some(mass);
        let nparams = p.len();
        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            None,
            nparams,
        )?;
        let eqn = OdeSolverEquations::new(rhs, mass, None, init, None, p);
        OdeSolverProblem::new(
            Rc::new(eqn),
            M::T::from(self.rtol),
            atol,
            self.sens_rtol.map(M::T::from),
            sens_atol,
            self.out_rtol.map(M::T::from),
            out_atol,
            self.param_rtol.map(M::T::from),
            param_atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            self.integrate_out,
        )
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn build_ode_with_mass_and_out<M, F, G, H, I, J, K>(
        self,
        rhs: F,
        rhs_jac: G,
        mass: H,
        init: I,
        out: J,
        out_jac: K,
        nout: usize,
    ) -> Result<
        OdeSolverProblem<
            OdeSolverEquations<
                M,
                Closure<M, F, G>,
                ConstantClosure<M, I>,
                LinearClosure<M, H>,
                UnitCallable<M>,
                Closure<M, J, K>,
            >,
        >,
        DiffsolError,
    >
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
        J: Fn(&M::V, &M::V, M::T, &mut M::V),
        K: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let p = Rc::new(Self::build_p(self.p));
        let t0 = M::T::from(self.t0);
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let nparams = p.len();
        let mut rhs = Closure::new(rhs, rhs_jac, nstates, nstates, p.clone());
        let out = Closure::new(out, out_jac, nstates, nout, p.clone());
        let mut mass = LinearClosure::new(mass, nstates, nstates, p.clone());
        let init = ConstantClosure::new(init, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
            mass.calculate_sparsity(t0);
        }
        let mass = Some(mass);
        let out = Some(out);
        let eqn = OdeSolverEquations::new(rhs, mass, None, init, out, p);
        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            Some(nout),
            nparams,
        )?;
        OdeSolverProblem::new(
            Rc::new(eqn),
            M::T::from(self.rtol),
            atol,
            self.sens_rtol.map(M::T::from),
            sens_atol,
            self.out_rtol.map(M::T::from),
            out_atol,
            self.param_rtol.map(M::T::from),
            param_atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            self.integrate_out,
        )
    }

    /// Build an ODE problem with a mass matrix that is the identity matrix.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    ///
    /// # Generic Arguments
    ///
    /// - `M`: Type that implements the `Matrix` trait. Often this must be provided explicitly (i.e. `type M = DMatrix<f64>; builder.build_ode::<M, _, _, _>`).
    ///
    /// # Example
    ///
    ///
    ///
    /// ```
    /// use diffsol::OdeBuilder;
    /// use nalgebra::DVector;
    /// type M = nalgebra::DMatrix<f64>;
    ///
    ///
    /// // dy/dt = y
    /// // y(0) = 0.1
    /// let problem = OdeBuilder::new()
    ///    .build_ode::<M, _, _, _>(
    ///        |x, _p, _t, y| y[0] = x[0],
    ///        |x, _p, _t, v , y| y[0] = v[0],
    ///        |p, _t| DVector::from_element(1, 0.1),
    ///    );
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn build_ode<M, F, G, I>(
        self,
        rhs: F,
        rhs_jac: G,
        init: I,
    ) -> Result<
        OdeSolverProblem<OdeSolverEquations<M, Closure<M, F, G>, ConstantClosure<M, I>>>,
        DiffsolError,
    >
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Rc::new(Self::build_p(self.p));
        let t0 = M::T::from(self.t0);
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let mut rhs = Closure::new(rhs, rhs_jac, nstates, nstates, p.clone());
        let init = ConstantClosure::new(init, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
        }
        let nparams = p.len();
        let eqn = OdeSolverEquations::new(rhs, None, None, init, None, p);
        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            None,
            nparams,
        )?;
        OdeSolverProblem::new(
            Rc::new(eqn),
            M::T::from(self.rtol),
            atol,
            self.sens_rtol.map(M::T::from),
            sens_atol,
            self.out_rtol.map(M::T::from),
            out_atol,
            self.param_rtol.map(M::T::from),
            param_atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            self.integrate_out,
        )
    }

    /// Build an ODE problem with a mass matrix that is the identity matrix and sensitivities.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    /// - `rhs_sens`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the partial derivative of the rhs wrt the parameters, with the vector v.
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    /// - `init_sens`: Function of type Fn(p: &V, t: S, y: &mut V) that computes the multiplication of the partial derivative of the initial state wrt the parameters, with the vector v.
    ///
    /// # Example
    ///
    /// ```
    /// use diffsol::OdeBuilder;
    /// use nalgebra::DVector;
    /// type M = nalgebra::DMatrix<f64>;
    ///
    ///
    /// // dy/dt = a y
    /// // y(0) = 0.1
    /// let problem = OdeBuilder::new()
    ///    .build_ode_with_sens::<M, _, _, _, _, _>(
    ///        |x, p, _t, y| y[0] = p[0] * x[0],
    ///        |x, p, _t, v, y| y[0] = p[0] * v[0],
    ///        |x, p, _t, v, y| y[0] = v[0] * x[0],
    ///        |p, _t| DVector::from_element(1, 0.1),
    ///        |p, t, v, y| y.fill(0.0),
    ///    );
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn build_ode_with_sens<M, F, G, I, J, K>(
        self,
        rhs: F,
        rhs_jac: G,
        rhs_sens: J,
        init: I,
        init_sens: K,
    ) -> Result<
        OdeSolverProblem<
            OdeSolverEquations<M, ClosureWithSens<M, F, G, J>, ConstantClosureWithSens<M, I, K>>,
        >,
        DiffsolError,
    >
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
        J: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        K: Fn(&M::V, M::T, &M::V, &mut M::V),
    {
        let p = Rc::new(Self::build_p(self.p));
        let t0 = M::T::from(self.t0);
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let init = ConstantClosureWithSens::new(init, init_sens, nstates, nstates, p.clone());
        let mut rhs = ClosureWithSens::new(rhs, rhs_jac, rhs_sens, nstates, nstates, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_jacobian_sparsity(&y0, t0);
            rhs.calculate_sens_sparsity(&y0, t0);
        }
        let nparams = p.len();
        let eqn = OdeSolverEquations::new(rhs, None, None, init, None, p);
        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            None,
            nparams,
        )?;
        OdeSolverProblem::new(
            Rc::new(eqn),
            M::T::from(self.rtol),
            atol,
            self.sens_rtol.map(M::T::from),
            sens_atol,
            self.out_rtol.map(M::T::from),
            out_atol,
            self.param_rtol.map(M::T::from),
            param_atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            self.integrate_out,
        )
    }

    /// Build an ODE problem with an event.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    /// - `init`: Function of type Fn(p: &V, t: S) -> V that computes the initial state.
    /// - `root`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the root function.
    /// - `nroots`: Number of roots (i.e. number of elements in the `y` arg in `root`), an event is triggered when any of the roots changes sign.
    ///
    /// # Generic Arguments
    ///
    /// - `M`: Type that implements the `Matrix` trait. Often this must be provided explicitly (i.e. `type M = DMatrix<f64>; builder.build_ode::<M, _, _, _, _>`).
    ///
    /// # Example
    ///
    ///
    ///
    /// ```
    /// use diffsol::OdeBuilder;
    /// use nalgebra::DVector;
    /// type M = nalgebra::DMatrix<f64>;
    ///
    ///
    /// // dy/dt = y
    /// // y(0) = 0.1
    /// // event at y = 0.5
    /// let problem = OdeBuilder::new()
    ///    .build_ode_with_root::<M, _, _, _, _>(
    ///        |x, _p, _t, y| y[0] = x[0],
    ///        |x, _p, _t, v , y| y[0] = v[0],
    ///        |p, _t| DVector::from_element(1, 0.1),
    ///        |x, _p, _t, y| y[0] = x[0] - 0.5,
    ///        1,
    ///    );
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn build_ode_with_root<M, F, G, I, H>(
        self,
        rhs: F,
        rhs_jac: G,
        init: I,
        root: H,
        nroots: usize,
    ) -> Result<
        OdeSolverProblem<
            OdeSolverEquations<
                M,
                Closure<M, F, G>,
                ConstantClosure<M, I>,
                UnitCallable<M>,
                ClosureNoJac<M, H>,
            >,
        >,
        DiffsolError,
    >
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Rc::new(Self::build_p(self.p));
        let t0 = M::T::from(self.t0);
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let mut rhs = Closure::new(rhs, rhs_jac, nstates, nstates, p.clone());
        let root = ClosureNoJac::new(root, nstates, nroots, p.clone());
        let init = ConstantClosure::new(init, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
        }
        let nparams = p.len();
        let eqn = OdeSolverEquations::new(rhs, None, Some(root), init, None, p);
        let (atol, sens_atol, out_atol, param_atol) = Self::build_atols(
            self.atol,
            self.sens_atol,
            self.out_atol,
            self.param_atol,
            nstates,
            None,
            nparams,
        )?;
        OdeSolverProblem::new(
            Rc::new(eqn),
            M::T::from(self.rtol),
            atol,
            self.sens_rtol.map(M::T::from),
            sens_atol,
            self.out_rtol.map(M::T::from),
            out_atol,
            self.param_rtol.map(M::T::from),
            param_atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            self.integrate_out,
        )
    }

    /// Build an ODE problem using the default dense matrix (see [Self::build_ode]).
    #[allow(clippy::type_complexity)]
    pub fn build_ode_dense<V, F, G, I>(
        self,
        rhs: F,
        rhs_jac: G,
        init: I,
    ) -> Result<
        OdeSolverProblem<OdeSolverEquations<V::M, Closure<V::M, F, G>, ConstantClosure<V::M, I>>>,
        DiffsolError,
    >
    where
        V: Vector + DefaultDenseMatrix,
        F: Fn(&V, &V, V::T, &mut V),
        G: Fn(&V, &V, V::T, &V, &mut V),
        I: Fn(&V, V::T) -> V,
    {
        self.build_ode(rhs, rhs_jac, init)
    }

    /// Build an ODE problem from a set of equations
    pub fn build_from_eqn<Eqn>(self, eqn: Rc<Eqn>) -> Result<OdeSolverProblem<Eqn>, DiffsolError>
    where
        Eqn: OdeEquations,
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
        OdeSolverProblem::new(
            eqn,
            Eqn::T::from(self.rtol),
            atol,
            self.sens_rtol.map(Eqn::T::from),
            sens_atol,
            self.out_rtol.map(Eqn::T::from),
            out_atol,
            self.param_rtol.map(Eqn::T::from),
            param_atol,
            Eqn::T::from(self.t0),
            Eqn::T::from(self.h0),
            self.integrate_out,
        )
    }
}
