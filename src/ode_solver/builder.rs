use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    vector::DefaultDenseMatrix,
    Closure, ClosureNoJac, ClosureWithSens, ConstantClosure, ConstantClosureWithSens,
    LinearClosure, LinearClosureWithSens, Matrix, OdeEquations, OdeSolverProblem, Op, UnitCallable,
    Vector,
};

use super::equations::OdeSolverEquations;

/// Builder for ODE problems. Use methods to set parameters and then call one of the build methods when done.
pub struct OdeBuilder {
    t0: f64,
    h0: f64,
    rtol: f64,
    atol: Vec<f64>,
    p: Vec<f64>,
    use_coloring: bool,
    sensitivities: bool,
    sensitivities_error_control: bool,
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
/// while solver.state().unwrap().t() <= t {
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
        Self {
            t0: 0.0,
            h0: 1.0,
            rtol: 1e-6,
            atol: vec![1e-6],
            p: vec![],
            use_coloring: false,
            sensitivities: false,
            sensitivities_error_control: false,
        }
    }

    /// Set the initial time.
    pub fn t0(mut self, t0: f64) -> Self {
        self.t0 = t0;
        self
    }

    pub fn sensitivities(mut self, sensitivities: bool) -> Self {
        self.sensitivities = sensitivities;
        self
    }

    pub fn sensitivities_error_control(mut self, sensitivities_error_control: bool) -> Self {
        self.sensitivities_error_control = sensitivities_error_control;
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

    fn build_atol<V: Vector>(atol: Vec<f64>, nstates: usize) -> Result<V, DiffsolError> {
        if atol.len() == 1 {
            Ok(V::from_element(nstates, V::T::from(atol[0])))
        } else if atol.len() != nstates {
            Err(ode_solver_error!(AtolLengthMismatch))
        } else {
            let mut v = V::zeros(nstates);
            for (i, &a) in atol.iter().enumerate() {
                v[i] = V::T::from(a);
            }
            Ok(v)
        }
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
        let mass = Some(Rc::new(mass));
        let rhs = Rc::new(rhs);
        let init = Rc::new(init);
        let eqn = OdeSolverEquations::new(rhs, mass, None, init, None, p);
        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            false,
            self.sensitivities_error_control,
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
        let mut rhs = Closure::new(rhs, rhs_jac, nstates, nstates, p.clone());
        let out = Closure::new(out, out_jac, nstates, nout, p.clone());
        let mut mass = LinearClosure::new(mass, nstates, nstates, p.clone());
        let init = ConstantClosure::new(init, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
            mass.calculate_sparsity(t0);
        }
        let mass = Some(Rc::new(mass));
        let rhs = Rc::new(rhs);
        let init = Rc::new(init);
        let out = Some(Rc::new(out));
        let eqn = OdeSolverEquations::new(rhs, mass, None, init, out, p);
        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            false,
            self.sensitivities_error_control,
        )
    }

    /// Build an ODE problem with a mass matrix and sensitivities.
    ///
    /// # Arguments
    ///
    /// - `rhs`: Function of type Fn(x: &V, p: &V, t: S, y: &mut V) that computes the right-hand side of the ODE.
    /// - `rhs_jac`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the Jacobian of the right-hand side with the vector v.
    /// - `rhs_sens`: Function of type Fn(x: &V, p: &V, t: S, v: &V, y: &mut V) that computes the multiplication of the partial derivative of the rhs wrt the parameters, with the vector v.
    /// - `mass`: Function of type Fn(v: &V, p: &V, t: S, beta: S, y: &mut V) that computes a gemv multiplication of the mass matrix with the vector v (i.e. y = M * v + beta * y).
    /// - `mass_sens`: Function of type Fn(v: &V, p: &V, t: S, y: &mut V) that computes the multiplication of the partial derivative of the mass matrix wrt the parameters, with the vector v.
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
    /// // dy/dt = a y
    /// // 0 = z - y
    /// // y(0) = 0.1
    /// // z(0) = 0.1
    /// let problem = OdeBuilder::new()
    ///   .build_ode_with_mass_and_sens::<M, _, _, _, _, _, _, _>(
    ///       |x, p, _t, y| {
    ///           y[0] = p[0] * x[0];
    ///           y[1] = x[1] - x[0];
    ///       },
    ///       |x, p, _t, v, y|  {
    ///           y[0] = p[0] * v[0];
    ///           y[1] = v[1] - v[0];
    ///       },
    ///       |x, _p, _t, v, y| {
    ///           y[0] = v[0] * x[0];
    ///           y[1] = 0.0;
    ///       },
    ///       |x, _p, _t, beta, y| {
    ///           y[0] = x[0] + beta * y[0];
    ///           y[1] = beta * y[1];
    ///       },
    ///       |x, p, t, v, y| {
    ///           y.fill(0.0);
    ///       },
    ///       |p, _t| DVector::from_element(2, 0.1),
    ///       |p, t, v, y| {
    ///           y.fill(0.0);
    ///       }
    /// );
    /// ```
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn build_ode_with_mass_and_sens<M, F, G, H, I, J, K, L>(
        self,
        rhs: F,
        rhs_jac: G,
        rhs_sens: J,
        mass: H,
        mass_sens: L,
        init: I,
        init_sens: K,
    ) -> Result<
        OdeSolverProblem<
            OdeSolverEquations<
                M,
                ClosureWithSens<M, F, G, J>,
                ConstantClosureWithSens<M, I, K>,
                LinearClosureWithSens<M, H, L>,
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
        J: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        K: Fn(&M::V, M::T, &M::V, &mut M::V),
        L: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    {
        let p = Rc::new(Self::build_p(self.p));
        let t0 = M::T::from(self.t0);
        let y0 = init(&p, t0);
        let nstates = y0.len();
        let mut rhs = ClosureWithSens::new(rhs, rhs_jac, rhs_sens, nstates, nstates, p.clone());
        let mut mass = LinearClosureWithSens::new(mass, mass_sens, nstates, nstates, p.clone());
        let init = ConstantClosureWithSens::new(init, init_sens, nstates, nstates, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
            mass.calculate_sparsity(t0);
        }
        let mass = Some(Rc::new(mass));
        let rhs = Rc::new(rhs);
        let init = Rc::new(init);
        let eqn = OdeSolverEquations::new(rhs, mass, None, init, None, p);
        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            true,
            self.sensitivities_error_control,
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
        let rhs = Rc::new(rhs);
        let init = Rc::new(init);
        let eqn = OdeSolverEquations::new(rhs, None, None, init, None, p);
        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            false,
            self.sensitivities_error_control,
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
            rhs.calculate_sparsity(&y0, t0);
        }
        let rhs = Rc::new(rhs);
        let init = Rc::new(init);
        let eqn = OdeSolverEquations::new(rhs, None, None, init, None, p);
        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            true,
            self.sensitivities_error_control,
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
        let root = Rc::new(ClosureNoJac::new(root, nstates, nroots, p.clone()));
        let init = ConstantClosure::new(init, p.clone());
        if self.use_coloring || M::is_sparse() {
            rhs.calculate_sparsity(&y0, t0);
        }
        let rhs = Rc::new(rhs);
        let init = Rc::new(init);
        let eqn = OdeSolverEquations::new(rhs, None, Some(root), init, None, p);
        let atol = Self::build_atol(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
            false,
            self.sensitivities_error_control,
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

    /// Build an ODE problem using the DiffSL language (requires either the `diffsl-cranelift` or `diffls-llvm` features).
    /// The source code is provided as a string, please see the [DiffSL documentation](https://martinjrobins.github.io/diffsl/) for more information.
    #[cfg(feature = "diffsl")]
    pub fn build_diffsl<M, CG>(
        self,
        context: &crate::ode_solver::diffsl::DiffSlContext<M, CG>,
    ) -> Result<OdeSolverProblem<crate::ode_solver::diffsl::DiffSl<'_, M, CG>>, DiffsolError>
    where
        M: Matrix<T = crate::ode_solver::diffsl::T>,
        CG: diffsl::execution::module::CodegenModule,
    {
        use crate::ode_solver::diffsl;
        let p = Self::build_p::<M::V>(self.p);
        let mut eqn = diffsl::DiffSl::new(context, self.use_coloring || M::is_sparse());
        eqn.set_params(p);
        let atol = Self::build_atol::<M::V>(self.atol, eqn.rhs().nstates())?;
        OdeSolverProblem::new(
            eqn,
            self.rtol,
            atol,
            self.t0,
            self.h0,
            self.sensitivities,
            self.sensitivities_error_control,
        )
    }
}
