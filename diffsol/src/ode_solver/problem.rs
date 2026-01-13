use num_traits::FromPrimitive;
use std::{cell::RefCell, rc::Rc};

use crate::{
    error::DiffsolError, vector::Vector, AdjointContext, AdjointEquations, AugmentedOdeEquations,
    AugmentedOdeEquationsImplicit, Bdf, BdfState, Checkpointing, DefaultDenseMatrix, DenseMatrix,
    ExplicitRk, LinearSolver, MatrixRef, NewtonNonlinearSolver, NoLineSearch, NonLinearOp,
    OdeEquations, OdeEquationsAdjoint, OdeEquationsImplicit, OdeEquationsImplicitAdjoint,
    OdeEquationsImplicitSens, OdeSolverMethod, OdeSolverState, RkState, Scalar, Sdirk,
    SensEquations, Tableau, VectorRef,
};

/// Options for the initial condition solver used to find consistent initial conditions
/// (i.e. when a mass matrix with zeros on the diagonal is present)
pub struct InitialConditionSolverOptions<T: Scalar> {
    /// use a backtracking linesearch in the Newton solver (default: true)
    pub use_linesearch: bool,
    /// maximum number of iterations of the linesearch (default: 10)
    pub max_linesearch_iterations: usize,
    /// maximum number of Newton iterations for the initial condition solve (default: 10)
    pub max_newton_iterations: usize,
    /// maximum number of linear solver setups during the initial condition solve (default: 4)
    /// This compounds with the max_newton_iterations to limit the total number of newton
    /// iterations (i.e. max_newton_iterations * max_linear_solver_setups)
    pub max_linear_solver_setups: usize,
    /// factor to reduce the step size during the linesearch (default: 0.5)
    pub step_reduction_factor: T,
    /// Armijo constant for the linesearch (default: 1e-4)
    pub armijo_constant: T,
}

impl<T: Scalar> Default for InitialConditionSolverOptions<T> {
    fn default() -> Self {
        Self {
            use_linesearch: true,
            max_linesearch_iterations: 10,
            max_linear_solver_setups: 4,
            max_newton_iterations: 10,
            step_reduction_factor: T::from_f64(0.5).unwrap(),
            armijo_constant: T::from_f64(1e-4).unwrap(),
        }
    }
}

/// Options for the ODE solver. These options control various aspects of the solver's behavior.
/// Some options may not be applicable to all solver methods (e.g. implicit vs explicit methods).
pub struct OdeSolverOptions<T: Scalar> {
    /// maximum number of nonlinear solver iterations per solve (default: 10)
    pub max_nonlinear_solver_iterations: usize,
    /// maximum number of error test failures before aborting the solve and returning an error (default: 40)
    pub max_error_test_failures: usize,
    /// maximum number of nonlinear solver failures before aborting the solve and returning an error (default: 50)
    pub max_nonlinear_solver_failures: usize,
    /// minimum allowed timestep size (default: 1e-13)
    pub min_timestep: T,
    /// maximum number of steps after which to update the Jacobian (default: 20).
    /// This only requires an additional linear solver setup, not evaluation of the full Jacobian.
    pub update_jacobian_after_steps: usize,
    /// maximum number of steps after which to update the RHS Jacobian (default: 50).
    /// This evaluates the full Jacobian of the RHS function and requires an additional linear solver setup.
    pub update_rhs_jacobian_after_steps: usize,
    /// threshold on the change in timestep size |dt_new / dt_old - 1| to trigger a Jacobian update (default: 0.3)
    pub threshold_to_update_jacobian: T,
    /// threshold on the change in timestep size |dt_new / dt_old - 1| to trigger a RHS Jacobian update (default: 0.2)
    pub threshold_to_update_rhs_jacobian: T,
}

impl<T: Scalar> Default for OdeSolverOptions<T> {
    fn default() -> Self {
        Self {
            max_nonlinear_solver_iterations: 10,
            max_error_test_failures: 40,
            max_nonlinear_solver_failures: 50,
            min_timestep: T::from_f64(1e-13).unwrap(),
            update_jacobian_after_steps: 20,
            update_rhs_jacobian_after_steps: 50,
            threshold_to_update_jacobian: T::from_f64(0.3).unwrap(),
            threshold_to_update_rhs_jacobian: T::from_f64(0.2).unwrap(),
        }
    }
}

/// Struct representing an ODE solver problem, encapsulating the equations,
/// tolerances, initial conditions, and solver options. This struct can be used
/// to create individual solvers for different methods (e.g., BDF, Runge-Kutta),
/// solving the same underlying problem with consistent settings.
/// 
/// This struct is normally generated via the `OdeBuilder` API, which provides
/// a more user-friendly interface for constructing ODE solver problems.
pub struct OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
{
    /// The ODE equations to be solved, which satisfy the `OdeEquations` trait.
    pub eqn: Eqn,
    /// Relative tolerance for the solver. The state equations are solved to this and the absolute tolerance, given by the norm `sum_i(y_i / (atol_i + rtol * |y0_i|)) < 1`.
    pub rtol: Eqn::T,
    /// Absolute tolerance for the solver. The state equations are solved to this and the relative tolerance, given by the norm `sum_i(y_i / (atol_i + rtol * |y0_i|)) < 1`.
    pub atol: Eqn::V,
    /// Initial time for the ODE solve.
    pub t0: Eqn::T,
    /// Initial step size for the ODE solver.
    pub h0: Eqn::T,
    /// Whether to integrate the output equations alongside the state equations.
    pub integrate_out: bool,
    /// Relative tolerance for the forward sensitivity equations or the adjoint equations, if sensitivities are being computed. If `None`, sensitivities are not included in error control.
    pub sens_rtol: Option<Eqn::T>,
    /// Absolute tolerance for the forward sensitivity equations or the adjoint equations, if sensitivities are being computed. If `None`, sensitivities are not included in error control.
    pub sens_atol: Option<Eqn::V>,
    /// Relative tolerance for output equations, if outputs are being integrated and used in error control.
    pub out_rtol: Option<Eqn::T>,
    /// Absolute tolerance for output equations, if outputs are being integrated and used in error control.
    pub out_atol: Option<Eqn::V>,
    /// Relative tolerance for the adjoint gradient wrt each parameter, if adjoint sensitivities are being computed and used in error control.
    pub param_rtol: Option<Eqn::T>,
    /// Absolute tolerance for the adjoint gradient wrt each parameter, if adjoint sensitivities are being computed and used in error control.
    pub param_atol: Option<Eqn::V>,
    /// Options for the initial condition solver.
    pub ic_options: InitialConditionSolverOptions<Eqn::T>,
    /// Options for the ODE solver.
    pub ode_options: OdeSolverOptions<Eqn::T>,
}

macro_rules! sdirk_solver_from_tableau {
    ($method:ident, $method_sens:ident, $method_solver:ident, $method_solver_sens:ident, $method_solver_adjoint:ident, $tableau:ident) => {
        #[doc = concat!("Create a new ", stringify!($tableau), " SDIRK solver instance with the given initial state.\n\n",
            "This method uses the built-in ", stringify!($tableau), " Butcher tableau.\n\n",
            "# Type Parameters\n",
            "- `LS`: The linear solver type\n\n",
            "# Arguments\n",
            "- `state`: The initial state for the solver\n\n",
            "# Returns\n",
            "An SDIRK solver instance configured with the ", stringify!($tableau), " method")]
        pub fn $method_solver<LS: LinearSolver<Eqn::M>>(
            &self,
            state: RkState<Eqn::V>,
        ) -> Result<Sdirk<'_, Eqn, LS>, DiffsolError>
        where
            Eqn: OdeEquationsImplicit,
        {
            self.sdirk_solver(
                state,
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone()),
            )
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " SDIRK solver instance with forward sensitivities, given the initial state.\n\n",
            "This method uses the built-in ", stringify!($tableau), " Butcher tableau and simultaneously solves\n",
            "the state equations and forward sensitivity equations.\n\n",
            "# Type Parameters\n",
            "- `LS`: The linear solver type\n\n",
            "# Arguments\n",
            "- `state`: The initial state for the solver (including sensitivities)\n\n",
            "# Returns\n",
            "An SDIRK solver instance configured for forward sensitivity analysis using ", stringify!($tableau))]
        pub fn $method_solver_sens<LS: LinearSolver<Eqn::M>>(
            &self,
            state: RkState<Eqn::V>,
        ) -> Result<
            Sdirk<'_, Eqn, LS, <Eqn::V as DefaultDenseMatrix>::M, SensEquations<'_, Eqn>>,
            DiffsolError,
        >
        where
            Eqn: OdeEquationsImplicitSens,
        {
            self.sdirk_solver_sens(
                state,
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone()),
            )
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " SDIRK solver instance for adjoint sensitivity analysis.\n\n",
            "This method creates a solver for the backward adjoint equations using the ", stringify!($tableau), " method.\n",
            "Requires a checkpointer to provide the forward solution during the backward solve.\n\n",
            "# Type Parameters\n",
            "- `LS`: The linear solver type\n",
            "- `S`: The forward solver method type used for checkpointing\n\n",
            "# Arguments\n",
            "- `checkpointer`: The checkpointing object containing the forward solution\n",
            "- `nout_override`: Optional override for the number of output equations\n\n",
            "# Returns\n",
            "An SDIRK solver instance configured for adjoint sensitivity analysis using ", stringify!($tableau))]
        pub fn $method_solver_adjoint<'a, LS: LinearSolver<Eqn::M>, S: OdeSolverMethod<'a, Eqn>>(
            &'a self,
            checkpointer: Checkpointing<'a, Eqn, S>,
            nout_override: Option<usize>,
        ) -> Result<
            Sdirk<'a, Eqn, LS, <Eqn::V as DefaultDenseMatrix>::M, AdjointEquations<'a, Eqn, S>>,
            DiffsolError,
        >
        where
            Eqn: OdeEquationsImplicitAdjoint,
        {
            self.sdirk_solver_adjoint(
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone()),
                checkpointer,
                nout_override,
            )
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " SDIRK solver instance with a consistent initial state.\n\n",
            "This convenience method combines state creation and solver initialization using the\n",
            "built-in ", stringify!($tableau), " Butcher tableau. It will create a consistent initial state,\n",
            "which may require solving a nonlinear system if a mass matrix is present.\n\n",
            "# Type Parameters\n",
            "- `LS`: The linear solver type\n\n",
            "# Returns\n",
            "An SDIRK solver instance configured with the ", stringify!($tableau), " method and consistent initial state")]
        pub fn $method<LS: LinearSolver<Eqn::M>>(&self) -> Result<Sdirk<'_, Eqn, LS>, DiffsolError>
        where
            Eqn: OdeEquationsImplicit,
        {
            let tableau =
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone());
            let state = self.rk_state_and_consistent::<LS, _>(&tableau)?;
            self.sdirk_solver(state, tableau)
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " SDIRK solver instance with forward sensitivities and consistent initial state.\n\n",
            "This convenience method combines state creation and solver initialization for forward\n",
            "sensitivity analysis using the built-in ", stringify!($tableau), " Butcher tableau. It will create\n",
            "a consistent initial state, which may require solving a nonlinear system if a mass matrix is present.\n\n",
            "# Type Parameters\n",
            "- `LS`: The linear solver type\n\n",
            "# Returns\n",
            "An SDIRK solver instance configured for forward sensitivity analysis using ", stringify!($tableau))]
        pub fn $method_sens<LS: LinearSolver<Eqn::M>>(
            &self,
        ) -> Result<
            Sdirk<'_, Eqn, LS, <Eqn::V as DefaultDenseMatrix>::M, SensEquations<'_, Eqn>>,
            DiffsolError,
        >
        where
            Eqn: OdeEquationsImplicitSens,
        {
            let tableau =
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone());
            let state = self.rk_state_sens_and_consistent::<LS, _>(&tableau)?;
            self.sdirk_solver_sens(state, tableau)
        }
    };
}

macro_rules! rk_solver_from_tableau {
    ($method:ident, $method_sens:ident, $method_solver:ident, $method_solver_sens:ident, $method_solver_adjoint:ident, $tableau:ident) => {
        #[doc = concat!("Create a new ", stringify!($tableau), " explicit Runge-Kutta solver instance with the given initial state.\n\n",
            "This method uses the built-in ", stringify!($tableau), " Butcher tableau.\n\n",
            "# Arguments\n",
            "- `state`: The initial state for the solver\n\n",
            "# Returns\n",
            "An explicit RK solver instance configured with the ", stringify!($tableau), " method")]
        pub fn $method_solver(
            &self,
            state: RkState<Eqn::V>,
        ) -> Result<ExplicitRk<'_, Eqn>, DiffsolError>
        where
            Eqn: OdeEquations,
        {
            self.explicit_rk_solver(
                state,
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone()),
            )
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " explicit Runge-Kutta solver instance with forward sensitivities, given the initial state.\n\n",
            "This method uses the built-in ", stringify!($tableau), " Butcher tableau and simultaneously solves\n",
            "the state equations and forward sensitivity equations.\n\n",
            "# Arguments\n",
            "- `state`: The initial state for the solver (including sensitivities)\n\n",
            "# Returns\n",
            "An explicit RK solver instance configured for forward sensitivity analysis using ", stringify!($tableau))]
        pub fn $method_solver_sens(
            &self,
            state: RkState<Eqn::V>,
        ) -> Result<
            ExplicitRk<'_, Eqn, <Eqn::V as DefaultDenseMatrix>::M, SensEquations<'_, Eqn>>,
            DiffsolError,
        >
        where
            Eqn: OdeEquationsImplicitSens,
        {
            self.explicit_rk_solver_sens(
                state,
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone()),
            )
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " explicit Runge-Kutta solver instance for adjoint sensitivity analysis.\n\n",
            "This method creates a solver for the backward adjoint equations using the ", stringify!($tableau), " method.\n",
            "Requires a checkpointer to provide the forward solution during the backward solve.\n\n",
            "# Type Parameters\n",
            "- `S`: The forward solver method type used for checkpointing (this can be auto-deduced fromt the `checkpointer`\n\n",
            "# Arguments\n",
            "- `checkpointer`: The checkpointing object containing the forward solution\n",
            "- `nout_override`: Optional override for the number of output equations\n\n",
            "# Returns\n",
            "An explicit RK solver instance configured for adjoint sensitivity analysis using ", stringify!($tableau))]
        pub fn $method_solver_adjoint<'a, S: OdeSolverMethod<'a, Eqn>>(
            &'a self,
            checkpointer: Checkpointing<'a, Eqn, S>,
            nout_override: Option<usize>,
        ) -> Result<
            ExplicitRk<'a, Eqn, <Eqn::V as DefaultDenseMatrix>::M, AdjointEquations<'a, Eqn, S>>,
            DiffsolError,
        >
        where
            Eqn: OdeEquationsAdjoint,
        {
            self.explicit_rk_solver_adjoint(
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone()),
                checkpointer,
                nout_override,
            )
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " explicit Runge-Kutta solver instance with initial state.\n\n",
            "This convenience method combines state creation and solver initialization using the\n",
            "built-in ", stringify!($tableau), " Butcher tableau.\n\n",
            "# Returns\n",
            "An explicit RK solver instance configured with the ", stringify!($tableau), " method")]
        pub fn $method(&self) -> Result<ExplicitRk<'_, Eqn>, DiffsolError>
        where
            Eqn: OdeEquations,
        {
            let tableau =
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone());
            let state = self.rk_state(&tableau)?;
            self.explicit_rk_solver(state, tableau)
        }

        #[doc = concat!("Create a new ", stringify!($tableau), " explicit Runge-Kutta solver instance with forward sensitivities.\n\n",
            "This convenience method combines state creation and solver initialization for forward\n",
            "sensitivity analysis using the built-in ", stringify!($tableau), " Butcher tableau.\n\n",
            "# Returns\n",
            "An explicit RK solver instance configured for forward sensitivity analysis using ", stringify!($tableau))]
        pub fn $method_sens(
            &self,
        ) -> Result<
            ExplicitRk<'_, Eqn, <Eqn::V as DefaultDenseMatrix>::M, SensEquations<'_, Eqn>>,
            DiffsolError,
        >
        where
            Eqn: OdeEquationsImplicitSens,
        {
            let tableau =
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone());
            let state = self.rk_state_sens(&tableau)?;
            self.explicit_rk_solver_sens(state, tableau)
        }
    };
}

impl<Eqn> OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
{
    /// Returns whether outputs are included in the error control.
    /// 
    /// This returns `true` if all of the following conditions are met:
    /// - Output integration is enabled (`integrate_out` is true)
    /// - The equations have output functions defined
    /// - Output relative tolerance is specified
    /// - Output absolute tolerance is specified
    pub fn output_in_error_control(&self) -> bool {
        self.integrate_out
            && self.eqn.out().is_some()
            && self.out_rtol.is_some()
            && self.out_atol.is_some()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        eqn: Eqn,
        rtol: Eqn::T,
        atol: Eqn::V,
        sens_rtol: Option<Eqn::T>,
        sens_atol: Option<Eqn::V>,
        out_rtol: Option<Eqn::T>,
        out_atol: Option<Eqn::V>,
        param_rtol: Option<Eqn::T>,
        param_atol: Option<Eqn::V>,
        t0: Eqn::T,
        h0: Eqn::T,
        integrate_out: bool,
        ic_options: InitialConditionSolverOptions<Eqn::T>,
        ode_options: OdeSolverOptions<Eqn::T>,
    ) -> Result<Self, DiffsolError> {
        Ok(Self {
            eqn,
            rtol,
            atol,
            out_atol,
            out_rtol,
            param_atol,
            param_rtol,
            sens_atol,
            sens_rtol,
            t0,
            h0,
            integrate_out,
            ic_options,
            ode_options,
        })
    }

    /// Returns a reference to the ODE equations being solved.
    pub fn eqn(&self) -> &Eqn {
        &self.eqn
    }
    /// Returns a mutable reference to the ODE equations being solved.
    pub fn eqn_mut(&mut self) -> &mut Eqn {
        &mut self.eqn
    }
    /// Returns a reference to the context associated with the ODE equations.
    pub fn context(&self) -> &Eqn::C {
        self.eqn.context()
    }
}

impl<Eqn> OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    /// Create a new state for the Bdf solver. This will provide a consistent initial state,
    /// so might require solving a nonlinear system if a mass matrix is present.
    pub fn bdf_state<LS: LinearSolver<Eqn::M>>(&self) -> Result<BdfState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        BdfState::new_and_consistent::<LS, Eqn>(self, 1)
    }

    /// Create a new state for the Bdf solver with sensitivities. This will provide a consistent initial state,
    /// so might require solving a nonlinear system if a mass matrix is present.
    pub fn bdf_state_sens<LS: LinearSolver<Eqn::M>>(&self) -> Result<BdfState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        BdfState::new_with_sensitivities_and_consistent::<LS, Eqn>(self, 1)
    }

    /// Create a new BDF solver instance for the problem with the given initial state.
    /// 
    /// This method creates a BDF solver with a Newton nonlinear solver using the specified
    /// linear solver type. The state must be provided and should typically be created using
    /// [`Self::bdf_state`], which ensures consistency for problems with mass matrices.
    /// 
    /// # Type Parameters
    /// - `LS`: The linear solver type used in the Newton solver
    /// 
    /// # Arguments
    /// - `state`: The initial state for the solver
    /// 
    /// # Returns
    /// A BDF solver instance configured for this problem
    #[allow(clippy::type_complexity)]
    pub fn bdf_solver<LS: LinearSolver<Eqn::M>>(
        &self,
        state: BdfState<Eqn::V>,
    ) -> Result<Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let newton_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
        Bdf::new(self, state, newton_solver)
    }

    /// Create a new BDF solver instance for the problem. This will create a consistent initial state,
    /// so might require solving a nonlinear system if a mass matrix is present.
    /// 
    /// This is a convenience method that combines [`Self::bdf_state`] and [`Self::bdf_solver`].
    /// 
    /// # Type Parameters
    /// - `LS`: The linear solver type used in the Newton solver
    /// 
    /// # Returns
    /// A BDF solver instance configured for this problem with a consistent initial state
    #[allow(clippy::type_complexity)]
    pub fn bdf<LS: LinearSolver<Eqn::M>>(
        &self,
    ) -> Result<Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let state = self.bdf_state::<LS>()?;
        self.bdf_solver(state)
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn bdf_solver_aug<
        LS: LinearSolver<Eqn::M>,
        Aug: AugmentedOdeEquationsImplicit<Eqn>,
    >(
        &self,
        state: BdfState<Eqn::V>,
        aug_eqn: Aug,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
            <Eqn::V as DefaultDenseMatrix>::M,
            Aug,
        >,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicit,
    {
        let newton_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
        Bdf::new_augmented(state, self, aug_eqn, newton_solver)
    }

    /// Create a new BDF solver instance for the backwards solve for the adjoint equations. This requires
    /// a checkpointer to provide the forward solution during the backward solve. 
    /// 
    /// If you are computing adjoint sensitivites of the continuous integral of the outputs, the number
    /// of output equations are taken from the equations being solved. If you are computing adjoint sensitivities
    /// for a discrete sum of outputs not in your equations, you must override the number of outputs via `nout_override`.
    /// 
    /// # Type Parameters
    /// - `LS`: The linear solver type used in the Newton solver
    /// - `S`: The forward solver method type used for checkpointing (this can be auto-deduced from the `checkpointer`)
    /// 
    /// # Arguments
    /// - `checkpointer`: The checkpointing object containing the forward solution
    /// - `nout_override`: Optional override for the number of output equations
    /// 
    /// # Returns
    /// A BDF solver instance configured for adjoint sensitivity analysis
    #[allow(clippy::type_complexity)]
    pub fn bdf_solver_adjoint<'a, LS: LinearSolver<Eqn::M>, S: OdeSolverMethod<'a, Eqn>>(
        &'a self,
        checkpointer: Checkpointing<'a, Eqn, S>,
        nout_override: Option<usize>,
    ) -> Result<
        Bdf<
            'a,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
            <Eqn::V as DefaultDenseMatrix>::M,
            AdjointEquations<'a, Eqn, S>,
        >,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicitAdjoint,
    {
        let h = checkpointer.last_h();
        let t = checkpointer.last_t();
        let nout = nout_override.unwrap_or_else(|| self.eqn.nout());
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer, nout)));
        let mut augmented_eqn = AdjointEquations::new(self, context, self.integrate_out);
        let mut newton_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
        let mut state = BdfState::new_without_initialise_augmented(self, &mut augmented_eqn)?;
        *state.as_mut().t = t;
        if let Some(h) = h {
            *state.as_mut().h = -h;
        }
        state.set_consistent(self, &mut newton_solver)?;
        state.set_consistent_augmented(self, &mut augmented_eqn, &mut newton_solver)?;
        state.set_step_size(
            state.h,
            augmented_eqn.atol().unwrap(),
            augmented_eqn.rtol().unwrap(),
            &augmented_eqn,
            1,
        );
        Bdf::new_augmented(state, self, augmented_eqn, newton_solver)
    }

    /// Create a new BDF solver instance for the problem with forward sensitivities, given the initial state.
    /// 
    /// This method creates a BDF solver that simultaneously solves the state equations and the
    /// forward sensitivity equations with respect to the parameters.
    /// 
    /// # Type Parameters
    /// - `LS`: The linear solver type used in the Newton solver
    /// 
    /// # Arguments
    /// - `state`: The initial state for the solver (including sensitivities)
    /// 
    /// # Returns
    /// A BDF solver instance configured for forward sensitivity analysis
    #[allow(clippy::type_complexity)]
    pub fn bdf_solver_sens<LS: LinearSolver<Eqn::M>>(
        &self,
        state: BdfState<Eqn::V>,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
            <Eqn::V as DefaultDenseMatrix>::M,
            SensEquations<'_, Eqn>,
        >,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let sens_eqn = SensEquations::new(self);
        self.bdf_solver_aug(state, sens_eqn)
    }

    /// Create a new BDF solver instance for the problem with forward sensitivities. This will create
    /// a consistent initial state, so might require solving a nonlinear system if a mass matrix is present.
    /// 
    /// This is a convenience method that combines [`Self::bdf_state_sens`] and [`Self::bdf_solver_sens`].
    /// 
    /// # Type Parameters
    /// - `LS`: The linear solver type used in the Newton solver
    /// 
    /// # Returns
    /// A BDF solver instance configured for forward sensitivity analysis with a consistent initial state
    #[allow(clippy::type_complexity)]
    pub fn bdf_sens<LS: LinearSolver<Eqn::M>>(
        &self,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS, NoLineSearch>,
            <Eqn::V as DefaultDenseMatrix>::M,
            SensEquations<'_, Eqn>,
        >,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let state = self.bdf_state_sens::<LS>()?;
        self.bdf_solver_sens(state)
    }

    /// Create a new state for the Runge-Kutta solvers (explict or implicit).
    /// Note: This function will not provide a consistent initial state for
    /// problems with a mass matrix, for this case please use [Self::rk_state_and_consistent]
    /// or initialise the state manually.
    /// 
    /// Note that in-built tableaus (e.g. TR-BDF2, ESDIRK34) have their own methods, so
    /// only use this method for custom tableaus.
    pub fn rk_state<DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquations,
    {
        RkState::new(self, tableau.order())
    }

    /// Create a new state for the Runge-Kutta solvers (explict or implicit). This will provide
    /// a consistent initial state for problems with a mass matrix, so might require solving
    /// a nonlinear system if a mass matrix is present.
    pub fn rk_state_and_consistent<LS: LinearSolver<Eqn::M>, DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        RkState::new_and_consistent::<LS, _>(self, tableau.order())
    }

    /// Create a new state for the Runge-Kutta solvers with sensitivities. Note: This function will not
    /// provide a consistent initial state for problems with a mass matrix, for this case please use
    /// [Self::rk_state_sens_and_consistent] or initialise the state manually.
    pub fn rk_state_sens<DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        RkState::new_with_sensitivities(self, tableau.order())
    }

    /// Create a new state for the Runge-Kutta solvers with sensitivities. This will provide
    /// a consistent initial state for problems with a mass matrix, so might require solving
    /// a nonlinear system if a mass matrix is present.
    pub fn rk_state_sens_and_consistent<LS: LinearSolver<Eqn::M>, DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        RkState::new_with_sensitivities_and_consistent::<LS, _>(self, tableau.order())
    }

    pub fn sdirk_solver<
        LS: LinearSolver<Eqn::M>,
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    >(
        &self,
        state: RkState<Eqn::V>,
        tableau: Tableau<DM>,
    ) -> Result<Sdirk<'_, Eqn, LS, DM>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let linear_solver = LS::default();
        Sdirk::new(self, state, tableau, linear_solver)
    }

    pub(crate) fn sdirk_solver_aug<
        LS: LinearSolver<Eqn::M>,
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
        Aug: AugmentedOdeEquationsImplicit<Eqn>,
    >(
        &self,
        state: RkState<Eqn::V>,
        tableau: Tableau<DM>,
        aug_eqn: Aug,
    ) -> Result<Sdirk<'_, Eqn, LS, DM, Aug>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        Sdirk::new_augmented(self, state, tableau, LS::default(), aug_eqn)
    }

    pub(crate) fn sdirk_solver_adjoint<
        'a,
        LS: LinearSolver<Eqn::M>,
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
        S: OdeSolverMethod<'a, Eqn>,
    >(
        &'a self,
        tableau: Tableau<DM>,
        checkpointer: Checkpointing<'a, Eqn, S>,
        nout_override: Option<usize>,
    ) -> Result<Sdirk<'a, Eqn, LS, DM, AdjointEquations<'a, Eqn, S>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitAdjoint,
    {
        let t = checkpointer.last_t();
        let h = checkpointer.last_h();
        let nout = nout_override.unwrap_or_else(|| self.eqn.nout());
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer, nout)));
        let mut augmented_eqn = AdjointEquations::new(self, context, self.integrate_out);
        let mut state = RkState::new_without_initialise_augmented(self, &mut augmented_eqn)?;
        *state.as_mut().t = t;
        if let Some(h) = h {
            *state.as_mut().h = -h;
        }
        let mut newton_solver = NewtonNonlinearSolver::new(LS::default(), NoLineSearch);
        state.set_consistent(self, &mut newton_solver)?;
        state.set_consistent_augmented(self, &mut augmented_eqn, &mut newton_solver)?;
        state.set_step_size(
            state.h,
            augmented_eqn.atol().unwrap(),
            augmented_eqn.rtol().unwrap(),
            &augmented_eqn,
            tableau.order(),
        );
        Sdirk::new_augmented(self, state, tableau, LS::default(), augmented_eqn)
    }

    pub fn sdirk_solver_sens<
        LS: LinearSolver<Eqn::M>,
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
    >(
        &self,
        state: RkState<Eqn::V>,
        tableau: Tableau<DM>,
    ) -> Result<Sdirk<'_, Eqn, LS, DM, SensEquations<'_, Eqn>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let sens_eqn = SensEquations::new(self);
        self.sdirk_solver_aug::<LS, DM, _>(state, tableau, sens_eqn)
    }

    sdirk_solver_from_tableau!(
        tr_bdf2,
        tr_bdf2_sens,
        tr_bdf2_solver,
        tr_bdf2_solver_sens,
        tr_bdf2_solver_adjoint,
        tr_bdf2
    );
    sdirk_solver_from_tableau!(
        esdirk34,
        esdirk34_sens,
        esdirk34_solver,
        esdirk34_solver_sens,
        esdirk34_solver_adjoint,
        esdirk34
    );

    /// Create a new explicit Runge-Kutta solver instance with a custom tableau.
    /// 
    /// This method creates an explicit RK solver using the provided Butcher tableau. For built-in
    /// tableaus like Tsitouras 4(5), use the specialized method [`Self::tsit45_solver`].
    /// 
    /// # Type Parameters
    /// - `DM`: The dense matrix type for the tableau (this can be auto-deduced from the `tableau`)
    /// 
    /// # Arguments
    /// - `state`: The initial state for the solver
    /// - `tableau`: The Butcher tableau defining the explicit RK method
    /// 
    /// # Returns
    /// An explicit RK solver instance configured for this problem
    pub fn explicit_rk_solver<DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>>(
        &self,
        state: RkState<Eqn::V>,
        tableau: Tableau<DM>,
    ) -> Result<ExplicitRk<'_, Eqn, DM>, DiffsolError>
    where
        Eqn: OdeEquations,
    {
        ExplicitRk::new(self, state, tableau)
    }

    pub(crate) fn explicit_rk_solver_aug<
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
        Aug: AugmentedOdeEquations<Eqn>,
    >(
        &self,
        state: RkState<Eqn::V>,
        tableau: Tableau<DM>,
        aug_eqn: Aug,
    ) -> Result<ExplicitRk<'_, Eqn, DM, Aug>, DiffsolError>
    where
        Eqn: OdeEquations,
    {
        ExplicitRk::new_augmented(self, state, tableau, aug_eqn)
    }

    pub(crate) fn explicit_rk_solver_adjoint<
        'a,
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>,
        S: OdeSolverMethod<'a, Eqn>,
    >(
        &'a self,
        tableau: Tableau<DM>,
        checkpointer: Checkpointing<'a, Eqn, S>,
        nout_override: Option<usize>,
    ) -> Result<ExplicitRk<'a, Eqn, DM, AdjointEquations<'a, Eqn, S>>, DiffsolError>
    where
        Eqn: OdeEquationsAdjoint,
    {
        let t = checkpointer.last_t();
        let h = checkpointer.last_h();
        let nout = nout_override.unwrap_or_else(|| self.eqn.nout());
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer, nout)));
        let mut augmented_eqn = AdjointEquations::new(self, context, self.integrate_out);
        let mut state = RkState::new_without_initialise_augmented(self, &mut augmented_eqn)?;
        *state.as_mut().t = t;
        if let Some(h) = h {
            *state.as_mut().h = -h;
        }

        // eval the rhs since we're not calling set_consistent_augmented
        let state_mut = state.as_mut();
        augmented_eqn.update_rhs_out_state(state_mut.y, state_mut.dy, *state_mut.t);
        let naug = augmented_eqn.max_index();
        for i in 0..naug {
            augmented_eqn.set_index(i);
            augmented_eqn
                .rhs()
                .call_inplace(&state_mut.s[i], *state_mut.t, &mut state_mut.ds[i]);
        }

        state.set_step_size(
            state.h,
            augmented_eqn.atol().unwrap(),
            augmented_eqn.rtol().unwrap(),
            &augmented_eqn,
            tableau.order(),
        );
        ExplicitRk::new_augmented(self, state, tableau, augmented_eqn)
    }

    /// Create a new explicit Runge-Kutta solver instance with forward sensitivities using a custom tableau.
    /// 
    /// This method creates an explicit RK solver that simultaneously solves the state equations and the
    /// forward sensitivity equations. For built-in tableaus, use specialized methods like [`Self::tsit45_solver_sens`].
    /// 
    /// # Type Parameters
    /// - `DM`: The dense matrix type for the tableau (this can be auto-deduced from the `tableau`)
    /// 
    /// # Arguments
    /// - `state`: The initial state for the solver (including sensitivities)
    /// - `tableau`: The Butcher tableau defining the explicit RK method
    /// 
    /// # Returns
    /// An explicit RK solver instance configured for forward sensitivity analysis
    pub fn explicit_rk_solver_sens<DM: DenseMatrix<V = Eqn::V, T = Eqn::T, C = Eqn::C>>(
        &self,
        state: RkState<Eqn::V>,
        tableau: Tableau<DM>,
    ) -> Result<ExplicitRk<'_, Eqn, DM, SensEquations<'_, Eqn>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        let sens_eqn = SensEquations::new(self);
        self.explicit_rk_solver_aug::<DM, _>(state, tableau, sens_eqn)
    }

    rk_solver_from_tableau!(
        tsit45,
        tsit45_sens,
        tsit45_solver,
        tsit45_solver_sens,
        tsit45_solver_adjoint,
        tsit45
    );
}

/// A single point in the ODE solver solution, consisting of the state vector and time.
#[derive(Debug, Clone)]
pub struct OdeSolverSolutionPoint<V: Vector> {
    /// The state vector at this solution point
    pub state: V,
    /// The time at this solution point
    pub t: V::T,
}

/// Container for the complete ODE solver solution, including state and sensitivity solution points.
/// 
/// This struct stores the solution trajectory, keeping points sorted by time. It supports both
/// forward-time integration (increasing time) and backward-time integration (decreasing time).
pub struct OdeSolverSolution<V: Vector> {
    /// The solution points for the state variables, sorted by time
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
    /// Optional sensitivity solution points, one vector per parameter
    pub sens_solution_points: Option<Vec<Vec<OdeSolverSolutionPoint<V>>>>,
    /// Relative tolerance used for the solution
    pub rtol: V::T,
    /// Absolute tolerance used for the solution
    pub atol: V,
    /// Whether this is a backward-time integration (decreasing time)
    pub negative_time: bool,
}

impl<V: Vector> OdeSolverSolution<V> {
    /// Add a new solution point to the solution, maintaining time-sorted order.
    /// 
    /// The point is inserted at the correct position to keep the solution points
    /// sorted by time (increasing for forward integration, decreasing for backward).
    /// 
    /// # Arguments
    /// - `state`: The state vector at this solution point
    /// - `t`: The time at this solution point
    pub fn push(&mut self, state: V, t: V::T) {
        // find the index to insert the new point keeping the times sorted
        let index = self.get_index(t);
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
    }
    fn get_index(&self, t: V::T) -> usize {
        if self.negative_time {
            self.solution_points
                .iter()
                .position(|x| x.t < t)
                .unwrap_or(self.solution_points.len())
        } else {
            self.solution_points
                .iter()
                .position(|x| x.t > t)
                .unwrap_or(self.solution_points.len())
        }
    }
    /// Add a new solution point with sensitivities to the solution, maintaining time-sorted order.
    /// 
    /// This method adds both the state and its sensitivities with respect to parameters.
    /// The points are inserted at the correct position to keep all solution points sorted by time.
    /// 
    /// # Arguments
    /// - `state`: The state vector at this solution point
    /// - `t`: The time at this solution point
    /// - `sens`: The sensitivity vectors at this solution point (one per parameter)
    pub fn push_sens(&mut self, state: V, t: V::T, sens: &[V]) {
        // find the index to insert the new point keeping the times sorted
        let index = self.get_index(t);
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
        // if the sensitivity solution is not initialized, initialize it
        if self.sens_solution_points.is_none() {
            self.sens_solution_points = Some(vec![vec![]; sens.len()]);
        }
        // insert the new sensitivity point at that index
        for (i, s) in sens.iter().enumerate() {
            self.sens_solution_points.as_mut().unwrap()[i].insert(
                index,
                OdeSolverSolutionPoint {
                    state: s.clone(),
                    t,
                },
            );
        }
    }
}

impl<V: Vector> Default for OdeSolverSolution<V> {
    fn default() -> Self {
        Self {
            solution_points: Vec::new(),
            sens_solution_points: None,
            rtol: V::T::from_f64(1e-6).unwrap(),
            atol: V::from_element(1, V::T::from_f64(1e-6).unwrap(), V::C::default()),
            negative_time: false,
        }
    }
}
