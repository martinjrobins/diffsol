use num_traits::FromPrimitive;
use std::{cell::RefCell, rc::Rc};

use crate::{
    error::DiffsolError, vector::Vector, AdjointContext, AdjointEquations, AugmentedOdeEquations,
    AugmentedOdeEquationsImplicit, Bdf, BdfState, Checkpointing, DefaultDenseMatrix, DenseMatrix,
    ExplicitRk, LinearSolver, MatrixRef, NewtonNonlinearSolver, NonLinearOp, OdeEquations,
    OdeEquationsAdjoint, OdeEquationsImplicit, OdeEquationsImplicitAdjoint,
    OdeEquationsImplicitSens, OdeSolverMethod, OdeSolverState, RkState, Sdirk, SensEquations,
    Tableau, VectorRef,
};

pub struct OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
{
    pub eqn: Eqn,
    pub rtol: Eqn::T,
    pub atol: Eqn::V,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
    pub integrate_out: bool,
    pub sens_rtol: Option<Eqn::T>,
    pub sens_atol: Option<Eqn::V>,
    pub out_rtol: Option<Eqn::T>,
    pub out_atol: Option<Eqn::V>,
    pub param_rtol: Option<Eqn::T>,
    pub param_atol: Option<Eqn::V>,
}

macro_rules! sdirk_solver_from_tableau {
    ($method:ident, $method_sens:ident, $method_solver:ident, $method_solver_sens:ident, $method_solver_adjoint:ident, $tableau:ident) => {
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

        pub fn $method<LS: LinearSolver<Eqn::M>>(&self) -> Result<Sdirk<'_, Eqn, LS>, DiffsolError>
        where
            Eqn: OdeEquationsImplicit,
        {
            let tableau =
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone());
            let state = self.rk_state_and_consistent::<LS, _>(&tableau)?;
            self.sdirk_solver(state, tableau)
        }

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

        pub fn $method(&self) -> Result<ExplicitRk<'_, Eqn>, DiffsolError>
        where
            Eqn: OdeEquations,
        {
            let tableau =
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(self.context().clone());
            let state = self.rk_state(&tableau)?;
            self.explicit_rk_solver(state, tableau)
        }

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
        })
    }

    pub fn eqn(&self) -> &Eqn {
        &self.eqn
    }
    pub fn eqn_mut(&mut self) -> &mut Eqn {
        &mut self.eqn
    }
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
    pub fn bdf_state<LS: LinearSolver<Eqn::M>>(&self) -> Result<BdfState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        BdfState::new_and_consistent::<LS, Eqn>(self, 1)
    }

    pub fn bdf_state_sens<LS: LinearSolver<Eqn::M>>(&self) -> Result<BdfState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        BdfState::new_with_sensitivities_and_consistent::<LS, Eqn>(self, 1)
    }

    /// Create a new state for the Bdf solver.
    /// The initial state is made consistent with any provided mass matrix.
    pub fn bdf_solver<LS: LinearSolver<Eqn::M>>(
        &self,
        state: BdfState<Eqn::V>,
    ) -> Result<Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let newton_solver = NewtonNonlinearSolver::new(LS::default());
        Bdf::new(self, state, newton_solver)
    }

    pub fn bdf<LS: LinearSolver<Eqn::M>>(
        &self,
    ) -> Result<Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS>>, DiffsolError>
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
        Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS>, <Eqn::V as DefaultDenseMatrix>::M, Aug>,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicit,
    {
        let newton_solver = NewtonNonlinearSolver::new(LS::default());
        Bdf::new_augmented(state, self, aug_eqn, newton_solver)
    }

    #[allow(clippy::type_complexity)]
    pub fn bdf_solver_adjoint<'a, LS: LinearSolver<Eqn::M>, S: OdeSolverMethod<'a, Eqn>>(
        &'a self,
        checkpointer: Checkpointing<'a, Eqn, S>,
        nout_override: Option<usize>,
    ) -> Result<
        Bdf<
            'a,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS>,
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
        let mut newton_solver = NewtonNonlinearSolver::new(LS::default());
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

    #[allow(clippy::type_complexity)]
    pub fn bdf_solver_sens<LS: LinearSolver<Eqn::M>>(
        &self,
        state: BdfState<Eqn::V>,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS>,
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

    #[allow(clippy::type_complexity)]
    pub fn bdf_sens<LS: LinearSolver<Eqn::M>>(
        &self,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS>,
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
    pub fn rk_state<DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquations,
    {
        RkState::new(self, tableau.order())
    }

    /// Create a new state for the Runge-Kutta solvers (explict or implicit).
    pub fn rk_state_and_consistent<LS: LinearSolver<Eqn::M>, DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        RkState::new_and_consistent::<LS, _>(self, tableau.order())
    }

    pub fn rk_state_sens<DM: DenseMatrix>(
        &self,
        tableau: &Tableau<DM>,
    ) -> Result<RkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicitSens,
    {
        RkState::new_with_sensitivities(self, tableau.order())
    }

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
        let mut newton_solver = NewtonNonlinearSolver::new(LS::default());
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

#[derive(Debug, Clone)]
pub struct OdeSolverSolutionPoint<V: Vector> {
    pub state: V,
    pub t: V::T,
}

pub struct OdeSolverSolution<V: Vector> {
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
    pub sens_solution_points: Option<Vec<Vec<OdeSolverSolutionPoint<V>>>>,
    pub rtol: V::T,
    pub atol: V,
    pub negative_time: bool,
}

impl<V: Vector> OdeSolverSolution<V> {
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
