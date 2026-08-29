use num_traits::{One, Signed, Zero};
use std::{
    cell::{Cell, RefCell},
    ops::SubAssign,
    rc::Rc,
};

use crate::{
    error::DiffsolError, op::nonlinear_op::NonLinearOpJacobian, AugmentedOdeEquations,
    CheckpointingPath, ConstantOp, ConstantOpSensAdjoint, Context, LinearOp, LinearOpTranspose,
    Matrix, NonLinearOp, NonLinearOpAdjoint, NonLinearOpSensAdjoint, OdeEquations,
    OdeEquationsAdjoint, OdeEquationsRef, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op,
    Scalar, Vector,
};

pub struct AdjointContext<'a, Eqn, State, Method>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
    Method: OdeSolverMethod<'a, Eqn, State = State>,
{
    eqn: &'a Eqn,
    t0: Eqn::T,
    checkpointers: CheckpointingPath<Eqn, State>,
    active_checkpointer: usize,
    solver: Option<RefCell<Method>>,
    x: Eqn::V,
    max_index: usize,
    last_t: Option<Eqn::T>,
    aug_ctx: Eqn::C,
}

impl<'a, Eqn, State, Method> AdjointContext<'a, Eqn, State, Method>
where
    Eqn: OdeEquations,
    State: OdeSolverState<Eqn::V>,
    Method: OdeSolverMethod<'a, Eqn, State = State>,
{
    /// `aug_ctx` must hold `eqn.context().nbatch() * max_index` batches, see
    /// [`AugmentedOdeEquations`].
    pub fn new(
        eqn: &'a Eqn,
        t0: Eqn::T,
        checkpointers: CheckpointingPath<Eqn, State>,
        solver: Option<Method>,
        max_index: usize,
        aug_ctx: Eqn::C,
    ) -> Self {
        let active_checkpointer = checkpointers
            .len()
            .checked_sub(1)
            .expect("adjoint checkpointing path must not be empty");
        let ctx = eqn.context();
        let x = <Eqn::V as Vector>::zeros(eqn.rhs().nstates(), ctx.clone());
        Self {
            eqn,
            t0,
            checkpointers,
            active_checkpointer,
            solver: solver.map(RefCell::new),
            x,
            max_index,
            aug_ctx,
            last_t: None,
        }
    }

    fn active_checkpointer(&self) -> &crate::Checkpointing<Eqn, State> {
        &self.checkpointers[self.active_checkpointer]
    }

    pub(crate) fn pop_last_checkpointing(&mut self) -> Option<crate::Checkpointing<Eqn, State>> {
        if self.active_checkpointer == self.checkpointers.len() - 1 {
            self.active_checkpointer = self.active_checkpointer.saturating_sub(1);
        }
        self.checkpointers.pop()
    }

    pub fn set_state(&mut self, t: Eqn::T) {
        if let Some(last_t) = self.last_t {
            if last_t == t {
                return;
            }
        }
        // clamp tiny boundary overshoots to the boundary values to avoid interpolation errors
        let t0 = self.t0;
        let t1 = self.checkpointers[self.checkpointers.len() - 1].end_t();
        let boundary_tol = Eqn::T::EPSILON.sqrt() * (t.abs() + t1.abs() + Eqn::T::one());
        let t_interp = if t > t1 && t - t1 <= boundary_tol {
            t1
        } else if t < t0 && t0 - t <= boundary_tol {
            t0
        } else {
            t
        };
        self.last_t = Some(t_interp);
        while self.active_checkpointer > 0
            && t_interp + boundary_tol < self.active_checkpointer().first_t()
        {
            self.active_checkpointer -= 1;
        }
        while self.active_checkpointer + 1 < self.checkpointers.len()
            && t_interp > self.active_checkpointer().end_t() + boundary_tol
        {
            self.active_checkpointer += 1;
        }
        let active_checkpointer = self.active_checkpointer;
        let x = &mut self.x;
        match self.solver.as_ref() {
            Some(solver) => {
                let mut solver = solver.borrow_mut();
                self.checkpointers[active_checkpointer]
                    .interpolate(Some(&mut *solver), t_interp, x)
                    .unwrap();
            }
            None => self.checkpointers[active_checkpointer]
                .interpolate::<Method>(None, t_interp, x)
                .unwrap(),
        }
        // for diffsl, we need to set data for the adjoint state!
        // basically just involves calling the normal rhs function with the new self.x
        // todo: this seems a bit hacky, perhaps a dedicated function on the trait for this?
        self.eqn.rhs().call(&self.x, t_interp);
    }

    pub fn state(&self) -> &Eqn::V {
        &self.x
    }

    /// The context of the augmented state, holding `nbatch * max_index` batches.
    pub fn aug_context(&self) -> &Eqn::C {
        &self.aug_ctx
    }
}

pub struct AdjointMass<'a, Eqn>
where
    Eqn: OdeEquations,
{
    eqn: &'a Eqn,
}

impl<'a, Eqn> AdjointMass<'a, Eqn>
where
    Eqn: OdeEquations,
{
    pub fn new(eqn: &'a Eqn) -> Self {
        Self { eqn }
    }
}

impl<Eqn> Op for AdjointMass<'_, Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type C = Eqn::C;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<Eqn> LinearOp for AdjointMass<'_, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        self.eqn
            .mass()
            .unwrap()
            .gemv_transpose_inplace(x, t, beta, y);
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.eqn.mass().unwrap().transpose_inplace(t, y);
    }

    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.eqn.mass().unwrap().transpose_sparsity()
    }
}

pub struct AdjointInit<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    _marker: std::marker::PhantomData<Method>,
}

impl<'a, Eqn, Method> AdjointInit<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub fn new(
        eqn: &'a Eqn,
        _context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
        _with_out: bool,
    ) -> Self {
        Self {
            eqn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, Eqn, Method> Op for AdjointInit<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type C = Eqn::C;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<'a, Eqn, Method> ConstantOp for AdjointInit<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.fill(Eqn::T::zero());
    }
}

/// Right-hand side of the adjoint equations is:
///
/// F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
///
/// f_x is the partial derivative of the right-hand side with respect to the state vector.
/// g_x is the partial derivative of the functional g with respect to the state vector.
///
/// We need the current state x(t), which is obtained from the checkpointed forward solve at the current time step.
pub struct AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
    /// `-g_y^T`, at the problem's own batch count, one column per output channel
    out_adjoint: RefCell<Eqn::M>,
    /// the time `out_adjoint` was last filled at
    cached_t: Cell<Option<Eqn::T>>,
    with_out: bool,
}

impl<'a, Eqn, Method> AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub fn new(
        eqn: &'a Eqn,
        context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
        with_out: bool,
    ) -> Self {
        let nstates = eqn.rhs().nstates();
        let nout = context.borrow().max_index;
        let ctx = eqn.context().clone();
        let out_adjoint = match (with_out, eqn.out()) {
            (false, _) => <Eqn::M as Matrix>::zeros(0, 0, ctx),
            (true, Some(out)) => {
                <Eqn::M as Matrix>::new_from_sparsity(nstates, nout, out.adjoint_sparsity(), ctx)
            }
            (true, None) => {
                // The default output is the identity on the state, so `g_y = I` and this operator
                // contributes `-g_y^T = -I` — this is currently dead code as you can't integrate
                // the output without defining one, kept in case this changes.
                let n = nstates.min(nout);
                let indices = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
                let values = vec![-Eqn::T::one(); n * ctx.nbatch()];
                <Eqn::M as Matrix>::try_from_triplets(nstates, nout, indices, values, ctx)
                    .expect("failed to build the identity output Jacobian")
            }
        };
        Self {
            eqn,
            context,
            out_adjoint: RefCell::new(out_adjoint),
            cached_t: Cell::new(None),
            with_out,
        }
    }
}

impl<'a, Eqn, Method> Op for AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type C = Eqn::C;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<'a, Eqn, Method> NonLinearOp for AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    /// F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
    fn call_inplace(&self, lambda: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();

        // y = -f^T_x(x, t) λ
        self.eqn.rhs().jac_transpose_mul_inplace(x, t, lambda, y);

        // y = -f^T_x(x, t) λ - g^T_x(x,t): column `i` of `-g_y^T` is channel `i`'s contribution,
        // so it goes straight onto lane `b * nout + i`
        if self.with_out {
            let mut out_adjoint = self.out_adjoint.borrow_mut();
            // with no out operator the matrix is the constant identity and is never refilled
            if let Some(out) = self.eqn.out() {
                if self.cached_t.get() != Some(t) {
                    // the forward state only moves when `t` does, so refill once per time point
                    out.adjoint_inplace(x, t, &mut out_adjoint);
                    self.cached_t.set(Some(t));
                }
            }
            out_adjoint.add_columns_to_batched_vector(y);
        }
    }
}

impl<'a, Eqn, Method> NonLinearOpJacobian for AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    // J = -f^T_x(x, t)
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().jac_transpose_mul_inplace(x, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().adjoint_inplace(x, t, y);
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.eqn.rhs().adjoint_sparsity()
    }
}

/// Output of the adjoint equations is:
///
/// F(λ, x, t) = -g_p^T(x, t) - f_p^T(x, t) λ
///
/// f_p is the partial derivative of the right-hand side with respect to the parameter vector
/// g_p is the partial derivative of the functional g with respect to the parameter vector
///
/// We need the current state x(t), which is obtained from the checkpointed forward solve at the current time step.
pub struct AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
    /// `-g_p^T`, at the problem's own batch count, one column per output channel
    out_sens_adjoint: RefCell<Eqn::M>,
    /// the time `out_sens_adjoint` was last filled at
    cached_t: Cell<Option<Eqn::T>>,
    with_out: bool,
}

impl<'a, Eqn, Method> AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub fn new(
        eqn: &'a Eqn,
        context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
        with_out: bool,
    ) -> Self {
        let ctx = eqn.context().clone();
        // with no out operator there is no `g_p` term at all
        let out_sens_adjoint = match (with_out, eqn.out()) {
            (true, Some(out)) => <Eqn::M as Matrix>::new_from_sparsity(
                eqn.rhs().nparams(),
                context.borrow().max_index,
                out.sens_adjoint_sparsity(),
                ctx,
            ),
            _ => <Eqn::M as Matrix>::zeros(0, 0, ctx),
        };
        Self {
            eqn,
            context,
            out_sens_adjoint: RefCell::new(out_sens_adjoint),
            cached_t: Cell::new(None),
            with_out,
        }
    }
}

impl<'a, Eqn, Method> Op for AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type C = Eqn::C;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<'a, Eqn, Method> NonLinearOp for AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    /// F(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t)
    fn call_inplace(&self, lambda: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_transpose_mul_inplace(x, t, lambda, y);

        // column `i` of `-g_p^T` is channel `i`'s contribution, see `AdjointRhs::call_inplace`
        if self.with_out {
            if let Some(out) = self.eqn.out() {
                let mut out_sens_adjoint = self.out_sens_adjoint.borrow_mut();
                if self.cached_t.get() != Some(t) {
                    out.sens_adjoint_inplace(x, t, &mut out_sens_adjoint);
                    self.cached_t.set(Some(t));
                }
                out_sens_adjoint.add_columns_to_batched_vector(y);
            }
        }
    }
}

impl<'a, Eqn, Method> NonLinearOpJacobian for AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    // J = -f_p(x, t)
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_transpose_mul_inplace(x, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_adjoint_inplace(x, t, y);
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.eqn.rhs().sens_adjoint_sparsity()
    }
}

/// Adjoint equations for ODEs
///
/// M * dλ/dt = -f^T_x(x, t) λ - g^T_x(x,t)
/// λ(T) = 0
/// g(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t)
///
pub struct AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    rhs: AdjointRhs<'a, Eqn, Method>,
    out: AdjointOut<'a, Eqn, Method>,
    mass: Option<AdjointMass<'a, Eqn>>,
    context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
    tmp: RefCell<Eqn::V>,
    tmp2: RefCell<Eqn::V>,
    aug_ctx: Eqn::C,
    init: AdjointInit<'a, Eqn, Method>,
    atol: Option<&'a Eqn::V>,
    rtol: Option<Eqn::T>,
    out_rtol: Option<Eqn::T>,
    out_atol: Option<&'a Eqn::V>,
}

impl<'a, Eqn, Method> Clone for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn clone(&self) -> Self {
        let context_ref = self.context.borrow();
        let context = Rc::new(RefCell::new(AdjointContext::new(
            context_ref.eqn,
            context_ref.t0,
            context_ref.checkpointers.clone(),
            context_ref
                .solver
                .as_ref()
                .map(|solver| solver.borrow().clone()),
            context_ref.max_index,
            context_ref.aug_ctx.clone(),
        )));
        let rhs = AdjointRhs::new(self.eqn, context.clone(), self.rhs.with_out);
        let init = AdjointInit::new(self.eqn, context.clone(), self.rhs.with_out);
        let out = AdjointOut::new(self.eqn, context.clone(), self.out.with_out);
        let tmp = self.tmp.clone();
        let tmp2 = self.tmp2.clone();
        let atol = self.atol;
        let rtol = self.rtol;
        let out_atol = self.out_atol;
        let out_rtol = self.out_rtol;
        let mass = self.eqn.mass().map(|_m| AdjointMass::new(self.eqn));
        let aug_ctx = self.aug_ctx.clone();
        Self {
            rhs,
            init,
            mass,
            context,
            out,
            tmp,
            tmp2,
            aug_ctx,
            eqn: self.eqn,
            atol,
            rtol,
            out_rtol,
            out_atol,
        }
    }
}

impl<'a, Eqn, Method> AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub(crate) fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        context: Rc<RefCell<AdjointContext<'a, Eqn, Method::State, Method>>>,
        with_out: bool,
    ) -> Self {
        let eqn = &problem.eqn;
        let rhs = AdjointRhs::new(eqn, context.clone(), with_out);
        let init = AdjointInit::new(eqn, context.clone(), with_out);
        let out = AdjointOut::new(eqn, context.clone(), with_out);
        let aug_ctx = context.borrow().aug_context().clone();
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(
            eqn.rhs().nparams(),
            aug_ctx.clone(),
        ));
        let tmp2 = RefCell::new(<Eqn::V as Vector>::zeros(
            eqn.rhs().nstates(),
            aug_ctx.clone(),
        ));
        let atol = Some(&problem.atol);
        let rtol = Some(problem.rtol);
        let out_atol = problem.param_atol.as_ref();
        let out_rtol = problem.param_rtol;
        let mass = eqn.mass().map(|_m| AdjointMass::new(eqn));
        Self {
            rhs,
            init,
            mass,
            context,
            out,
            tmp,
            tmp2,
            aug_ctx,
            eqn,
            atol,
            rtol,
            out_rtol,
            out_atol,
        }
    }

    pub fn eqn(&self) -> &'a Eqn {
        self.eqn
    }

    pub fn last_t(&self) -> Eqn::T {
        self.context.borrow().checkpointers.last().unwrap().last_t()
    }

    pub fn last_h(&self) -> Option<Eqn::T> {
        self.context.borrow().checkpointers.last().unwrap().last_h()
    }

    pub(crate) fn checkpointing_len(&self) -> usize {
        self.context.borrow().checkpointers.len()
    }

    pub(crate) fn checkpointing_bounds(&self, index: usize) -> (Eqn::T, Eqn::T) {
        let context = self.context.borrow();
        let checkpointer = &context.checkpointers[index];
        (checkpointer.first_t(), checkpointer.end_t())
    }

    pub(crate) fn checkpointing_terminal_reset_root_idx(&self, index: usize) -> Option<usize> {
        self.context.borrow().checkpointers[index].terminal_reset_root_idx()
    }

    pub fn with_out(&self) -> bool {
        self.rhs.with_out
    }

    pub fn correct_sg_for_init(&self, t: Eqn::T, s: &Eqn::V, sg: &mut Eqn::V) {
        let mut tmp = self.tmp.borrow_mut();
        if let Some(mass) = self.eqn.mass() {
            let mut tmp2 = self.tmp2.borrow_mut();
            mass.call_transpose_inplace(s, t, &mut tmp2);
            self.eqn
                .init()
                .sens_transpose_mul_inplace(t, &tmp2, &mut tmp);
        } else {
            self.eqn.init().sens_transpose_mul_inplace(t, s, &mut tmp);
        }
        sg.sub_assign(&*tmp);
    }

    pub fn interpolate_forward_state(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        let mut context = self.context.borrow_mut();
        context.set_state(t);
        y.copy_from(context.state());
        Ok(())
    }

    pub fn checkpointing_last_state(&self, index: usize) -> Method::State {
        self.context.borrow().checkpointers[index]
            .last_checkpoint()
            .clone()
    }

    pub fn checkpointing_first_state(&self, index: usize) -> Method::State {
        self.context.borrow().checkpointers[index]
            .first_checkpoint()
            .clone()
    }

    pub fn pop_last_checkpointing(
        &mut self,
    ) -> Result<crate::Checkpointing<Eqn, Method::State>, DiffsolError> {
        let mut context = self.context.borrow_mut();
        context
            .pop_last_checkpointing()
            .ok_or_else(|| DiffsolError::Other("No more checkpointing to pop".to_string()))
    }

    pub fn into_checkpointing(self) -> CheckpointingPath<Eqn, Method::State> {
        let Self {
            rhs,
            out,
            context,
            eqn: _,
            mass: _,
            tmp: _,
            tmp2: _,
            aug_ctx: _,
            init: _,
            atol: _,
            rtol: _,
            out_rtol: _,
            out_atol: _,
        } = self;

        drop(rhs);
        drop(out);

        match Rc::try_unwrap(context) {
            Ok(context) => context.into_inner().checkpointers,
            Err(_) => {
                panic!("adjoint context should be uniquely owned after consuming AdjointEquations")
            }
        }
    }
}

impl<'a, Eqn, Method> std::fmt::Debug for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdjointEquations").finish()
    }
}

impl<'a, Eqn, Method> Op for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type C = Eqn::C;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nout()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<'a, 'b, Eqn, Method> OdeEquationsRef<'a> for AdjointEquations<'b, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'b, Eqn>,
{
    type Rhs = &'a AdjointRhs<'b, Eqn, Method>;
    type Mass = &'a AdjointMass<'b, Eqn>;
    type Root = <Eqn as OdeEquationsRef<'a>>::Root;
    type Init = &'a AdjointInit<'b, Eqn, Method>;
    type Out = &'a AdjointOut<'b, Eqn, Method>;
    type Reset = <Eqn as OdeEquationsRef<'a>>::Reset;
}

impl<'a, Eqn, Method> OdeEquations for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn rhs(&self) -> &AdjointRhs<'a, Eqn, Method> {
        &self.rhs
    }
    fn mass(&self) -> Option<&AdjointMass<'a, Eqn>> {
        self.mass.as_ref()
    }
    fn root(&self) -> Option<<Eqn as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn init(&self) -> &AdjointInit<'a, Eqn, Method> {
        &self.init
    }
    fn out(&self) -> Option<&AdjointOut<'a, Eqn, Method>> {
        Some(&self.out)
    }
    fn set_params(&mut self, p: &Self::V) {
        self.eqn.set_params(p);
    }
    fn set_model_index(&mut self, m: usize) {
        self.eqn.set_model_index(m);
    }
    fn get_params(&self, p: &mut Self::V) {
        self.eqn.get_params(p);
    }
}

impl<'a, Eqn, Method> AugmentedOdeEquations<Eqn> for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn include_in_error_control(&self) -> bool {
        self.atol.is_some() && self.rtol.is_some()
    }
    fn include_out_in_error_control(&self) -> bool {
        self.out().is_some() && self.out_atol.is_some() && self.out_rtol.is_some()
    }

    fn atol(&self) -> Option<&Eqn::V> {
        self.atol
    }
    fn out_atol(&self) -> Option<&Eqn::V> {
        self.out_atol
    }
    fn out_rtol(&self) -> Option<Eqn::T> {
        self.out_rtol
    }
    fn rtol(&self) -> Option<Eqn::T> {
        self.rtol
    }

    fn max_index(&self) -> usize {
        self.context.borrow().max_index
    }

    fn aug_context(&self) -> &Eqn::C {
        &self.aug_ctx
    }

    fn update_rhs_out_state(&mut self, _y: &Eqn::V, _dy: &Eqn::V, _t: Eqn::T) {}

    fn integrate_main_eqn(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::{
            adjoint_equations::AdjointEquations,
            test_models::{
                exponential_decay::exponential_decay_problem_adjoint,
                logistic::logistic_problem_adjoint_no_out,
            },
        },
        AdjointContext, AugmentedOdeEquations, Checkpointing, Context, DenseMatrix, FaerSparseLU,
        FaerSparseMat, FaerVec, Matrix, MatrixCommon, NalgebraVec, NonLinearOp,
        NonLinearOpJacobian, OdeEquations, Op, RkState, Vector,
    };
    type Mcpu = NalgebraMat<f64>;
    type Vcpu = NalgebraVec<f64>;
    type LS = crate::NalgebraLU<f64>;

    /// With no explicit output operator the output is the identity on the state, so `g_y = I` and
    /// each adjoint channel picks up `-g_y^T e_i = -e_i`. That term used to be a one-hot basis
    /// vector added with the *opposite* sign; it is now `-I`, taking the same reshape path as an
    /// explicit `-g_y^T`.
    #[test]
    fn test_rhs_identity_output() {
        // dy/dt = r*u*(1 - u/k), no out operator, so nout == nstates == 1
        let (problem, _soln) = logistic_problem_adjoint_no_out::<Mcpu>();
        let ctx = problem.eqn.context();
        let state = RkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![2.0], *ctx),
            dy: Vcpu::from_vec(vec![0.0], *ctx),
            g: Vcpu::zeros(0, *ctx),
            dg: Vcpu::zeros(0, *ctx),
            sg: Vcpu::zeros(0, *ctx),
            dsg: Vcpu::zeros(0, *ctx),
            s: Vcpu::zeros(0, *ctx),
            ds: Vcpu::zeros(0, *ctx),
            h: 0.0,
        };
        let nout = problem.eqn.nout();
        assert_eq!(nout, 1);
        let mut solver = problem.esdirk34_solver::<LS>(state.clone()).unwrap();
        let checkpointer = Checkpointing::new(
            Some(&mut solver),
            0,
            vec![state.clone(), state.clone()],
            None,
        );
        let aug_ctx = ctx.clone_with_nbatch(nout).unwrap();
        let context = Rc::new(RefCell::new(AdjointContext::new(
            &problem.eqn,
            problem.t0,
            vec![checkpointer],
            Some(solver.clone()),
            nout,
            aug_ctx,
        )));
        // `with_out` with no out operator: the identity-output branch
        let adj_eqn = AdjointEquations::new(&problem, context, true);

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_y e_0, with r = k = 1, x = 2 and g_y = I:
        //   -f_x = -r(1 - 2x/k) = 3, so F = 3λ - 1 = 2 at λ = 1
        let lambda = Vcpu::from_vec(vec![1.0], *ctx);
        let mut f = Vcpu::zeros(1, *ctx);
        adj_eqn.rhs.call_inplace(&lambda, state.t, &mut f);
        f.assert_eq_st(&Vcpu::from_vec(vec![2.0], *ctx), 1e-10);
    }

    #[test]
    fn test_rhs_exponential() {
        // dy/dt = -ay (p = [a])
        // a = 0.1
        let (problem, _soln) = exponential_decay_problem_adjoint::<Mcpu>(true, true);
        let ctx = problem.eqn.context();
        let state = RkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0], *ctx),
            dy: Vcpu::from_vec(vec![1.0, 1.0], *ctx),
            g: Vcpu::zeros(0, *ctx),
            dg: Vcpu::zeros(0, *ctx),
            sg: Vcpu::zeros(0, *ctx),
            dsg: Vcpu::zeros(0, *ctx),
            s: Vcpu::zeros(0, *ctx),
            ds: Vcpu::zeros(0, *ctx),
            h: 0.0,
        };
        let nout = problem.eqn.out().unwrap().nout();
        let mut solver = problem.esdirk34_solver::<LS>(state.clone()).unwrap();
        let checkpointer = Checkpointing::new(
            Some(&mut solver),
            0,
            vec![state.clone(), state.clone()],
            None,
        );
        // the adjoint state holds one channel per output in its batch lanes
        let aug_ctx = ctx.clone_with_nbatch(nout).unwrap();
        let context = Rc::new(RefCell::new(AdjointContext::new(
            &problem.eqn,
            problem.t0,
            vec![checkpointer],
            Some(solver.clone()),
            nout,
            aug_ctx,
        )));
        let adj_eqn = AdjointEquations::new(&problem, context.clone(), false);
        // F(λ, x, t) = -f^T_x(x, t) λ
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| = |a| = |0.1|
        //              |0 a| |2|   |2a| = |0.2|
        let v = Vcpu::from_vec(vec![1.0, 2.0], *ctx);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![0.1, 0.2], *ctx);
        f.assert_eq_st(&f_expect, 1e-10);

        let adj_eqn = AdjointEquations::new(&problem, context, true);

        // f_x^T = |-a 0|
        //         |0 -a|
        // J = -f_x^T
        let adjoint = adj_eqn.rhs.jacobian(&state.y, state.t);
        assert_eq!(adjoint.nrows(), 2);
        assert_eq!(adjoint.ncols(), 2);
        assert_eq!(adjoint.get_index(0, 0), 0.1);
        assert_eq!(adjoint.get_index(1, 1), 0.1);

        // g_x = |1 2|
        //       |3 4|
        // S = -g^T_x(x,t)
        // so S = |-1 -3|
        //        |-2 -4|

        // f_p^T = |-x_1 -x_2 |
        //         |0   0 |
        // g_p = |0 0|
        //       |0 0|
        // g(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t)
        //            = |1  1| |1| + |0| = |3|
        //              |0  0| |2|  |0|  = |0|
        // one lambda per output channel, all channels evaluated at once
        let aug_ctx = *adj_eqn.aug_context();
        let v_aug = Vcpu::from_vec(vec![1.0, 2.0, 1.0, 2.0], aug_ctx);
        let mut out = Vcpu::zeros(2, aug_ctx);
        adj_eqn.out.call_inplace(&v_aug, state.t, &mut out);
        // g_p = 0, so both channels give the same result
        let out_expect = Vcpu::from_vec(vec![3.0, 0.0, 3.0, 0.0], aug_ctx);
        out.assert_eq_st(&out_expect, 1e-10);

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t) E
        // f_x = |-a 0|
        //       |0 -a|
        // channel 0 = |a 0| |1| - |1.0| = | a - 1| = |-0.9|
        //             |0 a| |2|   |2.0|   |2a - 2|   |-1.8|
        // channel 1 = |a 0| |1| - |3.0| = | a - 3| = |-2.9|
        //             |0 a| |2|   |4.0|   |2a - 4|   |-3.8|
        let mut f = Vcpu::zeros(2, aug_ctx);
        adj_eqn.rhs.call_inplace(&v_aug, state.t, &mut f);
        let f_expect = Vcpu::from_vec(vec![-0.9, -1.8, -2.9, -3.8], aug_ctx);
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_exponential_sparse() {
        // dy/dt = -ay (p = [a])
        // a = 0.1
        let (problem, _soln) = exponential_decay_problem_adjoint::<FaerSparseMat<f64>>(true, true);
        let ctx = problem.eqn.context();
        let state = RkState {
            t: 0.0,
            y: FaerVec::from_vec(vec![1.0, 1.0], *ctx),
            dy: FaerVec::from_vec(vec![1.0, 1.0], *ctx),
            g: FaerVec::zeros(0, *ctx),
            dg: FaerVec::zeros(0, *ctx),
            sg: FaerVec::zeros(0, *ctx),
            dsg: FaerVec::zeros(0, *ctx),
            s: FaerVec::zeros(0, *ctx),
            ds: FaerVec::zeros(0, *ctx),
            h: 0.0,
        };
        let nout = problem.eqn.out().unwrap().nout();
        let mut solver = problem
            .esdirk34_solver::<FaerSparseLU<f64>>(state.clone())
            .unwrap();
        let checkpointer = Checkpointing::new(
            Some(&mut solver),
            0,
            vec![state.clone(), state.clone()],
            None,
        );
        // the adjoint state holds one channel per output in its batch lanes
        let aug_ctx = ctx.clone_with_nbatch(nout).unwrap();
        let context = Rc::new(RefCell::new(AdjointContext::new(
            &problem.eqn,
            problem.t0,
            vec![checkpointer],
            Some(solver.clone()),
            nout,
            aug_ctx,
        )));
        let adj_eqn = AdjointEquations::new(&problem, context, true);

        // f_x^T = |-a 0|
        //         |0 -a|
        // J = -f_x^T
        let adjoint = adj_eqn.rhs.jacobian(&state.y, state.t);
        assert_eq!(adjoint.nrows(), 2);
        assert_eq!(adjoint.ncols(), 2);
        let (idx, vals) = adjoint.triplet_iter();
        for ((i, j), v) in idx.zip(vals) {
            if i == j {
                assert_eq!(v, 0.1);
            } else {
                assert_eq!(v, 0.0);
            }
        }

        // g_x = |1 2|
        //       |3 4|
        // S = -g^T_x(x,t)
        // so S = |-1 -3|
        //        |-2 -4|

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| - |1.0| = |a - 1| = |-0.9|
        //              |0 a| |2|   |2.0|   |2a - 2| = |-1.8|
        let aug_ctx = *adj_eqn.aug_context();
        let v = FaerVec::from_vec(vec![1.0, 2.0, 1.0, 2.0], aug_ctx);
        let mut f = FaerVec::zeros(2, aug_ctx);
        adj_eqn.rhs.call_inplace(&v, state.t, &mut f);
        let f_expect = FaerVec::from_vec(vec![-0.9, -1.8, -2.9, -3.8], aug_ctx);
        f.assert_eq_st(&f_expect, 1e-10);
    }
}
