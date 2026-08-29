use num_traits::Zero;
use std::cell::RefCell;

use crate::{
    op::nonlinear_op::NonLinearOpJacobian, AugmentedOdeEquations, ConstantOp, ConstantOpSens,
    Context, Matrix, NonLinearOp, NonLinearOpSens, OdeEquations, OdeEquationsImplicitSens,
    OdeEquationsRef, OdeSolverProblem, Op, Vector,
};

/// Initial condition of the sensitivity equations, for every parameter at once:
///
/// S(0) = dy(0)/dp
///
/// `dy(0)/dp` is evaluated as a matrix at the problem's own batch count and reshaped onto the
/// augmented lanes: its column `p` belongs in lane `b * nparams + p` (see
/// [`AugmentedOdeEquations`]). This runs once per state, not per step.
pub struct SensInit<'a, Eqn>
where
    Eqn: OdeEquations,
{
    eqn: &'a Eqn,
    /// `dy(0)/dp`, at the problem's batch count
    sens: RefCell<Eqn::M>,
    t0: Eqn::T,
}

impl<'a, Eqn> SensInit<'a, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    pub fn new(eqn: &'a Eqn, t0: Eqn::T) -> Self {
        let init_sens = Eqn::M::new_from_sparsity(
            eqn.rhs().nstates(),
            eqn.rhs().nparams(),
            eqn.init().sens_sparsity(),
            eqn.context().clone(),
        );
        Self {
            eqn,
            sens: RefCell::new(init_sens),
            t0,
        }
    }
}

impl<Eqn> Op for SensInit<'_, Eqn>
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

impl<Eqn> ConstantOp for SensInit<'_, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        let mut sens = self.sens.borrow_mut();
        self.eqn.init().sens_inplace(self.t0, &mut sens);
        y.fill(Self::T::zero());
        sens.add_columns_to_batched_vector(y);
    }
}

/// Right-hand side of the sensitivity equations is (we assume M_p = 0):
///
/// F(S, t) = J * S + f_p
///
/// f_p is the partial derivative of the right-hand side with respect to the parameters,
/// this is constant over a step and can be precomputed. It is a matrix of size nstates x nparams.
///
/// M_p * dy/dt is the partial derivative of the mass matrix wrt the parameters,
/// multiplied by the derivative of the state wrt time. It is a matrix of size nstates x nparams.
///
/// `S` holds one sensitivity vector per parameter in its batch lanes (see
/// [`AugmentedOdeEquations`]), which is the same layout as the columns of `f_p`: column `p` of
/// batch `b` belongs in lane `b * nparams + p`. So adding `f_p` to the sensitivities is a reshape,
/// not a product: the right-hand side is one batched Jacobian product plus one batched add. `f_p` and the cached
/// state are held at the problem's batch count and broadcast over the parameter lanes; so is the
/// Jacobian, which is shared by every lane.
///
/// Pre-compute `f_p` from the state at the given time step with [Self::update_state].
pub struct SensRhs<'a, Eqn>
where
    Eqn: OdeEquations,
{
    eqn: &'a Eqn,
    sens: RefCell<Eqn::M>,
    y: RefCell<Eqn::V>,
}

impl<'a, Eqn> SensRhs<'a, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    pub fn new(eqn: &'a Eqn, allocate: bool) -> Self {
        if !allocate {
            return Self {
                eqn,
                sens: RefCell::new(<Eqn::M as Matrix>::zeros(0, 0, eqn.context().clone())),
                y: RefCell::new(<Eqn::V as Vector>::zeros(0, eqn.context().clone())),
            };
        }
        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let rhs_sens = Eqn::M::new_from_sparsity(
            nstates,
            nparams,
            eqn.rhs().sens_sparsity().map(|s| s.to_owned()),
            eqn.context().clone(),
        );
        let y = RefCell::new(<Eqn::V as Vector>::zeros(nstates, eqn.context().clone()));
        Self {
            eqn,
            sens: RefCell::new(rhs_sens),
            y,
        }
    }

    /// pre-compute f_p from the state
    pub fn update_state(&mut self, y: &Eqn::V, _dy: &Eqn::V, t: Eqn::T) {
        let mut sens = self.sens.borrow_mut();
        self.eqn.rhs().sens_inplace(y, t, &mut sens);
        let mut state_y = self.y.borrow_mut();
        state_y.copy_from(y);
    }
}

impl<Eqn> Op for SensRhs<'_, Eqn>
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

impl<Eqn> NonLinearOp for SensRhs<'_, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    /// F(S, t) = J * S + f_p, evaluated for every parameter lane at once
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let state_y = self.y.borrow();
        let sens = self.sens.borrow();
        self.eqn.rhs().jac_mul_inplace(&state_y, t, x, y);
        sens.add_columns_to_batched_vector(y);
    }
}

impl<Eqn> NonLinearOpJacobian for SensRhs<'_, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let state_y = self.y.borrow();
        self.eqn.rhs().jac_mul_inplace(&state_y, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        let state_y = self.y.borrow();
        self.eqn.rhs().jacobian_inplace(&state_y, t, y);
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.eqn.rhs().jacobian_sparsity()
    }
}

/// Sensitivity & adjoint equations for ODEs (we assume M_p = 0):
///
/// Sensitivity equations are linear:
/// M * dS/dt = J * S + f_p
/// S(0) = dy(0)/dp
/// where
///  M is the mass matrix
///  M_p is the partial derivative of the mass matrix wrt the parameters
///  dy/dt is the derivative of the state wrt time
///  J is the Jacobian of the right-hand side
///  S holds the sensitivity wrt every parameter, one per batch lane, matching the columns of f_p
///  f_p is the partial derivative of the right-hand side with respect to the parameters
///  dy(0)/dp is the partial derivative of the state at the initial time wrt the parameters
///
pub struct SensEquations<'a, Eqn>
where
    Eqn: OdeEquations,
{
    eqn: &'a Eqn,
    rhs: SensRhs<'a, Eqn>,
    init: SensInit<'a, Eqn>,
    aug_ctx: Eqn::C,
    /// borrowed from the problem, already laid out one lane per (batch, parameter)
    atol: Option<&'a Eqn::V>,
    rtol: Option<Eqn::T>,
}

impl<Eqn> Clone for SensEquations<'_, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    fn clone(&self) -> Self {
        Self {
            eqn: self.eqn,
            rhs: SensRhs::new(self.eqn, false),
            init: SensInit::new(self.eqn, self.init.t0),
            aug_ctx: self.aug_ctx.clone(),
            rtol: self.rtol,
            atol: self.atol,
        }
    }
}

impl<Eqn> std::fmt::Debug for SensEquations<'_, Eqn>
where
    Eqn: OdeEquations,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SensEquations")
    }
}

impl<'a, Eqn> SensEquations<'a, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    pub(crate) fn new(
        problem: &'a OdeSolverProblem<Eqn>,
    ) -> Result<Self, crate::error::DiffsolError> {
        let eqn = &problem.eqn;
        let nparams = eqn.rhs().nparams();
        let nbatch = eqn.context().nbatch();
        let aug_ctx = eqn.context().clone_with_nbatch(nbatch * nparams)?;
        let rtol = problem.sens_rtol;
        // already laid out one lane per (batch, parameter) by the builder, so it matches the
        // augmented state and the whole thing is error-controlled in one norm
        let atol = problem.sens_atol.as_ref();
        let rhs = SensRhs::new(eqn, true);
        let init = SensInit::new(eqn, problem.t0);
        Ok(Self {
            rhs,
            init,
            eqn,
            aug_ctx,
            rtol,
            atol,
        })
    }
}

impl<Eqn> Op for SensEquations<'_, Eqn>
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
        self.eqn.rhs().nout()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<'a, 'b, Eqn> OdeEquationsRef<'a> for SensEquations<'b, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    type Rhs = &'a SensRhs<'b, Eqn>;
    type Mass = <Eqn as OdeEquationsRef<'a>>::Mass;
    type Root = <Eqn as OdeEquationsRef<'a>>::Root;
    type Init = &'a SensInit<'b, Eqn>;
    type Out = <Eqn as OdeEquationsRef<'a>>::Out;
    type Reset = <Eqn as OdeEquationsRef<'a>>::Reset;
}

impl<'a, Eqn> OdeEquations for SensEquations<'a, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    fn rhs(&self) -> &SensRhs<'a, Eqn> {
        &self.rhs
    }
    fn mass(&self) -> Option<<Eqn as OdeEquationsRef<'_>>::Mass> {
        self.eqn.mass()
    }
    fn root(&self) -> Option<<Eqn as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn init(&self) -> &SensInit<'a, Eqn> {
        &self.init
    }
    fn out(&self) -> Option<<Eqn as OdeEquationsRef<'_>>::Out> {
        None
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

impl<Eqn: OdeEquationsImplicitSens> AugmentedOdeEquations<Eqn> for SensEquations<'_, Eqn> {
    fn include_in_error_control(&self) -> bool {
        self.rtol.is_some() && self.atol.is_some()
    }
    fn include_out_in_error_control(&self) -> bool {
        false
    }
    fn rtol(&self) -> Option<Eqn::T> {
        self.rtol
    }
    fn atol(&self) -> Option<&Eqn::V> {
        self.atol
    }
    fn out_atol(&self) -> Option<&Eqn::V> {
        None
    }
    fn out_rtol(&self) -> Option<Eqn::T> {
        None
    }

    fn max_index(&self) -> usize {
        self.nparams()
    }
    fn aug_context(&self) -> &Eqn::C {
        &self.aug_ctx
    }
    fn update_rhs_out_state(&mut self, y: &Eqn::V, dy: &Eqn::V, t: Eqn::T) {
        self.rhs.update_state(y, dy, t);
    }
    fn integrate_main_eqn(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            exponential_decay::exponential_decay_problem_sens,
            exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_sens,
            robertson::robertson_sens,
        },
        AugmentedOdeEquations, Context, DenseMatrix, MatrixCommon, NalgebraVec, NonLinearOp,
        RkState, SensEquations, Vector,
    };
    type Mcpu = NalgebraMat<f64>;
    type Vcpu = NalgebraVec<f64>;

    /// A state with empty sensitivity vectors, for tests that only need `y`, `dy` and `t`.
    fn state_at(y: Vcpu, dy: Vcpu, ctx: <Mcpu as MatrixCommon>::C) -> RkState<Vcpu> {
        RkState {
            t: 0.0,
            y,
            dy,
            g: Vcpu::zeros(0, ctx),
            dg: Vcpu::zeros(0, ctx),
            sg: Vcpu::zeros(0, ctx),
            dsg: Vcpu::zeros(0, ctx),
            s: Vcpu::zeros(0, ctx),
            ds: Vcpu::zeros(0, ctx),
            h: 0.0,
        }
    }

    #[test]
    fn test_rhs_exponential() {
        // dy/dt = -ay (p = [a, y0])
        let (problem, _soln) = exponential_decay_problem_sens::<Mcpu>(false);
        let ctx = *problem.context();
        let mut sens_eqn = SensEquations::new(&problem).unwrap();
        let state = state_at(
            Vcpu::from_vec(vec![1.0, 1.0], ctx),
            Vcpu::from_vec(vec![1.0, 1.0], ctx),
            ctx,
        );
        // f_p = |-y  0|
        // M_p = 0
        // so f_p = |-1.0 0|
        //          |-1.0 0|
        sens_eqn.update_rhs_out_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 2);
        assert_eq!(sens.get_index(0, 0), -1.0);
        assert_eq!(sens.get_index(1, 0), -1.0);
        assert_eq!(sens.get_index(0, 1), 0.0);
        assert_eq!(sens.get_index(1, 1), 0.0);
        drop(sens);

        // F(S, t) = J * S + f_p * E, one lane per parameter
        // J = |-a 0|
        //     |0 -a|
        // lane 0 = |-a 0| |1| + |-1.0| = |-1.1|
        //          |0 -a| |2|   |-1.0|   |-1.2|
        // lane 1 = |-a 0| |1| + | 0  | = |-0.1|
        //          |0 -a| |2|   | 0  |   |-0.2|
        let aug_ctx = *sens_eqn.aug_context();
        assert_eq!(aug_ctx.nbatch(), 2);
        let s = Vcpu::from_vec(vec![1.0, 2.0, 1.0, 2.0], aug_ctx);
        let mut f = Vcpu::zeros(2, aug_ctx);
        sens_eqn.rhs.call_inplace(&s, state.t, &mut f);
        let f_expect = Vcpu::from_vec(vec![-1.1, -1.2, -0.1, -0.2], aug_ctx);
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_exponential_algebraic() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem_sens::<Mcpu>();
        let ctx = *problem.context();
        let mut sens_eqn = SensEquations::new(&problem).unwrap();
        let state = state_at(
            Vcpu::from_vec(vec![1.0, 1.0, 1.0], ctx),
            Vcpu::from_vec(vec![1.0, 1.0, 1.0], ctx),
            ctx,
        );

        // f_p = |-y|
        //       |-y|
        //       | 0|
        // M_p = 0
        // so f_p = |-1.0|
        //          |-1.0|
        //          | 0  |
        sens_eqn.update_rhs_out_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 3);
        assert_eq!(sens.ncols(), 1);
        assert_eq!(sens.get_index(0, 0), -1.0);
        assert_eq!(sens.get_index(1, 0), -1.0);
        assert_eq!(sens.get_index(2, 0), 0.0);
        sens_eqn.rhs.y.borrow().assert_eq_st(&state.y, 1e-10);
        drop(sens);

        // F(S, t) = J * S + f_p * E
        // J = |-a 0 0|
        //     |0 -a 0|
        //     |0 0 0 |
        // lane 0 = |-a 0 0| |1| + |-1.0| = |-1.1|
        //          |0 -a 0| |1|   |-1.0|   |-1.1|
        //          |0 0 0 | |1|   | 0  |   | 0 |
        let aug_ctx = *sens_eqn.aug_context();
        assert_eq!(aug_ctx.nbatch(), 1);
        let s = Vcpu::from_vec(vec![1.0, 1.0, 1.0], aug_ctx);
        let f = sens_eqn.rhs.call(&s, state.t);
        let f_expect = Vcpu::from_vec(vec![-1.1, -1.1, 0.0], aug_ctx);
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_robertson() {
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let ctx = *problem.context();
        let mut sens_eqn = SensEquations::new(&problem).unwrap();
        let state = state_at(
            Vcpu::from_vec(vec![1.0, 2.0, 3.0], ctx),
            Vcpu::from_vec(vec![1.0, 1.0, 1.0], ctx),
            ctx,
        );

        // f_p = |-x0 x1*x2 0|
        //       |x0 -x1*x2 -x1*x1|
        //       | 0   0    0|
        // M_p = 0
        sens_eqn.update_rhs_out_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 3);
        assert_eq!(sens.ncols(), 3);
        assert_eq!(sens.get_index(0, 0), -state.y[0]);
        assert_eq!(sens.get_index(0, 1), state.y[1] * state.y[2]);
        assert_eq!(sens.get_index(0, 2), 0.0);
        assert_eq!(sens.get_index(1, 0), state.y[0]);
        assert_eq!(sens.get_index(1, 1), -state.y[1] * state.y[2]);
        assert_eq!(sens.get_index(1, 2), -state.y[1] * state.y[1]);
        assert_eq!(sens.get_index(2, 0), 0.0);
        assert_eq!(sens.get_index(2, 1), 0.0);
        assert_eq!(sens.get_index(2, 2), 0.0);
    }
}
