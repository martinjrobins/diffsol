use num_traits::{One, Zero};
use std::cell::RefCell;

use crate::{
    op::nonlinear_op::NonLinearOpJacobian, AugmentedOdeEquations, ConstantOp, ConstantOpSens,
    Matrix, NonLinearOp, NonLinearOpSens, OdeEquations, OdeEquationsImplicitSens, OdeEquationsRef,
    OdeSolverProblem, Op, Vector,
};

pub struct SensInit<'a, Eqn>
where
    Eqn: OdeEquations,
{
    eqn: &'a Eqn,
    index: usize,
    tmp: Eqn::V,
    t0: Eqn::T,
}

impl<'a, Eqn> SensInit<'a, Eqn>
where
    Eqn: OdeEquationsImplicitSens,
{
    pub fn new(eqn: &'a Eqn, t0: Eqn::T) -> Self {
        let index = 0;
        let nparams = eqn.rhs().nparams();
        let tmp = Eqn::V::zeros(nparams, eqn.context().clone());
        Self {
            tmp,
            eqn,
            index,
            t0,
        }
    }
    pub fn set_param_index(&mut self, index: usize) {
        self.tmp.set_index(self.index, Eqn::T::zero());
        self.index = index;
        self.tmp.set_index(self.index, Eqn::T::one());
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
        self.eqn.init().sens_mul_inplace(self.t0, &self.tmp, y);
    }
}

/// Right-hand side of the sensitivity equations is (we assume M_p = 0):
///
/// F(s, t) = J * s + f_p
///
/// f_p is the partial derivative of the right-hand side with respect to the parameters,
/// this is constant and can be precomputed. It is a matrix of size nstates x nparams.
///
/// M_p * dy/dt is the partial derivative of the mass matrix wrt the parameters,
/// multiplied by the derivative of the state wrt time. It is a matrix of size nstates x nparams.
///
/// Strategy is to pre-compute S = f_p from the state at given time step and store it in a matrix using [Self::update_state].
/// Then the ith column of function F(s, t) is evaluated as J * s_i + S_i, where s_i is the ith column of the sensitivity matrix
/// and S_i is the ith column of the matrix S. The column to evaluate is set using [Self::set_param_index].
pub struct SensRhs<'a, Eqn>
where
    Eqn: OdeEquations,
{
    eqn: &'a Eqn,
    sens: RefCell<Eqn::M>,
    y: RefCell<Eqn::V>,
    index: RefCell<usize>,
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
                index: RefCell::new(0),
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
        let index = RefCell::new(0);
        Self {
            eqn,
            sens: RefCell::new(rhs_sens),
            y,
            index,
        }
    }

    /// pre-compute S = f_p from the state
    pub fn update_state(&mut self, y: &Eqn::V, _dy: &Eqn::V, t: Eqn::T) {
        let mut sens = self.sens.borrow_mut();
        self.eqn.rhs().sens_inplace(y, t, &mut sens);
        let mut state_y = self.y.borrow_mut();
        state_y.copy_from(y);
    }
    pub fn set_param_index(&self, index: usize) {
        self.index.replace(index);
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
    /// the ith column of function F(s, t) is evaluated as J * s_i + S_i, where s_i is the ith column of the sensitivity matrix
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let state_y = self.y.borrow();
        let sens = self.sens.borrow();
        let index = *self.index.borrow();
        self.eqn.rhs().jac_mul_inplace(&state_y, t, x, y);
        sens.add_column_to_vector(index, y);
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
/// M * ds/dt = J * s + f_p
/// s(0) = dy(0)/dp
/// where
///  M is the mass matrix
///  M_p is the partial derivative of the mass matrix wrt the parameters
///  dy/dt is the derivative of the state wrt time
///  J is the Jacobian of the right-hand side
///  s is the sensitivity
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
    pub(crate) fn new(problem: &'a OdeSolverProblem<Eqn>) -> Self {
        let eqn = &problem.eqn;
        // `n_sens` cannot change after construction, so check the contract once here.
        let (n_sens, nparams) = (eqn.rhs().n_sens(), eqn.rhs().nparams());
        assert!(
            n_sens <= nparams,
            "Op::n_sens() must be <= Op::nparams(), got {n_sens} > {nparams}"
        );
        for sens_col in 0..n_sens {
            let param_index = eqn.rhs().sens_param_index(sens_col);
            assert!(
                param_index < nparams,
                "Op::sens_param_index({sens_col}) must be < Op::nparams(), \
                 got {param_index} >= {nparams}"
            );
        }
        let rtol = problem.sens_rtol;
        let atol = problem.sens_atol.as_ref();
        let rhs = SensRhs::new(eqn, true);
        let init = SensInit::new(eqn, problem.t0);
        Self {
            rhs,
            init,
            eqn,
            rtol,
            atol,
        }
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
        self.eqn.rhs().n_sens()
    }
    fn update_rhs_out_state(&mut self, y: &Eqn::V, dy: &Eqn::V, t: Eqn::T) {
        self.rhs.update_state(y, dy, t);
    }
    fn set_index(&mut self, index: usize) {
        // `index` is a sensitivity column; both ops want a parameter index.
        let param_index = self.eqn.rhs().sens_param_index(index);
        self.rhs.set_param_index(param_index);
        self.init.set_param_index(param_index);
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
        AugmentedOdeEquations, DenseMatrix, MatrixCommon, NalgebraVec, NonLinearOp,
        NonLinearOpJacobian, NonLinearOpSens, OdeEquations, OdeEquationsImplicit,
        OdeEquationsImplicitSens, OdeEquationsRef, OdeSolverProblem, Op, RkState, SensEquations,
        Vector, VectorView,
    };
    type Mcpu = NalgebraMat<f64>;
    type Vcpu = NalgebraVec<f64>;

    #[test]
    fn test_rhs_exponential() {
        // dy/dt = -ay (p = [a])
        let (problem, _soln) = exponential_decay_problem_sens::<Mcpu>(false);
        let mut sens_eqn = SensEquations::new(&problem);
        let state = RkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0], *problem.context()),
            dy: Vcpu::from_vec(vec![1.0, 1.0], *problem.context()),
            g: Vcpu::zeros(0, *problem.context()),
            dg: Vcpu::zeros(0, *problem.context()),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        // S = f_p - M_p * dy/dt
        // f_p = -y (a = 0.1)
        // M_p = 0
        // so S = |-1.0|
        //        |-1.0|
        sens_eqn.update_rhs_out_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 2);
        assert_eq!(sens.get_index(0, 0), -1.0);
        assert_eq!(sens.get_index(1, 0), -1.0);

        // F(s, t)_i = J * s_i + S_i
        // J = |-a 0|
        //     |0 -a|
        // F(s, t)_0 = |-a 0| |1| + |-1.0| = |-1.1|
        //             |0 -a| |2|   |-1.0|   |-1.2|
        sens_eqn.rhs.set_param_index(0);
        let s = Vcpu::from_vec(vec![1.0, 2.0], *problem.context());
        let f = sens_eqn.rhs.call(&s, state.t);
        let f_expect = Vcpu::from_vec(vec![-1.1, -1.2], *problem.context());
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_exponential_algebraic() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem_sens::<Mcpu>();
        let mut sens_eqn = SensEquations::new(&problem);
        let state = RkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0, 1.0], *problem.context()),
            dy: Vcpu::from_vec(vec![1.0, 1.0, 1.0], *problem.context()),
            g: Vcpu::zeros(0, *problem.context()),
            dg: Vcpu::zeros(0, *problem.context()),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };

        // S = f_p - M_p * dy/dt
        // f_p = |-y|
        //       |-y|
        //       | 0|
        // M_p = 0
        // so S = |-0.1|
        //        |-0.1|
        //        | 0 |
        sens_eqn.update_rhs_out_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 3);
        assert_eq!(sens.ncols(), 1);
        assert_eq!(sens.get_index(0, 0), -1.0);
        assert_eq!(sens.get_index(1, 0), -1.0);
        assert_eq!(sens.get_index(2, 0), 0.0);
        sens_eqn.rhs.y.borrow().assert_eq_st(&state.y, 1e-10);

        // F(s, t)_i = J * s_i + S_i
        // J = |-a 0 0|
        //     |0 -a 0|
        //     |0 0 0 |
        // F(s, t)_0 = |-a 0 0| |1| + |-1.0| = |-1.1|
        //             |0 -a 0| |1|   |-1.0|   |-1.1|
        //             |0 0 0 | |1|   | 0  |   | 0 |
        sens_eqn.rhs.set_param_index(0);
        assert_eq!(sens_eqn.rhs.index.borrow().clone(), 0);
        let s = Vcpu::from_vec(vec![1.0, 1.0, 1.0], *problem.context());
        let f = sens_eqn.rhs.call(&s, state.t);
        let f_expect = Vcpu::from_vec(vec![-1.1, -1.1, 0.0], *problem.context());
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_robertson() {
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let mut sens_eqn = SensEquations::new(&problem);
        let state = RkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 2.0, 3.0], *problem.context()),
            dy: Vcpu::from_vec(vec![1.0, 1.0, 1.0], *problem.context()),
            g: Vcpu::zeros(0, *problem.context()),
            dg: Vcpu::zeros(0, *problem.context()),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };

        // S = f_p - M_p * dy/dt
        // f_p = |-x0 x1*x2 0|
        //       |x0 -x1*x2 -x1*x1|
        //       | 0   0    0|
        // M_p = 0
        // so S = f_p
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

    /// Rhs wrapper overriding `n_sens` and the column -> parameter mapping, delegating
    /// everything else, so a test can decouple them from `nparams`.
    struct NSensOverrideRhs<'a, Eqn: OdeEquations> {
        inner: <Eqn as OdeEquationsRef<'a>>::Rhs,
        n_sens: usize,
        sens_params: Option<Vec<usize>>,
    }

    impl<Eqn: OdeEquations> Op for NSensOverrideRhs<'_, Eqn> {
        type T = Eqn::T;
        type V = Eqn::V;
        type M = Eqn::M;
        type C = Eqn::C;
        fn nstates(&self) -> usize {
            self.inner.nstates()
        }
        fn nout(&self) -> usize {
            self.inner.nout()
        }
        fn nparams(&self) -> usize {
            self.inner.nparams()
        }
        fn context(&self) -> &Self::C {
            self.inner.context()
        }
        fn n_sens(&self) -> usize {
            self.n_sens
        }
        fn sens_param_index(&self, sens_col: usize) -> usize {
            match &self.sens_params {
                Some(map) => map[sens_col],
                None => sens_col,
            }
        }
    }

    impl<Eqn: OdeEquations> NonLinearOp for NSensOverrideRhs<'_, Eqn> {
        fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
            self.inner.call_inplace(x, t, y)
        }
    }

    impl<Eqn: OdeEquationsImplicit> NonLinearOpJacobian for NSensOverrideRhs<'_, Eqn> {
        fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
            self.inner.jac_mul_inplace(x, t, v, y)
        }
    }

    impl<Eqn: OdeEquationsImplicitSens> NonLinearOpSens for NSensOverrideRhs<'_, Eqn> {
        fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
            self.inner.sens_mul_inplace(x, t, v, y)
        }
    }

    /// Equations wrapper that swaps in [`NSensOverrideRhs`] for the rhs op.
    struct NSensOverrideEqn<Eqn> {
        inner: Eqn,
        n_sens: usize,
        sens_params: Option<Vec<usize>>,
    }

    impl<Eqn: OdeEquations> Op for NSensOverrideEqn<Eqn> {
        type T = Eqn::T;
        type V = Eqn::V;
        type M = Eqn::M;
        type C = Eqn::C;
        fn nstates(&self) -> usize {
            self.inner.nstates()
        }
        fn nout(&self) -> usize {
            self.inner.nout()
        }
        fn nparams(&self) -> usize {
            self.inner.nparams()
        }
        fn context(&self) -> &Self::C {
            self.inner.context()
        }
    }

    impl<'a, Eqn: OdeEquationsImplicitSens> OdeEquationsRef<'a> for NSensOverrideEqn<Eqn> {
        type Mass = <Eqn as OdeEquationsRef<'a>>::Mass;
        type Rhs = NSensOverrideRhs<'a, Eqn>;
        type Root = <Eqn as OdeEquationsRef<'a>>::Root;
        type Init = <Eqn as OdeEquationsRef<'a>>::Init;
        type Out = <Eqn as OdeEquationsRef<'a>>::Out;
        type Reset = <Eqn as OdeEquationsRef<'a>>::Reset;
    }

    impl<Eqn: OdeEquationsImplicitSens> OdeEquations for NSensOverrideEqn<Eqn> {
        fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
            NSensOverrideRhs {
                inner: self.inner.rhs(),
                n_sens: self.n_sens,
                sens_params: self.sens_params.clone(),
            }
        }
        fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
            self.inner.mass()
        }
        fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
            self.inner.root()
        }
        fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
            self.inner.out()
        }
        fn reset(&self) -> Option<<Self as OdeEquationsRef<'_>>::Reset> {
            self.inner.reset()
        }
        fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
            self.inner.init()
        }
        fn set_params(&mut self, p: &Self::V) {
            self.inner.set_params(p)
        }
        fn get_params(&self, p: &mut Self::V) {
            self.inner.get_params(p)
        }
    }

    /// Rebuild `problem` around an [`NSensOverrideEqn`], keeping every tolerance.
    fn with_n_sens_override<Eqn: OdeEquationsImplicitSens>(
        problem: OdeSolverProblem<Eqn>,
        n_sens: usize,
    ) -> OdeSolverProblem<NSensOverrideEqn<Eqn>> {
        with_sens_override(problem, n_sens, None)
    }

    /// As [`with_n_sens_override`], but also installing a column -> parameter mapping.
    fn with_sens_override<Eqn: OdeEquationsImplicitSens>(
        problem: OdeSolverProblem<Eqn>,
        n_sens: usize,
        sens_params: Option<Vec<usize>>,
    ) -> OdeSolverProblem<NSensOverrideEqn<Eqn>> {
        OdeSolverProblem::new(
            NSensOverrideEqn {
                inner: problem.eqn,
                n_sens,
                sens_params,
            },
            problem.rtol,
            problem.atol,
            problem.sens_rtol,
            problem.sens_atol,
            problem.out_rtol,
            problem.out_atol,
            problem.param_rtol,
            problem.param_atol,
            problem.t0,
            problem.h0,
            problem.integrate_out,
            problem.ic_options,
            problem.ode_options,
        )
        .unwrap()
    }

    #[test]
    fn max_index_respects_n_sens_override() {
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let wrapped_problem = with_n_sens_override(problem, 1);
        let sens_eqn = SensEquations::new(&wrapped_problem);
        assert_eq!(sens_eqn.nparams(), 3);
        assert_eq!(sens_eqn.max_index(), 1);
    }

    /// Checked at construction, so it fires in release too rather than as an opaque
    /// out-of-bounds panic mid-solve.
    #[test]
    #[should_panic(expected = "Op::n_sens() must be <= Op::nparams(), got 5 > 3")]
    fn sens_equations_new_rejects_n_sens_above_nparams() {
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let wrapped_problem = with_n_sens_override(problem, 5);
        let _ = SensEquations::new(&wrapped_problem);
    }

    /// Likewise for an out-of-range column -> parameter mapping.
    #[test]
    #[should_panic(expected = "Op::sens_param_index(0) must be < Op::nparams(), got 7 >= 3")]
    fn sens_equations_new_rejects_out_of_range_sens_param_index() {
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let wrapped_problem = with_sens_override(problem, 1, Some(vec![7]));
        let _ = SensEquations::new(&wrapped_problem);
    }

    const SENS_T_EVAL: [f64; 3] = [1e-4, 1e-2, 1.0];

    /// Solve `robertson_sens`, optionally with a reduced `n_sens`. Returns the state, the
    /// sensitivity columns at [`SENS_T_EVAL`], and the tolerances for comparing solves.
    fn solve_robertson_sens(n_sens_override: Option<usize>) -> (Mcpu, Vec<Mcpu>, Vcpu, f64) {
        solve_robertson_sens_mapped(n_sens_override, None)
    }

    /// As [`solve_robertson_sens`], but able to install a column -> parameter mapping.
    fn solve_robertson_sens_mapped(
        n_sens_override: Option<usize>,
        sens_params: Option<Vec<usize>>,
    ) -> (Mcpu, Vec<Mcpu>, Vcpu, f64) {
        use crate::{NalgebraLU, SensitivitiesOdeSolverMethod};
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let (atol, rtol) = (problem.atol.clone(), problem.rtol);
        let (y, sens) = match n_sens_override {
            None => {
                let (y, sens, _) = problem
                    .bdf_sens::<NalgebraLU<f64>>()
                    .unwrap()
                    .solve_dense_sensitivities(&SENS_T_EVAL)
                    .unwrap();
                (y, sens)
            }
            Some(n_sens) => {
                let wrapped = with_sens_override(problem, n_sens, sens_params);
                let (y, sens, _) = wrapped
                    .bdf_sens::<NalgebraLU<f64>>()
                    .unwrap()
                    .solve_dense_sensitivities(&SENS_T_EVAL)
                    .unwrap();
                (y, sens)
            }
        };
        (y, sens, atol, rtol)
    }

    /// With `n_sens == nparams` the override plumbing must reproduce the unwrapped solve
    /// bit-for-bit, so the default path is provably untouched.
    #[test]
    fn solve_dense_sensitivities_bit_identical_when_n_sens_equals_nparams() {
        let (y_full, sens_full, _, _) = solve_robertson_sens(None);
        let (y_wrapped, sens_wrapped, _, _) = solve_robertson_sens(Some(3));

        assert_eq!(sens_full.len(), 3, "robertson_sens has 3 parameters");
        assert_eq!(y_full, y_wrapped, "state trajectory must be bit-identical");
        assert_eq!(
            sens_full, sens_wrapped,
            "sens columns must be bit-identical"
        );
    }

    /// A non-identity mapping must integrate the selected parameter, not the one sharing
    /// the column index. Robertson's parameters span orders of magnitude (0.04, 1e4, 3e7),
    /// so a mix-up is unmissable.
    #[test]
    fn solve_dense_sensitivities_respects_sens_param_index_mapping() {
        let (_, sens_full, atol, rtol) = solve_robertson_sens(None);
        let (_, sens_mapped, _, _) = solve_robertson_sens_mapped(Some(1), Some(vec![2]));

        assert_eq!(sens_mapped.len(), 1, "one column was selected");
        for j in 0..SENS_T_EVAL.len() {
            sens_mapped[0].column(j).into_owned().assert_eq_norm(
                &sens_full[2].column(j).into_owned(),
                &atol,
                rtol,
                10.0,
            );
        }
    }

    /// A reduced `n_sens` must integrate fewer columns while the leading column still
    /// tracks the full solve. Only to solver tolerance, not bit-exact: fewer columns shifts
    /// the step sequence, so both solves are valid to the requested accuracy.
    #[test]
    fn solve_dense_sensitivities_respects_n_sens_override() {
        let (_, sens_full, atol, rtol) = solve_robertson_sens(None);
        let (_, sens_reduced, _, _) = solve_robertson_sens(Some(1));

        assert_eq!(
            sens_reduced.len(),
            1,
            "only the leading sensitivity column should be integrated"
        );
        assert_eq!(sens_reduced[0].ncols(), SENS_T_EVAL.len());
        assert_eq!(sens_reduced[0].nrows(), sens_full[0].nrows());

        // Compare under the solver's own weighted norm, as other solve tests do.
        for j in 0..SENS_T_EVAL.len() {
            sens_reduced[0].column(j).into_owned().assert_eq_norm(
                &sens_full[0].column(j).into_owned(),
                &atol,
                rtol,
                10.0,
            );
        }
    }

    #[test]
    fn max_index_defaults_to_nparams() {
        let (problem, _soln) = robertson_sens::<Mcpu>();
        let sens_eqn = SensEquations::new(&problem);
        assert_eq!(sens_eqn.max_index(), sens_eqn.nparams());
    }

    /// `dx/dt = -p[0]·x`, `x(0) = 1`, with a reset `x -> x + p[1]` firing at `x = 0.5`.
    /// Parameter 1 enters the problem only through the reset, so its post-event
    /// sensitivity comes entirely from the parameter-space seed in the reset correction:
    /// s_1(t) = exp(-p[0]·(t - t_root)) for t > t_root = ln(2)/p[0], and 0 before.
    fn decay_with_param_dependent_reset_problem() -> OdeSolverProblem<
        impl OdeEquationsImplicitSens<M = Mcpu, V = Vcpu, T = f64, C = crate::NalgebraContext>,
    > {
        use crate::OdeBuilder;
        OdeBuilder::<Mcpu>::new()
            .p([1.0, 0.3])
            .rtol(1e-6)
            .atol([1e-6])
            .sens_rtol(1e-6)
            .sens_atol([1e-6])
            .rhs_sens_implicit(
                |x: &Vcpu, p: &Vcpu, _t, y: &mut Vcpu| y[0] = -p[0] * x[0],
                |_x: &Vcpu, p: &Vcpu, _t, v: &Vcpu, y: &mut Vcpu| y[0] = -p[0] * v[0],
                |x: &Vcpu, _p: &Vcpu, _t, v: &Vcpu, y: &mut Vcpu| y[0] = -x[0] * v[0],
            )
            .init_sens(
                |_p: &Vcpu, _t, y: &mut Vcpu| y[0] = 1.0,
                |_p: &Vcpu, _t, _v: &Vcpu, y: &mut Vcpu| y[0] = 0.0,
                1,
            )
            .root_sens_implicit(
                |x: &Vcpu, _p: &Vcpu, _t, y: &mut Vcpu| y[0] = x[0] - 0.5,
                |_x: &Vcpu, _p: &Vcpu, _t, v: &Vcpu, y: &mut Vcpu| y[0] = v[0],
                |_x: &Vcpu, _p: &Vcpu, _t, _v: &Vcpu, y: &mut Vcpu| y[0] = 0.0,
                1,
            )
            .reset_sens_implicit(
                |x: &Vcpu, p: &Vcpu, _t, y: &mut Vcpu| y[0] = x[0] + p[1],
                |_x: &Vcpu, _p: &Vcpu, _t, v: &Vcpu, y: &mut Vcpu| y[0] = v[0],
                // dreset/dp · v = v[1]: reads the parameter-space seed at parameter 1.
                |_x: &Vcpu, _p: &Vcpu, _t, v: &Vcpu, y: &mut Vcpu| y[0] = v[1],
            )
            .build()
            .unwrap()
    }

    const RESET_SENS_T_EVAL: [f64; 3] = [0.2, 1.0, 1.5];

    /// Solve [`decay_with_param_dependent_reset_problem`], optionally with only
    /// parameter 1's sensitivity integrated (column 0 -> parameter 1).
    fn solve_decay_reset_sens(use_bdf: bool, mapped: bool) -> (Vec<Mcpu>, Vcpu, f64) {
        use crate::{NalgebraLU, SensitivitiesOdeSolverMethod};
        let problem = decay_with_param_dependent_reset_problem();
        let (atol, rtol) = (problem.atol.clone(), problem.rtol);
        let sens = match (mapped, use_bdf) {
            (false, false) => {
                problem
                    .tsit45_sens()
                    .unwrap()
                    .solve_dense_sensitivities(&RESET_SENS_T_EVAL)
                    .unwrap()
                    .1
            }
            (false, true) => {
                problem
                    .bdf_sens::<NalgebraLU<f64>>()
                    .unwrap()
                    .solve_dense_sensitivities(&RESET_SENS_T_EVAL)
                    .unwrap()
                    .1
            }
            (true, use_bdf) => {
                let wrapped = with_sens_override(problem, 1, Some(vec![1]));
                if use_bdf {
                    wrapped
                        .bdf_sens::<NalgebraLU<f64>>()
                        .unwrap()
                        .solve_dense_sensitivities(&RESET_SENS_T_EVAL)
                        .unwrap()
                        .1
                } else {
                    wrapped
                        .tsit45_sens()
                        .unwrap()
                        .solve_dense_sensitivities(&RESET_SENS_T_EVAL)
                        .unwrap()
                        .1
                }
            }
        };
        (sens, atol, rtol)
    }

    fn assert_reset_sens_correction_respects_mapping(use_bdf: bool) {
        let (sens_full, atol, rtol) = solve_decay_reset_sens(use_bdf, false);
        let (sens_mapped, _, _) = solve_decay_reset_sens(use_bdf, true);

        // Guard: parameter 1's sensitivity is nonzero after the event (s_1 = 2/e at
        // t = 1), so the comparison below cannot pass vacuously.
        let expected = 2.0 / std::f64::consts::E;
        assert!(
            (sens_full[1].get_index(0, 1) - expected).abs() < 1e-3,
            "full solve s_1(1.0) should be ~{expected}, got {}",
            sens_full[1].get_index(0, 1)
        );

        assert_eq!(sens_mapped.len(), 1, "one column was selected");
        for j in 0..RESET_SENS_T_EVAL.len() {
            sens_mapped[0].column(j).into_owned().assert_eq_norm(
                &sens_full[1].column(j).into_owned(),
                &atol,
                rtol,
                100.0,
            );
        }
    }

    /// The reset correction seeds parameter space per column; it must seed
    /// `sens_param_index(j)`, not column `j`. Tsit45 routes through
    /// `apply_reset_with_sens`.
    #[test]
    fn reset_sens_correction_respects_mapping_tsit45() {
        assert_reset_sens_correction_respects_mapping(false);
    }

    /// As above through BDF, which routes through `apply_reset_with_sens_mass`.
    #[test]
    fn reset_sens_correction_respects_mapping_bdf() {
        assert_reset_sens_correction_respects_mapping(true);
    }
}
