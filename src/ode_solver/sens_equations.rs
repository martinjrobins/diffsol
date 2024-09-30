use num_traits::{One, Zero};
use std::{cell::RefCell, rc::Rc};

use crate::{
    matrix::sparsity::MatrixSparsityRef, AugmentedOdeEquations, ConstantOp, LinearOp, Matrix, MatrixSparsity, NonLinearOp, OdeEquations, Op, Vector
};

pub struct SensInit<Eqn>
where
    Eqn: OdeEquations,
{
    eqn: Rc<Eqn>,
    init_sens: Eqn::M,
    index: usize,
}

impl<Eqn> SensInit<Eqn>
where
    Eqn: OdeEquations,
{
    pub fn new(eqn: &Rc<Eqn>) -> Self {
        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let init_sens = Eqn::M::new_from_sparsity(
            nstates,
            nparams,
            eqn.init().sparsity_sens().map(|s| s.to_owned()),
        );
        let index = 0;
        Self {
            eqn: eqn.clone(),
            init_sens,
            index,
        }
    }
    pub fn update_state(&mut self, t: Eqn::T) {
        self.eqn.init().sens_inplace(t, &mut self.init_sens);
    }
    pub fn set_param_index(&mut self, index: usize) {
        self.index = index;
    }
}

impl<Eqn> Op for SensInit<Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> ConstantOp for SensInit<Eqn>
where
    Eqn: OdeEquations,
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.fill(Eqn::T::zero());
        self.init_sens.add_column_to_vector(self.index, y);
    }
}

/// Right-hand side of the sensitivity equations is:
///
/// F(s, t) = J * s + f_p - M_p * dy/dt
///
/// f_p is the partial derivative of the right-hand side with respect to the parameters,
/// this is constant and can be precomputed. It is a matrix of size nstates x nparams.
///
/// M_p * dy/dt is the partial derivative of the mass matrix wrt the parameters,
/// multiplied by the derivative of the state wrt time. It is a matrix of size nstates x nparams.
///
/// Strategy is to pre-compute S = f_p - M_p * dy/dt from the state at given time step and store it in a matrix using [Self::update_state].
/// Then the ith column of function F(s, t) is evaluated as J * s_i + S_i, where s_i is the ith column of the sensitivity matrix
/// and S_i is the ith column of the matrix S. The column to evaluate is set using [Self::set_param_index].
pub struct SensRhs<Eqn>
where
    Eqn: OdeEquations,
{
    eqn: Rc<Eqn>,
    sens: RefCell<Eqn::M>,
    rhs_sens: Option<RefCell<Eqn::M>>,
    mass_sens: Option<RefCell<Eqn::M>>,
    y: RefCell<Eqn::V>,
    index: RefCell<usize>,
}

impl<Eqn> SensRhs<Eqn>
where
    Eqn: OdeEquations,
{
    pub fn new(eqn: &Rc<Eqn>, allocate: bool) -> Self {
        if !allocate {
            return Self {
                eqn: eqn.clone(),
                sens: RefCell::new(<Eqn::M as Matrix>::zeros(0, 0)),
                rhs_sens: None,
                mass_sens: None,
                y: RefCell::new(<Eqn::V as Vector>::zeros(0)),
                index: RefCell::new(0),
            };
        }
        let nstates = eqn.rhs().nstates();
        let nparams = eqn.rhs().nparams();
        let rhs_sens = Eqn::M::new_from_sparsity(
            nstates,
            nparams,
            eqn.rhs().sparsity_sens().map(|s| s.to_owned()),
        );
        let y = RefCell::new(<Eqn::V as Vector>::zeros(nstates));
        let index = RefCell::new(0);
        if let Some(mass) = eqn.mass() {
            let mass_sens = Eqn::M::new_from_sparsity(
                nstates,
                nparams,
                mass.sparsity_sens().map(|s| s.to_owned()),
            );
            let sens = if rhs_sens.sparsity().is_some() && mass_sens.sparsity().is_some() {
                // union of sparsity patterns
                let sparsity = rhs_sens
                    .sparsity()
                    .unwrap()
                    .to_owned()
                    .union(mass_sens.sparsity().unwrap())
                    .unwrap();
                Eqn::M::new_from_sparsity(nstates, nparams, Some(sparsity))
            } else {
                Eqn::M::new_from_sparsity(nstates, nparams, None)
            };
            Self {
                eqn: eqn.clone(),
                sens: RefCell::new(sens),
                rhs_sens: Some(RefCell::new(rhs_sens)),
                mass_sens: Some(RefCell::new(mass_sens)),
                y,
                index,
            }
        } else {
            Self {
                eqn: eqn.clone(),
                sens: RefCell::new(rhs_sens),
                rhs_sens: None,
                mass_sens: None,
                y,
                index,
            }
        }
    }

    /// pre-compute S = f_p - M_p * dy/dt from the state
    pub fn update_state(&mut self, y: &Eqn::V, dy: &Eqn::V, t: Eqn::T) {
        if self.rhs_sens.is_some() {
            let mut rhs_sens = self.rhs_sens.as_ref().unwrap().borrow_mut();
            let mut mass_sens = self.mass_sens.as_ref().unwrap().borrow_mut();
            let mut sens = self.sens.borrow_mut();
            self.eqn.rhs().sens_inplace(y, t, &mut rhs_sens);
            self.eqn.mass().unwrap().sens_inplace(dy, t, &mut mass_sens);
            sens.scale_add_and_assign(&rhs_sens, -Eqn::T::one(), &mass_sens);
        } else {
            let mut sens = self.sens.borrow_mut();
            self.eqn.rhs().sens_inplace(y, t, &mut sens);
        }
        let mut state_y = self.y.borrow_mut();
        state_y.copy_from(y);
    }
    pub fn set_param_index(&self, index: usize) {
        self.index.replace(index);
    }
}

impl<Eqn> Op for SensRhs<Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> NonLinearOp for SensRhs<Eqn>
where
    Eqn: OdeEquations,
{
    /// the ith column of function F(s, t) is evaluated as J * s_i + S_i, where s_i is the ith column of the sensitivity matrix
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let state_y = self.y.borrow();
        let sens = self.sens.borrow();
        let index = *self.index.borrow();
        self.eqn.rhs().jac_mul_inplace(&state_y, t, x, y);
        sens.add_column_to_vector(index, y);
    }
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let state_y = self.y.borrow();
        self.eqn.rhs().jac_mul_inplace(&state_y, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        let state_y = self.y.borrow();
        self.eqn.rhs().jacobian_inplace(&state_y, t, y);
    }
}
/// Sensitivity & adjoint equations for ODEs
///
/// Sensitivity equations are linear:
/// M * ds/dt = J * s + f_p - M_p * dy/dt
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
pub struct SensEquations<Eqn>
where
    Eqn: OdeEquations
{
    eqn: Rc<Eqn>,
    rhs: Rc<SensRhs<Eqn>>,
    init: Rc<SensInit<Eqn>>,
    include_in_error_control: bool,
}

impl<Eqn> std::fmt::Debug for SensEquations<Eqn>
where
    Eqn: OdeEquations,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SensEquations")
    }
}

impl<Eqn> SensEquations<Eqn>
where
    Eqn: OdeEquations,
{
    pub(crate) fn new(eqn: &Rc<Eqn>) -> Self {
        let rhs = Rc::new(SensRhs::new(eqn, true));
        let init = Rc::new(SensInit::new(eqn));
        Self {
            rhs,
            init,
            eqn: eqn.clone(),
            include_in_error_control: false,
        }
    }
}

impl<Eqn> Op for SensEquations<Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nout()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> OdeEquations for SensEquations<Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type Rhs = SensRhs<Eqn>;
    type Mass = Eqn::Mass;
    type Root = Eqn::Root;
    type Init = SensInit<Eqn>;
    type Out = Eqn::Out;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn mass(&self) -> Option<&Rc<Self::Mass>> {
        self.eqn.mass()
    }
    fn root(&self) -> Option<&Rc<Self::Root>> {
        self.eqn.root()
    }
    fn init(&self) -> &Rc<Self::Init> {
        &self.init
    }
    fn set_params(&mut self, _p: Self::V) {
        panic!("Not implemented for SensEquations");
    }
    fn out(&self) -> Option<&Rc<Self::Out>> {
        self.eqn.out()
    }
}

impl<Eqn: OdeEquations> AugmentedOdeEquations<Eqn> for SensEquations<Eqn>  {
    fn include_in_error_control(&self) -> bool {
        self.include_in_error_control
    }
    fn set_include_in_error_control(&mut self, include: bool) {
        self.include_in_error_control = include;
    }
    fn max_index(&self) -> usize {
        self.nparams()
    }
    fn update_rhs_state(&mut self, y: &Eqn::V, dy: &Eqn::V, t: Eqn::T) {
        Rc::get_mut(&mut self.rhs).unwrap().update_state(y, dy, t);
    }
    fn update_init_state(&mut self, t: Eqn::T) {
        Rc::get_mut(&mut self.init).unwrap().update_state(t);
    }
    fn set_index(&mut self, index: usize) {
        Rc::get_mut(&mut self.rhs).unwrap().set_param_index(index);
        Rc::get_mut(&mut self.init).unwrap().set_param_index(index);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ode_solver::test_models::{
            exponential_decay::exponential_decay_problem_sens,
            exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem_sens,
            robertson_sens::robertson_sens,
        },
        AugmentedOdeEquations,
        NonLinearOp, SdirkState, SensEquations, Vector,
    };
    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn test_rhs_exponential() {
        // dy/dt = -ay (p = [a])
        let (problem, _soln) = exponential_decay_problem_sens::<Mcpu>(false);
        let mut sens_eqn = SensEquations::new(&problem.eqn);
        let state = SdirkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0]),
            dy: Vcpu::from_vec(vec![1.0, 1.0]),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        // S = f_p - M_p * dy/dt
        // f_p = -y (a = 0.1)
        // M_p = 0
        // so S = |-1.0|
        //        |-1.0|
        sens_eqn.update_rhs_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 2);
        assert_eq!(sens[(0, 0)], -1.0);
        assert_eq!(sens[(1, 0)], -1.0);

        // F(s, t)_i = J * s_i + S_i
        // J = |-a 0|
        //     |0 -a|
        // F(s, t)_0 = |-a 0| |1| + |-1.0| = |-1.1|
        //             |0 -a| |2|   |-1.0|   |-1.2|
        sens_eqn.rhs.set_param_index(0);
        let s = Vcpu::from_vec(vec![1.0, 2.0]);
        let f = sens_eqn.rhs.call(&s, state.t);
        let f_expect = Vcpu::from_vec(vec![-1.1, -1.2]);
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_exponential_algebraic() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem_sens::<Mcpu>(false);
        let mut sens_eqn = SensEquations::new(&problem.eqn);
        let state = SdirkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0, 1.0]),
            dy: Vcpu::from_vec(vec![1.0, 1.0, 1.0]),
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
        sens_eqn.update_rhs_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 3);
        assert_eq!(sens.ncols(), 1);
        assert_eq!(sens[(0, 0)], -1.0);
        assert_eq!(sens[(1, 0)], -1.0);
        assert_eq!(sens[(2, 0)], 0.0);
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
        let s = Vcpu::from_vec(vec![1.0, 1.0, 1.0]);
        let f = sens_eqn.rhs.call(&s, state.t);
        let f_expect = Vcpu::from_vec(vec![-1.1, -1.1, 0.0]);
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_robertson() {
        let (problem, _soln) = robertson_sens::<Mcpu>(false);
        let mut sens_eqn = SensEquations::new(&problem.eqn);
        let state = SdirkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 2.0, 3.0]),
            dy: Vcpu::from_vec(vec![1.0, 1.0, 1.0]),
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
        sens_eqn.update_rhs_state(&state.y, &state.dy, state.t);
        let sens = sens_eqn.rhs.sens.borrow();
        assert_eq!(sens.nrows(), 3);
        assert_eq!(sens.ncols(), 3);
        assert_eq!(sens[(0, 0)], -state.y[0]);
        assert_eq!(sens[(0, 1)], state.y[1] * state.y[2]);
        assert_eq!(sens[(0, 2)], 0.0);
        assert_eq!(sens[(1, 0)], state.y[0]);
        assert_eq!(sens[(1, 1)], -state.y[1] * state.y[2]);
        assert_eq!(sens[(1, 2)], -state.y[1] * state.y[1]);
        assert_eq!(sens[(2, 0)], 0.0);
        assert_eq!(sens[(2, 1)], 0.0);
        assert_eq!(sens[(2, 2)], 0.0);
    }
}
