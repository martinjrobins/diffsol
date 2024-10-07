use num_traits::{Zero, One};
use std::{cell::RefCell, rc::Rc, ops::AddAssign, ops::SubAssign};



use crate::{
    Checkpointing, ConstantOp, NonLinearOp, OdeEquations, OdeSolverMethod, Op, Vector, Matrix, AugmentedOdeEquations, LinearOp
};

pub struct AdjointContext<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    checkpointer: Checkpointing<Eqn, Method>,
    x: Eqn::V,
    index: usize,
    last_t: Option<Eqn::T>,
    col: Eqn::V,
}

impl <Eqn, Method> AdjointContext<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub fn new(checkpointer: Checkpointing<Eqn, Method>) -> Self {
        let x = <Eqn::V as Vector>::zeros(checkpointer.problem.eqn.rhs().nstates());
        let mut col = <Eqn::V as Vector>::zeros(checkpointer.problem.eqn.rhs().nout());
        let index = 0;
        col[0] = Eqn::T::one();
        Self {
            checkpointer,
            x,
            index,
            col,
            last_t: None,
        }
    }

    pub fn set_state(&mut self, t: Eqn::T) {
        if let Some(last_t) = self.last_t {
            if last_t == t {
                return;
            }
        }
        self.last_t = Some(t);
        self.checkpointer.interpolate(t, &mut self.x).unwrap();
    }

    pub fn state(&self) -> &Eqn::V {
        &self.x
    }

    pub fn col(&self) -> &Eqn::V {
        &self.col
    }

    pub fn set_index(&mut self, index: usize) {
        self.col[self.index] = Eqn::T::zero();
        self.index = index;
        self.col[self.index] = Eqn::T::one();
    }
}

pub struct AdjointInit<Eqn>
where
    Eqn: OdeEquations,
{
    eqn: Rc<Eqn>,
}

impl<Eqn> AdjointInit<Eqn>
where
    Eqn: OdeEquations,
{
    pub fn new(eqn: &Rc<Eqn>) -> Self {
        Self {
            eqn: eqn.clone(),
        }
    }
}

impl<Eqn> Op for AdjointInit<Eqn>
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

impl<Eqn> ConstantOp for AdjointInit<Eqn>
where
    Eqn: OdeEquations,
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
pub struct AdjointRhs<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    eqn: Rc<Eqn>,
    context: Rc<RefCell<AdjointContext<Eqn, Method>>>,
    tmp: RefCell<Eqn::V>,
    with_out: bool,
}

impl<Eqn, Method> AdjointRhs<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub fn new(eqn: &Rc<Eqn>, context: Rc<RefCell<AdjointContext<Eqn, Method>>>, with_out: bool) -> Self {
        let tmp_n = if with_out { eqn.rhs().nstates() } else { 0 };
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(tmp_n));
        Self {
            eqn: eqn.clone(),
            context,
            tmp,
            with_out,
        }
    }
}

impl<Eqn, Method> Op for AdjointRhs<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
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
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.eqn.rhs().sparsity_adjoint()
    }
}

impl<Eqn, Method> NonLinearOp for AdjointRhs<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    /// F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
    fn call_inplace(&self, lambda: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();

        // y = -f^T_x(x, t) λ
        self.eqn.rhs().jac_transpose_mul_inplace(x, t, lambda, y);

        // y = -f^T_x(x, t) λ - g^T_x(x,t)
        if self.with_out {
            let col = context.col();
            let mut tmp = self.tmp.borrow_mut();
            self.eqn.out().unwrap().jac_transpose_mul_inplace(x, t, col, &mut tmp);
            y.add_assign(&*tmp);
        }
    }
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
}

/// Output of the adjoint equations is:
/// 
/// F(λ, x, t) = -g_p^T(x, t) - f_p^T(x, t) λ 
/// 
/// f_p is the partial derivative of the right-hand side with respect to the parameter vector
/// g_p is the partial derivative of the functional g with respect to the parameter vector
/// 
/// We need the current state x(t), which is obtained from the checkpointed forward solve at the current time step.
pub struct AdjointOut<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    eqn: Rc<Eqn>,
    context: Rc<RefCell<AdjointContext<Eqn, Method>>>,
    tmp: RefCell<Eqn::V>,
    with_out: bool,
}

impl<Eqn, Method> AdjointOut<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub fn new(eqn: &Rc<Eqn>, context: Rc<RefCell<AdjointContext<Eqn, Method>>>, with_out: bool) -> Self {
        let tmp_n = if with_out { eqn.rhs().nparams() } else { 0 };
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(tmp_n));
        Self {
            eqn: eqn.clone(),
            context,
            tmp,
            with_out,
        }
    }
}

impl<Eqn, Method> Op for AdjointOut<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.eqn.rhs().sparsity_sens_adjoint()
    }
}

impl<Eqn, Method> NonLinearOp for AdjointOut<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    /// F(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t) 
    fn call_inplace(&self, lambda: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_transpose_mul_inplace(x, t, lambda, y);

        if self.with_out {
            let col = context.col();
            let mut tmp = self.tmp.borrow_mut();
            self.eqn.out().unwrap().sens_transpose_mul_inplace(x, t, col, &mut tmp);
            y.add_assign(&*tmp);
        }
    }
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
}


/// Adjoint equations for ODEs
/// 
/// M * dλ/dt = -f^T_x(x, t) λ - g^T_x(x,t)
/// λ(T) = 0
/// g(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t) 
///
pub struct AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    eqn: Rc<Eqn>,
    rhs: Rc<AdjointRhs<Eqn, Method>>,
    out: Option<Rc<AdjointOut<Eqn, Method>>>,
    context: Rc<RefCell<AdjointContext<Eqn, Method>>>,
    tmp: RefCell<Eqn::V>,
    tmp2: RefCell<Eqn::V>,
    init: Rc<AdjointInit<Eqn>>,
    include_in_error_control: bool,
}

impl<Eqn, Method> AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub(crate) fn new(eqn: &Rc<Eqn>, context: Rc<RefCell<AdjointContext<Eqn, Method>>>, with_out: bool) -> Self {
        let rhs = Rc::new(AdjointRhs::new(eqn, context.clone(), with_out));
        let init = Rc::new(AdjointInit::new(eqn));
        let out = if with_out {
            Some(Rc::new(AdjointOut::new(eqn, context.clone(), with_out)))
        } else {
            None
        };
        let tmp = if with_out {
            RefCell::new(<Eqn::V as Vector>::zeros(0))
        } else {
            RefCell::new(<Eqn::V as Vector>::zeros(eqn.rhs().nparams()))
        };
        let tmp2 = if with_out {
            RefCell::new(<Eqn::V as Vector>::zeros(0))
        } else {
            RefCell::new(<Eqn::V as Vector>::zeros(eqn.rhs().nparams()))
        };
        Self {
            rhs,
            init,
            context,
            out,
            tmp,
            tmp2,
            eqn: eqn.clone(),
            include_in_error_control: false,
        }
    }

    pub fn correct_sg_for_init(&self, t: Eqn::T, s: &[Eqn::V], sg: &mut [Eqn::V]) {
        let mut tmp = self.tmp.borrow_mut();
        for (s_i, sg_i) in s.iter().zip(sg.iter_mut()) {
            if let Some(mass) = self.eqn.mass() {
                let mut tmp2 = self.tmp2.borrow_mut();
                mass.call_transpose_inplace(s_i, t, &mut tmp2);
                self.eqn.init().sens_mul_transpose_inplace(t, &tmp2, &mut tmp);
                sg_i.add_assign(&*tmp);

            } else {
                self.eqn.init().sens_mul_transpose_inplace(t, s_i, &mut tmp);
                sg_i.sub_assign(&*tmp);
            }
        }
    }
    
}

impl<Eqn, Method> std::fmt::Debug for AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdjointEquations").finish()
    }
}

impl<Eqn, Method> Op for AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
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

impl<Eqn, Method> OdeEquations for AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;
    type Rhs = AdjointRhs<Eqn, Method>;
    type Mass = Eqn::Mass;
    type Root = Eqn::Root;
    type Init = AdjointInit<Eqn>;
    type Out = AdjointOut<Eqn, Method>;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn mass(&self) -> Option<&Rc<Self::Mass>> {
        self.eqn.mass()
    }
    fn root(&self) -> Option<&Rc<Self::Root>> {
        None
    }
    fn init(&self) -> &Rc<Self::Init> {
        &self.init
    }
    fn set_params(&mut self, _p: Self::V) {
        panic!("Not implemented for SensEquations");
    }
    fn out(&self) -> Option<&Rc<Self::Out>> {
        self.out.as_ref()
    }
}


impl<Eqn, Method> AugmentedOdeEquations<AdjointEquations<Eqn, Method>> for AdjointEquations<Eqn, Method> 
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    
    fn max_index(&self) -> usize {
        self.eqn.out().map(|o| o.nout()).unwrap_or(0)
    }
    
    fn set_index(&mut self, index: usize) {
        self.context.borrow_mut().set_index(index);
    }

    fn update_rhs_out_state(&mut self, _y: &Eqn::V, _dy: &Eqn::V, _t: Eqn::T) {
    }

    
    
    fn include_in_error_control(&self) -> bool {
        self.include_in_error_control
    }
    
    fn set_include_in_error_control(&mut self, include: bool) {
        self.include_in_error_control = include;
    }
    
    fn update_init_state(&mut self, _t: <Eqn as OdeEquations>::T) {
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        ode_solver::{adjoint_equations::AdjointEquations, test_models::
            exponential_decay::exponential_decay_problem_adjoint
        }, AdjointContext, AugmentedOdeEquations, Checkpointing, FaerSparseLU, Matrix, MatrixCommon, NalgebraLU, NonLinearOp, Sdirk, SdirkState, SparseColMat, Tableau, Vector
    };
    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn test_rhs_exponential() {
        // dy/dt = -ay (p = [a])
        // a = 0.1
        let (problem, _soln) = exponential_decay_problem_adjoint::<Mcpu>();
        let solver = Sdirk::<Mcpu, _, _>::new(Tableau::esdirk34(), NalgebraLU::default());
        let state = SdirkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0]),
            dy: Vcpu::from_vec(vec![1.0, 1.0]),
            g: Vcpu::zeros(0),
            dg: Vcpu::zeros(0),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let checkpointer = Checkpointing::new(&problem, solver, 0, vec![state.clone(), state.clone()]);
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer)));
        let adj_eqn = AdjointEquations::new(&problem.eqn, context.clone(), false);
        // F(λ, x, t) = -f^T_x(x, t) λ
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| = |a| = |0.1|
        //              |0 a| |2|   |2a| = |0.2|
        let v = Vcpu::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![0.1, 0.2]);
        f.assert_eq_st(&f_expect, 1e-10);

        let mut adj_eqn = AdjointEquations::new(&problem.eqn, context.clone(), true);

        // f_x^T = |-a 0|
        //         |0 -a|
        // J = -f_x^T
        let adjoint = adj_eqn.rhs.jacobian(&state.y, state.t);
        assert_eq!(adjoint.nrows(), 2);
        assert_eq!(adjoint.ncols(), 2);
        assert_eq!(adjoint[(0, 0)], 0.1);
        assert_eq!(adjoint[(1, 1)], 0.1);
        
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
        adj_eqn.set_index(0);
        let out = adj_eqn.out.unwrap().call(&v, state.t);
        let out_expect = Vcpu::from_vec(vec![3.0, 0.0]);
        out.assert_eq_st(&out_expect, 1e-10);

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| - |1.0| = | a - 1| = |-0.9|
        //              |0 a| |2|   |2.0|   |2a - 2| = |-1.8|
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);

        
    }

    #[test]
    fn test_rhs_exponential_sparse() {
        // dy/dt = -ay (p = [a])
        // a = 0.1
        let (problem, _soln) = exponential_decay_problem_adjoint::<SparseColMat<f64>>();
        let solver = Sdirk::<faer::Mat<f64>, _, _>::new(Tableau::esdirk34(), FaerSparseLU::default());
        let state = SdirkState {
            t: 0.0,
            y: faer::Col::from_vec(vec![1.0, 1.0]),
            dy: faer::Col::from_vec(vec![1.0, 1.0]),
            g: faer::Col::zeros(0),
            dg: faer::Col::zeros(0),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let checkpointer = Checkpointing::new(&problem, solver, 0, vec![state.clone(), state.clone()]);
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer)));
        let mut adj_eqn = AdjointEquations::new(&problem.eqn, context, true);

        // f_x^T = |-a 0|
        //         |0 -a|
        // J = -f_x^T
        let adjoint = adj_eqn.rhs.jacobian(&state.y, state.t);
        assert_eq!(adjoint.nrows(), 2);
        assert_eq!(adjoint.ncols(), 2);
        for (i, j, v) in adjoint.triplet_iter() {
            if i == j {
                assert_eq!(*v, 0.1);
            } else {
                assert_eq!(*v, 0.0);
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
        adj_eqn.set_index(0);
        let v = faer::Col::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = faer::Col::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);
    }
}
