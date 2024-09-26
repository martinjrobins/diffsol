use num_traits::Zero;
use std::{cell::RefCell, rc::Rc};


use crate::{
    matrix::sparsity::MatrixSparsityRef, Checkpointing, ConstantOp, NonLinearOp, OdeEquations, OdeSolverMethod, Op, Vector, Matrix
};

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
    checkpointer: RefCell<Checkpointing<Eqn, Method>>,
    g_x: RefCell<Eqn::M>,
    x: RefCell<Eqn::V>,
    index: RefCell<Option<usize>>,
}

impl<Eqn, Method> AdjointRhs<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub fn new(eqn: &Rc<Eqn>, checkpointer: Checkpointing<Eqn, Method>) -> Self {
        let g_x = if let Some(g) = eqn.out() {
            let g_x_sparsity = g.sparsity_adjoint();
            Eqn::M::new_from_sparsity(
                g.nout(),
                g.nstates(),
                g_x_sparsity.map(|s| s.to_owned()),
            )
        } else {
            Eqn::M::zeros(0, 0)
        };
        let x = <Eqn::V as Vector>::zeros(eqn.rhs().nstates());
        let index = None;
        Self {
            eqn: eqn.clone(),
            g_x: RefCell::new(g_x),
            checkpointer: RefCell::new(checkpointer),
            x: RefCell::new(x),
            index: RefCell::new(index),
        }
    }

    /// precompute S = g^T_x(x,t) and the state x(t) from t
    pub fn update_state(&self, t: Eqn::T) {
        let mut checkpointer = self.checkpointer.borrow_mut();
        let check_x = checkpointer.interpolate(t).unwrap();

        // update -g_x^T
        if let Some(g) = self.eqn.out() {
            let mut g_x = self.g_x.borrow_mut();
            g.adjoint_inplace(&check_x, t, &mut g_x);
        }

        let mut x = self.x.borrow_mut();
        x.copy_from(&check_x);
    }
    pub fn set_param_index(&self, index: Option<usize>) {
        if index.is_some() && self.eqn.out().is_none() {
            panic!("Cannot set parameter index for problem without output");
        }
        self.index.replace(index);
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
        // y = -f^T_x(x, t) λ
        let x_ref= self.x.borrow();
        self.eqn.rhs().jac_transpose_mul_inplace(&x_ref, t, lambda, y);

        // y = -f^T_x(x, t) λ - g^T_x(x,t)
        let g_x_ref = self.g_x.borrow();
        let index_ref = self.index.borrow();
        if let Some(index_ref) = *index_ref {
            g_x_ref.add_column_to_vector(index_ref, y);
        }
    }
    // J = -f^T_x(x, t)
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let x_ref = self.x.borrow();
        self.eqn.rhs().jac_transpose_mul_inplace(&x_ref, t, v, y);
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.eqn.rhs().adjoint_inplace(x, t, y);
    }
}
/// Sensitivity equations for ODEs
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
pub struct AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    eqn: Rc<Eqn>,
    rhs: Rc<AdjointRhs<Eqn, Method>>,
    init: Rc<AdjointInit<Eqn>>,
}

impl<Eqn, Method> AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub fn new(eqn: &Rc<Eqn>, checkpointer: Checkpointing<Eqn, Method>) -> Self {
        let rhs = Rc::new(AdjointRhs::new(eqn, checkpointer));
        let init = Rc::new(AdjointInit::new(eqn));
        Self {
            rhs,
            init,
            eqn: eqn.clone(),
        }
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
    type Out = Eqn::Out;

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
        self.eqn.out()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ode_solver::{adjoint_equations::AdjointEquations, test_models::
            exponential_decay::exponential_decay_problem_adjoint
        }, Checkpointing, FaerSparseLU, NalgebraLU, NonLinearOp, Sdirk, SdirkState, SparseColMat, Tableau, Vector,
        MatrixCommon, Matrix
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
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let checkpointer = Checkpointing::new(&problem, solver, vec![state.clone()]);
        let adj_eqn = AdjointEquations::new(&problem.eqn, checkpointer);

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
        adj_eqn.rhs.update_state(state.t);
        let sens = adj_eqn.rhs.g_x.borrow();
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 2);
        assert_eq!(sens[(0, 0)], -1.0);
        assert_eq!(sens[(1, 0)], -2.0);
        assert_eq!(sens[(0, 1)], -3.0);
        assert_eq!(sens[(1, 1)], -4.0);

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| - |1.0| = |a - 1| = |-0.9|
        //              |0 a| |2|   |2.0|   |2a - 2| = |-1.8|
        adj_eqn.rhs.set_param_index(Some(0));
        let v = Vcpu::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);

        adj_eqn.rhs.set_param_index(None);
        // F(λ, x, t) = -f^T_x(x, t) λ
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| = |a| = |0.1|
        //              |0 a| |2|   |2a| = |0.2|
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![0.1, 0.2]);
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
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let checkpointer = Checkpointing::new(&problem, solver, vec![state.clone()]);
        let adj_eqn = AdjointEquations::new(&problem.eqn, checkpointer);

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
        adj_eqn.rhs.update_state(state.t);
        let sens = adj_eqn.rhs.g_x.borrow();
        assert_eq!(sens.nrows(), 2);
        assert_eq!(sens.ncols(), 2);
        for (i, j, v) in sens.triplet_iter() {
            match (i, j) {
                (0, 0) => assert_eq!(*v, -1.0),
                (1, 0) => assert_eq!(*v, -2.0),
                (0, 1) => assert_eq!(*v, -3.0),
                (1, 1) => assert_eq!(*v, -4.0),
                _ => panic!("Invalid index"),
            }
        }

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| - |1.0| = |a - 1| = |-0.9|
        //              |0 a| |2|   |2.0|   |2a - 2| = |-1.8|
        adj_eqn.rhs.set_param_index(Some(0));
        let v = faer::Col::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = faer::Col::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);
    }
}