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
    checkpointer: Rc<Checkpointing<Eqn, Method>>,
    g_x: Option<Eqn::M>,
    tmp: RefCell<Eqn::V>,
    tmp_t: RefCell<Option<Eqn::T>>,
    index: Option<usize>,
}

impl<Eqn, Method> AdjointRhs<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub fn new(eqn: &Rc<Eqn>, checkpointer: Rc<Checkpointing<Eqn, Method>>, with_out: bool) -> Self {
        let (g_x, index) = if with_out {
            let g_x = if let Some(g) = eqn.out() {
                let g_x_sparsity = g.sparsity_adjoint();
                Eqn::M::new_from_sparsity(
                    g.nout(),
                    g.nstates(),
                    g_x_sparsity.map(|s| s.to_owned()),
                )
            } else {
                panic!("Cannot call AdjointRhs::new with output without output");
            };
            (Some(g_x), Some(0))
        } else {
            (None, None)
        };

        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(eqn.rhs().nstates()));
        let tmp_t = RefCell::new(None);
        Self {
            eqn: eqn.clone(),
            g_x,
            checkpointer,
            tmp,
            index,
            tmp_t,
        }
    }

    fn update_tmp(&self, t: Eqn::T) {
        {
            let tmp_t = self.tmp_t.borrow();
            if let Some(tmp_t) = *tmp_t {
                if t == tmp_t {
                    return;
                }
            }
        }
        let mut x = self.tmp.borrow_mut();
        self.checkpointer.interpolate(t, &mut x).unwrap();
        self.tmp_t.replace(Some(t));
    }

    /// precompute S = g^T_x(x,t) and the state x(t) from t
    pub fn update_state(&mut self, t: Eqn::T) {
        // update -g_x^T
        let g = self.eqn.out().expect("Cannot call update_state without output");
        self.update_tmp(t);
        let tmp = self.tmp.borrow();
        let g_x = self.g_x.as_mut().expect("Cannot call update_state without output");
        g.adjoint_inplace(&tmp, t, g_x);
    }

    pub fn set_out_index(&mut self, new_index: usize) {
        let index = self.index.as_mut().expect("Cannot set parameter index without output");
        *index = new_index;
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
        self.update_tmp(t);
        let tmp = self.tmp.borrow();

        self.eqn.rhs().jac_transpose_mul_inplace(&tmp, t, lambda, y);

        // y = -f^T_x(x, t) λ - g^T_x(x,t)
        if let (Some(g_x), Some(index)) = (&self.g_x, self.index) {
            g_x.add_column_to_vector(index, y);
        }
    }
    // J = -f^T_x(x, t)
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.update_tmp(t);
        let tmp = self.tmp.borrow();
        self.eqn.rhs().jac_transpose_mul_inplace(&tmp, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.update_tmp(t);
        let tmp = self.tmp.borrow();
        self.eqn.rhs().adjoint_inplace(&tmp, t, y);
    }
}

/// Adjoint equations for ODEs
/// 
/// M * dλ/dt = -f^T_x(x, t) λ - g^T_x(x,t)
/// λ(T) = 0
///
pub struct AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    eqn: Rc<Eqn>,
    rhs: Rc<AdjointRhs<Eqn, Method>>,
    init: Rc<AdjointInit<Eqn>>,
    include_in_error_control: bool,
}

impl<Eqn, Method> AdjointEquations<Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<Eqn>,
{
    pub(crate) fn new(eqn: &Rc<Eqn>, checkpointer: Rc<Checkpointing<Eqn, Method>>, with_out: bool) -> Self {
        let rhs = Rc::new(AdjointRhs::new(eqn, checkpointer, with_out));
        let init = Rc::new(AdjointInit::new(eqn));
        Self {
            rhs,
            init,
            eqn: eqn.clone(),
            include_in_error_control: false,
        }
    }
    pub(crate) fn set_out_index(&mut self, index: usize) {
        Rc::get_mut(&mut self.rhs).unwrap().set_out_index(index);
    }

    pub(crate) fn update_rhs_state(&mut self, t: Eqn::T) {
        Rc::get_mut(&mut self.rhs).unwrap().update_state(t);
    }
    
    pub fn include_in_error_control(&self) -> bool {
        self.include_in_error_control
    }
    
    pub fn set_include_in_error_control(&mut self, include: bool) {
        self.include_in_error_control = include;
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
    use std::rc::Rc;

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
        let checkpointer = Rc::new(Checkpointing::new(&problem, solver, 0, vec![state.clone(), state.clone()]));
        let adj_eqn = AdjointEquations::new(&problem.eqn, checkpointer.clone(), false);
        // F(λ, x, t) = -f^T_x(x, t) λ
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| = |a| = |0.1|
        //              |0 a| |2|   |2a| = |0.2|
        let v = Vcpu::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![0.1, 0.2]);
        f.assert_eq_st(&f_expect, 1e-10);

        let mut adj_eqn = AdjointEquations::new(&problem.eqn, checkpointer.clone(), true);

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
        adj_eqn.update_rhs_state(state.t);
        let sens = adj_eqn.rhs.g_x.as_ref().unwrap();
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
        adj_eqn.set_out_index(0);
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
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let checkpointer = Rc::new(Checkpointing::new(&problem, solver, 0, vec![state.clone(), state.clone()]));
        let mut adj_eqn = AdjointEquations::new(&problem.eqn, checkpointer, true);

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
        adj_eqn.update_rhs_state(state.t);
        let sens = adj_eqn.rhs.g_x.as_ref().unwrap();
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
        adj_eqn.set_out_index(0);
        let v = faer::Col::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = faer::Col::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);
    }
}
