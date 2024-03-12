
use std::cell::RefCell;

use diffsl::execution::Compiler;
use anyhow::Result;

use crate::OdeEquations;

use super::{NonLinearOp, Op};

type T = f64;
type V = nalgebra::DVector<T>;
type M = nalgebra::DMatrix<T>;


struct DiffSl {
    compiler: Compiler,
    data: RefCell<Vec<T>>,
    nstates: usize,
    nparams: usize,
}

impl DiffSl {
    pub fn new(text: &str, p: V) -> Result<Self> {
        let compiler = Compiler::from_discrete_str(text)?;
        let mut data = compiler.get_new_data();
        compiler.set_inputs(p.as_slice(), data.as_mut_slice());
        let data = RefCell::new(data);
        let (nstates, nparams, _nout, _ndata, _stop) = compiler.get_dims();
        Ok(Self {
            compiler,
            data,
            nparams, 
            nstates,
        })
    }
}

impl Op for DiffSl {
    type V = V;
    type T = T;
    type M = M;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
}

impl OdeEquations for DiffSl {
    fn set_params(&mut self, p: Self::V) {
        self.compiler.set_inputs(p.as_slice(), self.data.borrow_mut().as_mut_slice());
    }

    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        todo!()
    }

    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        todo!()
    }

    fn init(&self, _t: Self::T) -> Self::V {
        let mut ret_y = Self::V::zeros(self.nstates());
        let mut ret_yp = Self::V::zeros(self.nstates());
        self.compiler.set_u0(ret_y.as_mut_slice(), ret_yp.as_mut_slice(), self.data.borrow_mut().as_mut_slice());
        ret_y
    }

    fn residual_inplace(&self, t: Self::T, y_mass: &Self::V, y_rhs: &Self::V, c: Self::T, res: &mut Self::V) {
        todo!()
    }

    fn residual_jac_inplace(&self, t: Self::T, y: &Self::V, c: Self::T, v: &Self::V, jac: &mut Self::V) {
        todo!()
    }

    fn algebraic_indices(&self) -> <Self::V as crate::vector::Vector>::Index {
        todo!()
    }

}

