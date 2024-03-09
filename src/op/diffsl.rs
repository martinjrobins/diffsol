
use std::cell::RefCell;

use diffsl::execution::Compiler;
use anyhow::Result;

use super::{NonLinearOp, Op};

type T = f64;
type V = nalgebra::DVector<T>;
type M = nalgebra::DMatrix<T>;

struct DiffslRhsOp<'a> {
  diffsl: &'a DiffSl,
  nstates: usize,
  nout: usize,
  nparams: usize,
}

impl <'a> DiffslRhsOp<'a> {
  pub fn new(diffsl: &'a DiffSl) -> Self {
    Self {
      diffsl,
      nstates: diffsl.nstates(),
      nout: diffsl.nout(),
      nparams: diffsl.nparams(),
    }
  }
}

impl Op for DiffslRhsOp<'_> {
  type T = T;
  type V = V;
  type M = M;
  fn nstates(&self) -> usize {
    self.nstates
  }
  fn nout(&self) -> usize {
    self.nout
  }
  fn nparams(&self) -> usize {
    self.nparams
  }
}

impl NonLinearOp for DiffslRhsOp<'_> {
  fn call_inplace(&self, x: &V, t: T, y: &mut V) {
    self.diffsl.rhs_call_inplace(x, t, y);
  }
  fn jac_mul_inplace(&self, x: &V, t: T, v: &V, y: &mut V) {
    self.diffsl.rhs_jac_mul_inplace(x, t, v, y);
  }
}

struct DiffSl {
  compiler: Compiler,
  data: RefCell<Vec<T>>,
  nstates: usize,
  nout: usize,
  nparams: usize,
}

impl DiffSl {
  pub fn new(text: &str, p: V) -> Result<Self> {
    let compiler = Compiler::from_discrete_str(text)?;
    let data = compiler.get_new_data();
    compiler.set_inputs(p, data);
    Ok(Self {
        compiler,
        data,
    })
  }

  pub fn rhs_call_inplace(&self, x: &V, t: T, y: &mut V) {
    let data = self.data.borrow_mut();
    self.compiler.residual(t, x, x, data.as_mut_slice(), y);
  }

  pub fn mass_call_inplace(&self, x: &V, t: T, y: &mut V) {
    let data = self.data.borrow_mut();
    self.compiler.residual(t, x, x, data.as_mut_slice(), y);
  }

  pub fn init_call_inplace(&self, t: T, y: &mut V) {
    let data = self.data.borrow_mut();
    self.compiler.set_u0(y, y, data.as_mut_slice());
  }

  pub fn nstates(&self) -> usize {
    self.compiler.number_of_states()
  }

  pub fn nout(&self) -> usize {
    self.compiler.number_of_outputs()
  }

  pub fn nparams(&self) -> usize {
    self.compiler.number_of_parameters()
  }

  pub fn rhs(&self) -> DiffslRhsOp {
    DiffslRhsOp::new(self)
  }
}