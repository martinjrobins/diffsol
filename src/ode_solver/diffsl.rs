use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use diffsl::execution::Compiler;

use crate::{
    jacobian::{find_non_zeros_linear, find_non_zeros_nonlinear, JacobianColoring},
    op::{LinearOp, NonLinearOp, Op},
    OdeEquations,
};

pub type T = f64;
pub type V = nalgebra::DVector<T>;
pub type M = nalgebra::DMatrix<T>;

/// Context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
/// This contains the compiled code and the data structures needed to evaluate the ODE equations.
///
/// # Example
///
/// ```rust
/// use diffsol::{OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod, DiffSlContext};
/// type M = nalgebra::DMatrix<f64>;
///         
/// // dy/dt = -ay
/// // y(0) = 1
/// let context = DiffSlContext::new("
///     in = [a]
///     a { 1 }
///     u { 1.0 }
///     F { -a*u }
///     out { u }
/// ").unwrap();
/// let problem = OdeBuilder::new()
///  .rtol(1e-6)
///  .p([0.1])
///  .build_diffsl(&context).unwrap();
/// let mut solver = Bdf::default();
/// let t = 0.4;
/// let state = OdeSolverState::new(&problem).unwrap();
/// solver.set_problem(state, &problem);
/// while solver.state().unwrap().t <= t {
///    solver.step().unwrap();
/// }
/// let y = solver.interpolate(t);
/// ```
pub struct DiffSlContext {
    compiler: Compiler,
    data: RefCell<Vec<T>>,
    ddata: RefCell<Vec<T>>,
    tmp: RefCell<V>,
    nstates: usize,
    nroots: usize,
    nparams: usize,
}

impl DiffSlContext {
    /// Create a new context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
    /// The input parameters are not initialized and must be set using the [OdeEquations::set_params] function before solving the ODE.
    pub fn new(text: &str) -> Result<Self> {
        let compiler = Compiler::from_discrete_str(text)?;
        let (nstates, nparams, _nout, _ndata, nroots) = compiler.get_dims();
        let data = RefCell::new(compiler.get_new_data());
        let ddata = RefCell::new(compiler.get_new_data());
        let tmp = RefCell::new(V::zeros(nstates));

        Ok(Self {
            compiler,
            data,
            ddata,
            nparams,
            nstates,
            tmp,
            nroots,
        })
    }
    pub fn out(&self, t: T, y: &V) -> &[T] {
        self.compiler
            .calc_out(t, y.as_slice(), self.data.borrow_mut().as_mut_slice());
        self.compiler.get_out(self.data.borrow().as_slice())
    }
}

pub struct DiffSl<'a> {
    context: &'a DiffSlContext,
    rhs: Rc<DiffSlRhs<'a>>,
    mass: Rc<DiffSlMass<'a>>,
    root: Rc<DiffSlRoot<'a>>,
}

impl<'a> DiffSl<'a> {
    pub fn new(context: &'a DiffSlContext, use_coloring: bool) -> Self {
        let rhs = Rc::new(DiffSlRhs::new(context, use_coloring));
        let mass = Rc::new(DiffSlMass::new(context, use_coloring));
        let root = Rc::new(DiffSlRoot::new(context));
        Self {
            context,
            rhs,
            mass,
            root,
        }
    }
}

pub struct DiffSlRoot<'a> {
    context: &'a DiffSlContext,
}

pub struct DiffSlRhs<'a> {
    context: &'a DiffSlContext,
    coloring: Option<JacobianColoring<M>>,
}

pub struct DiffSlMass<'a> {
    context: &'a DiffSlContext,
    coloring: Option<JacobianColoring<M>>,
}

impl<'a> DiffSlRoot<'a> {
    pub fn new(context: &'a DiffSlContext) -> Self {
        Self { context }
    }
}

impl<'a> DiffSlRhs<'a> {
    pub fn new(context: &'a DiffSlContext, use_coloring: bool) -> Self {
        let mut ret = Self {
            context,
            coloring: None,
        };

        if use_coloring {
            let x0 = V::zeros(context.nstates);
            let t0 = 0.0;
            let non_zeros = find_non_zeros_nonlinear(&ret, &x0, t0);
            ret.coloring = Some(JacobianColoring::new_from_non_zeros(&ret, non_zeros));
        }
        ret
    }
}

impl<'a> DiffSlMass<'a> {
    pub fn new(context: &'a DiffSlContext, use_coloring: bool) -> Self {
        let mut ret = Self {
            context,
            coloring: None,
        };

        if use_coloring {
            let t0 = 0.0;
            let non_zeros = find_non_zeros_linear(&ret, t0);
            ret.coloring = Some(JacobianColoring::new_from_non_zeros(&ret, non_zeros));
        }
        ret
    }
}

macro_rules! impl_op_for_diffsl {
    ($name:ident) => {
        impl Op for $name<'_> {
            type M = M;
            type T = T;
            type V = V;

            fn nstates(&self) -> usize {
                self.context.nstates
            }
            fn nout(&self) -> usize {
                self.context.nstates
            }
            fn nparams(&self) -> usize {
                self.context.nparams
            }
        }
    };
}

impl_op_for_diffsl!(DiffSlRhs);
impl_op_for_diffsl!(DiffSlMass);

impl Op for DiffSlRoot<'_> {
    type M = M;
    type T = T;
    type V = V;

    fn nstates(&self) -> usize {
        self.context.nstates
    }
    fn nout(&self) -> usize {
        self.context.nroots
    }
    fn nparams(&self) -> usize {
        self.context.nparams
    }
}

impl NonLinearOp for DiffSlRoot<'_> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.compiler.calc_stop(
            t,
            x.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }

    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(0.0);
    }
}

impl NonLinearOp for DiffSlRhs<'_> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.compiler.rhs(
            t,
            x.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }

    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut dummy_rhs = Self::V::zeros(self.nstates());
        self.context.compiler.rhs_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            self.context.ddata.borrow_mut().as_mut_slice(),
            dummy_rhs.as_mut_slice(),
            y.as_mut_slice(),
        );
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.coloring {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
}

impl LinearOp for DiffSlMass<'_> {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self.context.tmp.borrow_mut();
        self.context.compiler.mass(
            t,
            x.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            tmp.as_mut_slice(),
        );

        // y = tmp + beta * y
        y.axpy(1.0, &tmp, beta);
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
}

impl<'a> OdeEquations for DiffSl<'a> {
    type M = M;
    type T = T;
    type V = V;
    type Mass = DiffSlMass<'a>;
    type Rhs = DiffSlRhs<'a>;
    type Root = DiffSlRoot<'a>;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }

    fn mass(&self) -> &Rc<Self::Mass> {
        &self.mass
    }

    fn root(&self) -> Option<&Rc<Self::Root>> {
        Some(&self.root)
    }

    fn set_params(&mut self, p: Self::V) {
        self.context
            .compiler
            .set_inputs(p.as_slice(), self.context.data.borrow_mut().as_mut_slice());
    }

    fn init(&self, _t: Self::T) -> Self::V {
        let mut ret_y = Self::V::zeros(self.rhs().nstates());
        self.context.compiler.set_u0(
            ret_y.as_mut_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
        );
        ret_y
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::{
        linear_solver::NalgebraLU, nonlinear_solver::newton::NewtonNonlinearSolver, Bdf, LinearOp,
        NonLinearOp, OdeBuilder, OdeEquations, OdeSolverMethod, Vector,
    };

    use super::{DiffSl, DiffSlContext};

    #[test]
    fn diffsl_expontential_decay() {
        // TODO: put this example into the docs once https://github.com/rust-lang/cargo/pull/13490 makes it to stable
        let code = "
            u { y = 1 }
            F { -y }
            out { y }
        ";
        let context = DiffSlContext::new(code).unwrap();
        let problem = OdeBuilder::new().build_diffsl(&context).unwrap();
        let mut solver = Bdf::default();
        let _y = solver.solve(&problem, 1.0).unwrap();
    }

    #[test]
    fn diffsl_logistic_growth() {
        let text = "
            in = [r, k]
            r { 1 }
            k { 1 }
            u_i {
                y = 0.1,
                z = 0,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                3 * y,
                4 * z,
            }
        ";

        let k = 1.0;
        let r = 1.0;
        let context = DiffSlContext::new(text).unwrap();
        let mut eqn = DiffSl::new(&context, false);
        let p = DVector::from_vec(vec![r, k]);
        eqn.set_params(p);

        // test that the initial values look ok
        let y0 = 0.1;
        let init = eqn.init(0.0);
        let init_expect = DVector::from_vec(vec![y0, 0.0]);
        init.assert_eq_st(&init_expect, 1e-10);
        let rhs = eqn.rhs().call(&init, 0.0);
        let rhs_expect = DVector::from_vec(vec![r * y0 * (1.0 - y0 / k), 2.0 * y0]);
        rhs.assert_eq_st(&rhs_expect, 1e-10);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let rhs_jac = eqn.rhs().jac_mul(&init, 0.0, &v);
        let rhs_jac_expect = DVector::from_vec(vec![r * (1.0 - y0 / k) - r * y0 / k, 1.0]);
        rhs_jac.assert_eq_st(&rhs_jac_expect, 1e-10);
        let mut mass_y = DVector::from_vec(vec![0.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        eqn.mass().call_inplace(&v, 0.0, &mut mass_y);
        let mass_y_expect = DVector::from_vec(vec![1.0, 0.0]);
        mass_y.assert_eq_st(&mass_y_expect, 1e-10);

        // solver a bit and check the state and output
        let problem = OdeBuilder::new().p([r, k]).build_diffsl(&context).unwrap();
        let mut solver = Bdf::default();
        let mut root_solver = NewtonNonlinearSolver::new(NalgebraLU::default());
        let t = 1.0;
        let state = solver
            .make_consistent_and_solve(&problem, t, &mut root_solver)
            .unwrap();
        let y_expect = k / (1.0 + (k - y0) * (-r * t).exp() / y0);
        let z_expect = 2.0 * y_expect;
        let expected_state = DVector::from_vec(vec![y_expect, z_expect]);
        state.assert_eq_st(&expected_state, 1e-5);
        let out = context.out(t, &state);
        let out = DVector::from_vec(out.to_vec());
        let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
        out.assert_eq_st(&expected_out, 1e-5);
    }
}
