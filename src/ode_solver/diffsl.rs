use std::{cell::RefCell, rc::Rc};

use diffsl::{execution::module::CodegenModule, Compiler};

use crate::{
    error::DiffsolError, find_jacobian_non_zeros, find_matrix_non_zeros,
    jacobian::JacobianColoring, matrix::sparsity::MatrixSparsity,
    op::nonlinear_op::NonLinearOpJacobian, ConstantOp, LinearOp, Matrix, NonLinearOp, OdeEquations,
    Op, Vector,
};

pub type T = f64;

/// Context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
///
/// This contains the compiled code and the data structures needed to evaluate the ODE equations.
/// Note that the example below uses the LLVM backend (requires one of the `diffsl-llvm` features),
/// but the Cranelift backend can also be used using
/// `diffsl::CraneliftModule` instead of `diffsl::LlvmModule` (requires `diffsl-cranelift` feature).
///
/// # Example
///
/// ```rust
/// use diffsol::{OdeBuilder, Bdf, OdeSolverState, OdeSolverMethod, DiffSlContext, diffsl::LlvmModule};
///         
/// // dy/dt = -ay
/// // y(0) = 1
/// let context = DiffSlContext::<nalgebra::DMatrix<f64>, LlvmModule>::new("
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
/// let state = OdeSolverState::new(&problem, &solver).unwrap();
/// solver.set_problem(state, &problem);
/// while solver.state().unwrap().t <= t {
///    solver.step().unwrap();
/// }
/// let y = solver.interpolate(t);
/// ```
pub struct DiffSlContext<M: Matrix<T = T>, CG: CodegenModule> {
    compiler: Compiler<CG>,
    data: RefCell<Vec<M::T>>,
    ddata: RefCell<Vec<M::T>>,
    tmp: RefCell<M::V>,
    nstates: usize,
    nroots: usize,
    nparams: usize,
    nout: usize,
}

impl<M: Matrix<T = T>, CG: CodegenModule> DiffSlContext<M, CG> {
    /// Create a new context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
    /// The input parameters are not initialized and must be set using the [OdeEquations::set_params] function before solving the ODE.
    pub fn new(text: &str) -> Result<Self, DiffsolError> {
        let compiler =
            Compiler::from_discrete_str(text).map_err(|e| DiffsolError::Other(e.to_string()))?;
        let (nstates, nparams, nout, _ndata, nroots) = compiler.get_dims();
        let data = RefCell::new(compiler.get_new_data());
        let ddata = RefCell::new(compiler.get_new_data());
        let tmp = RefCell::new(M::V::zeros(nstates));

        Ok(Self {
            compiler,
            data,
            ddata,
            nparams,
            nstates,
            tmp,
            nroots,
            nout,
        })
    }

    pub fn recompile(&mut self, text: &str) -> Result<(), DiffsolError> {
        self.compiler =
            Compiler::from_discrete_str(text).map_err(|e| DiffsolError::Other(e.to_string()))?;
        let (nstates, nparams, nout, _ndata, nroots) = self.compiler.get_dims();
        self.data = RefCell::new(self.compiler.get_new_data());
        self.ddata = RefCell::new(self.compiler.get_new_data());
        self.tmp = RefCell::new(M::V::zeros(nstates));
        self.nparams = nparams;
        self.nstates = nstates;
        self.nout = nout;
        self.nroots = nroots;
        Ok(())
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Default for DiffSlContext<M, CG> {
    fn default() -> Self {
        Self::new(
            "
            u { y = 1 }
            F { -y }
            out { y }
        ",
        )
        .unwrap()
    }
}

pub struct DiffSl<'a, M: Matrix<T = T>, CG: CodegenModule> {
    context: &'a DiffSlContext<M, CG>,
    rhs: Rc<DiffSlRhs<'a, M, CG>>,
    mass: Option<Rc<DiffSlMass<'a, M, CG>>>,
    root: Rc<DiffSlRoot<'a, M, CG>>,
    init: Rc<DiffSlInit<'a, M, CG>>,
    out: Rc<DiffSlOut<'a, M, CG>>,
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> DiffSl<'a, M, CG> {
    pub fn new(context: &'a DiffSlContext<M, CG>, use_coloring: bool) -> Self {
        let rhs = Rc::new(DiffSlRhs::new(context, use_coloring));
        let mass = DiffSlMass::new(context, use_coloring).map(Rc::new);
        let root = Rc::new(DiffSlRoot::new(context));
        let init = Rc::new(DiffSlInit::new(context));
        let out = Rc::new(DiffSlOut::new(context));
        Self {
            context,
            rhs,
            mass,
            root,
            init,
            out,
        }
    }
}

pub struct DiffSlRoot<'a, M: Matrix<T = T>, CG: CodegenModule> {
    context: &'a DiffSlContext<M, CG>,
}

pub struct DiffSlOut<'a, M: Matrix<T = T>, CG: CodegenModule> {
    context: &'a DiffSlContext<M, CG>,
}

pub struct DiffSlRhs<'a, M: Matrix<T = T>, CG: CodegenModule> {
    context: &'a DiffSlContext<M, CG>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
}

pub struct DiffSlMass<'a, M: Matrix<T = T>, CG: CodegenModule> {
    context: &'a DiffSlContext<M, CG>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
}

pub struct DiffSlInit<'a, M: Matrix<T = T>, CG: CodegenModule> {
    context: &'a DiffSlContext<M, CG>,
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> DiffSlOut<'a, M, CG> {
    pub fn new(context: &'a DiffSlContext<M, CG>) -> Self {
        Self { context }
    }
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> DiffSlRoot<'a, M, CG> {
    pub fn new(context: &'a DiffSlContext<M, CG>) -> Self {
        Self { context }
    }
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> DiffSlInit<'a, M, CG> {
    pub fn new(context: &'a DiffSlContext<M, CG>) -> Self {
        Self { context }
    }
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> DiffSlRhs<'a, M, CG> {
    pub fn new(context: &'a DiffSlContext<M, CG>, use_coloring: bool) -> Self {
        let mut ret = Self {
            context,
            coloring: None,
            sparsity: None,
        };

        if use_coloring {
            let x0 = M::V::zeros(context.nstates);
            let t0 = 0.0;
            let non_zeros = find_jacobian_non_zeros(&ret, &x0, t0);
            ret.sparsity = Some(
                MatrixSparsity::try_from_indices(ret.nout(), ret.nstates(), non_zeros.clone())
                    .expect("invalid sparsity pattern"),
            );
            ret.coloring = Some(JacobianColoring::new_from_non_zeros(&ret, non_zeros));
        }
        ret
    }
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> DiffSlMass<'a, M, CG> {
    pub fn new(context: &'a DiffSlContext<M, CG>, use_coloring: bool) -> Option<Self> {
        if !context.compiler.has_mass() {
            return None;
        }
        let mut ret = Self {
            context,
            coloring: None,
            sparsity: None,
        };

        if use_coloring {
            let t0 = 0.0;
            let non_zeros = find_matrix_non_zeros(&ret, t0);
            ret.sparsity = Some(
                MatrixSparsity::try_from_indices(ret.nout(), ret.nstates(), non_zeros.clone())
                    .expect("invalid sparsity pattern"),
            );
            ret.coloring = Some(JacobianColoring::new_from_non_zeros(&ret, non_zeros));
        }
        Some(ret)
    }
}

macro_rules! impl_op_for_diffsl {
    ($name:ident) => {
        impl<M: Matrix<T = T>, CG: CodegenModule> Op for $name<'_, M, CG> {
            type M = M;
            type T = T;
            type V = M::V;

            fn nstates(&self) -> usize {
                self.context.nstates
            }
            #[allow(clippy::misnamed_getters)]
            fn nout(&self) -> usize {
                self.context.nstates
            }
            fn nparams(&self) -> usize {
                self.context.nparams
            }
            fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
                self.sparsity.as_ref().map(|s| s.as_ref())
            }
        }
    };
}

impl_op_for_diffsl!(DiffSlRhs);
impl_op_for_diffsl!(DiffSlMass);

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSlInit<'_, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.context.nstates
    }
    fn nparams(&self) -> usize {
        self.context.nparams
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSlRoot<'_, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.context.nroots
    }
    fn nparams(&self) -> usize {
        self.context.nparams
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSlOut<'_, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.context.nstates
    }
    fn nout(&self) -> usize {
        self.context.nout
    }
    fn nparams(&self) -> usize {
        self.context.nparams
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> ConstantOp for DiffSlInit<'_, M, CG> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        self.context.compiler.set_u0(
            y.as_mut_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOp for DiffSlRoot<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.compiler.calc_stop(
            t,
            x.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpJacobian for DiffSlRoot<'_, M, CG> {
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(0.0);
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOp for DiffSlOut<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.compiler.calc_out(
            t,
            x.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
        );
        let out = self
            .context
            .compiler
            .get_out(self.context.data.borrow().as_slice());
        y.copy_from_slice(out);
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpJacobian for DiffSlOut<'_, M, CG> {
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.context.compiler.calc_out_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            self.context.ddata.borrow_mut().as_mut_slice(),
        );
        let out_grad = self
            .context
            .compiler
            .get_out(self.context.ddata.borrow().as_slice());
        y.copy_from_slice(out_grad);
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOp for DiffSlRhs<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.compiler.rhs(
            t,
            x.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpJacobian for DiffSlRhs<'_, M, CG> {
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

impl<M: Matrix<T = T>, CG: CodegenModule> LinearOp for DiffSlMass<'_, M, CG> {
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

impl<'a, M: Matrix<T = T>, CG: CodegenModule> OdeEquations for DiffSl<'a, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;
    type Mass = DiffSlMass<'a, M, CG>;
    type Rhs = DiffSlRhs<'a, M, CG>;
    type Root = DiffSlRoot<'a, M, CG>;
    type Init = DiffSlInit<'a, M, CG>;
    type Out = DiffSlOut<'a, M, CG>;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }

    fn mass(&self) -> Option<&Rc<Self::Mass>> {
        self.mass.as_ref()
    }

    fn root(&self) -> Option<&Rc<Self::Root>> {
        Some(&self.root)
    }

    fn set_params(&mut self, p: Self::V) {
        // set the parameters in data
        self.context
            .compiler
            .set_inputs(p.as_slice(), self.context.data.borrow_mut().as_mut_slice());

        // set_u0 will calculate all the constants in the equations based on the params
        let mut dummy = M::V::zeros(self.context.nstates);
        self.context.compiler.set_u0(
            dummy.as_mut_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
        );
    }

    fn init(&self) -> &Rc<Self::Init> {
        &self.init
    }

    fn out(&self) -> Option<&Rc<Self::Out>> {
        Some(&self.out)
    }
}

#[cfg(test)]
mod tests {
    use diffsl::{execution::module::CodegenModule, CraneliftModule};
    use nalgebra::DVector;

    use crate::{
        Bdf, ConstantOp, LinearOp, NonLinearOp, NonLinearOpJacobian, OdeBuilder, OdeEquations,
        OdeSolverMethod, OdeSolverState, Vector,
    };

    use super::{DiffSl, DiffSlContext};

    #[test]
    fn diffsl_logistic_growth_cranelift() {
        diffsl_logistic_growth::<CraneliftModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn diffsl_logistic_growth_llvm() {
        diffsl_logistic_growth::<diffsl::LlvmModule>();
    }

    fn diffsl_logistic_growth<CG: CodegenModule>() {
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
        let context = DiffSlContext::<nalgebra::DMatrix<f64>, CG>::new(text).unwrap();
        let mut eqn = DiffSl::new(&context, false);
        let p = DVector::from_vec(vec![r, k]);
        eqn.set_params(p);

        // test that the initial values look ok
        let y0 = 0.1;
        let init = eqn.init().call(0.0);
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
        eqn.mass().unwrap().call_inplace(&v, 0.0, &mut mass_y);
        let mass_y_expect = DVector::from_vec(vec![1.0, 0.0]);
        mass_y.assert_eq_st(&mass_y_expect, 1e-10);

        // solver a bit and check the state and output
        let problem = OdeBuilder::new().p([r, k]).build_diffsl(&context).unwrap();
        let mut solver = Bdf::default();
        let t = 1.0;
        let state = OdeSolverState::new(&problem, &solver).unwrap();
        let (ys, ts) = solver.solve(&problem, state, t).unwrap();
        for (i, t) in ts.iter().enumerate() {
            let y_expect = k / (1.0 + (k - y0) * (-r * t).exp() / y0);
            let z_expect = 2.0 * y_expect;
            let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
            ys.column(i).into_owned().assert_eq_st(&expected_out, 1e-4);
        }

        // do it again with some explicit t_evals
        let t_evals = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0];
        let state = OdeSolverState::new(&problem, &solver).unwrap();
        let ys = solver.solve_dense(&problem, state, &t_evals).unwrap();
        for (i, t) in t_evals.iter().enumerate() {
            let y_expect = k / (1.0 + (k - y0) * (-r * t).exp() / y0);
            let z_expect = 2.0 * y_expect;
            let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
            ys.column(i).into_owned().assert_eq_st(&expected_out, 1e-4);
        }
    }
}
