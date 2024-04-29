use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use diffsl::execution::Compiler;

use crate::{
    JacobianColoring,
    op::{LinearOp, NonLinearOp, Op},
    OdeEquations
};

pub type T = f64;
pub type V = nalgebra::DVector<T>;
pub type M = nalgebra::DMatrix<T>;

pub struct DiffSl {
    compiler: Compiler,
    data: RefCell<Vec<T>>,
    ddata: RefCell<Vec<T>>,
    tmp: RefCell<V>,
    nstates: usize,
    nparams: usize,
    nout: usize,
    rhs_coloring: Option<JacobianColoring<M>>,
    mass_coloring: Option<JacobianColoring<M>>,
}

impl DiffSl {
    pub fn new(text: &str, p: V, use_coloring: bool) -> Result<Self> {
        let p = Rc::new(p);
        let compiler = Compiler::from_discrete_str(text)?;
        let mut data = compiler.get_new_data();

        compiler.set_inputs(p.as_slice(), data.as_mut_slice());

        let ddata = compiler.get_new_data();
        compiler.set_inputs(p.as_slice(), data.as_mut_slice());
        let data = RefCell::new(data);
        let ddata = RefCell::new(ddata);
        let (nstates, nparams, nout, _ndata, _stop) = compiler.get_dims();

        let tmp = RefCell::new(V::zeros(nstates));
        

        let mut ret = Self {
            compiler,
            data,
            ddata,
            nparams,
            nstates,
            nout,
            tmp,
            rhs_coloring: None,
            mass_coloring: None,
        };

        if use_coloring {
            let t0 = 0.;
            let mut y0 = V::zeros(nstates);
            let rhs = DiffSlRhs(&ret);
            let mass = DiffSlMass(&ret);
            ret.rhs_coloring = Some(JacobianColoring::new(&rhs, &y0, t0));
            ret.mass_coloring = Some(JacobianColoring::new(&mass, &y0, t0));
        }
        Ok(ret)
    }
    pub fn out(&self, t: T, y: &V) -> &[T] {
        self.compiler
            .calc_out(t, y.as_slice(), self.data.borrow_mut().as_mut_slice());
        self.compiler.get_out(self.data.borrow().as_slice())
    }
}

struct DiffSlRhs<'a>(&'a DiffSl);
struct DiffSlMass<'a>(&'a DiffSl);


macro_rules! impl_op_for_diffsl {
    ($name:ident) => {
        impl Op for $name<'_> {
            type M = M;
            type T = T;
            type V = V;

            fn nstates(&self) -> usize {
                self.0.nstates
            }
            fn nout(&self) -> usize {
                self.0.nstates
            }
            fn nparams(&self) -> usize {
                self.0.nparams
            }
        }
    };
}

impl_op_for_diffsl!(DiffSlRhs);
impl_op_for_diffsl!(DiffSlMass);

impl NonLinearOp for DiffSlRhs<'_> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.compiler.rhs(
            t,
            y.as_slice(),
            self.0.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }

    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut dummy_rhs = Self::V::zeros(self.nstates());
        self.0.compiler.rhs_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.data.borrow_mut().as_mut_slice(),
            self.0.ddata.borrow_mut().as_mut_slice(),
            dummy_rhs.as_mut_slice(),
            y.as_mut_slice(),
        );
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.rhs_coloring {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            NonLinearOp::jacobian_inplace(self, x, t, y);
        }
    
    }
}

impl LinearOp for DiffSlMass<'_> {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self.0.tmp.borrow_mut();
        self.0.compiler.mass(
            t,
            x.as_slice(),
            self.0.data.borrow_mut().as_mut_slice(),
            tmp.as_mut_slice(),
        );

        // y = tmp + beta * y
        y.axpy(1.0, &tmp, beta);
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.mass_coloring {
            let dummy = V::zeros(0);
            coloring.matrix_inplace(self, &dummy, t, y);
        } else {
            LinearOp::matrix_inplace(self, t, y);
        }
    }
}

impl OdeEquations for DiffSl {
    type M = M;
    type T = T;
    type V = V;
    type Mass<'a> = DiffSlMass<'a>;
    type Rhs<'a> = DiffSlRhs<'a>;

    fn rhs(&self) -> Self::Rhs<'_> {
        DiffSlRhs(self)
    }

    fn mass(&self) -> Self::Mass<'_> {
        DiffSlMass(self)
    }

    fn set_params(&mut self, p: Self::V) {
        self.compiler
            .set_inputs(p.as_slice(), self.data.borrow_mut().as_mut_slice());
    }


    fn init(&self, _t: Self::T) -> Self::V {
        let mut ret_y = Self::V::zeros(self.rhs().nstates());
        self.compiler
            .set_u0(ret_y.as_mut_slice(), self.data.borrow_mut().as_mut_slice());
        ret_y
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::{
        linear_solver::NalgebraLU, nonlinear_solver::newton::NewtonNonlinearSolver, Bdf,
        OdeBuilder, OdeEquations, OdeSolverMethod, Vector, NonLinearOp, LinearOp
    };

    use super::DiffSl;

    #[test]
    fn diffsl_expontential_decay() {
        // TODO: put this example into the docs once https://github.com/rust-lang/cargo/pull/13490 makes it to stable
        let code = "
            u { y = 1 }
            F { -y }
            out { y }
        ";
        let problem = OdeBuilder::new().build_diffsl(code).unwrap();
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
        let p = DVector::from_vec(vec![r, k]);
        let eqn = DiffSl::new(text, p.clone(), false).unwrap();

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
        let problem = OdeBuilder::new().p([r, k]).build_diffsl(text).unwrap();
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
        let out = problem.eqn.out(t, &state);
        let out = DVector::from_vec(out.to_vec());
        let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
        out.assert_eq_st(&expected_out, 1e-5);
    }
}
