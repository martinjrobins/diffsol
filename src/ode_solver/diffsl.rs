use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use diffsl::execution::Compiler;

use crate::{
    jacobian::{find_non_zero_entries, JacobianColoring},
    op::{closure::Closure, linear_closure::LinearClosure, Op},
    vector::Vector,
    Matrix, OdeEquations,
};

pub type T = f64;
pub type V = nalgebra::DVector<T>;
pub type M = nalgebra::DMatrix<T>;

pub struct DiffSl {
    compiler: Compiler,
    data: RefCell<Vec<T>>,
    ddata: RefCell<Vec<T>>,
    nstates: usize,
    nparams: usize,
    nout: usize,
    jacobian_coloring: Option<JacobianColoring>,
    mass_coloring: Option<JacobianColoring>,
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

        let (jacobian_coloring, mass_coloring) = if use_coloring {
            let rhs_inplace = |x: &V, _p: &V, t: T, y_rhs: &mut V| {
                compiler.rhs(
                    t,
                    x.as_slice(),
                    data.borrow_mut().as_mut_slice(),
                    y_rhs.as_mut_slice(),
                );
            };

            let rhs_jac_inplace = |x: &V, _p: &V, t: T, v: &V, y: &mut V| {
                let mut dummy_rhs = V::zeros(nstates);
                compiler.rhs_grad(
                    t,
                    x.as_slice(),
                    v.as_slice(),
                    data.borrow_mut().as_mut_slice(),
                    ddata.borrow_mut().as_mut_slice(),
                    dummy_rhs.as_mut_slice(),
                    y.as_mut_slice(),
                );
            };
            let mass_inplace = |x: &V, _p: &V, t: T, y: &mut V| {
                compiler.mass(
                    t,
                    x.as_slice(),
                    data.borrow_mut().as_mut_slice(),
                    y.as_mut_slice(),
                );
            };
            let t0 = 0.;
            let mut y0 = V::zeros(nstates);
            compiler.set_u0(y0.as_mut_slice(), data.borrow_mut().as_mut_slice());

            let op =
                Closure::<M, _, _>::new(rhs_inplace, rhs_jac_inplace, nstates, nstates, p.clone());
            let jacobian_coloring = Some(JacobianColoring::new(&op, &y0, t0));
            let op = LinearClosure::<M, _>::new(mass_inplace, nstates, nstates, p.clone());
            let mass_coloring = Some(JacobianColoring::new(&op, &y0, t0));
            (jacobian_coloring, mass_coloring)
        } else {
            (None, None)
        };
        Ok(Self {
            compiler,
            data,
            ddata,
            nparams,
            nstates,
            nout,
            jacobian_coloring,
            mass_coloring,
        })
    }
    pub fn out(&self, t: T, y: &V) -> &[T] {
        self.compiler
            .calc_out(t, y.as_slice(), self.data.borrow_mut().as_mut_slice());
        self.compiler.get_out(self.data.borrow().as_slice())
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
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
}

impl OdeEquations for DiffSl {
    fn set_params(&mut self, p: Self::V) {
        self.compiler
            .set_inputs(p.as_slice(), self.data.borrow_mut().as_mut_slice());
    }

    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        self.compiler.rhs(
            t,
            y.as_slice(),
            self.data.borrow_mut().as_mut_slice(),
            rhs_y.as_mut_slice(),
        );
    }

    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let mut dummy_rhs = Self::V::zeros(self.nstates());
        self.compiler.rhs_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.data.borrow_mut().as_mut_slice(),
            self.ddata.borrow_mut().as_mut_slice(),
            dummy_rhs.as_mut_slice(),
            y.as_mut_slice(),
        );
    }

    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        let mut ids = vec![0.; self.nstates()];
        self.compiler.set_id(ids.as_mut_slice());
        let mut indices = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if *id == 0. {
                indices.push(i);
            }
        }
        <Self::V as Vector>::Index::from_vec(indices)
    }

    fn mass_matrix(&self, t: Self::T) -> Self::M {
        let mass_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y: &mut Self::V| {
            self.mass_inplace(t, x, y);
        };
        let dummy_p = Rc::new(V::zeros(0));
        let op = LinearClosure::<M, _>::new(mass_inplace, self.nstates, self.nstates, dummy_p);
        let triplets = if let Some(coloring) = &self.mass_coloring {
            coloring.find_non_zero_entries(&op, &self.init(t), t)
        } else {
            find_non_zero_entries(&op, &self.init(t), t)
        };
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }

    fn jacobian_matrix(&self, x: &Self::V, t: Self::T) -> Self::M {
        let rhs_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y_rhs: &mut Self::V| {
            self.rhs_inplace(t, x, y_rhs);
        };
        let rhs_jac_inplace =
            |x: &Self::V, _p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V| {
                self.rhs_jac_inplace(t, x, v, y);
            };
        let dummy_p = Rc::new(V::zeros(0));
        let op = Closure::<M, _, _>::new(
            rhs_inplace,
            rhs_jac_inplace,
            self.nstates,
            self.nstates,
            dummy_p,
        );
        let triplets = if let Some(coloring) = &self.jacobian_coloring {
            coloring.find_non_zero_entries(&op, x, t)
        } else {
            find_non_zero_entries(&op, x, t)
        };
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }

    fn init(&self, _t: Self::T) -> Self::V {
        let mut ret_y = Self::V::zeros(self.nstates());
        self.compiler
            .set_u0(ret_y.as_mut_slice(), self.data.borrow_mut().as_mut_slice());
        ret_y
    }
    fn mass_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.compiler.mass(
            t,
            v.as_slice(),
            self.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::{
        nonlinear_solver::newton::NewtonNonlinearSolver, Bdf, NalgebraLU, OdeBuilder, OdeEquations,
        OdeSolverMethod, Vector,
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
        init.assert_eq(&init_expect, 1e-10);
        let rhs = eqn.rhs(0.0, &init);
        let rhs_expect = DVector::from_vec(vec![r * y0 * (1.0 - y0 / k), 2.0 * y0]);
        rhs.assert_eq(&rhs_expect, 1e-10);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let rhs_jac = eqn.jac_mul(0.0, &init, &v);
        let rhs_jac_expect = DVector::from_vec(vec![r * (1.0 - y0 / k) - r * y0 / k, 1.0]);
        rhs_jac.assert_eq(&rhs_jac_expect, 1e-10);
        let mut mass_y = DVector::from_vec(vec![0.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        eqn.mass_inplace(0.0, &v, &mut mass_y);
        let mass_y_expect = DVector::from_vec(vec![1.0, 0.0]);
        mass_y.assert_eq(&mass_y_expect, 1e-10);

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
        state.assert_eq(&expected_state, 1e-5);
        let out = problem.eqn.out(t, &state);
        let out = DVector::from_vec(out.to_vec());
        let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
        out.assert_eq(&expected_out, 1e-5);
    }
}
