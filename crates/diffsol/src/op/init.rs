use crate::{
    jacobian::JacobianColoring, scale, Context, LinearOp, Matrix, MatrixSparsityRef,
    NonLinearOpJacobian, OdeEquationsImplicit, Vector, VectorIndex,
};
use num_traits::{One, Zero};
use std::cell::RefCell;

use super::{NonLinearOp, Op};

enum AlgebraicColumns<M: Matrix> {
    Coloring(JacobianColoring<M>),
    Copy {
        rhs_jac: RefCell<M>,
        col: RefCell<M::V>,
        cols: Vec<usize>,
    },
}

/// NonLinearOp implementation of consistent initial conditions for an ODE system.
///
/// We calculate consistent initial conditions following the approach of
/// Brown, P. N., Hindmarsh, A. C., & Petzold, L. R. (1998). Consistent initial condition calculation for differential-algebraic systems. SIAM Journal on Scientific Computing, 19(5), 1495-1512.
pub struct InitOp<'a, Eqn: OdeEquationsImplicit> {
    eqn: &'a Eqn,
    pub y0: RefCell<Eqn::V>,
    pub algebraic_indices: <Eqn::V as Vector>::Index,
    neg_mass: Eqn::M,
    jac: Eqn::M,
    alg_columns: AlgebraicColumns<Eqn::M>,
    v_alg: RefCell<Eqn::V>,
}

impl<'a, Eqn: OdeEquationsImplicit> InitOp<'a, Eqn> {
    pub fn new(
        eqn: &'a Eqn,
        t0: Eqn::T,
        y0: &Eqn::V,
        algebraic_indices: <Eqn::V as Vector>::Index,
    ) -> Self {
        let n = eqn.rhs().nstates();
        let ctx = eqn.context().clone();
        let mass = eqn.mass().unwrap().matrix(t0);

        // equations are:
        // h(t, u, v, du) = 0
        // g(t, u, v) = 0
        // where u are differential states, v are algebraic states.
        // choose h = -M_u du + f(u, v), where M_u are the differential states of the mass matrix
        // want to solve for du, v, so jacobian is
        // J = (-M_u, df/dv)
        //     (0,    dg/dv)
        // note rhs_jac = (df/du df/dv)
        //                (dg/du dg/dv)
        // according to the algebraic indices.
        let [(m_u, _), _, _, _] = mass.split(&algebraic_indices);
        let neg_mass_u = m_u * scale(-Eqn::T::one());
        let zero_ll = <Eqn::M as Matrix>::zeros(
            algebraic_indices.len(),
            n - algebraic_indices.len(),
            ctx.clone(),
        );
        let zero_ur = <Eqn::M as Matrix>::zeros(
            n - algebraic_indices.len(),
            algebraic_indices.len(),
            ctx.clone(),
        );
        let zero_lr = <Eqn::M as Matrix>::zeros(
            algebraic_indices.len(),
            algebraic_indices.len(),
            ctx.clone(),
        );
        let neg_mass = Eqn::M::combine(
            &neg_mass_u,
            &zero_ur,
            &zero_ll,
            &zero_lr,
            &algebraic_indices,
        );

        let rhs_jac = Eqn::M::new_from_sparsity(n, n, eqn.rhs().jacobian_sparsity(), ctx.clone());
        let alg_cols = algebraic_indices.clone_as_vec();
        let alg_entries = rhs_jac.sparsity().map(|sparsity| {
            let mut is_algebraic = vec![false; n];
            for &j in &alg_cols {
                is_algebraic[j] = true;
            }
            sparsity
                .indices()
                .into_iter()
                .filter(|&(_, j)| is_algebraic[j])
                .collect::<Vec<_>>()
        });
        let (jac, alg_columns) = match alg_entries {
            Some(alg_entries) => {
                let [_, (dfdv, _), _, (dgdv, _)] = rhs_jac.split(&algebraic_indices);
                let jac = Eqn::M::combine(&neg_mass_u, &dfdv, &zero_ll, &dgdv, &algebraic_indices);
                let jac_sparsity = jac
                    .sparsity()
                    .map(|s| s.to_owned())
                    .expect("a sparse jacobian has a sparsity pattern");
                let coloring = JacobianColoring::new(&jac_sparsity, &alg_entries, ctx.clone());
                (jac, AlgebraicColumns::Coloring(coloring))
            }
            None => (
                neg_mass.clone(),
                AlgebraicColumns::Copy {
                    rhs_jac: RefCell::new(rhs_jac),
                    col: RefCell::new(Eqn::V::zeros(n, ctx)),
                    cols: alg_cols,
                },
            ),
        };

        let v_alg = RefCell::new(Eqn::V::zeros(n, y0.context().clone()));
        let y0 = y0.clone();
        let y0 = RefCell::new(y0);
        Self {
            eqn,
            y0,
            neg_mass,
            jac,
            alg_columns,
            algebraic_indices,
            v_alg,
        }
    }

    pub fn scatter_soln(&self, soln: &Eqn::V, y: &mut Eqn::V, dy: &mut Eqn::V) {
        let tmp = dy.clone();
        dy.copy_from(soln);
        dy.copy_from_indices(&tmp, &self.algebraic_indices);
        y.copy_from_indices(soln, &self.algebraic_indices);
    }
}

impl<Eqn: OdeEquationsImplicit> Op for InitOp<'_, Eqn> {
    type V = Eqn::V;
    type T = Eqn::T;
    type M = Eqn::M;
    type C = Eqn::C;
    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn context(&self) -> &Self::C {
        self.eqn.context()
    }
}

impl<Eqn: OdeEquationsImplicit> NonLinearOp for InitOp<'_, Eqn> {
    // -M_u du + f(u, v)
    // g(t, u, v)
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        // input x = (du, v)
        // self.y0 = (u, v)
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(x, &self.algebraic_indices);

        // y = (f; g)
        self.eqn.rhs().call_inplace(&y0, t, y);

        // y = -M x + y
        self.neg_mass.gemv(Eqn::T::one(), x, Eqn::T::one(), y);
    }
}

impl<Eqn: OdeEquationsImplicit> NonLinearOpJacobian for InitOp<'_, Eqn> {
    // J v = (-M_u v_u + df/dv v_v; dg/dv v_v) at x = (du, v)
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(x, &self.algebraic_indices);

        // v can have more batch lanes than the equations
        let mut v_alg = self.v_alg.borrow_mut();
        if v_alg.context().nbatch() != v.context().nbatch() {
            *v_alg = Eqn::V::zeros(v.len(), v.context().clone());
        }
        v_alg.copy_from_indices(v, &self.algebraic_indices);
        self.eqn.rhs().jac_mul_inplace(&y0, t, &v_alg, y);

        // y = -M v + y
        self.neg_mass.gemv(Eqn::T::one(), v, Eqn::T::one(), y);
    }

    // J at x = (du, v)
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(x, &self.algebraic_indices);
        y.copy_from(&self.jac);
        match &self.alg_columns {
            AlgebraicColumns::Coloring(coloring) => {
                coloring.jacobian_inplace(&self.eqn.rhs(), &y0, t, y)
            }
            AlgebraicColumns::Copy { rhs_jac, col, cols } => {
                let mut rhs_jac = rhs_jac.borrow_mut();
                let mut col = col.borrow_mut();
                self.eqn.rhs().jacobian_inplace(&y0, t, &mut rhs_jac);
                for &j in cols {
                    col.fill(Eqn::T::zero());
                    rhs_jac.add_column_to_vector(j, &mut col);
                    y.set_column(j, &col);
                }
            }
        }
    }

    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.jac.sparsity().map(|s| s.to_owned())
    }
}

#[cfg(test)]
mod tests {

    use crate::ode_equations::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::ode_equations::test_models::nonlinear_algebraic::nonlinear_algebraic_problem;
    use crate::op::init::InitOp;
    use crate::vector::Vector;
    use crate::{
        Context, DenseMatrix, FaerSparseMat, LinearOp, Matrix, NalgebraMat, NalgebraVec,
        NonLinearOp, NonLinearOpJacobian, OdeEquations,
    };
    use num_traits::{FromPrimitive, One, Zero};

    type Mcpu = NalgebraMat<f64>;
    type Vcpu = NalgebraVec<f64>;

    #[test]
    fn test_initop() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y0 = Vcpu::from_vec(vec![1.0, 2.0, 3.0], *problem.context());
        let dy0 = Vcpu::from_vec(vec![4.0, 5.0, 6.0], *problem.context());
        let t = 0.0;
        let (algebraic_indices, _) = problem
            .eqn()
            .mass()
            .unwrap()
            .matrix(t)
            .partition_indices_by_zero_diagonal();

        let initop = InitOp::new(&problem.eqn, t, &y0, algebraic_indices);
        // check that the init function is correct
        let mut y_out = Vcpu::from_vec(vec![0.0, 0.0, 0.0], *problem.context());

        // -M_u du + f(u, v)
        // g(t, u, v)
        // M = |1 0 0|
        //     |0 1 0|
        //     |0 0 0|
        //
        // y = |1| (u)
        //     |2| (u)
        //     |3| (v)
        // dy = |4| (du)
        //      |5| (du)
        //      |6| (dv)
        // i.e. f(u, v) = -0.1 u = |-0.1|
        //                         |-0.2|
        //      g(u, v) = v - u = |1|
        //      M_u = |1 0|
        //            |0 1|
        //  i.e. F(y) = |-1 * 4 + -0.1 * 1| = |-4.1|
        //              |-1 * 5 + -0.1 * 2|   |-5.2|
        //              |2 - 1|               |1|
        let du_v = Vcpu::from_vec(vec![dy0[0], dy0[1], y0[2]], *problem.context());
        initop.call_inplace(&du_v, t, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![-4.1, -5.2, 1.0], *problem.context());
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        // df/dv = |0|
        //         |0|
        // dg/dv = |1|
        // J = (-M_u, df/dv) = |-1 0 0|
        //                   = |0 -1 0|
        //     (0,    dg/dv) = |0 0 1|
        let jac = initop.jacobian(&du_v, t);
        assert_eq!(jac.get_index(0, 0), -1.0);
        assert_eq!(jac.get_index(0, 1), 0.0);
        assert_eq!(jac.get_index(0, 2), 0.0);
        assert_eq!(jac.get_index(1, 0), 0.0);
        assert_eq!(jac.get_index(1, 1), -1.0);
        assert_eq!(jac.get_index(1, 2), 0.0);
        assert_eq!(jac.get_index(2, 0), 0.0);
        assert_eq!(jac.get_index(2, 1), 0.0);
        assert_eq!(jac.get_index(2, 2), 1.0);
    }

    #[test]
    fn test_initop_jacobian_follows_the_iterate() {
        let problem = nonlinear_algebraic_problem::<Mcpu>();
        let y0 = Vcpu::from_vec(vec![2.0, 5.0], *problem.context());
        let t = 0.0;
        let (algebraic_indices, _) = problem
            .eqn()
            .mass()
            .unwrap()
            .matrix(t)
            .partition_indices_by_zero_diagonal();
        let initop = InitOp::new(&problem.eqn, t, &y0, algebraic_indices);

        // J = (-1, df/dv)  = (-1,  1)
        //     (0,  dg/dv)    (0, -(1 + 3 v^2))
        for v in [5.0, 1.0] {
            let du_v = Vcpu::from_vec(vec![0.0, v], *problem.context());
            let jac = initop.jacobian(&du_v, t);
            assert_eq!(jac.get_index(0, 0), -1.0);
            assert_eq!(jac.get_index(0, 1), 1.0);
            assert_eq!(jac.get_index(1, 0), 0.0);
            assert_eq!(jac.get_index(1, 1), -(1.0 + 3.0 * v * v));
        }
    }

    #[test]
    fn test_initop_jac_mul_nalgebra() {
        test_initop_jac_mul_matches_jacobian::<Mcpu>();
    }

    #[test]
    fn test_initop_jac_mul_faer_sparse() {
        test_initop_jac_mul_matches_jacobian::<FaerSparseMat<f64>>();
    }

    fn test_initop_jac_mul_matches_jacobian<M: Matrix + 'static>() {
        let problem = nonlinear_algebraic_problem::<M>();
        let ctx = problem.context().clone();
        let from = |values: Vec<f64>, ctx: M::C| {
            M::V::from_vec(
                values
                    .into_iter()
                    .map(|v| M::T::from_f64(v).unwrap())
                    .collect(),
                ctx,
            )
        };
        let t = M::T::zero();
        let y0 = from(vec![2.0, 5.0], ctx.clone());
        let (algebraic_indices, _) = problem
            .eqn()
            .mass()
            .unwrap()
            .matrix(t)
            .partition_indices_by_zero_diagonal();
        let initop = InitOp::new(&problem.eqn, t, &y0, algebraic_indices);

        let iterates = [5.0, 1.0].map(|v| from(vec![0.0, v], ctx.clone()));
        let jacs = iterates
            .iter()
            .map(|du_v| initop.jacobian(du_v, t))
            .collect::<Vec<_>>();
        for nbatch in [1, 2] {
            let lanes = ctx.clone_with_nbatch(nbatch).unwrap();
            let v = from((1..=2 * nbatch).map(|i| i as f64).collect(), lanes.clone());
            for (du_v, jac) in iterates.iter().zip(&jacs) {
                let mut jv_expect = M::V::zeros(2, lanes.clone());
                jac.gemv(M::T::one(), &v, M::T::zero(), &mut jv_expect);

                let mut jv = from(vec![100.0; 2 * nbatch], lanes.clone());
                initop.jac_mul_inplace(du_v, t, &v, &mut jv);
                jv.assert_eq_st(&jv_expect, M::T::from_f64(1e-12).unwrap());
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(deprecated)]
    #[test]
    fn test_initop_batched() {
        use crate::{
            ode_equations::test_models::exponential_decay_with_algebraic::{
                exponential_decay_with_algebraic_batched,
                exponential_decay_with_algebraic_init_batched,
                exponential_decay_with_algebraic_jacobian_batched,
                exponential_decay_with_algebraic_mass_batched,
            },
            CudaContext, CudaMat, CudaVec, OdeBuilder,
        };

        let nbatch = 2;
        let ctx = CudaContext::default().with_nbatch(nbatch);
        let p_f64 = vec![0.1, 0.2];
        let problem = OdeBuilder::<CudaMat<f64>>::new()
            .context(ctx.clone())
            .p(p_f64)
            .rhs_implicit(
                exponential_decay_with_algebraic_batched::<CudaMat<f64>>,
                exponential_decay_with_algebraic_jacobian_batched::<CudaMat<f64>>,
            )
            .mass(exponential_decay_with_algebraic_mass_batched::<CudaMat<f64>>)
            .init(
                exponential_decay_with_algebraic_init_batched::<CudaMat<f64>>,
                3,
            )
            .build()
            .unwrap();

        let y0 = CudaVec::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ctx.clone());
        let t = 0.0;
        let (algebraic_indices, _) = problem
            .eqn()
            .mass()
            .unwrap()
            .matrix(t)
            .partition_indices_by_zero_diagonal();

        let initop = InitOp::new(&problem.eqn, t, &y0, algebraic_indices);

        let du_v = CudaVec::from_vec(vec![4.0, 5.0, 1.0, 4.0, 5.0, 1.0], ctx.clone());
        let mut y_out = CudaVec::zeros(3, ctx.clone());
        initop.call_inplace(&du_v, t, &mut y_out);
        let expect = CudaVec::from_vec(vec![-4.1, -5.1, 0.0, -4.2, -5.2, 0.0], ctx.clone());
        y_out.assert_eq_st(&expect, 1e-10);

        let x0 = CudaVec::from_vec(vec![-0.1, -0.1, 1.0, -0.2, -0.2, 1.0], ctx.clone());
        let mut zero_out = CudaVec::zeros(3, ctx);
        initop.call_inplace(&x0, t, &mut zero_out);
        let expect_zero = CudaVec::from_vec(vec![0.0; 6], zero_out.context().clone());
        zero_out.assert_eq_st(&expect_zero, 1e-10);
    }
}
