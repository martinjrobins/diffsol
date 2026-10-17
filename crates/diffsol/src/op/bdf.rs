use crate::{
    matrix::DenseMatrix, LinearOp, LinearOpTranspose, Matrix, MatrixSparsity, NonLinearOp,
    NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint,
    OdeEquationsImplicit, OdeEquationsImplicitAdjoint, OdeEquationsImplicitSens, Op, Scale, Vector,
    VectorViewMut,
};
use log::debug;
use num_traits::{One, ToPrimitive, Zero};
use std::{
    cell::{Ref, RefCell},
    ops::{Deref, MulAssign, SubAssign},
};

/// A NonLinearOp implementation of the BDF implicit equation system.
/// i.e. F(y) = M (y' + psi) - c * f(y) = 0
pub struct BdfCallable<Eqn: OdeEquationsImplicit> {
    pub(crate) eqn: Eqn,
    psi_neg_y0: Eqn::V,
    /// The coefficient `c = h * alpha` of `f` in the BDF equation.
    pub(crate) c: Eqn::T,
    tmp: RefCell<Eqn::V>,
    rhs_jac: RefCell<Eqn::M>,
    mass_jac: RefCell<Eqn::M>,
    jacobian_is_stale: RefCell<bool>,
    sparsity: Option<<Eqn::M as Matrix>::Sparsity>,
}

impl<Eqn: OdeEquationsImplicit> BdfCallable<Eqn> {
    pub fn clone_state(&self, eqn: Eqn) -> Self {
        Self {
            eqn,
            psi_neg_y0: self.psi_neg_y0.clone(),
            c: self.c,
            tmp: RefCell::new(self.tmp.borrow().clone()),
            rhs_jac: RefCell::new(self.rhs_jac.borrow().clone()),
            mass_jac: RefCell::new(self.mass_jac.borrow().clone()),
            jacobian_is_stale: RefCell::new(*self.jacobian_is_stale.borrow()),
            sparsity: self.sparsity.clone(),
        }
    }

    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    // M = I
    // dg = f(y)
    // g - y0 + psi = c * dg
    // g - y0 = c * dg - psi
    pub fn integrate_out<M: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        &self,
        dg: &Eqn::V,
        diff: &M,
        gamma: &[Eqn::T],
        alpha: &[Eqn::T],
        order: usize,
        d: &mut Eqn::V,
    ) {
        Self::set_psi(diff, gamma, alpha, order, d);
        d.axpy(self.c, dg, -Eqn::T::one());
    }
    /// `vec_ctx` is the context to allocate the state-shaped working vectors with. It is the
    /// equation's own context for the main equations, and the augmented context (one batch lane
    /// per augmented channel) when `eqn` is an augmented equation set.
    pub fn new_no_jacobian(eqn: Eqn, vec_ctx: Eqn::C) -> Self {
        let n = eqn.rhs().nstates();
        let ctx = eqn.context();
        let c = Eqn::T::zero();
        let psi_neg_y0 = <Eqn::V as Vector>::zeros(n, vec_ctx.clone());
        let jacobian_is_stale = RefCell::new(true);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(n, vec_ctx));
        let rhs_jac = RefCell::new(<Eqn::M as Matrix>::zeros(0, 0, ctx.clone()));
        let mass_jac = RefCell::new(<Eqn::M as Matrix>::zeros(0, 0, ctx.clone()));
        let sparsity = None;
        Self {
            eqn,
            psi_neg_y0,
            c,
            rhs_jac,
            mass_jac,
            jacobian_is_stale,
            tmp,
            sparsity,
        }
    }
    pub fn eqn(&self) -> &Eqn {
        &self.eqn
    }
    pub fn eqn_mut(&mut self) -> &mut Eqn {
        &mut self.eqn
    }
    pub fn jacobian_algebraic(&self) -> Option<(Eqn::M, Eqn::M)> {
        let rhs_jac = self.rhs_jac.borrow();
        let mass_jac = self.mass_jac.borrow();
        Some((rhs_jac.clone(), mass_jac.clone()))
    }
    pub fn rhs_jac(&self, x: &Eqn::V, t: Eqn::T) -> Ref<'_, Eqn::M> {
        {
            let mut rhs_jac = self.rhs_jac.borrow_mut();
            self.eqn.rhs().jacobian_inplace(x, t, &mut rhs_jac);
        }
        self.rhs_jac.borrow()
    }
    pub fn mass(&self, t: Eqn::T) -> Ref<'_, Eqn::M> {
        {
            let mut mass_jac = self.mass_jac.borrow_mut();
            self.eqn.mass().unwrap().matrix_inplace(t, &mut mass_jac);
        }
        self.mass_jac.borrow()
    }
    /// See [`Self::new_no_jacobian`] for `vec_ctx`.
    pub fn new(eqn: Eqn, vec_ctx: Eqn::C) -> Self {
        let n = eqn.rhs().nstates();
        let ctx = eqn.context();
        let c = Eqn::T::zero();
        let psi_neg_y0 = <Eqn::V as Vector>::zeros(n, vec_ctx.clone());
        let jacobian_is_stale = RefCell::new(true);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(n, vec_ctx));

        // create the mass and rhs jacobians according to the sparsity pattern
        let rhs_jac_sparsity = eqn.rhs().jacobian_sparsity();
        let rhs_jac = RefCell::new(Eqn::M::new_from_sparsity(
            n,
            n,
            rhs_jac_sparsity.map(|s| s.to_owned()),
            ctx.clone(),
        ));
        let sparsity = if let Some(rhs_jac_sparsity) = eqn.rhs().jacobian_sparsity() {
            if let Some(mass) = eqn.mass() {
                if let Some(mass_sparsity) = mass.sparsity() {
                    // have mass, use the union of the mass and rhs jacobians sparse patterns
                    Some(mass_sparsity.union(rhs_jac_sparsity.as_ref()).unwrap())
                } else {
                    // no mass sparsity, panic
                    panic!("Mass matrix must have a sparsity pattern if the rhs jacobian has a sparsity pattern");
                }
            } else {
                // no mass, use the identity
                let mass_sparsity = <Eqn::M as Matrix>::Sparsity::new_diagonal(n);
                Some(mass_sparsity.union(rhs_jac_sparsity.as_ref()).unwrap())
            }
        } else {
            None
        };

        let mass_jac = if eqn.mass().is_none() {
            // no mass matrix, so just use the identity
            Eqn::M::from_diagonal(&Eqn::V::from_element(n, Eqn::T::one(), ctx.clone()))
        } else {
            // mass is not constant, so just create a matrix with the correct sparsity
            Eqn::M::new_from_sparsity(
                n,
                n,
                eqn.mass().unwrap().sparsity().map(|s| s.to_owned()),
                ctx.clone(),
            )
        };

        let mass_jac = RefCell::new(mass_jac);

        Self {
            eqn,
            psi_neg_y0,
            c,
            rhs_jac,
            mass_jac,
            jacobian_is_stale,
            tmp,
            sparsity,
        }
    }

    #[cfg(test)]
    fn set_psi_neg_y0_direct(&mut self, psi_neg_y0: Eqn::V) {
        self.psi_neg_y0 = psi_neg_y0;
    }

    pub fn tmp(&self) -> Ref<'_, Eqn::V> {
        self.tmp.borrow()
    }

    pub fn set_c(&mut self, h: Eqn::T, alpha: Eqn::T) {
        self.c = h * alpha;
    }
    /// `gamma` must be `gamma[1..]`, i.e. offset to line up with the `1..order + 1` column
    /// range psi sums over.
    fn set_psi<M: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        diff: &M,
        gamma: &[Eqn::T],
        alpha: &[Eqn::T],
        order: usize,
        psi: &mut Eqn::V,
    ) {
        // update psi term as defined in second equation on page 9 of [1]
        debug_assert!(order >= 1, "BDF order is always at least one");
        diff.gemv_cols(1, order + 1, alpha[order], gamma, Eqn::T::zero(), psi);
    }
    pub fn set_psi_and_y0<M: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        &mut self,
        diff: &M,
        gamma: &[Eqn::T],
        alpha: &[Eqn::T],
        order: usize,
        y0: &Eqn::V,
    ) {
        Self::psi_neg_y0_into(diff, gamma, alpha, order, y0, &mut self.psi_neg_y0);
    }
    /// The history input `psi - y0` of the operator, from the differences `diff` and the
    /// prediction `y0`, written into `out`. This is linear in `(diff, y0)`, so it also maps their
    /// tangents. See [Self::set_psi] for `gamma`.
    pub fn psi_neg_y0_into<M: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        diff: &M,
        gamma: &[Eqn::T],
        alpha: &[Eqn::T],
        order: usize,
        y0: &Eqn::V,
        out: &mut Eqn::V,
    ) {
        Self::set_psi(diff, gamma, alpha, order, out);
        out.sub_assign(y0);
    }
    pub fn set_jacobian_is_stale(&self) {
        self.jacobian_is_stale.replace(true);
    }
}

impl<Eqn: OdeEquationsImplicit> Op for BdfCallable<Eqn> {
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

// dF(y)/dp = dM/dp (y - y0 + psi) + Ms - c * df(y)/dp - c df(y)/dy s = 0
// jac is M - c * df(y)/dy, same
// callable to solve for F(y) = M (y' + psi) - f(y) = 0
impl<Eqn: OdeEquationsImplicit> NonLinearOp for BdfCallable<Eqn> {
    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        let psi_neg_y0 = &self.psi_neg_y0;

        self.eqn.rhs().call_inplace(x, t, y);

        let c = self.c;
        // y = M tmp - c * y, for tmp = x + psi_neg_y0
        if let Some(mass) = self.eqn.mass() {
            let mut tmp = self.tmp.borrow_mut();
            tmp.for_each_elem([x, psi_neg_y0], |tmp, [x, psi], _lane, i| {
                *tmp = x[i] + psi[i]
            });
            mass.gemv_inplace(&tmp, t, -c, y);
        } else {
            // no mass matrix, so don't need `tmp`
            Eqn::V::for_each_elem_mut([y], [x, psi_neg_y0], move |[y], [x, psi], _lane, i| {
                *y = x[i] + psi[i] - c * *y
            });
        }
    }
}

impl<Eqn: OdeEquationsImplicit> NonLinearOpJacobian for BdfCallable<Eqn> {
    // (M - c * f'(y)) v
    fn jac_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.eqn.rhs().jac_mul_inplace(x, t, v, y);
        let c = self.c;
        // y = Mv - c y
        if let Some(mass) = self.eqn.mass() {
            mass.gemv_inplace(v, t, -c, y);
        } else {
            y.axpy(Eqn::T::one(), v, -c);
        }
    }

    // M - c * f'(y)
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if *self.jacobian_is_stale.borrow() {
            // calculate the mass and rhs jacobians
            let mut rhs_jac = self.rhs_jac.borrow_mut();
            self.eqn.rhs().jacobian_inplace(x, t, &mut rhs_jac);
            let c = self.c;
            debug!("Recomputing RHS Jacobian, c = {}", c.to_f64().unwrap());
            if self.eqn.mass().is_none() {
                let mass_jac = self.mass_jac.borrow();
                y.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
            } else {
                let mut mass_jac = self.mass_jac.borrow_mut();
                self.eqn.mass().unwrap().matrix_inplace(t, &mut mass_jac);
                y.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
            }
            self.jacobian_is_stale.replace(false);
        } else {
            // only c has changed, so just do the addition
            let rhs_jac = self.rhs_jac.borrow();
            let mass_jac = self.mass_jac.borrow();
            let c = self.c;
            debug!(
                "Reusing Jacobian, updating with new c = {}",
                c.to_f64().unwrap()
            );
            y.scale_add_and_assign(mass_jac.deref(), -c, rhs_jac.deref());
        }
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.sparsity.clone()
    }
}

/// `-(M - c f'(y))^T v`
impl<Eqn: OdeEquationsImplicitAdjoint> NonLinearOpAdjoint for BdfCallable<Eqn> {
    fn jac_transpose_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        // y = -f'(y)^T v, then y = -M^T v - c y
        self.eqn.rhs().jac_transpose_mul_inplace(x, t, v, y);
        let c = self.c;
        match self.eqn.mass() {
            Some(mass) => {
                mass.gemv_transpose_inplace(v, t, c, y);
                y.mul_assign(Scale(-Eqn::T::one()));
            }
            None => y.axpy(-Eqn::T::one(), v, -c),
        }
    }
}

/// `dF/dp v = -c f_p v`, taking the mass matrix to be independent of the parameters.
impl<Eqn: OdeEquationsImplicitSens> NonLinearOpSens for BdfCallable<Eqn> {
    fn sens_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.eqn.rhs().sens_mul_inplace(x, t, v, y);
        y.mul_assign(Scale(-self.c));
    }
}

/// `-(dF/dp)^T v = c f_p^T v`, taking the mass matrix to be independent of the parameters.
impl<Eqn: OdeEquationsImplicitAdjoint> NonLinearOpSensAdjoint for BdfCallable<Eqn> {
    fn sens_transpose_mul_inplace(&self, x: &Eqn::V, t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        // the rhs gives -f_p^T v
        self.eqn.rhs().sens_transpose_mul_inplace(x, t, v, y);
        y.mul_assign(Scale(-self.c));
    }
}

impl<Eqn: OdeEquationsImplicitSens> BdfCallable<Eqn> {
    /// The right-hand side `b` of the tangent of `F(y) = 0` at its solution `y = x`,
    /// `(M - c f'(x)) y_dot = b`, given the tangents of the history `diff_dot`, of the
    /// prediction `y0_dot` and of the parameters `p_dot`:
    /// `b = -(dF/d(psi - y0) (psi_dot - y0_dot) + dF/dp p_dot)`, with `dF/d(psi - y0) = M`.
    #[allow(clippy::too_many_arguments)]
    pub fn tangent_rhs<M: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        &self,
        x: &Eqn::V,
        t: Eqn::T,
        diff_dot: &M,
        gamma: &[Eqn::T],
        alpha: &[Eqn::T],
        order: usize,
        y0_dot: &Eqn::V,
        p_dot: &Eqn::V,
        b: &mut Eqn::V,
    ) {
        let mut psi_dot = y0_dot.clone();
        Self::psi_neg_y0_into(diff_dot, gamma, alpha, order, y0_dot, &mut psi_dot);
        self.sens_mul_inplace(x, t, p_dot, b);
        match self.eqn.mass() {
            Some(mass) => mass.gemv_inplace(&psi_dot, t, Eqn::T::one(), b),
            None => b.axpy(Eqn::T::one(), &psi_dot, Eqn::T::one()),
        }
        b.mul_assign(Scale(-Eqn::T::one()));
    }
}

impl<Eqn: OdeEquationsImplicitAdjoint> BdfCallable<Eqn> {
    /// The transpose of [Self::tangent_rhs]: given the cotangent `w` of `b` (so
    /// `(M - c f'(x))^T w` is the cotangent of `y`), add the cotangents of the history columns
    /// `1..=order` to `diff_bar`, of the prediction to `y0_bar` and of the parameters to `p_bar`.
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_inputs<M: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        &self,
        x: &Eqn::V,
        t: Eqn::T,
        w: &Eqn::V,
        gamma: &[Eqn::T],
        alpha: &[Eqn::T],
        order: usize,
        diff_bar: &mut M,
        y0_bar: &mut Eqn::V,
        p_bar: &mut Eqn::V,
    ) {
        let one = Eqn::T::one();
        // cotangent of psi - y0 is -M^T w
        let mut psi_bar = w.clone();
        if let Some(mass) = self.eqn.mass() {
            mass.call_transpose_inplace(w, t, &mut psi_bar);
        }
        psi_bar.mul_assign(Scale(-one));
        // psi - y0 = alpha[order] sum_{j=1}^{order} gamma_j diff_j - y0
        for j in 1..=order {
            diff_bar
                .column_mut(j)
                .axpy(alpha[order] * gamma[j - 1], &psi_bar, one);
        }
        y0_bar.axpy(-one, &psi_bar, one);

        // cotangent of p is -(dF/dp)^T w
        let mut fp_w = p_bar.clone();
        self.sens_transpose_mul_inplace(x, t, w, &mut fp_w);
        p_bar.axpy(one, &fp_w, one);
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::dense_nalgebra_serial::NalgebraMat;
    use crate::ode_equations::test_models::exponential_decay::exponential_decay_problem;
    use crate::vector::Vector;
    use crate::{
        DenseMatrix, Matrix, MatrixCommon, NalgebraVec, NonLinearOp, NonLinearOpJacobian,
        VectorView,
    };

    use super::BdfCallable;
    type Mcpu = NalgebraMat<f64>;
    type Vcpu = NalgebraVec<f64>;

    fn rand(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|i| ((i as f64 + 1.0) * seed).sin()).collect()
    }

    fn dot(a: &Vcpu, b: &Vcpu) -> f64 {
        let (a, b) = (a.clone_as_vec(), b.clone_as_vec());
        a.iter().zip(&b).map(|(x, y)| x * y).sum()
    }

    fn dot_cols(a: &Mcpu, b: &Mcpu) -> f64 {
        (0..a.ncols())
            .map(|j| dot(&a.column(j).into_owned(), &b.column(j).into_owned()))
            .sum()
    }

    /// The transposed and parameter derivatives of the BDF operator against its Jacobian and
    /// each other, with and without a mass matrix, on a model whose Jacobian is not symmetric.
    #[test]
    fn test_bdf_callable_derivatives() {
        use crate::ode_equations::test_models::ad_test_model::ad_test_problem;
        use crate::{
            NonLinearOpAdjoint, NonLinearOpSens, NonLinearOpSensAdjoint, OdeEquations, Op,
        };
        for dae in [false, true] {
            let problem = ad_test_problem::<Mcpu>([1.5, 3.0, 1.0], dae, 1e-8);
            let mut op = BdfCallable::new(&problem.eqn, *problem.context());
            let ctx = *problem.context();
            op.set_c(0.1, 0.7);
            let (n, np) = (problem.eqn.rhs().nstates(), problem.eqn.rhs().nparams());
            let x = Vcpu::from_vec(vec![0.6, 0.3, 0.1], ctx);
            let t = 0.0;
            let w = Vcpu::from_vec(rand(n, 1.7), ctx);
            let p_dot = Vcpu::from_vec(rand(np, 2.3), ctx);

            // -J^T w against the dense Jacobian
            let jac = op.jacobian(&x, t);
            let mut jt_w = Vcpu::zeros(n, ctx);
            op.jac_transpose_mul_inplace(&x, t, &w, &mut jt_w);
            let w_vals = w.clone_as_vec();
            for (i, got) in jt_w.clone_as_vec().into_iter().enumerate() {
                let expected: f64 = -(0..n).map(|k| jac.get_index(k, i) * w_vals[k]).sum::<f64>();
                assert!(
                    (got - expected).abs() < 1e-12,
                    "dae = {dae}: {got} != {expected}"
                );
            }

            // <w, F_p p_dot> = -<-F_p^T w, p_dot>
            let mut fp = Vcpu::zeros(n, ctx);
            op.sens_mul_inplace(&x, t, &p_dot, &mut fp);
            let mut fpt = Vcpu::zeros(np, ctx);
            op.sens_transpose_mul_inplace(&x, t, &w, &mut fpt);
            assert!((dot(&w, &fp) + dot(&fpt, &p_dot)).abs() < 1e-12);

            // tangent_rhs against adjoint_inputs: <w, b> = <diff_bar, diff_dot> + <y0_bar, y0_dot>
            // + <p_bar, p_dot>
            let (gamma, alpha) = ([1.0, 1.5, 1.8], [0.0, 0.8, 0.9, 1.0]);
            let order = 2;
            let ncols = 6;
            let diff_dot = Mcpu::from_vec(n, ncols, rand(n * ncols, 0.4), ctx);
            let y0_dot = Vcpu::from_vec(rand(n, 3.1), ctx);
            let mut b = Vcpu::zeros(n, ctx);
            op.tangent_rhs(
                &x, t, &diff_dot, &gamma, &alpha, order, &y0_dot, &p_dot, &mut b,
            );
            let mut diff_bar = Mcpu::zeros(n, ncols, ctx);
            let mut y0_bar = Vcpu::zeros(n, ctx);
            let mut p_bar = Vcpu::zeros(np, ctx);
            op.adjoint_inputs(
                &x,
                t,
                &w,
                &gamma,
                &alpha,
                order,
                &mut diff_bar,
                &mut y0_bar,
                &mut p_bar,
            );
            let lhs = dot(&w, &b);
            let rhs = dot_cols(&diff_bar, &diff_dot) + dot(&y0_bar, &y0_dot) + dot(&p_bar, &p_dot);
            assert!((lhs - rhs).abs() < 1e-12, "dae = {dae}: {lhs} != {rhs}");
        }
    }

    #[test]
    fn test_bdf_callable() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let mut bdf_callable = BdfCallable::new(&problem.eqn, *problem.context());
        let ctx = problem.context();
        let c = 0.1;
        let phi_neg_y0 = Vcpu::from_vec(vec![1.1, 1.2], *ctx);
        bdf_callable.c = c;
        bdf_callable.set_psi_neg_y0_direct(phi_neg_y0);
        // check that the bdf function is correct
        let y = Vcpu::from_vec(vec![1.0, 1.0], *ctx);
        let t = 0.0;
        let mut y_out = Vcpu::from_vec(vec![0.0, 0.0], *ctx);

        // F(y) = M (y - y0 + psi) - c * f(y)
        // M = |1 0|
        //     |0 1|
        // y = |1|
        //     |1|
        // f(y) = |-0.1|
        //        |-0.1|
        //  i.e. F(y) = |1 0| |2.1| - 0.1 * |-0.1| =  |2.11|
        //              |0 1| |2.2|         |-0.1|    |2.21|
        bdf_callable.call_inplace(&y, t, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![2.11, 2.21], *ctx);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        let v = Vcpu::from_vec(vec![1.0, 1.0], *ctx);
        // f'(y)v = |-0.1|
        //          |-0.1|
        // Mv - c * f'(y) v = |1 0| |1| - 0.1 * |-0.1| = |1.01|
        //                    |0 1| |1|         |-0.1|   |1.01|
        bdf_callable.jac_mul_inplace(&y, t, &v, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![1.01, 1.01], *ctx);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        // J = M - c * f'(y) = |1 0| - 0.1 * |-0.1 0| = |1.01 0|
        //                     |0 1|         |0 -0.1|   |0 1.01|
        let jac = bdf_callable.jacobian(&y, t);
        assert_eq!(jac.get_index(0, 0), 1.01);
        assert_eq!(jac.get_index(0, 1), 0.0);
        assert_eq!(jac.get_index(1, 0), 0.0);
        assert_eq!(jac.get_index(1, 1), 1.01);
    }
}
