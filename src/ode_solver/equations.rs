use num_traits::Zero;
use std::{cell::RefCell, rc::Rc};

use crate::{
    jacobian::{find_non_zero_entries, JacobianColoring},
    op::{closure::Closure, linear_closure::LinearClosure, Op},
    Matrix, NonLinearOp, Vector, VectorIndex,
};
use num_traits::One;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct OdeEquationsStatistics {
    pub number_of_rhs_evals: usize,
    pub number_of_jac_mul_evals: usize,
    pub number_of_mass_evals: usize,
    pub number_of_mass_matrix_evals: usize,
    pub number_of_jacobian_matrix_evals: usize,
}

impl Default for OdeEquationsStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl OdeEquationsStatistics {
    pub fn new() -> Self {
        Self {
            number_of_rhs_evals: 0,
            number_of_jac_mul_evals: 0,
            number_of_mass_evals: 0,
            number_of_mass_matrix_evals: 0,
            number_of_jacobian_matrix_evals: 0,
        }
    }
}

pub trait OdeEquations: Op {
    /// This must be called first
    fn set_params(&mut self, p: Self::V);

    /// calculates $F(y)$ where $y$ is given in `y` and stores the result in `rhs_y`
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V);

    /// calculates $y = J(x)v$
    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V);

    /// initializes `y` with the initial condition
    fn init(&self, t: Self::T) -> Self::V;

    fn rhs(&self, t: Self::T, y: &Self::V) -> Self::V {
        let mut rhs_y = Self::V::zeros(self.nstates());
        self.rhs_inplace(t, y, &mut rhs_y);
        rhs_y
    }

    fn jac_mul(&self, t: Self::T, x: &Self::V, v: &Self::V) -> Self::V {
        let mut rhs_jac_y = Self::V::zeros(self.nstates());
        self.rhs_jac_inplace(t, x, v, &mut rhs_jac_y);
        rhs_jac_y
    }

    /// rhs jacobian matrix J(x), re-use jacobian calculation from NonLinearOp
    fn jacobian_matrix(&self, x: &Self::V, t: Self::T) -> Self::M {
        let rhs_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y_rhs: &mut Self::V| {
            self.rhs_inplace(t, x, y_rhs);
        };
        let rhs_jac_inplace =
            |x: &Self::V, _p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V| {
                self.rhs_jac_inplace(t, x, v, y);
            };
        let dummy_p = Rc::new(Self::V::zeros(0));
        let closure = Closure::new(
            rhs_inplace,
            rhs_jac_inplace,
            self.nstates(),
            self.nstates(),
            dummy_p,
        );
        closure.jacobian(x, t)
    }

    /// mass matrix action: y = M(t)
    fn mass_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        // assume identity mass matrix
        y.copy_from(v);
    }

    /// returns the indices of the algebraic state variables
    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        // assume identity mass matrix
        <Self::V as Vector>::Index::zeros(0)
    }

    fn mass_matrix(&self, _t: Self::T) -> Self::M {
        // assume identity mass matrix
        Self::M::from_diagonal(&Self::V::from_element(self.nstates(), Self::T::one()))
    }

    fn get_statistics(&self) -> OdeEquationsStatistics {
        OdeEquationsStatistics::new()
    }
}

pub struct OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: F,
    rhs_jac: G,
    mass: H,
    init: I,
    p: Rc<M::V>,
    nstates: usize,
    jacobian_coloring: Option<JacobianColoring>,
    mass_coloring: Option<JacobianColoring>,
    statistics: RefCell<OdeEquationsStatistics>,
}

impl<M, F, G, H, I> OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode_with_mass(
        rhs: F,
        rhs_jac: G,
        mass: H,
        init: I,
        p: M::V,
        t0: M::T,
        use_coloring: bool,
    ) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);
        let statistics = RefCell::default();
        let mut ret = Self {
            rhs,
            rhs_jac,
            mass,
            init,
            p: p.clone(),
            nstates,
            jacobian_coloring: None,
            mass_coloring: None,
            statistics,
        };
        let (jacobian_coloring, mass_coloring) = if use_coloring {
            let rhs_inplace = |x: &M::V, _p: &M::V, t: M::T, y_rhs: &mut M::V| {
                ret.rhs_inplace(t, x, y_rhs);
            };
            let rhs_jac_inplace = |x: &M::V, _p: &M::V, t: M::T, v: &M::V, y: &mut M::V| {
                ret.rhs_jac_inplace(t, x, v, y);
            };
            let mass_inplace = |x: &M::V, _p: &M::V, t: M::T, y: &mut M::V| {
                ret.mass_inplace(t, x, y);
            };
            let op =
                Closure::<M, _, _>::new(rhs_inplace, rhs_jac_inplace, nstates, nstates, p.clone());
            let jacobian_coloring = Some(JacobianColoring::new(&op, &y0, t0));
            let op = LinearClosure::<M, _>::new(mass_inplace, nstates, nstates, p.clone());
            let mass_coloring = Some(JacobianColoring::new(&op, &y0, t0));
            (jacobian_coloring, mass_coloring)
        } else {
            (None, None)
        };
        ret.jacobian_coloring = jacobian_coloring;
        ret.mass_coloring = mass_coloring;
        ret
    }
}

// impl Op
impl<M, F, G, H, I> Op for OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.p.len()
    }
    fn nstates(&self) -> usize {
        self.nstates
    }
}

impl<M, F, G, H, I> OdeEquations for OdeSolverEquations<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs)(y, p, t, rhs_y);
        self.statistics.borrow_mut().number_of_rhs_evals += 1;
    }

    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs_jac)(x, p, t, v, y);
        self.statistics.borrow_mut().number_of_jac_mul_evals += 1;
    }

    fn mass_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.mass)(v, p, t, y);
        self.statistics.borrow_mut().number_of_mass_evals += 1;
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }

    fn jacobian_matrix(&self, x: &Self::V, t: Self::T) -> Self::M {
        self.statistics.borrow_mut().number_of_jacobian_matrix_evals += 1;
        let rhs_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y_rhs: &mut Self::V| {
            self.rhs_inplace(t, x, y_rhs);
        };
        let rhs_jac_inplace =
            |x: &Self::V, _p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V| {
                self.rhs_jac_inplace(t, x, v, y);
            };
        let op = Closure::<M, _, _>::new(
            rhs_inplace,
            rhs_jac_inplace,
            self.nstates,
            self.nstates,
            self.p.clone(),
        );
        let triplets = if let Some(coloring) = &self.jacobian_coloring {
            coloring.find_non_zero_entries(&op, x, t)
        } else {
            find_non_zero_entries(&op, x, t)
        };
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }

    fn mass_matrix(&self, t: Self::T) -> Self::M {
        self.statistics.borrow_mut().number_of_mass_matrix_evals += 1;
        let mass_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y: &mut Self::V| {
            self.mass_inplace(t, x, y);
        };
        let op =
            LinearClosure::<M, _>::new(mass_inplace, self.nstates, self.nstates, self.p.clone());
        let triplets = if let Some(coloring) = &self.mass_coloring {
            coloring.find_non_zero_entries(&op, &self.init(t), t)
        } else {
            find_non_zero_entries(&op, &self.init(t), t)
        };
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }

    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        let mass = self.mass_matrix(Self::T::zero());
        let diag: Self::V = mass.diagonal();
        diag.filter_indices(|x| x == Self::T::zero())
    }

    fn get_statistics(&self) -> OdeEquationsStatistics {
        self.statistics.borrow().clone()
    }
}

pub struct OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    rhs: F,
    rhs_jac: G,
    init: I,
    p: Rc<M::V>,
    nstates: usize,
    coloring: Option<JacobianColoring>,
    statistics: RefCell<OdeEquationsStatistics>,
}

impl<M, F, G, I> OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new_ode(rhs: F, rhs_jac: G, init: I, p: M::V, t0: M::T, use_coloring: bool) -> Self {
        let y0 = init(&p, M::T::zero());
        let nstates = y0.len();
        let p = Rc::new(p);

        let statistics = RefCell::default();
        let mut ret = Self {
            rhs,
            rhs_jac,
            init,
            p: p.clone(),
            nstates,
            coloring: None,
            statistics,
        };
        let coloring = if use_coloring {
            let rhs_inplace = |x: &M::V, _p: &M::V, t: M::T, y_rhs: &mut M::V| {
                ret.rhs_inplace(t, x, y_rhs);
            };
            let rhs_jac_inplace = |x: &M::V, _p: &M::V, t: M::T, v: &M::V, y: &mut M::V| {
                ret.rhs_jac_inplace(t, x, v, y);
            };
            let op =
                Closure::<M, _, _>::new(rhs_inplace, rhs_jac_inplace, nstates, nstates, p.clone());
            Some(JacobianColoring::new(&op, &y0, t0))
        } else {
            None
        };
        ret.coloring = coloring;
        ret
    }
}

// impl Op
impl<M, F, G, I> Op for OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        self.p.len()
    }
    fn nstates(&self) -> usize {
        self.nstates
    }
}

impl<M, F, G, I> OdeEquations for OdeSolverEquationsMassI<M, F, G, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, M::T) -> M::V,
{
    fn rhs_inplace(&self, t: Self::T, y: &Self::V, rhs_y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs)(y, p, t, rhs_y);
        self.statistics.borrow_mut().number_of_rhs_evals += 1;
    }

    fn rhs_jac_inplace(&self, t: Self::T, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        let p = self.p.as_ref();
        (self.rhs_jac)(x, p, t, v, y);
        self.statistics.borrow_mut().number_of_jac_mul_evals += 1;
    }

    fn init(&self, t: Self::T) -> Self::V {
        let p = self.p.as_ref();
        (self.init)(p, t)
    }

    fn set_params(&mut self, p: Self::V) {
        self.p = Rc::new(p);
    }

    fn algebraic_indices(&self) -> <Self::V as Vector>::Index {
        <Self::V as Vector>::Index::zeros(0)
    }

    fn jacobian_matrix(&self, x: &Self::V, t: Self::T) -> Self::M {
        self.statistics.borrow_mut().number_of_jacobian_matrix_evals += 1;
        let rhs_inplace = |x: &Self::V, _p: &Self::V, t: Self::T, y_rhs: &mut Self::V| {
            self.rhs_inplace(t, x, y_rhs);
        };
        let rhs_jac_inplace =
            |x: &Self::V, _p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V| {
                self.rhs_jac_inplace(t, x, v, y);
            };
        let op = Closure::<M, _, _>::new(
            rhs_inplace,
            rhs_jac_inplace,
            self.nstates,
            self.nstates,
            self.p.clone(),
        );
        let triplets = if let Some(coloring) = &self.coloring {
            coloring.find_non_zero_entries(&op, x, t)
        } else {
            find_non_zero_entries(&op, x, t)
        };
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }
    fn get_statistics(&self) -> OdeEquationsStatistics {
        self.statistics.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::ode_solver::equations::OdeEquations;
    use crate::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use crate::ode_solver::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::vector::Vector;

    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn ode_equation_test() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let y = DVector::from_vec(vec![1.0, 1.0]);
        let rhs_y = problem.eqn.rhs(0.0, &y);
        let expect_rhs_y = DVector::from_vec(vec![-0.1, -0.1]);
        rhs_y.assert_eq(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.jac_mul(0.0, &y, &y);
        let expect_jac_rhs_y = Vcpu::from_vec(vec![-0.1, -0.1]);
        jac_rhs_y.assert_eq(&expect_jac_rhs_y, 1e-10);
        let mass = problem.eqn.mass_matrix(0.0);
        assert_eq!(mass[(0, 0)], 1.0);
        assert_eq!(mass[(1, 1)], 1.0);
        assert_eq!(mass[(0, 1)], 0.);
        assert_eq!(mass[(1, 0)], 0.);
        let jac = problem.eqn.jacobian_matrix(&y, 0.0);
        assert_eq!(jac[(0, 0)], -0.1);
        assert_eq!(jac[(1, 1)], -0.1);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
    }

    #[test]
    fn ode_with_mass_test() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let rhs_y = problem.eqn.rhs(0.0, &y);
        let expect_rhs_y = DVector::from_vec(vec![-0.1, -0.1, 0.0]);
        rhs_y.assert_eq(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.jac_mul(0.0, &y, &y);
        let expect_jac_rhs_y = Vcpu::from_vec(vec![-0.1, -0.1, 0.0]);
        jac_rhs_y.assert_eq(&expect_jac_rhs_y, 1e-10);
        let mass = problem.eqn.mass_matrix(0.0);
        assert_eq!(mass[(0, 0)], 1.);
        assert_eq!(mass[(1, 1)], 1.);
        assert_eq!(mass[(2, 2)], 0.);
        assert_eq!(mass[(0, 1)], 0.);
        assert_eq!(mass[(1, 0)], 0.);
        assert_eq!(mass[(0, 2)], 0.);
        assert_eq!(mass[(2, 0)], 0.);
        assert_eq!(mass[(1, 2)], 0.);
        assert_eq!(mass[(2, 1)], 0.);
        let jac = problem.eqn.jacobian_matrix(&y, 0.0);
        assert_eq!(jac[(0, 0)], -0.1);
        assert_eq!(jac[(1, 1)], -0.1);
        assert_eq!(jac[(2, 2)], 1.0);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(0, 2)], 0.0);
        assert_eq!(jac[(2, 0)], 0.0);
        assert_eq!(jac[(1, 2)], 0.0);
        assert_eq!(jac[(2, 1)], -1.0);
    }
}
