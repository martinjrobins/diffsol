//
//heatres: heat equation system residual function
//This uses 5-point central differencing on the interior points, and
//includes algebraic equations for the boundary values.
//So for each interior point, the residual component has the form
//   res_i = u'_i - (central difference)_i
//while for each boundary point, it is res_i = u_i.

use crate::{
    find_jacobian_non_zeros, find_matrix_non_zeros, ode_solver::problem::OdeSolverSolution,
    scalar::Scalar, ConstantOp, Context, JacobianColoring, LinearOp, Matrix, MatrixSparsity,
    NonLinearOp, NonLinearOpJacobian, OdeBuilder, OdeEquations, OdeEquationsImplicit,
    OdeEquationsRef, OdeSolverProblem, Op, ParameterisedOp, UnitCallable, Vector,
};
use num_traits::{FromPrimitive, One, Zero};

#[cfg(feature = "diffsl")]
use crate::{ConstantOp, LinearOp, NonLinearOpJacobian, OdeEquations};

#[cfg(feature = "diffsl")]
#[allow(clippy::type_complexity)]
pub fn heat2d_diffsl_problem<
    M: Matrix<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
    const MGRID: usize,
>() -> (
    OdeSolverProblem<impl crate::OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let (problem, _soln) = head2d_problem::<M, MGRID>();
    let u0 = problem.eqn.init().call(0.0);
    let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
    let mass = problem.eqn.mass().unwrap().matrix(0.0);
    let init = problem.eqn.init().call(0.0);
    let init_diffsl = init
        .clone_as_vec()
        .iter()
        .map(|v| format!("            {}", *v))
        .collect::<Vec<_>>()
        .join(",\n");
    let (jac_idx, jac_vals) = jac.triplet_iter();
    let jac_diffsl = jac_idx
        .zip(jac_vals)
        .map(|((i, j), v)| format!("            ({i}, {j}): {v}"))
        .collect::<Vec<_>>()
        .join(",\n");

    let (mass_idx, _mass_vals) = mass.triplet_iter();
    let mass_ones = mass_idx.map(|(i, _j)| i).collect::<Vec<_>>();
    let mut mass_diffsl = Vec::new();
    for i in 0..MGRID * MGRID {
        // check if i in mass_ones
        if mass_ones.contains(&i) {
            mass_diffsl.push(format!("            ({i}, {i}): 1"));
        } else {
            mass_diffsl.push(format!("            ({i}, {i}): 0"));
        }
    }
    let mass_diffsl = mass_diffsl.join(",\n");

    let code = format!(
        "
        D_ij {{
{}
        }}
        Mass_ij {{
{}
        }}
        init_i {{
{}
        }}
        u_i {{
            y = init_i,
        }}
        dudt_i {{
            (0:{n}): dydt = 0,
        }}
        M_i {{
            Mass_ij * dydt_j,
        }}
        F_i {{
            D_ij * y_j,
        }}
        out_i {{
            {dx2} * y_j * y_j,
        }}",
        jac_diffsl,
        mass_diffsl,
        init_diffsl,
        n = MGRID * MGRID,
        dx2 = (1.0 / (MGRID as f64 - 1.0)).powi(2),
    );

    let problem = OdeBuilder::<M>::new()
        .rtol(1e-7)
        .atol([1e-7])
        .build_from_diffsl::<CG>(code.as_str())
        .unwrap();
    let soln = soln::<M>(problem.context().clone());
    (problem, soln)
}

fn heat2d_rhs<M: Matrix, const MGRID: usize>(x: &[M::T], _p: &[M::T], _t: M::T, y: &mut [M::T]) {
    // Initialize y to x, to take care of boundary equations.
    y.copy_from_slice(x);
    let mm = M::T::from_f64(MGRID as f64).unwrap();
    let four = M::T::from_f64(4.0).unwrap();

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] =
                coeff * (x[loc - 1] + x[loc + 1] + x[loc - MGRID] + x[loc + MGRID] - four * x[loc]);
        }
    }
}

fn heat2d_jac_mul<M: Matrix, const MGRID: usize>(
    _x: &[M::T],
    _p: &[M::T],
    _t: M::T,
    v: &[M::T],
    y: &mut [M::T],
) {
    // Initialize y to v, to take care of boundary equations.
    y.copy_from_slice(v);
    let mm = M::T::from_f64(MGRID as f64).unwrap();
    let four = M::T::from_f64(4.0).unwrap();

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] =
                coeff * (v[loc - 1] + v[loc + 1] + v[loc - MGRID] + v[loc + MGRID] - four * v[loc]);
        }
    }
}

fn heat2d_init<M: Matrix, const MGRID: usize>(_p: &[M::T], _t: M::T, uu: &mut [M::T]) {
    let mm = M::T::from_f64(MGRID as f64).unwrap();
    let bval = M::T::zero();
    let one = M::T::one();
    let dx = one / (mm - one);
    let mm1 = MGRID - 1;
    let sixteen = M::T::from_f64(16.0).unwrap();

    /* Initialize uu on all grid points. */
    for j in 0..MGRID {
        let yfact = dx * M::T::from_f64(j as f64).unwrap();
        let offset = MGRID * j;
        for i in 0..MGRID {
            let xfact = dx * M::T::from_f64(i as f64).unwrap();
            let loc = offset + i;
            uu[loc] = sixteen * xfact * (one - xfact) * yfact * (one - yfact);
        }
    }

    /* Finally, set values of u at boundary points. */
    for j in 0..MGRID {
        let offset = MGRID * j;
        for i in 0..MGRID {
            let loc = offset + i;
            if j == 0 || j == mm1 || i == 0 || i == mm1 {
                uu[loc] = bval;
            }
        }
    }
}

fn heat2d_mass<M: Matrix, const MGRID: usize>(
    x: &[M::T],
    _p: &[M::T],
    _t: M::T,
    beta: M::T,
    y: &mut [M::T],
) {
    let mm = MGRID;
    let mm1 = mm - 1;
    for j in 0..mm {
        let offset = mm * j;
        for i in 0..mm {
            let loc = offset + i;
            if j == 0 || j == mm1 || i == 0 || i == mm1 {
                y[loc] *= beta;
            } else {
                y[loc] = x[loc] + beta * y[loc];
            }
        }
    }
}

fn heat2d_out<M: Matrix, const MGRID: usize>(x: &[M::T], _p: &[M::T], _t: M::T, y: &mut [M::T]) {
    let dx = M::T::one() / (M::T::from_f64(MGRID as f64).unwrap() - M::T::one());
    let squared_norm = x.iter().fold(M::T::zero(), |acc, xi| acc + *xi * *xi);
    y[0] = squared_norm * dx * dx;
}

fn heat2d_out_jac_mul<M: Matrix, const MGRID: usize>(
    _x: &[M::T],
    _p: &[M::T],
    _t: M::T,
    _v: &[M::T],
    _y: &mut [M::T],
) {
    unimplemented!()
}

fn _pde_solution<T: Scalar>(x: T, y: T, t: T, max_terms: usize) -> T {
    let mut u = T::zero();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    let four = T::from_f64(4.0).unwrap();
    let two = T::from_f64(2.0).unwrap();
    let sixteen = T::from_f64(16.0).unwrap();

    for n in 1..=max_terms {
        let nt = T::from_f64(n as f64).unwrap();
        for m in 1..=max_terms {
            let mt = T::from_f64(m as f64).unwrap();
            let pi_mt = pi * mt;
            let pi_nt = pi * nt;
            let pi_cubed = pi * pi * pi;
            let mt_cubed = mt * mt * mt;
            let nt_cubed = nt * nt * nt;
            let ii = (-pi * mt * pi_mt.sin() - two * pi_mt.cos() + two) / (pi_cubed * mt_cubed);
            let jj = (-pi * nt * pi_nt.sin() - two * pi_nt.cos() + two) / (pi_cubed * nt_cubed);
            let coefficient = four * sixteen * ii * jj;

            let sin_term = (pi_nt * x).sin() * (pi_mt * y).sin();
            let nt_pi_sq = pi_nt * pi_nt;
            let mt_pi_sq = pi_mt * pi_mt;
            let exp_term = ((-nt_pi_sq - mt_pi_sq) * t).exp();

            u += coefficient * sin_term * exp_term;
        }
    }

    u
}

#[allow(clippy::type_complexity)]
pub fn head2d_problem<M: Matrix + 'static, const MGRID: usize>() -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let nstates = MGRID * MGRID;
    let problem = OdeBuilder::<M>::new()
        .rtol(1e-7)
        .atol([1e-7])
        .rhs_implicit(heat2d_rhs::<M, MGRID>, heat2d_jac_mul::<M, MGRID>)
        .mass(heat2d_mass::<M, MGRID>)
        .init(heat2d_init::<M, MGRID>, nstates)
        .out_implicit(heat2d_out::<M, MGRID>, heat2d_out_jac_mul::<M, MGRID>, 1)
        .build()
        .unwrap();
    let ctx = problem.context().clone();

    (problem, soln::<M>(ctx))
}

fn soln<M: Matrix>(ctx: M::C) -> OdeSolverSolution<M::V> {
    let mut soln = OdeSolverSolution {
        solution_points: Vec::new(),
        sens_solution_points: None,
        rtol: M::T::from_f64(1e-5).unwrap(),
        atol: M::V::from_element(1, M::T::from_f64(1e-5).unwrap(), ctx.clone()),
        negative_time: false,
    };
    let data = vec![
        (vec![0.28435774340267284], 0.0),
        (vec![0.19195491512700597], 0.01),
        (vec![0.12979676270145094], 0.02),
        (vec![0.05939666913561712], 0.04),
        (vec![0.012441804151214689], 0.08),
        (vec![0.0005459318925793768], 0.16),
        (vec![1.05130465235137e-6], 0.32),
        (vec![3.983888966577838e-12], 0.64),
        (vec![7.015395128730499e-16], 1.28),
        (vec![5.06965159341517e-17], 2.56),
        (vec![1.735145399301106e-18], 5.12),
        (vec![3.259034338585213e-17], 10.24),
    ];
    let nbatch = ctx.nbatch();
    for (values, time) in data {
        // every lane solves the same problem, so each output value repeats per lane
        let mut per_lane = Vec::with_capacity(values.len() * nbatch);
        for _ in 0..nbatch {
            per_lane.extend(values.iter().map(|v| M::T::from_f64(*v).unwrap()));
        }
        let values = M::V::from_vec(per_lane, ctx.clone());
        let time = M::T::from_f64(time).unwrap();
        soln.push(values, time);
    }
    soln
}

#[cfg(test)]
mod tests {
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat, ConstantOp, LinearOp, MatrixCommon,
        NonLinearOpJacobian, OdeEquations,
    };

    use super::*;

    #[test]
    fn test_jacobian() {
        //let jac = heat2d_jacobian::<nalgebra::DMatrix<f64>, 10>();
        let (problem, _soln) = head2d_problem::<NalgebraMat<f64>, 10>();
        let u0 = problem.eqn.init().call(0.0);
        let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
        insta::assert_yaml_snapshot!(jac.inner().to_string());
    }

    #[test]
    fn test_mass() {
        let (problem, _soln) = head2d_problem::<NalgebraMat<f64>, 10>();
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        insta::assert_yaml_snapshot!(mass.inner().to_string());
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn test_mass_diffsl() {
        use crate::{FaerSparseMat, FaerVec};
        use diffsl::CraneliftJitModule;

        let (problem, _soln) = heat2d_diffsl_problem::<FaerSparseMat<f64>, CraneliftJitModule, 5>();
        let u = FaerVec::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
            *problem.context(),
        );
        let mut y = FaerVec::zeros(25, *problem.context());
        problem.eqn.mass().unwrap().call_inplace(&u, 0.0, &mut y);
        let expect = FaerVec::from_vec(
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 12.0, 13.0, 14.0, 0.0, 0.0,
                17.0, 18.0, 19.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            *problem.context(),
        );
        y.assert_eq_st(&expect, 1.0e-10);
    }

    #[test]
    fn test_soln() {
        let (_problem, _soln) = head2d_problem::<NalgebraMat<f64>, 10>();
    }
}

// ============================================================
// Element-parallel implementation
// ============================================================

/// The constants of the discretisation, precomputed on the host.
///
/// A device closure captures by value and cannot afford a panic path, so
/// `T::from_f64(x).unwrap()` has to happen out here rather than in the kernel.
#[derive(Clone, Copy)]
struct Heat2dConsts<T: Scalar> {
    /// `1 / dx^2`
    coeff: T,
    /// `dx`
    dx: T,
    four: T,
    sixteen: T,
}

impl<T: Scalar> Heat2dConsts<T> {
    fn new(mgrid: usize) -> Self {
        let dx = 1.0 / (mgrid as f64 - 1.0);
        Self {
            coeff: T::from_f64(1.0 / (dx * dx)).unwrap(),
            dx: T::from_f64(dx).unwrap(),
            four: T::from_f64(4.0).unwrap(),
            sixteen: T::from_f64(16.0).unwrap(),
        }
    }
}

/// Is element `i` of the grid on the boundary?
#[inline]
fn heat2d_is_boundary(mgrid: usize, i: usize) -> bool {
    let (jx, jy) = (i % mgrid, i / mgrid);
    jx == 0 || jy == 0 || jx == mgrid - 1 || jy == mgrid - 1
}

/// One element of the 5-point central difference, holding the boundary values fixed.
///
/// The element form of [`heat2d_rhs`], and of [`heat2d_jac_mul`] too: the operator is linear, so
/// the same body serves `f(x)` and `J v`.
fn heat2d_elem_rhs<T: Scalar>(y: &mut T, x: &[T], c: Heat2dConsts<T>, mgrid: usize, i: usize) {
    *y = if heat2d_is_boundary(mgrid, i) {
        // the boundary equations are algebraic: the residual is the value itself
        x[i]
    } else {
        c.coeff * (x[i - 1] + x[i + 1] + x[i - mgrid] + x[i + mgrid] - c.four * x[i])
    };
}

/// One element of the mass action `y = M x + beta y`, the element form of [`heat2d_mass`].
fn heat2d_elem_mass<T: Scalar>(y: &mut T, x: &[T], beta: T, mgrid: usize, i: usize) {
    // the boundary rows of the mass matrix are zero
    *y = if heat2d_is_boundary(mgrid, i) {
        beta * *y
    } else {
        x[i] + beta * *y
    };
}

/// One element of the initial condition, the element form of [`heat2d_init`].
fn heat2d_elem_init<T: Scalar>(y: &mut T, c: Heat2dConsts<T>, mgrid: usize, i: usize) {
    *y = if heat2d_is_boundary(mgrid, i) {
        T::zero()
    } else {
        let (jx, jy) = (i % mgrid, i / mgrid);
        let xfact = c.dx * T::from_usize(jx).unwrap();
        let yfact = c.dx * T::from_usize(jy).unwrap();
        c.sixteen * xfact * (T::one() - xfact) * yfact * (T::one() - yfact)
    };
}

/// The heat equation on an `MGRID x MGRID` grid, with every operator written one element at a
/// time so [`Vector::for_each_elem`] can run it a thread per `(lane, element)` on a device
/// backend.
pub struct Heat2dElem<M: Matrix, const MGRID: usize> {
    consts: Heat2dConsts<M::T>,
    rhs_sparsity: Option<M::Sparsity>,
    rhs_coloring: Option<JacobianColoring<M>>,
    mass_sparsity: Option<M::Sparsity>,
    mass_coloring: Option<JacobianColoring<M>>,
    ctx: M::C,
}

impl<M: Matrix, const MGRID: usize> Heat2dElem<M, MGRID> {
    const NSTATES: usize = MGRID * MGRID;

    pub fn new(ctx: M::C, t0: M::T) -> Self {
        let mut ret = Self {
            consts: Heat2dConsts::new(MGRID),
            rhs_sparsity: None,
            rhs_coloring: None,
            mass_sparsity: None,
            mass_coloring: None,
            ctx,
        };
        let y0 = Heat2dElemInit { eqn: &ret }.call(t0);

        let rhs = Heat2dElemRhs { eqn: &ret };
        let non_zeros = find_jacobian_non_zeros(&rhs, &y0, t0);
        ret.rhs_sparsity = Some(
            MatrixSparsity::try_from_indices(rhs.nout(), rhs.nstates(), non_zeros.clone()).unwrap(),
        );
        ret.rhs_coloring = Some(JacobianColoring::new(
            ret.rhs_sparsity.as_ref().unwrap(),
            &non_zeros,
            ret.ctx.clone(),
        ));

        let mass = Heat2dElemMass { eqn: &ret };
        let non_zeros = find_matrix_non_zeros(&mass, t0);
        ret.mass_sparsity = Some(
            MatrixSparsity::try_from_indices(mass.nout(), mass.nstates(), non_zeros.clone())
                .unwrap(),
        );
        ret.mass_coloring = Some(JacobianColoring::new(
            ret.mass_sparsity.as_ref().unwrap(),
            &non_zeros,
            ret.ctx.clone(),
        ));
        ret
    }
}

pub struct Heat2dElemRhs<'a, M: Matrix, const MGRID: usize> {
    eqn: &'a Heat2dElem<M, MGRID>,
}
pub struct Heat2dElemMass<'a, M: Matrix, const MGRID: usize> {
    eqn: &'a Heat2dElem<M, MGRID>,
}
pub struct Heat2dElemInit<'a, M: Matrix, const MGRID: usize> {
    eqn: &'a Heat2dElem<M, MGRID>,
}
pub struct Heat2dElemOut<'a, M: Matrix, const MGRID: usize> {
    eqn: &'a Heat2dElem<M, MGRID>,
}

macro_rules! impl_heat2d_elem_op {
    ($name:ident, $nout:expr) => {
        impl<M: Matrix, const MGRID: usize> Op for $name<'_, M, MGRID> {
            type M = M;
            type V = M::V;
            type T = M::T;
            type C = M::C;

            fn nstates(&self) -> usize {
                Heat2dElem::<M, MGRID>::NSTATES
            }
            fn nout(&self) -> usize {
                $nout
            }
            fn nparams(&self) -> usize {
                0
            }
            fn context(&self) -> &Self::C {
                &self.eqn.ctx
            }
        }
    };
}

impl_heat2d_elem_op!(Heat2dElemRhs, Heat2dElem::<M, MGRID>::NSTATES);
impl_heat2d_elem_op!(Heat2dElemMass, Heat2dElem::<M, MGRID>::NSTATES);
impl_heat2d_elem_op!(Heat2dElemInit, Heat2dElem::<M, MGRID>::NSTATES);
impl_heat2d_elem_op!(Heat2dElemOut, 1);

impl<M: Matrix, const MGRID: usize> NonLinearOp for Heat2dElemRhs<'_, M, MGRID> {
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        let c = self.eqn.consts;
        y.for_each_elem(
            [x],
            move |y: &mut M::T, [x]: [&[M::T]; 1], _lane: usize, i: usize| {
                heat2d_elem_rhs(y, x, c, MGRID, i)
            },
        );
    }
}

impl<M: Matrix, const MGRID: usize> NonLinearOpJacobian for Heat2dElemRhs<'_, M, MGRID> {
    fn jac_mul_inplace(&self, _x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        let c = self.eqn.consts;
        // the operator is linear, so the action of the jacobian is the operator itself
        y.for_each_elem(
            [v],
            move |y: &mut M::T, [v]: [&[M::T]; 1], _lane: usize, i: usize| {
                heat2d_elem_rhs(y, v, c, MGRID, i)
            },
        );
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.eqn.rhs_coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
    fn jacobian_sparsity(&self) -> Option<M::Sparsity> {
        self.eqn.rhs_sparsity.clone()
    }
}

impl<M: Matrix, const MGRID: usize> LinearOp for Heat2dElemMass<'_, M, MGRID> {
    fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.for_each_elem(
            [x],
            move |y: &mut M::T, [x]: [&[M::T]; 1], _lane: usize, i: usize| {
                heat2d_elem_mass(y, x, beta, MGRID, i)
            },
        );
    }
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.eqn.mass_coloring.as_ref() {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn sparsity(&self) -> Option<M::Sparsity> {
        self.eqn.mass_sparsity.clone()
    }
}

impl<M: Matrix, const MGRID: usize> ConstantOp for Heat2dElemInit<'_, M, MGRID> {
    fn call_inplace(&self, _t: M::T, y: &mut M::V) {
        let c = self.eqn.consts;
        y.for_each_elem(
            [],
            move |y: &mut M::T, _: [&[M::T]; 0], _lane: usize, i: usize| {
                heat2d_elem_init(y, c, MGRID, i)
            },
        );
    }
}

impl<M: Matrix, const MGRID: usize> NonLinearOp for Heat2dElemOut<'_, M, MGRID> {
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        // `dx^2 * sum_i x_i^2`, one output per lane, so one thread per lane walks the lane
        let dx2 = self.eqn.consts.dx * self.eqn.consts.dx;
        y.for_each_elem(
            [x],
            move |y: &mut M::T, [x]: [&[M::T]; 1], _lane: usize, _i: usize| {
                let mut acc = M::T::zero();
                for xk in x.iter() {
                    acc += *xk * *xk;
                }
                *y = acc * dx2;
            },
        );
    }
}

impl<M: Matrix, const MGRID: usize> NonLinearOpJacobian for Heat2dElemOut<'_, M, MGRID> {
    fn jac_mul_inplace(&self, _x: &M::V, _t: M::T, _v: &M::V, _y: &mut M::V) {
        // as in the builder version, the output jacobian is not needed by any solve here
        unimplemented!()
    }
}

impl<M: Matrix, const MGRID: usize> Op for Heat2dElem<M, MGRID> {
    type M = M;
    type V = M::V;
    type T = M::T;
    type C = M::C;

    fn nstates(&self) -> usize {
        Self::NSTATES
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<'a, M: Matrix, const MGRID: usize> OdeEquationsRef<'a> for Heat2dElem<M, MGRID> {
    type Rhs = Heat2dElemRhs<'a, M, MGRID>;
    type Mass = Heat2dElemMass<'a, M, MGRID>;
    type Init = Heat2dElemInit<'a, M, MGRID>;
    type Out = Heat2dElemOut<'a, M, MGRID>;
    type Root = ParameterisedOp<'a, UnitCallable<M>>;
    type Reset = ParameterisedOp<'a, UnitCallable<M>>;
}

impl<M: Matrix, const MGRID: usize> OdeEquations for Heat2dElem<M, MGRID> {
    fn rhs(&self) -> Heat2dElemRhs<'_, M, MGRID> {
        Heat2dElemRhs { eqn: self }
    }
    fn mass(&self) -> Option<Heat2dElemMass<'_, M, MGRID>> {
        Some(Heat2dElemMass { eqn: self })
    }
    fn init(&self) -> Heat2dElemInit<'_, M, MGRID> {
        Heat2dElemInit { eqn: self }
    }
    fn out(&self) -> Option<Heat2dElemOut<'_, M, MGRID>> {
        Some(Heat2dElemOut { eqn: self })
    }
    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn set_params(&mut self, _p: &Self::V) {
        unimplemented!()
    }
    fn get_params(&self, _p: &mut Self::V) {
        unimplemented!()
    }
}

/// [`head2d_problem`] with every operator element-parallel, over `nbatch` lanes.
#[allow(clippy::type_complexity)]
pub fn heat2d_elem_problem<M: Matrix + 'static, const MGRID: usize>(
    nbatch: usize,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    let ctx = M::C::default().clone_with_nbatch(nbatch).unwrap();
    let rtol = M::T::from_f64(1e-7).unwrap();
    let atol = M::V::from_element(MGRID * MGRID, M::T::from_f64(1e-7).unwrap(), ctx.clone());
    let t0 = M::T::zero();
    let h0 = M::T::one();
    let eqn = Heat2dElem::<M, MGRID>::new(ctx.clone(), t0);
    let problem = OdeSolverProblem::new(
        eqn,
        rtol,
        atol,
        None,
        None,
        None,
        None,
        None,
        None,
        t0,
        h0,
        false,
        Default::default(),
        Default::default(),
    )
    .unwrap();
    let soln = soln::<M>(ctx);
    (problem, soln)
}

#[cfg(test)]
mod elem_tests {
    use super::*;
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat, DenseMatrix, MatrixCommon, NalgebraVec,
    };

    const MGRID: usize = 10;

    /// The element operators and the whole-lane operators of [`head2d_problem`] are two copies of
    /// the same maths, so check them against each other.
    #[test]
    fn test_elem_matches_lane() {
        type M = NalgebraMat<f64>;
        let (elem, _) = heat2d_elem_problem::<M, MGRID>(1);
        let (lane, _) = head2d_problem::<M, MGRID>();

        let y0 = elem.eqn.init().call(0.0);
        y0.assert_eq_st(&lane.eqn.init().call(0.0), 1e-14);

        elem.eqn
            .rhs()
            .call(&y0, 0.0)
            .assert_eq_st(&lane.eqn.rhs().call(&y0, 0.0), 1e-9);

        let v = NalgebraVec::from_element(MGRID * MGRID, 0.5, *elem.context());
        elem.eqn
            .rhs()
            .jac_mul(&y0, 0.0, &v)
            .assert_eq_st(&lane.eqn.rhs().jac_mul(&y0, 0.0, &v), 1e-9);

        let mass_elem = elem.eqn.mass().unwrap().matrix(0.0);
        let mass_lane = lane.eqn.mass().unwrap().matrix(0.0);
        for i in 0..mass_elem.nrows() {
            for j in 0..mass_elem.ncols() {
                assert_eq!(
                    mass_elem.get_index(i, j),
                    mass_lane.get_index(i, j),
                    "mass[{i}, {j}]"
                );
            }
        }

        elem.eqn
            .out()
            .unwrap()
            .call(&y0, 0.0)
            .assert_eq_st(&lane.eqn.out().unwrap().call(&y0, 0.0), 1e-12);
    }
}
