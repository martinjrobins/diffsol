use std::rc::Rc;

use crate::{
    find_non_zeros_linear, find_non_zeros_nonlinear, ode_solver::problem::OdeSolverSolution,
    ConstantOp, DenseMatrix, JacobianColoring, LinearOp, Matrix, MatrixSparsity, NonLinearOp,
    OdeEquations, OdeSolverProblem, Op, UnitCallable, Vector,
};
use num_traits::Zero;

const NPREY: usize = 1;
const NUM_SPECIES: usize = 2 * NPREY;
const NSMX: usize = NUM_SPECIES * MX;
const MX: usize = 10;
const MY: usize = 10;
const AX: f64 = 1.0;
const AY: f64 = 1.0;
const DX: f64 = AX / (MX as f64 - 1.0);
const DY: f64 = AY / (MY as f64 - 1.0);

const AA: f64 = 1.0;
const EE: f64 = 10000.0;
const GG: f64 = 0.5e-6;
const BB: f64 = 1.0;
const DPREY: f64 = 1.0;
const DPRED: f64 = 0.05;

const ALPHA: f64 = 50.0;
const BETA: f64 = 1000.0;

pub struct FoodWebContext<MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    acoef: MD,
    bcoef: M::V,
    cox: M::V,
    coy: M::V,
    nstates: usize,
}

// Following is the description of this model from the Sundials examples:
//
/* -----------------------------------------------------------------
 *
 * The mathematical problem solved in this example is a DAE system
 * that arises from a system of partial differential equations after
 * spatial discretization. The PDE system is a food web population
 * model, with predator-prey interaction and diffusion on the unit
 * square in two dimensions. The dependent variable vector is:
 *
 *         1   2         ns
 *   c = (c , c ,  ..., c  ) , ns = 2 * np
 *
 * and the PDE's are as follows:
 *
 *     i             i      i
 *   dc /dt = d(i)*(c    + c  )  +  R (x,y,c)   (i = 1,...,np)
 *                   xx     yy       i
 *
 *              i      i
 *   0 = d(i)*(c    + c  )  +  R (x,y,c)   (i = np+1,...,ns)
 *              xx     yy       i
 *
 *   where the reaction terms R are:
 *
 *                   i             ns         j
 *   R  (x,y,c)  =  c  * (b(i)  + sum a(i,j)*c )
 *    i                           j=1
 *
 * The number of species is ns = 2 * np, with the first np being
 * prey and the last np being predators. The coefficients a(i,j),
 * b(i), d(i) are:
 *
 *  a(i,i) = -AA   (all i)
 *  a(i,j) = -GG   (i <= np , j >  np)
 *  a(i,j) =  EE   (i >  np, j <= np)
 *  all other a(i,j) = 0
 *  b(i) = BB*(1+ alpha * x*y + beta*sin(4 pi x)*sin(4 pi y)) (i <= np)
 *  b(i) =-BB*(1+ alpha * x*y + beta*sin(4 pi x)*sin(4 pi y)) (i  > np)
 *  d(i) = DPREY   (i <= np)
 *  d(i) = DPRED   (i > np)
 *
 * The various scalar parameters required are set using '#define'
 * statements or directly in routine InitUserData. In this program,
 * np = 1, ns = 2. The boundary conditions are homogeneous Neumann:
 * normal derivative = 0.
 *
 * A polynomial in x and y is used to set the initial values of the
 * first np variables (the prey variables) at each x,y location,
 * while initial values for the remaining (predator) variables are
 * set to a flat value, which is corrected by IDACalcIC.
 *
 * The PDEs are discretized by central differencing on a MX by MY
 * mesh.
 *
 * -----------------------------------------------------------------
 * References:
 * [1] Peter N. Brown and Alan C. Hindmarsh,
 *     Reduced Storage Matrix Methods in Stiff ODE systems, Journal
 *     of Applied Mathematics and Computation, Vol. 31 (May 1989),
 *     pp. 40-91.
 *
 * [2] Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
 *     Using Krylov Methods in the Solution of Large-Scale
 *     Differential-Algebraic Systems, SIAM J. Sci. Comput., 15
 *     (1994), pp. 1467-1488.
 *
 * [3] Peter N. Brown, Alan C. Hindmarsh, and Linda R. Petzold,
 *     Consistent Initial Condition Calculation for Differential-
 *     Algebraic Systems, SIAM J. Sci. Comput., 19 (1998),
 *     pp. 1495-1512.
 * -----------------------------------------------------------------*/
impl<MD, M> FoodWebContext<MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new() -> Self {
        let mut acoef = MD::zeros(NUM_SPECIES, NUM_SPECIES);
        let mut bcoef = M::V::zeros(NUM_SPECIES);
        let mut cox = M::V::zeros(NUM_SPECIES);
        let mut coy = M::V::zeros(NUM_SPECIES);
        let nstates = NUM_SPECIES * MX * MY;

        for i in 0..NPREY {
            for j in 0..NPREY {
                acoef[(i, NPREY + j)] = M::T::from(-GG);
                acoef[(i + NPREY, j)] = M::T::from(EE);
                acoef[(i, j)] = M::T::from(0.0);
                acoef[(i + NPREY, NPREY + j)] = M::T::from(0.0);
            }

            acoef[(i, i)] = M::T::from(-AA);
            acoef[(i + NPREY, i + NPREY)] = M::T::from(-AA);

            bcoef[i] = M::T::from(BB);
            bcoef[i + NPREY] = M::T::from(-BB);
            cox[i] = M::T::from(DPREY / DX.powi(2));
            cox[i + NPREY] = M::T::from(DPRED / DX.powi(2));
            coy[i] = M::T::from(DPREY / DY.powi(2));
            coy[i + NPREY] = M::T::from(DPRED / DY.powi(2));
        }

        Self {
            acoef,
            bcoef,
            cox,
            coy,
            nstates,
        }
    }
}

impl<MD, M> Default for FoodWebContext<MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    fn default() -> Self {
        Self::new()
    }
}

struct FoodWebInit<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M>,
}

// macro for bringing in constants from Context
macro_rules! context_consts {
    ($name:ident) => {
        impl<'a, MD, M> $name<'a, MD, M>
        where
            MD: DenseMatrix,
            M: Matrix<V = MD::V, T = MD::T>,
        {
            pub fn new(context: &'a FoodWebContext<MD, M>) -> Self {
                Self { context }
            }
        }
    };
}

// macro for impl ops
macro_rules! impl_op {
    ($name:ident) => {
        impl<'a, MD, M> Op for $name<'a, MD, M>
        where
            MD: DenseMatrix,
            M: Matrix<V = MD::V, T = MD::T>,
        {
            type M = M;
            type V = M::V;
            type T = M::T;

            fn nout(&self) -> usize {
                self.context.nstates
            }
            fn nparams(&self) -> usize {
                0
            }
            fn nstates(&self) -> usize {
                self.context.nstates
            }
        }
    };
}

context_consts!(FoodWebInit);
impl_op!(FoodWebInit);

impl<'a, MD, M> ConstantOp for FoodWebInit<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    fn call_inplace(&self, _t: M::T, y: &mut M::V) {
        /* Loop over grid, load cc values and id values. */
        for jy in 0..MY {
            let yy = jy as f64 * DY;
            let yloc = NSMX * jy;
            for jx in 0..MX {
                let xx = jx as f64 * DX;
                let xyfactor = 16.0 * xx * (1.0 - xx) * yy * (1.0 - yy);
                let xyfactor = xyfactor.powi(2);
                let loc = yloc + NUM_SPECIES * jx;

                for is in 0..NUM_SPECIES {
                    if is < NPREY {
                        y[loc + is] = M::T::from(10.0 + (is + 1) as f64 * xyfactor);
                    } else {
                        y[loc + is] = M::T::from(1.0e5);
                    }
                }
            }
        }
    }
}

struct FoodWebRhs<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M>,
    pub sparsity: Option<M::Sparsity>,
    pub coloring: Option<JacobianColoring<M>>,
}

impl<'a, MD, M> FoodWebRhs<'a, MD, M>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new(context: &'a FoodWebContext<MD, M>, y0: &M::V, t0: M::T) -> Self {
        let mut ret = Self {
            context,
            sparsity: None,
            coloring: None,
        };
        let non_zeros = find_non_zeros_nonlinear(&ret, y0, t0);
        ret.sparsity = Some(
            MatrixSparsity::try_from_indices(ret.nout(), ret.nstates(), non_zeros.clone()).unwrap(),
        );
        ret.coloring = Some(JacobianColoring::new_from_non_zeros(&ret, non_zeros));
        ret
    }
}

impl<'a, MD, M> Op for FoodWebRhs<'a, MD, M>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        self.context.nstates
    }
    fn nparams(&self) -> usize {
        0
    }
    fn nstates(&self) -> usize {
        self.context.nstates
    }
    fn sparsity(&self) -> Option<M::SparsityRef<'_>> {
        self.sparsity.as_ref().map(|s| s.as_ref())
    }
}

impl<'a, MD, M> NonLinearOp for FoodWebRhs<'a, MD, M>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    /*
     * Fweb: Rate function for the food-web problem.
     * This routine computes the right-hand sides of the system equations,
     * consisting of the diffusion term and interaction term.
     * The interaction term is computed by the function WebRates.
     */
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        let mut rates = [M::T::zero(); NUM_SPECIES];
        /* Loop over grid points, evaluate interaction vector (length ns),
        form diffusion difference terms, and load crate.                    */
        for jy in 0..MY {
            let yy = jy as f64 * DY;
            let idyu = if jy != MY - 1 {
                NSMX as i32
            } else {
                -(NSMX as i32)
            };
            let idyl = if jy != 0 { NSMX as i32 } else { -(NSMX as i32) };

            for jx in 0..MX {
                let xx = jx as f64 * DX;
                let idxu = if jx != MX - 1 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let idxl = if jx != 0 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let loc = NUM_SPECIES * jx + NSMX * jy;

                /*
                 * WebRates: Evaluate reaction rates at a given spatial point.
                 * At a given (x,y), evaluate the array of ns reaction terms R.
                 */
                for (is, rate) in rates.iter_mut().enumerate().take(NUM_SPECIES) {
                    let mut dp = M::T::zero();
                    for js in 0..NUM_SPECIES {
                        dp += self.context.acoef[(is, js)] * x[loc + js];
                    }
                    *rate = dp;
                }
                let fac = M::T::from(
                    1.0 + ALPHA * xx * yy
                        + BETA
                            * (4.0 * std::f64::consts::PI * xx).sin()
                            * (4.0 * std::f64::consts::PI * yy).sin(),
                );

                for is in 0..NUM_SPECIES {
                    rates[is] = x[loc + is] * (self.context.bcoef[is] * fac + rates[is]);
                }

                /* Loop over species, do differencing, load crate segment. */
                for is in 0..NUM_SPECIES {
                    /* Differencing in y. */
                    let dcyli =
                        x[loc + is] - x[usize::try_from(loc as i32 - idyl + is as i32).unwrap()];
                    let dcyui =
                        x[usize::try_from(loc as i32 + idyu + is as i32).unwrap()] - x[loc + is];

                    /* Differencing in x. */
                    let dcxli =
                        x[loc + is] - x[usize::try_from(loc as i32 - idxl + is as i32).unwrap()];
                    let dcxui =
                        x[usize::try_from(loc as i32 + idxu + is as i32).unwrap()] - x[loc + is];

                    /* Compute the crate values at (xx,yy). */
                    y[loc + is] = self.context.coy[is] * (dcyui - dcyli)
                        + self.context.cox[is] * (dcxui - dcxli)
                        + rates[is];
                }
            }
        }
    }

    fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        let mut rates = [M::T::zero(); NUM_SPECIES];
        let mut drates = [M::T::zero(); NUM_SPECIES];
        /* Loop over grid points, evaluate interaction vector (length ns),
        form diffusion difference terms, and load crate.                    */
        for jy in 0..MY {
            let yy = jy as f64 * DY;
            let idyu = if jy != MY - 1 {
                NSMX as i32
            } else {
                -(NSMX as i32)
            };
            let idyl = if jy != 0 { NSMX as i32 } else { -(NSMX as i32) };

            for jx in 0..MX {
                let xx = jx as f64 * DX;
                let idxu = if jx != MX - 1 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let idxl = if jx != 0 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let loc = NUM_SPECIES * jx + NSMX * jy;

                /*
                 * WebRates: Evaluate reaction rates at a given spatial point.
                 * At a given (x,y), evaluate the array of ns reaction terms R.
                 */
                for is in 0..NUM_SPECIES {
                    let mut ddp = M::T::zero();
                    let mut dp = M::T::zero();
                    for js in 0..NUM_SPECIES {
                        dp += self.context.acoef[(is, js)] * x[loc + js];
                        ddp += self.context.acoef[(is, js)] * v[loc + js];
                    }
                    rates[is] = dp;
                    drates[is] = ddp;
                }
                let fac = M::T::from(
                    1.0 + ALPHA * xx * yy
                        + BETA
                            * (4.0 * std::f64::consts::PI * xx).sin()
                            * (4.0 * std::f64::consts::PI * yy).sin(),
                );

                for is in 0..NUM_SPECIES {
                    rates[is] = x[loc + is] * (self.context.bcoef[is] * fac + rates[is]);
                    drates[is] = x[loc + is] * drates[is]
                        + v[loc + is] * (self.context.bcoef[is] * fac + rates[is]);
                }

                /* Loop over species, do differencing, load crate segment. */
                for is in 0..NUM_SPECIES {
                    /* Differencing in y. */
                    let dcyli =
                        v[loc + is] - v[usize::try_from(loc as i32 - idyl + is as i32).unwrap()];
                    let dcyui =
                        v[usize::try_from(loc as i32 + idyu + is as i32).unwrap()] - v[loc + is];

                    /* Differencing in x. */
                    let dcxli =
                        v[loc + is] - v[usize::try_from(loc as i32 - idxl + is as i32).unwrap()];
                    let dcxui =
                        v[usize::try_from(loc as i32 + idxu + is as i32).unwrap()] - v[loc + is];

                    /* Compute the crate values at (xx,yy). */
                    y[loc + is] = self.context.coy[is] * (dcyui - dcyli)
                        + self.context.cox[is] * (dcxui - dcxli)
                        + drates[is];
                }
            }
        }
    }
}

struct FoodWebMass<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M>,
    pub sparsity: Option<M::Sparsity>,
    pub coloring: Option<JacobianColoring<M>>,
}

impl<'a, MD, M> FoodWebMass<'a, MD, M>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new(context: &'a FoodWebContext<MD, M>, t0: M::T) -> Self {
        let mut ret = Self {
            context,
            sparsity: None,
            coloring: None,
        };
        let non_zeros = find_non_zeros_linear(&ret, t0);
        ret.sparsity = Some(
            MatrixSparsity::try_from_indices(ret.nout(), ret.nstates(), non_zeros.clone()).unwrap(),
        );
        ret.coloring = Some(JacobianColoring::new_from_non_zeros(&ret, non_zeros));
        ret
    }
}

impl<'a, MD, M> Op for FoodWebMass<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        self.context.nstates
    }
    fn nparams(&self) -> usize {
        0
    }
    fn nstates(&self) -> usize {
        self.context.nstates
    }
    fn sparsity(&self) -> Option<M::SparsityRef<'_>> {
        self.sparsity.as_ref().map(|s| s.as_ref())
    }
}

impl<'a, MD, M> LinearOp for FoodWebMass<'a, MD, M>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    #[allow(unused_mut)]
    fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, mut y: &mut Self::V) {
        /* Loop over all grid points, setting residual values appropriately
        for differential or algebraic components.                        */
        for jy in 0..MY {
            let yloc = NSMX * jy;
            for jx in 0..MX {
                let loc = yloc + NUM_SPECIES * jx;
                for is in 0..NUM_SPECIES {
                    if is < NPREY {
                        y[loc + is] = x[loc + is] + beta * y[loc + is];
                    } else {
                        y[loc + is] = beta * y[loc + is];
                    }
                }
            }
        }
    }
}

struct FoodWebOut<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M>,
}

context_consts!(FoodWebOut);

impl<'a, MD, M> Op for FoodWebOut<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        2 * NUM_SPECIES
    }
    fn nparams(&self) -> usize {
        0
    }
    fn nstates(&self) -> usize {
        self.context.nstates
    }
}

impl<'a, MD, M> NonLinearOp for FoodWebOut<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    #[allow(unused_mut)]
    fn call_inplace(&self, x: &M::V, _t: M::T, mut y: &mut M::V) {
        let jx_tl = 0;
        let jy_tl = 0;
        let jx_br = MX - 1;
        let jy_br = MY - 1;
        let loc_tl = NUM_SPECIES * jx_tl + NSMX * jy_tl;
        let loc_br = NUM_SPECIES * jx_br + NSMX * jy_br;
        for is in 0..NUM_SPECIES {
            y[2 * is] = x[loc_tl + is];
            y[2 * is + 1] = x[loc_br + is];
        }
    }

    #[allow(unused_mut)]
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, mut y: &mut Self::V) {
        let jx_tl = 0;
        let jy_tl = 0;
        let jx_br = MX - 1;
        let jy_br = MY - 1;
        let loc_tl = NUM_SPECIES * jx_tl + NSMX * jy_tl;
        let loc_br = NUM_SPECIES * jx_br + NSMX * jy_br;
        for is in 0..NUM_SPECIES {
            y[2 * is] = v[loc_tl + is];
            y[2 * is + 1] = v[loc_br + is];
        }
    }
}

struct FoodWeb<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub rhs: Rc<FoodWebRhs<'a, MD, M>>,
    pub mass: Rc<FoodWebMass<'a, MD, M>>,
    pub init: Rc<FoodWebInit<'a, MD, M>>,
    pub out: Rc<FoodWebOut<'a, MD, M>>,
}

impl<'a, MD, M> FoodWeb<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new(context: &'a FoodWebContext<MD, M>, t0: M::T) -> Self {
        let init = FoodWebInit::new(context);
        let y0 = init.call(t0);
        let rhs = FoodWebRhs::new(context, &y0, t0);
        let mass = FoodWebMass::new(context, t0);
        let out = FoodWebOut::new(context);

        let init = Rc::new(init);
        let rhs = Rc::new(rhs);
        let mass = Rc::new(mass);
        let out = Rc::new(out);

        Self {
            rhs,
            mass,
            init,
            out,
        }
    }
}

impl<'a, MD, M> OdeEquations for FoodWeb<'a, MD, M>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    type M = M;
    type V = M::V;
    type T = M::T;
    type Init = FoodWebInit<'a, MD, M>;
    type Rhs = FoodWebRhs<'a, MD, M>;
    type Mass = FoodWebMass<'a, MD, M>;
    type Root = UnitCallable<M>;
    type Out = FoodWebOut<'a, MD, M>;

    fn rhs(&self) -> &Rc<Self::Rhs> {
        &self.rhs
    }
    fn init(&self) -> &Rc<Self::Init> {
        &self.init
    }
    fn mass(&self) -> Option<&Rc<Self::Mass>> {
        Some(&self.mass)
    }
    fn out(&self) -> Option<&Rc<Self::Out>> {
        Some(&self.out)
    }
    fn set_params(&mut self, _p: Self::V) {}
}

fn soln<M: Matrix>() -> OdeSolverSolution<M::V> {
    let mut soln = OdeSolverSolution {
        solution_points: Vec::new(),
        sens_solution_points: None,
        rtol: M::T::from(1e-5),
        atol: M::V::from_element(2 * NUM_SPECIES, M::T::from(1e-5)),
    };
    let data = vec![
        (vec![1.0e1, 1.0e1, 1.0e5, 1.0e5], 0.0),
        (vec![1.0318e1, 1.0827e1, 1.0319e5, 1.0822e5], 1.0e-3),
        (vec![1.6189e2, 1.9735e2, 1.6189e6, 1.9735e6], 1.0e-2),
        (vec![2.4019e2, 2.7072e2, 2.4019e6, 2.7072e6], 1.0e-1),
        (vec![2.4019e2, 2.7072e2, 2.4019e6, 2.7072e6], 4.0e-1),
        (vec![2.4019e2, 2.7072e2, 2.4019e6, 2.7072e6], 7.0e-1),
        (vec![2.4019e2, 2.7072e2, 2.4019e6, 2.7072e6], 1.0),
    ];
    for (values, time) in data {
        let values = M::V::from_vec(values.iter().map(|v| M::T::from(*v)).collect::<Vec<_>>());
        let time = M::T::from(time);
        soln.push(values, time);
    }
    soln
}

pub fn foodweb_problem<MD, M>(
    context: &FoodWebContext<MD, M>,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T> + '_>,
    OdeSolverSolution<M::V>,
)
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    let rtol = M::T::from(1e-5);
    let atol = M::V::from_element(NUM_SPECIES * MX * MY, M::T::from(1e-5));
    let t0 = M::T::zero();
    let h0 = M::T::from(1.0);
    let eqn = FoodWeb::new(context, t0);
    let problem = OdeSolverProblem::new(eqn, rtol, atol, t0, h0, false, false).unwrap();
    let soln = soln::<M>();
    (problem, soln)
}

#[cfg(test)]
mod tests {
    use crate::{ConstantOp, LinearOp, NonLinearOp};

    use super::*;

    #[test]
    fn test_jacobian() {
        type M = nalgebra::DMatrix<f64>;
        let context = FoodWebContext::<M, M>::new();
        let (problem, _soln) = foodweb_problem::<M, M>(&context);
        let u0 = problem.eqn.init().call(0.0);
        let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
        println!("{}", jac);
        insta::assert_yaml_snapshot!(jac.to_string());
    }

    #[test]
    fn test_mass() {
        type M = nalgebra::DMatrix<f64>;
        let context = FoodWebContext::<M, M>::new();
        let (problem, _soln) = foodweb_problem::<M, M>(&context);
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        println!("{}", mass);
        insta::assert_yaml_snapshot!(mass.to_string());
    }
}
