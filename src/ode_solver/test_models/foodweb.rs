use std::rc::Rc;

use crate::{
    find_non_zeros_linear, find_non_zeros_nonlinear, ode_solver::problem::OdeSolverSolution,
    ConstantOp, DenseMatrix, JacobianColoring, LinearOp, Matrix, MatrixSparsity, NonLinearOp,
    OdeEquations, OdeSolverProblem, Op, UnitCallable, Vector,
};
use num_traits::Zero;

const NPREY: usize = 1;
const NUM_SPECIES: usize = 2 * NPREY;
const AX: f64 = 1.0;
const AY: f64 = 1.0;

const AA: f64 = 1.0;
const EE: f64 = 10000.0;
const GG: f64 = 0.5e-6;
const BB: f64 = 1.0;
const DPREY: f64 = 1.0;
const DPRED: f64 = 0.05;

const ALPHA: f64 = 50.0;
const BETA: f64 = 1000.0;

#[cfg(feature = "diffsl")]
pub fn foodweb_diffsl<MD, M, const NX: usize>(
    diffsl_context: &mut crate::DiffSlContext<M>,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T> + '_>,
    OdeSolverSolution<M::V>,
)
where
    MD: DenseMatrix<T = f64>,
    M: Matrix<V = MD::V, T = MD::T>,
{
    use crate::OdeBuilder;

    let context = FoodWebContext::<MD, M, NX>::default();
    let (problem, _soln) = foodweb_problem::<MD, M, NX>(&context);
    let u0 = problem.eqn.init().call(0.0);
    let diffop = FoodWebDiff::<M, NX>::new(&u0, 0.0);
    let diff = diffop.jacobian(&u0, 0.0);
    let diff_diffsl = diff
        .triplet_iter()
        .map(|(i, j, v)| format!("            ({}, {}): {}", i, j, *v))
        .collect::<Vec<_>>()
        .join(",\n");

    let mut xx = M::V::zeros(NX * NX);
    let mut yy = M::V::zeros(NX * NX);
    for jy in 0..NX {
        let y = jy as f64 * AY / (NX as f64 - 1.0);
        for jx in 0..NX {
            let x = jx as f64 * AX / (NX as f64 - 1.0);
            let loc = jx + NX * jy;
            xx[loc] = x;
            yy[loc] = y;
        }
    }
    let xx_diffsl = xx
        .as_slice()
        .iter()
        .map(|v| format!("            {}", v))
        .collect::<Vec<_>>()
        .join(",\n");
    let yy_diffsl = yy
        .as_slice()
        .iter()
        .map(|v| format!("            {}", v))
        .collect::<Vec<_>>()
        .join(",\n");

    let code = format!(
        "
        in = []
        AA {{ 1.0 }}
        EE {{ 10000.0 }}
        GG {{ 0.5e-6 }}
        BB {{ 1.0 }}
        ALPHA {{ 50.0 }}
        BETA {{ 1000.0 }}
        PI {{ 3.141592653589793 }}
        DPREY {{ 1.0 }}
        DPRED {{ 0.05 }}

        D_ij {{
{}
        }}
        xx_i {{
{}
        }}
        yy_i {{
{}
        }}
        tl_i {{ 
            (0): 1.0, 
            (1:{n}): 0.0,
        }}
        br_i {{ 
            (0:{n1}): 0.0, 
            ({n1}): 1.0,
        }}
        b_i {{
            (1.0 + ALPHA * xx_i * yy_i + BETA * sin(4.0 * PI * xx_i) * sin(4.0 * PI * yy_i))
        }}
        u_i {{
            c1 = 10.0 + pow(16.0 * xx_i * (1.0 - xx_i) * yy_i * (1.0 - yy_i), 2),
            ({n}:{n2}): c2 = 1.0e5,
        }}
        dudt_i {{
            (0:{n}): dc1dt = 0,
            ({n}:{n2}): dc2dt = 0,
        }}
        M_i {{
            dc1dt_i,
            ({n}:{n2}): 0,
        }}
        c1diff_i {{
            DPREY * D_ij * c1_j,
        }}
        c2diff_i {{
            DPRED * D_ij * c2_j,
        }}
        F_i {{
            c1diff_i + c1_i * (BB * b_i - AA * c1_i - GG * c2_i),
            c2diff_i + c2_i * (-BB * b_i + EE * c1_i - AA * c2_i),
        }}
        out_i {{
            tl_j * c1_j,
            br_j * c1_j,
            tl_j * c2_j,
            br_j * c2_j,
        }}",
        diff_diffsl,
        xx_diffsl,
        yy_diffsl,
        n = NX * NX,
        n1 = NX * NX - 1,
        n2 = 2 * NX * NX,
    );

    diffsl_context.recompile(code.as_str()).unwrap();

    let problem = OdeBuilder::new()
        .rtol(1e-5)
        .atol([1e-5])
        .build_diffsl(diffsl_context)
        .unwrap();
    let soln = soln::<M>();
    (problem, soln)
}

pub struct FoodWebContext<MD, M, const NX: usize>
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
impl<MD, M, const NX: usize> FoodWebContext<MD, M, NX>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    const DX: f64 = AX / (NX as f64 - 1.0);
    const DY: f64 = AY / (NX as f64 - 1.0);

    pub fn new() -> Self {
        let mut acoef = MD::zeros(NUM_SPECIES, NUM_SPECIES);
        let mut bcoef = M::V::zeros(NUM_SPECIES);
        let mut cox = M::V::zeros(NUM_SPECIES);
        let mut coy = M::V::zeros(NUM_SPECIES);
        let nstates = NUM_SPECIES * NX * NX;

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
            cox[i] = M::T::from(DPREY / Self::DX.powi(2));
            cox[i + NPREY] = M::T::from(DPRED / Self::DX.powi(2));
            coy[i] = M::T::from(DPREY / Self::DY.powi(2));
            coy[i + NPREY] = M::T::from(DPRED / Self::DY.powi(2));
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

impl<MD, M, const NX: usize> Default for FoodWebContext<MD, M, NX>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    fn default() -> Self {
        Self::new()
    }
}

struct FoodWebInit<'a, MD, M, const NX: usize>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M, NX>,
}

// macro for bringing in constants from Context
macro_rules! context_consts {
    ($name:ident) => {
        impl<'a, MD, M, const NX: usize> $name<'a, MD, M, NX>
        where
            MD: DenseMatrix,
            M: Matrix<V = MD::V, T = MD::T>,
        {
            pub fn new(context: &'a FoodWebContext<MD, M, NX>) -> Self {
                Self { context }
            }
        }
    };
}

// macro for impl ops
macro_rules! impl_op {
    ($name:ident) => {
        impl<'a, MD, M, const NX: usize> Op for $name<'a, MD, M, NX>
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

impl<'a, MD, M, const NX: usize> ConstantOp for FoodWebInit<'a, MD, M, NX>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    #[allow(unused_mut)]
    fn call_inplace(&self, _t: M::T, mut y: &mut M::V) {
        let nsmx: usize = NUM_SPECIES * NX;
        let dx: f64 = AX / (NX as f64 - 1.0);
        let dy: f64 = AY / (NX as f64 - 1.0);

        /* Loop over grid, load cc values and id values. */
        for jy in 0..NX {
            let yy = jy as f64 * dy;
            let yloc = nsmx * jy;
            for jx in 0..NX {
                let xx = jx as f64 * dx;
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

struct FoodWebRhs<'a, MD, M, const NX: usize>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M, NX>,
    pub sparsity: Option<M::Sparsity>,
    pub coloring: Option<JacobianColoring<M>>,
}

impl<'a, MD, M, const NX: usize> FoodWebRhs<'a, MD, M, NX>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new(context: &'a FoodWebContext<MD, M, NX>, y0: &M::V, t0: M::T) -> Self {
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

impl<'a, MD, M, const NX: usize> Op for FoodWebRhs<'a, MD, M, NX>
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

impl<'a, MD, M, const NX: usize> NonLinearOp for FoodWebRhs<'a, MD, M, NX>
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
    #[allow(unused_mut)]
    fn call_inplace(&self, x: &M::V, _t: M::T, mut y: &mut M::V) {
        let nsmx: usize = NUM_SPECIES * NX;
        let dx: f64 = AX / (NX as f64 - 1.0);
        let dy: f64 = AY / (NX as f64 - 1.0);

        let mut rates = [M::T::zero(); NUM_SPECIES];
        /* Loop over grid points, evaluate interaction vector (length ns),
        form diffusion difference terms, and load crate.                    */
        for jy in 0..NX {
            let yy = jy as f64 * dy;
            let idyu = if jy != NX - 1 {
                nsmx as i32
            } else {
                -(nsmx as i32)
            };
            let idyl = if jy != 0 { nsmx as i32 } else { -(nsmx as i32) };

            for jx in 0..NX {
                let xx = jx as f64 * dx;
                let idxu = if jx != NX - 1 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let idxl = if jx != 0 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let loc = NUM_SPECIES * jx + nsmx * jy;

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

    #[allow(unused_mut)]
    fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, mut y: &mut M::V) {
        let nsmx: usize = NUM_SPECIES * NX;
        let dx: f64 = AX / (NX as f64 - 1.0);
        let dy: f64 = AY / (NX as f64 - 1.0);

        let mut rates = [M::T::zero(); NUM_SPECIES];
        let mut drates = [M::T::zero(); NUM_SPECIES];
        /* Loop over grid points, evaluate interaction vector (length ns),
        form diffusion difference terms, and load crate.                    */
        for jy in 0..NX {
            let yy = jy as f64 * dy;
            let idyu = if jy != NX - 1 {
                nsmx as i32
            } else {
                -(nsmx as i32)
            };
            let idyl = if jy != 0 { nsmx as i32 } else { -(nsmx as i32) };

            for jx in 0..NX {
                let xx = jx as f64 * dx;
                let idxu = if jx != NX - 1 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let idxl = if jx != 0 {
                    NUM_SPECIES as i32
                } else {
                    -(NUM_SPECIES as i32)
                };
                let loc = NUM_SPECIES * jx + nsmx * jy;

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

struct FoodWebMass<'a, MD, M, const NX: usize>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M, NX>,
    pub sparsity: Option<M::Sparsity>,
    pub coloring: Option<JacobianColoring<M>>,
}

impl<'a, MD, M, const NX: usize> FoodWebMass<'a, MD, M, NX>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new(context: &'a FoodWebContext<MD, M, NX>, t0: M::T) -> Self {
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

impl<'a, MD, M, const NX: usize> Op for FoodWebMass<'a, MD, M, NX>
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

impl<'a, MD, M, const NX: usize> LinearOp for FoodWebMass<'a, MD, M, NX>
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
    #[allow(unused_mut)]
    fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, mut y: &mut Self::V) {
        let nsmx: usize = NUM_SPECIES * NX;
        /* Loop over all grid points, setting residual values appropriately
        for differential or algebraic components.                        */
        for jy in 0..NX {
            let yloc = nsmx * jy;
            for jx in 0..NX {
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

struct FoodWebOut<'a, MD, M, const NX: usize>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub context: &'a FoodWebContext<MD, M, NX>,
}

context_consts!(FoodWebOut);

impl<'a, MD, M, const NX: usize> Op for FoodWebOut<'a, MD, M, NX>
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

impl<'a, MD, M, const NX: usize> NonLinearOp for FoodWebOut<'a, MD, M, NX>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    #[allow(unused_mut)]
    fn call_inplace(&self, x: &M::V, _t: M::T, mut y: &mut M::V) {
        let nsmx: usize = NUM_SPECIES * NX;
        let jx_tl = 0;
        let jy_tl = 0;
        let jx_br = NX - 1;
        let jy_br = NX - 1;
        let loc_tl = NUM_SPECIES * jx_tl + nsmx * jy_tl;
        let loc_br = NUM_SPECIES * jx_br + nsmx * jy_br;
        for is in 0..NUM_SPECIES {
            y[2 * is] = x[loc_tl + is];
            y[2 * is + 1] = x[loc_br + is];
        }
    }

    #[allow(unused_mut)]
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, mut y: &mut Self::V) {
        let nsmx: usize = NUM_SPECIES * NX;

        let jx_tl = 0;
        let jy_tl = 0;
        let jx_br = NX - 1;
        let jy_br = NX - 1;
        let loc_tl = NUM_SPECIES * jx_tl + nsmx * jy_tl;
        let loc_br = NUM_SPECIES * jx_br + nsmx * jy_br;
        for is in 0..NUM_SPECIES {
            y[2 * is] = v[loc_tl + is];
            y[2 * is + 1] = v[loc_br + is];
        }
    }
}

struct FoodWeb<'a, MD, M, const NX: usize>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub rhs: Rc<FoodWebRhs<'a, MD, M, NX>>,
    pub mass: Rc<FoodWebMass<'a, MD, M, NX>>,
    pub init: Rc<FoodWebInit<'a, MD, M, NX>>,
    pub out: Rc<FoodWebOut<'a, MD, M, NX>>,
}

impl<'a, MD, M, const NX: usize> FoodWeb<'a, MD, M, NX>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    pub fn new(context: &'a FoodWebContext<MD, M, NX>, t0: M::T) -> Self {
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

impl<'a, MD, M, const NX: usize> OdeEquations for FoodWeb<'a, MD, M, NX>
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    type M = M;
    type V = M::V;
    type T = M::T;
    type Init = FoodWebInit<'a, MD, M, NX>;
    type Rhs = FoodWebRhs<'a, MD, M, NX>;
    type Mass = FoodWebMass<'a, MD, M, NX>;
    type Root = UnitCallable<M>;
    type Out = FoodWebOut<'a, MD, M, NX>;

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

struct FoodWebDiff<M, const NX: usize>
where
    M: Matrix,
{
    pub sparsity: Option<M::Sparsity>,
    pub coloring: Option<JacobianColoring<M>>,
}

impl<M, const NX: usize> FoodWebDiff<M, NX>
where
    M: Matrix,
{
    pub fn new(y0: &M::V, t0: M::T) -> Self {
        let mut ret = Self {
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

impl<M, const NX: usize> Op for FoodWebDiff<M, NX>
where
    M: Matrix,
{
    type M = M;
    type V = M::V;
    type T = M::T;

    fn nout(&self) -> usize {
        NX * NX
    }
    fn nparams(&self) -> usize {
        0
    }
    fn nstates(&self) -> usize {
        NX * NX
    }
    fn sparsity(&self) -> Option<M::SparsityRef<'_>> {
        self.sparsity.as_ref().map(|s| s.as_ref())
    }
}

impl<M, const NX: usize> NonLinearOp for FoodWebDiff<M, NX>
where
    M: Matrix,
{
    #[allow(unused_mut)]
    fn call_inplace(&self, x: &M::V, _t: M::T, mut y: &mut M::V) {
        let nsmx: usize = NX;
        let dx = AX / (NX as f64 - 1.0);
        let dy = AY / (NX as f64 - 1.0);
        let cox = M::T::from(1.0 / dx.powi(2));
        let coy = M::T::from(1.0 / dy.powi(2));

        /* Loop over grid points, evaluate interaction vector (length ns),
        form diffusion difference terms, and load crate.                    */
        for jy in 0..NX {
            let idyu = if jy != NX - 1 {
                nsmx as i32
            } else {
                -(nsmx as i32)
            };
            let idyl = if jy != 0 { nsmx as i32 } else { -(nsmx as i32) };

            for jx in 0..NX {
                let idxu = if jx != NX - 1 { 1 } else { -1 };
                let idxl = if jx != 0 { 1 } else { -1 };
                let loc = jx + nsmx * jy;

                /* Differencing in y. */
                let dcyli = x[loc] - x[usize::try_from(loc as i32 - idyl).unwrap()];
                let dcyui = x[usize::try_from(loc as i32 + idyu).unwrap()] - x[loc];

                /* Differencing in x. */
                let dcxli = x[loc] - x[usize::try_from(loc as i32 - idxl).unwrap()];
                let dcxui = x[usize::try_from(loc as i32 + idxu).unwrap()] - x[loc];

                /* Compute the crate values at (xx,yy). */
                y[loc] = coy * (dcyui - dcyli) + cox * (dcxui - dcxli);
            }
        }
    }

    #[allow(unused_mut)]
    fn jac_mul_inplace(&self, _x: &M::V, _t: M::T, v: &M::V, mut y: &mut M::V) {
        let nsmx: usize = NX;
        let dx = AX / (NX as f64 - 1.0);
        let dy = AY / (NX as f64 - 1.0);
        let cox = M::T::from(1.0 / dx.powi(2));
        let coy = M::T::from(1.0 / dy.powi(2));

        /* Loop over grid points, evaluate interaction vector (length ns),
        form diffusion difference terms, and load crate.                    */
        for jy in 0..NX {
            let idyu = if jy != NX - 1 {
                nsmx as i32
            } else {
                -(nsmx as i32)
            };
            let idyl = if jy != 0 { nsmx as i32 } else { -(nsmx as i32) };

            for jx in 0..NX {
                let idxu = if jx != NX - 1 { 1 } else { -1 };
                let idxl = if jx != 0 { 1 } else { -1 };
                let loc = jx + nsmx * jy;

                /* Differencing in y. */
                let dcyli = v[loc] - v[usize::try_from(loc as i32 - idyl).unwrap()];
                let dcyui = v[usize::try_from(loc as i32 + idyu).unwrap()] - v[loc];

                /* Differencing in x. */
                let dcxli = v[loc] - v[usize::try_from(loc as i32 - idxl).unwrap()];
                let dcxui = v[usize::try_from(loc as i32 + idxu).unwrap()] - v[loc];

                /* Compute the crate values at (xx,yy). */
                y[loc] = coy * (dcyui - dcyli) + cox * (dcxui - dcxli);
            }
        }
    }
}

fn soln<M: Matrix>() -> OdeSolverSolution<M::V> {
    let mut soln = OdeSolverSolution {
        solution_points: Vec::new(),
        sens_solution_points: None,
        rtol: M::T::from(1e-4),
        atol: M::V::from_element(2 * NUM_SPECIES, M::T::from(1e-4)),
    };
    let data = vec![
        (vec![10.0, 10.0, 99999.0, 99949.0], 0.0),
        (
            vec![
                9.997887753650794,
                10.498336872161198,
                99979.21262678975,
                104933.61130371751,
            ],
            0.001,
        ),
        (
            vec![
                116.7394053543608,
                141.3349347208864,
                1167406.222331898,
                1413309.7156706247,
            ],
            0.01,
        ),
        (
            vec![
                169.50991588474182,
                196.55298551613117,
                1695106.6267256583,
                1965486.1821950572,
            ],
            0.1,
        ),
        (
            vec![
                169.50991230736778,
                196.55298216342456,
                1695106.5909521726,
                1965486.1486681814,
            ],
            0.4,
        ),
        (
            vec![
                169.5099123071205,
                196.55298216319915,
                1695106.5909496995,
                1965486.1486659276,
            ],
            0.7,
        ),
        (
            vec![
                169.50991230687316,
                196.55298216297376,
                1695106.5909472264,
                1965486.1486636735,
            ],
            1.0,
        ),
    ];
    for (values, time) in data {
        let values = M::V::from_vec(values.iter().map(|v| M::T::from(*v)).collect::<Vec<_>>());
        let time = M::T::from(time);
        soln.push(values, time);
    }
    soln
}

pub fn foodweb_problem<MD, M, const NX: usize>(
    context: &FoodWebContext<MD, M, NX>,
) -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T> + '_>,
    OdeSolverSolution<M::V>,
)
where
    MD: DenseMatrix,
    M: Matrix<V = MD::V, T = MD::T>,
{
    let rtol = M::T::from(1e-5);
    let atol = M::V::from_element(NUM_SPECIES * NX * NX, M::T::from(1e-5));
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
        const NX: usize = 10;
        let context = FoodWebContext::<M, M, NX>::new();
        let (problem, _soln) = foodweb_problem::<M, M, NX>(&context);
        let u0 = problem.eqn.init().call(0.0);
        let jac = problem.eqn.rhs().jacobian(&u0, 0.0);

        // check the jacobian via finite differences
        let h = 1e-5;
        for i in 0..jac.ncols() {
            let mut vplus = u0.clone();
            let mut vminus = u0.clone();
            vplus[i] += h;
            vminus[i] -= h;
            let yplus = problem.eqn.rhs().call(&vplus, 0.0);
            let yminus = problem.eqn.rhs().call(&vminus, 0.0);
            let fdiff = (yplus - yminus) / (2.0 * h);
            for j in 0..jac.nrows() {
                assert!(
                    (jac[(j, i)] - fdiff[j]).abs() < 1e-1,
                    "jac[{}, {}] = {} (expect {})",
                    j,
                    i,
                    jac[(j, i)],
                    fdiff[j]
                );
            }
        }
    }

    #[cfg(feature = "diffsl")]
    #[test]
    fn test_diffsl() {
        type M = nalgebra::DMatrix<f64>;
        const NX: usize = 10;
        let context = FoodWebContext::<M, M, NX>::new();
        let (problem, _soln) = foodweb_problem::<M, M, NX>(&context);
        let u0 = problem.eqn.init().call(0.0);
        let jac = problem.eqn.rhs().jacobian(&u0, 0.0);
        let y0 = problem.eqn.rhs().call(&u0, 0.0);

        let mut diffsl_context = crate::DiffSlContext::<M>::default();
        let (problem_diffsl, _soln) = foodweb_diffsl::<M, M, NX>(&mut diffsl_context);
        let u0_diffsl = problem_diffsl.eqn.init().call(0.0);
        for i in 0..u0.len() {
            let i_diffsl = if i % NUM_SPECIES >= NPREY {
                NX * NX + i / NUM_SPECIES
            } else {
                i / NUM_SPECIES
            };
            assert!(
                ((u0[i] - u0_diffsl[i_diffsl]) / u0[i]).abs() < 1e-3,
                "u0[{}] = {} (expect {})",
                i,
                u0[i],
                u0_diffsl[i_diffsl]
            );
        }

        let y0_diffsl = problem_diffsl.eqn.rhs().call(&u0_diffsl, 0.0);
        for i in 0..y0.len() {
            let i_diffsl = if i % NUM_SPECIES >= NPREY {
                NX * NX + i / NUM_SPECIES
            } else {
                i / NUM_SPECIES
            };
            assert!(
                ((y0[i] - y0_diffsl[i_diffsl]) / y0[i]).abs() < 1e-3,
                "y0[{}] = {} (expect {})",
                i,
                y0[i],
                y0_diffsl[i_diffsl]
            );
        }

        let jac_diffsl = problem_diffsl.eqn.rhs().jacobian(&u0_diffsl, 0.0);
        for i in 0..jac.ncols() {
            for j in 0..jac.nrows() {
                let i_diffsl = if i % NUM_SPECIES >= NPREY {
                    NX * NX + i / NUM_SPECIES
                } else {
                    i / NUM_SPECIES
                };
                let j_diffsl = if j % NUM_SPECIES >= NPREY {
                    NX * NX + j / NUM_SPECIES
                } else {
                    j / NUM_SPECIES
                };
                assert!(
                    (jac[(j, i)] - jac_diffsl[(j_diffsl, i_diffsl)]).abs() < 1e-3,
                    "jac[{}, {}] = {} (expect {})",
                    j,
                    i,
                    jac[(j, i)],
                    jac_diffsl[(j_diffsl, i_diffsl)]
                );
            }
        }
    }

    #[test]
    fn test_mass() {
        type M = nalgebra::DMatrix<f64>;
        const NX: usize = 10;
        let context = FoodWebContext::<M, M, NX>::new();
        let (problem, _soln) = foodweb_problem::<M, M, NX>(&context);
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        for i in 0..mass.ncols() {
            for j in 0..mass.nrows() {
                if i == j && i % NUM_SPECIES < NPREY {
                    assert_eq!(mass[(i, j)], 1.0);
                } else {
                    assert_eq!(mass[(i, j)], 0.0);
                }
            }
        }
    }
}
