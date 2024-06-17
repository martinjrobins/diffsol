use std::rc::Rc;

use crate::{ConstantOp, DenseMatrix, LinearOp, Matrix, NonLinearOp, OdeEquations, Op, UnitCallable, Vector};
use num_traits::Zero;

const NPREY: usize = 3;
const NUM_SPECIES: usize = 2 * NPREY;
const NSMX: usize = NUM_SPECIES * MX;
const MX: usize = 20;
const MY: usize = 20;
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

struct FoodWebContext<MD, M> 
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
 * The DAE system is solved by IDA using the SPGMR linear solver.
 * Output is printed at t = 0, .001, .01, .1, .4, .7, 1.
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
    fn new() -> Self {
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
    

    fn call_inplace(&self, t: M::T, y: &mut M::V) {
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


        /* Set c' for predators to 0. */
        for jy in 0..MY {
            let yloc = NSMX * jy;
            for jx in 0..MX {
                let loc = yloc + NUM_SPECIES * jx;
                for is in NPREY..NUM_SPECIES {
                    y[loc + is] = M::T::zero();
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
}

context_consts!(FoodWebRhs);
impl_op!(FoodWebRhs);

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
            let idyu = if jy != MY - 1 { NSMX as i32 } else { -(NSMX as i32) };
            let idyl = if jy != 0 { NSMX as i32 } else { -(NSMX as i32) };

            for jx in 0..MX {
                let xx = jx as f64 * DX;
                let idxu = if jx != MX - 1 { NUM_SPECIES as i32 } else { -(NUM_SPECIES as i32) };
                let idxl = if jx != 0 { NUM_SPECIES as i32 } else { -(NUM_SPECIES as i32) };
                let loc = NUM_SPECIES * jx + NSMX * jy;

                /*
                * WebRates: Evaluate reaction rates at a given spatial point.
                * At a given (x,y), evaluate the array of ns reaction terms R.
                */
                for is in 0..NUM_SPECIES {
                    let mut dp = M::T::zero();
                    for js in 0..NUM_SPECIES {
                        dp += self.context.acoef[(is, js)] * x[loc + js];
                    }
                    rates[is] = dp;
                }
                let fac = M::T::from(1.0 + ALPHA * xx * yy + BETA * (4.0 * std::f64::consts::PI * xx).sin() * (4.0 * std::f64::consts::PI * yy).sin());

                for is in 0..NUM_SPECIES {
                    rates[is] = x[loc + is] * (self.context.bcoef[is] * fac + rates[is]);
                }

                /* Loop over species, do differencing, load crate segment. */
                for is in 0..NUM_SPECIES {
                    /* Differencing in y. */
                    let dcyli = x[loc + is] - x[(loc as i32 - idyl + is as i32) as usize];
                    let dcyui = x[(loc as i32 + idyu + is as i32) as usize] - x[loc + is];

                    /* Differencing in x. */
                    let dcxli = x[loc + is] - x[(loc as i32 - idxl + is as i32) as usize];
                    let dcxui = x[(loc as i32 + idxu + is as i32) as usize] - x[loc + is];

                    /* Compute the crate values at (xx,yy). */
                    y[loc + is] = self.context.coy[is] * (dcyui - dcyli) + self.context.cox[is] * (dcxui - dcxli) +
                        rates[is];
                }
            }
        }
    }

    fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        let mut rates = [M::T::zero(); NUM_SPECIES];
        let mut drates = [M::T::zero(); NUM_SPECIES];
        /* Loop over grid points, evaluate interaction vector (length ns),
            form diffusion difference terms, and load crate.                    */
        for jy in 0..MY {
            let yy = jy as f64 * DY;
            let idyu = if jy != MY - 1 { NSMX as i32 } else { -(NSMX as i32) };
            let idyl = if jy != 0 { NSMX as i32 } else { -(NSMX as i32) };

            for jx in 0..MX {
                let xx = jx as f64 * DX;
                let idxu = if jx != MX - 1 { NUM_SPECIES as i32 } else { -(NUM_SPECIES as i32) };
                let idxl = if jx != 0 { NUM_SPECIES as i32 } else { -(NUM_SPECIES as i32) };
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
                let fac = M::T::from(1.0 + ALPHA * xx * yy + BETA * (4.0 * std::f64::consts::PI * xx).sin() * (4.0 * std::f64::consts::PI * yy).sin());

                for is in 0..NUM_SPECIES {
                    rates[is] = x[loc + is] * (self.context.bcoef[is] * fac + rates[is]);
                    drates[is] = x[loc + is] * drates[is] + v[loc + is] * (self.context.bcoef[is] * fac + rates[is]);
                }

                /* Loop over species, do differencing, load crate segment. */
                for is in 0..NUM_SPECIES {
                    /* Differencing in y. */
                    let dcyli = v[loc + is] - v[(loc as i32 - idyl + is as i32) as usize];
                    let dcyui = v[(loc as i32 + idyu + is as i32) as usize] - v[loc + is];

                    /* Differencing in x. */
                    let dcxli = v[loc + is] - v[(loc as i32 - idxl + is as i32) as usize];
                    let dcxui = v[(loc as i32 + idxu + is as i32) as usize] - v[loc + is];

                    /* Compute the crate values at (xx,yy). */
                    y[loc + is] = self.context.coy[is] * (dcyui - dcyli) + self.context.cox[is] * (dcxui - dcxli) +
                        drates[is];
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
}

context_consts!(FoodWebMass);
impl_op!(FoodWebMass);

impl<'a, MD, M> LinearOp for FoodWebMass<'a, MD, M> 
where
    MD: DenseMatrix + 'a,
    M: Matrix<V = MD::V, T = MD::T>,
{
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

    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, mut y: &mut Self::V) {
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
    fn set_params(&mut self, _p: Self::V) {
    }
}