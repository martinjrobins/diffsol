//! A network of synchronous generators, each governed by the swing equation and tied to every
//! other through the network.
//!
//! Units: angles are in radians and the speed deviations are in rad/s, both in the
//! synchronously rotating frame.

use diffsol::{
    ConstantOp, Context, Matrix, NonLinearOp, OdeBuilder, OdeEquations, OdeEquationsRef,
    OdeSolverProblem, Op, ParameterisedOp, Scalar, UnitCallable, Vector,
};
use num_traits::{FromPrimitive, Zero};

/// Generator inertia `M`, damping `D` and per-line susceptance `B`, all per unit.
// ANCHOR: constants
pub const INERTIA: f64 = 4.0;
pub const DAMPING: f64 = 0.6;
pub const SUSCEPTANCE: f64 = 0.25;
// ANCHOR_END: constants

/// `nbuses` generators, each tied to every other through the network. State `i < nbuses` is the
/// rotor angle of bus `i` and state `nbuses + g` is the speed deviation of bus `g`, so a lane
/// holds `2 * nbuses` states.
// ANCHOR: equations
pub struct SwingEqn<M: Matrix> {
    ctx: M::C,
    demand: M::V,
    nbuses: usize,
    inertia: M::T,
    damping: M::T,
    susceptance: M::T,
}

impl<M: Matrix> SwingEqn<M> {
    pub fn new(nbuses: usize, ctx: M::C) -> Self {
        assert!(
            nbuses >= 3 && nbuses.is_multiple_of(2),
            "need an even network of 4+ buses"
        );
        // one parameter per lane; the builder fills it from `OdeBuilder::p`
        let demand = M::V::zeros(1, ctx.clone());
        Self {
            ctx,
            demand,
            nbuses,
            inertia: M::T::from_f64(INERTIA).unwrap(),
            damping: M::T::from_f64(DAMPING).unwrap(),
            susceptance: M::T::from_f64(SUSCEPTANCE).unwrap(),
        }
    }
}

pub struct SwingRhs<'a, M: Matrix> {
    eqn: &'a SwingEqn<M>,
}

pub struct SwingInit<'a, M: Matrix> {
    eqn: &'a SwingEqn<M>,
}

impl<M: Matrix> NonLinearOp for SwingRhs<'_, M> {
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        // captured by value so the closure stays `Copy + Send`
        // and compiles for the device
        let n = self.eqn.nbuses;
        let (m, d, b) = (self.eqn.inertia, self.eqn.damping, self.eqn.susceptance);
        y.for_each_elem(
            [x, &self.eqn.demand],
            move |y: &mut M::T, [x, demand]: [&[M::T]; 2], _lane: usize, i: usize| {
                if i < n {
                    // rotor angle: d(delta)/dt = omega
                    *y = x[n + i];
                } else {
                    let g = i - n;
                    // every bus is tied to every other, as a dense admittance matrix.
                    let mut flow = M::T::zero();
                    for j in 0..n {
                        flow += (x[g] - x[j]).sin();
                    }
                    let flow = b * flow;
                    // the uncertain load at bus 0 drives the system
                    let inj = if g == 0 { -demand[0] } else { M::T::zero() };
                    *y = (inj - d * x[n + g] - flow) / m;
                }
            },
        );
    }
}

impl<M: Matrix> ConstantOp for SwingInit<'_, M> {
    fn call_inplace(&self, _t: M::T, y: &mut M::V) {
        // the nominal grid is balanced, so it starts at rest
        y.for_each_elem(
            [],
            |y: &mut M::T, _: [&[M::T]; 0], _lane: usize, _i: usize| *y = M::T::zero(),
        );
    }
}
// ANCHOR_END: equations

macro_rules! impl_swing_op {
    ($name:ident) => {
        impl<M: Matrix> Op for $name<'_, M> {
            type M = M;
            type V = M::V;
            type T = M::T;
            type C = M::C;

            fn nstates(&self) -> usize {
                2 * self.eqn.nbuses
            }
            fn nout(&self) -> usize {
                2 * self.eqn.nbuses
            }
            fn nparams(&self) -> usize {
                1
            }
            fn context(&self) -> &Self::C {
                &self.eqn.ctx
            }
        }
    };
}

impl_swing_op!(SwingRhs);
impl_swing_op!(SwingInit);

impl<M: Matrix> Op for SwingEqn<M> {
    type M = M;
    type V = M::V;
    type T = M::T;
    type C = M::C;

    fn nstates(&self) -> usize {
        2 * self.nbuses
    }
    fn nout(&self) -> usize {
        2 * self.nbuses
    }
    fn nparams(&self) -> usize {
        1
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<'a, M: Matrix> OdeEquationsRef<'a> for SwingEqn<M> {
    type Rhs = SwingRhs<'a, M>;
    type Init = SwingInit<'a, M>;
    type Mass = ParameterisedOp<'a, UnitCallable<M>>;
    type Root = ParameterisedOp<'a, UnitCallable<M>>;
    type Out = ParameterisedOp<'a, UnitCallable<M>>;
    type Reset = ParameterisedOp<'a, UnitCallable<M>>;
}

impl<M: Matrix> OdeEquations for SwingEqn<M> {
    fn rhs(&self) -> SwingRhs<'_, M> {
        SwingRhs { eqn: self }
    }
    fn init(&self) -> SwingInit<'_, M> {
        SwingInit { eqn: self }
    }
    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        None
    }
    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        None
    }
    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn set_params(&mut self, p: &Self::V) {
        self.demand.copy_from(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        p.copy_from(&self.demand);
    }
}

// ANCHOR: problem
/// One problem holding `demands.len()` independent grids, one per batch lane.
#[allow(clippy::type_complexity)]
pub fn swing_problem<M: Matrix + 'static>(
    nbuses: usize,
    demands: &[f64],
) -> OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T, C = M::C>> {
    let ctx = M::C::default().clone_with_nbatch(demands.len()).unwrap();
    OdeBuilder::<M>::new()
        .context(ctx.clone())
        // one value per lane, so the parameter vector is `nparams * nbatch` long
        .p(demands.iter().copied())
        .rtol(1e-6)
        .atol([1e-8])
        .build_from_eqn(SwingEqn::new(nbuses, ctx))
        .unwrap()
}
// ANCHOR_END: problem

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{NalgebraMat, OdeSolverMethod, OdeSolverStopReason, Vector};

    type CpuM = NalgebraMat<f64>;

    /// With no secondary control, a demand step `dP` leaves the ring turning slow by
    /// `dP / (nbuses * D)` once the swing has damped out.
    #[test]
    fn settles_to_the_analytic_frequency_offset() {
        let nbuses = 8;
        let demand = 0.2;
        let problem = swing_problem::<CpuM>(nbuses, &[demand]);
        let mut solver = problem.tsit45().unwrap();
        solver.set_stop_time(60.0).unwrap();
        while !matches!(solver.step().unwrap(), OdeSolverStopReason::TstopReached) {}
        let final_state = solver.state().y.clone_as_vec();

        let expected = -demand / (nbuses as f64 * DAMPING);
        for omega in &final_state[nbuses..] {
            assert!(
                (omega - expected).abs() < 1e-3,
                "expected {expected}, got {omega}"
            );
        }
    }
}
