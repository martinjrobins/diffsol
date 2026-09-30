//! Adaptive eight-stage Rodas5P Rosenbrock-Wanner method for identity-mass ODEs.
//!
//! The fifth-order solution, fourth-order embedded estimator, and fourth-order
//! dense extension use coefficients published by Steinebach (2023). A single
//! matrix `I - gamma*h*J` is factored per attempted step. The partial time
//! derivative is approximated from inside the attempted step, including when
//! the step ends at a declared forcing discontinuity.

use std::cell::Ref;

use num_traits::{FromPrimitive, One, Pow, Signed, ToPrimitive, Zero};

use crate::{
    error::{DiffsolError, OdeSolverError},
    op::sdirk::SdirkCallable,
    scale, DefaultDenseMatrix, DenseMatrix, ExplicitRkConfig, LinearSolver, NonLinearOp,
    OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason,
    Op, RkState, Scalar, StateRef, StateRefMut, Tableau, TableauMat, TableauVec, Vector,
};

use super::{jacobian_update::SolverState, runge_kutta::Rk, OdeSolverStatistics};

const GAMMA: f64 = 0.21193756319429014;

// Natural row/column orientation: A[i][j] and C[i][j] are used only for j<i.
// Source: G. Steinebach, BIT Numerical Mathematics 63, article 27 (2023),
// https://doi.org/10.1007/s10543-023-00967-x; coefficient transcription
// checked against OrdinaryDiffEqRosenbrock 2.4.0's MIT-licensed Rodas5P
// tableau (`src/rosenbrock_tableaus.jl`, RODAS5PA/PC/Pc/Pd/PH). See
// https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock.
const A: [[f64; 8]; 8] = [
    [0.0; 8],
    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        2.849394379747939,
        0.45842242204463923,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -6.954028509809101,
        2.489845061869568,
        -10.358996098473584,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        2.8029986275628964,
        0.5072464736228206,
        -0.3988312541770524,
        -0.04721187230404641,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
        0.0,
        0.0,
        0.0,
    ],
    [
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
        1.0,
        0.0,
        0.0,
    ],
    [
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
        1.0,
        1.0,
        0.0,
    ],
];

const C: [[f64; 8]; 8] = [
    [0.0; 8],
    [-14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [
        -17.97296035885952,
        -2.859693295451294,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        147.12150275711716,
        -1.41221402718213,
        71.68940251302358,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        165.43517024871676,
        -0.4592823456491126,
        42.90938336958603,
        -5.961986721573306,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        24.854864614690072,
        -3.0009227002832186,
        47.4931110020768,
        5.5814197821558125,
        -0.6610691825249471,
        0.0,
        0.0,
        0.0,
    ],
    [
        30.91273214028599,
        -3.1208243349937974,
        77.79954646070892,
        34.28646028294783,
        -19.097331116725623,
        -28.087943162872662,
        0.0,
        0.0,
    ],
    [
        37.80277123390563,
        -3.2571969029072276,
        112.26918849496327,
        66.9347231244047,
        -40.06618937091002,
        -54.66780262877968,
        -9.48861652309627,
        0.0,
    ],
];

const TIMES: [f64; 8] = [
    0.0,
    0.6358126895828704,
    0.4095798393397535,
    0.9769306725060716,
    0.4288403609558664,
    1.0,
    1.0,
    1.0,
];
const D: [f64; 8] = [
    0.21193756319429014,
    -0.42387512638858027,
    -0.3384627126235924,
    1.8046452872882734,
    2.325825639765069,
    0.0,
    0.0,
    0.0,
];
const B: [f64; 8] = [
    -7.502846399306121,
    2.561846144803919,
    -11.627539656261098,
    -0.18268767659942256,
    0.030198172008377946,
    1.0,
    1.0,
    1.0,
];
// H rows define the continuous extension
// y(theta)=(1-theta)y0 + theta[y1+(1-theta)(K1+theta(K2+theta K3))].
const H: [[f64; 8]; 3] = [
    [
        25.948786856663858,
        -2.5579724845846235,
        10.433815404888879,
        -2.3679251022685204,
        0.524948541321073,
        1.1241088310450404,
        0.4272876194431874,
        -0.17202221070155493,
    ],
    [
        -9.91568850695171,
        -0.9689944594115154,
        3.0438037242978453,
        -24.495224566215796,
        20.176138334709044,
        15.98066361424651,
        -6.789040303419874,
        -6.710236069923372,
    ],
    [
        11.419903575922262,
        2.8879645146136994,
        72.92137995996029,
        80.12511834622643,
        -52.072871366152654,
        -59.78993625266729,
        -0.15582684282751913,
        4.883087185713722,
    ],
];

/// Fifth-order L-stable Rosenbrock-Wanner method with a fourth-order
/// embedded estimator and fourth-order dense output.
///
/// Supports identity-mass ODEs whose right-hand side provides a Jacobian
/// action. Mass matrices, integrated outputs, and augmented sensitivities are
/// rejected explicitly. The coefficient method itself also supports DAEs,
/// but those modes require a separate Diffsol implementation and tests.
pub struct Rodas5P<'a, Eqn, LS, M = <<Eqn as Op>::V as DefaultDenseMatrix>::M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    rk: Rk<'a, Eqn, M>,
    linear_solver: LS,
    op: SdirkCallable<&'a Eqn>,
    config: ExplicitRkConfig<Eqn::T>,
    discontinuity_stop: Option<Eqn::T>,
}

struct StepAttempt<V, T> {
    error_norm: T,
    y_end: V,
    f_end: V,
    stages: Vec<V>,
}

impl<Eqn, LS, M> Clone for Rodas5P<'_, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    fn clone(&self) -> Self {
        let op = self.op.clone_state(self.rk.problem().eqn());
        let mut linear_solver = LS::default();
        linear_solver.set_problem(&op);
        Self {
            rk: self.rk.clone(),
            linear_solver,
            op,
            config: self.config.clone(),
            discontinuity_stop: self.discontinuity_stop,
        }
    }
}

impl<'a, Eqn, LS, M> Rodas5P<'a, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    fn tableau() -> Tableau<Eqn::T> {
        // The RK tableau supplies shared storage, order and interpolation;
        // Rodas stages use A, C and D above rather than the RK stage kernel.
        let mut beta = [Eqn::T::zero(); 32];
        for i in 0..8 {
            let coeffs = [
                B[i] + H[0][i],
                -H[0][i] + H[1][i],
                -H[1][i] + H[2][i],
                -H[2][i],
            ];
            for p in 0..4 {
                beta[p * 8 + i] = Eqn::T::from_f64(coeffs[p]).unwrap();
            }
        }
        Tableau::new(
            TableauMat::zeros(8, 8),
            TableauVec::from_slice(&B.map(|x| Eqn::T::from_f64(x).unwrap())),
            TableauVec::from_slice(&TIMES.map(|x| Eqn::T::from_f64(x).unwrap())),
            TableauVec::zeros(8),
            5,
            Some(TableauMat::from_slice(8, 4, &beta)),
        )
    }

    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: RkState<Eqn::V>,
        mut linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        if problem.eqn.mass().is_some() {
            return Err(OdeSolverError::MassMatrixNotSupported.into());
        }
        if !state.s.is_empty() {
            return Err(OdeSolverError::SensitivityNotSupported.into());
        }
        if problem.integrate_out {
            return Err(OdeSolverError::IntegratedOutputNotSupported.into());
        }
        let rk = Rk::new(problem, state, Self::tableau())?;
        let op = SdirkCallable::new(&problem.eqn, Eqn::T::one(), problem.context().clone());
        linear_solver.set_problem(&op);
        Ok(Self {
            rk,
            linear_solver,
            op,
            config: ExplicitRkConfig::new(&problem.ode_options),
            discontinuity_stop: None,
        })
    }

    pub fn get_statistics(&self) -> &OdeSolverStatistics {
        self.rk.get_statistics()
    }

    /// Stop at a known time discontinuity in the right-hand side. Endpoint
    /// stages are evaluated on the incoming side; the accepted time is still
    /// exactly `tstop`. Ordinary output times should use `set_stop_time`.
    pub fn set_discontinuity_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(tstop)?;
        self.discontinuity_stop = Some(tstop);
        Ok(())
    }

    fn endpoint_eval_time(&self, h: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        let start = self.rk.state().t;
        let end = start + h;
        let Some(stop) = self.discontinuity_stop else {
            return Ok(end);
        };
        let roundoff = Eqn::T::from_f64(100.0).unwrap() * Eqn::T::EPSILON * (end.abs() + h.abs());
        if (end - stop).abs() > roundoff {
            return Ok(end);
        }
        // Use a representable point as close as possible to the incoming
        // side. sqrt(epsilon) shifts the forcing enough to bias stiff systems.
        let nominal = Eqn::T::EPSILON * (Eqn::T::one() + end.abs());
        let quarter = h.abs() / Eqn::T::from_f64(4.0).unwrap();
        let delta = if nominal < quarter { nominal } else { quarter };
        let shifted = if h >= Eqn::T::zero() {
            end - delta
        } else {
            end + delta
        };
        let inside = if h >= Eqn::T::zero() {
            start < shifted && shifted < end
        } else {
            end < shifted && shifted < start
        };
        if !inside {
            return Err(OdeSolverError::Other(
                "Rodas5P discontinuity has no representable incoming endpoint".into(),
            )
            .into());
        }
        Ok(shifted)
    }

    fn attempt(
        &mut self,
        h: Eqn::T,
        retry: bool,
    ) -> Result<StepAttempt<Eqn::V, Eqn::T>, DiffsolError> {
        let state = self.rk.state();
        let y0 = state.y.clone();
        let t = state.t;
        let n = y0.len();
        let ctx = self.rk.problem().context().clone();
        let gamma = Eqn::T::from_f64(GAMMA).unwrap();

        let mut f0 = Eqn::V::zeros(n, ctx.clone());
        self.rk.problem().eqn.rhs().call_inplace(&y0, t, &mut f0);
        // Three-point one-sided derivative has O(delta^2) truncation error.
        // Keep both probes strictly inside the attempted step, so neither
        // reads the far side of a stop-time discontinuity.
        let nominal =
            Eqn::T::EPSILON.pow(Eqn::T::from_f64(1.0 / 3.0).unwrap()) * (Eqn::T::one() + t.abs());
        let third = Eqn::T::from_f64(1.0 / 3.0).unwrap();
        let bounded = if nominal < h.abs() * third {
            nominal
        } else {
            h.abs() * third
        };
        let delta = if h >= Eqn::T::zero() {
            bounded
        } else {
            -bounded
        };
        let probe1 = t + delta;
        let probe2 = t + delta + delta;
        let end = t + h;
        let probes_inside = if h >= Eqn::T::zero() {
            t < probe1 && probe1 < probe2 && probe2 < end
        } else {
            end < probe2 && probe2 < probe1 && probe1 < t
        };
        if !probes_inside {
            return Err(OdeSolverError::Other(
                "Rodas5P time-derivative probe cannot advance within this step".into(),
            )
            .into());
        }
        let mut ft = Eqn::V::zeros(n, ctx.clone());
        let mut f2 = Eqn::V::zeros(n, ctx.clone());
        self.rk
            .problem()
            .eqn
            .rhs()
            .call_inplace(&y0, probe1, &mut ft);
        self.rk
            .problem()
            .eqn
            .rhs()
            .call_inplace(&y0, probe2, &mut f2);
        ft *= scale(Eqn::T::from_f64(4.0).unwrap());
        ft.axpy(-Eqn::T::one(), &f2, Eqn::T::one());
        ft.axpy(-Eqn::T::from_f64(3.0).unwrap(), &f0, Eqn::T::one());
        ft *= scale(Eqn::T::one() / (delta + delta));

        self.op.zero_phi();
        self.op.set_h(gamma * h);
        // The factorization is rebuilt for every attempt, so its nonlinear
        // Jacobian must be evaluated at this attempt's current state.
        self.op.set_jacobian_is_stale();
        LinearSolver::set_linearisation(&mut self.linear_solver, &self.op, &y0, t);
        self.rk
            .statistics_mut()
            .record_linear_solver_setup(if retry {
                SolverState::ErrorTestFail
            } else {
                SolverState::StepSuccess
            });

        let mut stages: Vec<Eqn::V> = Vec::with_capacity(8);
        let mut f = Eqn::V::zeros(n, ctx.clone());
        for i in 0..8 {
            let mut y_stage = y0.clone();
            for j in 0..i {
                y_stage.axpy(
                    Eqn::T::from_f64(A[i][j]).unwrap(),
                    &stages[j],
                    Eqn::T::one(),
                );
            }
            let c = Eqn::T::from_f64(TIMES[i]).unwrap();
            let stage_time = if TIMES[i] == 1.0 {
                self.endpoint_eval_time(h)?
            } else {
                t + c * h
            };
            if i == 0 {
                f.copy_from(&f0);
            } else {
                self.rk
                    .problem()
                    .eqn
                    .rhs()
                    .call_inplace(&y_stage, stage_time, &mut f);
            }
            let mut rhs = f.clone();
            rhs.axpy(h * Eqn::T::from_f64(D[i]).unwrap(), &ft, Eqn::T::one());
            for j in 0..i {
                rhs.axpy(
                    Eqn::T::from_f64(C[i][j]).unwrap() / h,
                    &stages[j],
                    Eqn::T::one(),
                );
            }
            self.linear_solver.solve_in_place(&mut rhs)?;
            rhs *= scale(gamma * h);
            stages.push(rhs);
        }

        let mut y1 = y0;
        for (i, stage) in stages.iter().enumerate() {
            y1.axpy(Eqn::T::from_f64(B[i]).unwrap(), stage, Eqn::T::one());
        }
        let mut f1 = Eqn::V::zeros(n, ctx);
        self.rk
            .problem()
            .eqn
            .rhs()
            .call_inplace(&y1, self.endpoint_eval_time(h)?, &mut f1);
        // Rodas5P's embedded difference is exactly its eighth increment.
        let error_norm =
            stages[7].squared_norm(&y1, &self.rk.problem().atol, self.rk.problem().rtol);
        Ok(StepAttempt {
            error_norm,
            y_end: y1,
            f_end: f1,
            stages,
        })
    }
}

impl<'a, Eqn, LS, M> OdeSolverMethod<'a, Eqn> for Rodas5P<'a, Eqn, LS, M>
where
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T, C = Eqn::C>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V, C = Eqn::C>,
    LS: LinearSolver<Eqn::M>,
{
    type State = RkState<Eqn::V>;
    type Config = ExplicitRkConfig<Eqn::T>;

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.rk.problem()
    }
    fn checkpoint(&mut self) -> Self::State {
        self.rk.checkpoint()
    }
    fn state_clone(&self) -> Self::State {
        self.rk.state().clone()
    }
    fn set_state(&mut self, state: Self::State) {
        self.rk.set_state(state);
    }
    fn into_state(self) -> Self::State {
        self.rk.into_state()
    }
    fn state(&self) -> StateRef<'_, Eqn::V> {
        self.rk.state().as_ref()
    }
    fn state_mut(&mut self) -> StateRefMut<'_, Eqn::V> {
        self.rk.state_mut().as_mut()
    }
    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn config_mut(&mut self) -> &mut Self::Config {
        &mut self.config
    }
    fn jacobian(&self) -> Option<Ref<'_, Eqn::M>> {
        Some(self.op.rhs_jac(&self.rk.state().y, self.rk.state().t))
    }
    fn mass(&self) -> Option<Ref<'_, Eqn::M>> {
        None
    }
    fn order(&self) -> usize {
        5
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut h = self.rk.start_step()?;
        if h.abs() < self.config.minimum_timestep {
            return Err(OdeSolverError::StepSizeTooSmall {
                time: self.rk.state().t.to_f64().unwrap(),
            }
            .into());
        }
        let mut attempts = 0;
        let (factor, accepted) = loop {
            let attempt = self.attempt(h, attempts > 0)?;
            let factor = self.rk.factor_with_error_order(
                attempt.error_norm,
                5,
                1.0,
                (
                    self.config.minimum_timestep_shrink,
                    self.config.maximum_timestep_shrink,
                    self.config.minimum_timestep_growth,
                    self.config.maximum_timestep_growth,
                ),
            );
            if attempt.error_norm < Eqn::T::one() {
                break (factor, attempt);
            }
            h *= factor;
            attempts += 1;
            self.rk.reset_prev_error();
            self.rk.error_test_fail(
                h,
                attempts,
                self.config.maximum_error_test_failures,
                self.config.minimum_timestep,
            )?;
        };
        self.rk
            .store_rosenbrock_stages(&accepted.y_end, &accepted.f_end, &accepted.stages);
        self.rk.set_prev_error(accepted.error_norm);
        let stop_reason = self.rk.step_accepted(h, h * factor, false)?;
        if matches!(stop_reason, OdeSolverStopReason::TstopReached) {
            self.discontinuity_stop = None;
        }
        Ok(stop_reason)
    }
    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.set_stop_time(tstop)?;
        self.discontinuity_stop = None;
        Ok(())
    }
    fn interpolate_inplace(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_inplace(t, y)
    }
    fn interpolate_dy_inplace(&self, t: Eqn::T, dy: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_dy_inplace(t, dy)
    }
    fn interpolate_out_inplace(&self, t: Eqn::T, g: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_out_inplace(t, g)
    }
    fn interpolate_sens_inplace(&self, t: Eqn::T, sens: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.rk.interpolate_sens_inplace(t, sens)
    }
    fn state_mut_back(&mut self, t: Eqn::T) -> Result<(), DiffsolError> {
        self.rk.state_mut_back(t, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::dense_nalgebra_serial::NalgebraMat,
        ode_equations::test_models::{
            exponential_decay::exponential_decay_problem, robertson_ode::robertson_ode,
        },
        ode_solver::tests::{
            test_checkpointing, test_config, test_interpolate, test_interpolate_dy,
            test_ode_solver, test_problem, test_state_mut,
        },
        NalgebraLU, OdeBuilder,
    };

    type Mat = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn shared_solver_contract() {
        test_state_mut(test_problem::<Mat>(false).rodas5p::<LS>().unwrap());
        test_interpolate(test_problem::<Mat>(false).rodas5p::<LS>().unwrap());
        test_interpolate_dy(test_problem::<Mat>(false).rodas5p::<LS>().unwrap());
        test_config(robertson_ode::<Mat>(false, 1).0.rodas5p::<LS>().unwrap());

        let (problem, solution) = exponential_decay_problem::<Mat>(false);
        test_checkpointing(
            solution,
            problem.rodas5p::<LS>().unwrap(),
            problem.rodas5p::<LS>().unwrap(),
        );
        let (problem, solution) = exponential_decay_problem::<Mat>(false);
        let mut solver = problem.rodas5p::<LS>().unwrap();
        test_ode_solver(&mut solver, solution, None, false, false);
    }

    #[test]
    fn robertson_dense_output_matches_reference() {
        let (problem, solution) = robertson_ode::<Mat>(false, 1);
        let mut solver = problem.rodas5p::<LS>().unwrap();
        test_ode_solver(&mut solver, solution, None, false, false);
    }

    fn advance_to<'a, Eqn, Method>(solver: &mut Method, t: f64)
    where
        Eqn: OdeEquationsImplicit<T = f64> + 'a,
        Method: OdeSolverMethod<'a, Eqn>,
    {
        solver.set_stop_time(t).unwrap();
        loop {
            if let OdeSolverStopReason::TstopReached = solver.step().unwrap() {
                break;
            }
        }
    }

    // Steinebach (2023), Section 4: a simple nonautonomous forcing exposes
    // unreliable embedded estimates in earlier Rodas5 coefficients.
    #[test]
    fn paper_cosine_forcing_tracks_sine() {
        let problem = OdeBuilder::<Mat>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        advance_to(&mut solver, 1.0);
        assert!((solver.state().y[0] - 1.0f64.sin()).abs() < 2e-7);
        assert!(solver.get_statistics().number_of_steps > 0);
    }

    #[test]
    fn tight_tolerance_cosine_forcing_has_no_large_time_derivative_floor() {
        let solve = |rtol, atol| {
            let problem = OdeBuilder::<Mat>::new()
                .rtol(rtol)
                .atol([atol])
                .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
                .init(|_, _, y| y[0] = 0.0, 1)
                .build()
                .unwrap();
            let mut solver = problem.rodas5p::<LS>().unwrap();
            advance_to(&mut solver, 1.0);
            (
                (solver.state().y[0] - 1.0f64.sin()).abs(),
                solver.get_statistics().number_of_steps,
                solver.get_statistics().number_of_error_test_failures,
            )
        };
        let loose = solve(1e-7, 1e-9);
        let tight = solve(1e-10, 1e-12);
        let very_tight = solve(1e-12, 1e-14);
        assert!(tight.0 < 1e-9, "tight error={}", tight.0);
        assert!(
            tight.0 < loose.0,
            "loose error={}, tight error={}",
            loose.0,
            tight.0
        );
        assert!(
            very_tight.0 < tight.0,
            "tight={tight:?}, very_tight={very_tight:?}"
        );
        assert!(very_tight.0 < 1e-11, "very tight error={}", very_tight.0);
    }

    #[test]
    fn controller_uses_embedded_local_error_order_five() {
        let problem = OdeBuilder::<Mat>::new()
            .rhs_implicit(|x, _, _, f| f[0] = -x[0], |_, _, _, v, jv| jv[0] = -v[0])
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let solver = problem.rodas5p::<LS>().unwrap();
        let factor = solver
            .rk
            .factor_with_error_order(0.01, 5, 1.0, (0.1, 1.0, 1.0, 10.0));
        let expected = 0.9 * 0.01f64.powf(-0.5 / 5.0);
        assert!((factor - expected).abs() < 1e-12);
    }

    #[test]
    fn nonlinear_jacobian_refreshes_after_an_accepted_step() {
        use std::{cell::RefCell, rc::Rc};

        let jacobian_states = Rc::new(RefCell::new(Vec::<(f64, f64)>::new()));
        let observed = jacobian_states.clone();
        let problem = OdeBuilder::<Mat>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .h0(0.05)
            .rhs_implicit(
                |x, _, _, f| f[0] = -x[0] * x[0],
                move |x, _, t, v, jv| {
                    observed.borrow_mut().push((t, x[0]));
                    jv[0] = -2.0 * x[0] * v[0];
                },
            )
            .init(|_, _, y| y[0] = 2.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        solver.step().unwrap();
        let first_time = solver.state().t;
        let first_value = solver.state().y[0];
        assert!(first_time > 0.0 && first_value < 2.0);
        solver.step().unwrap();
        let second_time = solver.state().t;
        let second_value = solver.state().y[0];
        assert!(second_time > first_time);
        assert!(
            jacobian_states
                .borrow()
                .iter()
                .any(|(t, y)| { *t == first_time && (*y - first_value).abs() < 1e-12 }),
            "the nonlinear Jacobian was not evaluated at the first accepted state"
        );
        let exact = 2.0 / (1.0 + 2.0 * second_time);
        assert!((second_value - exact).abs() < 1e-5);
    }

    #[test]
    fn ordinary_stop_evaluates_exact_endpoint_and_jump_stop_uses_incoming_side() {
        let problem = OdeBuilder::<Mat>::new()
            .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        solver.set_stop_time(1.0).unwrap();
        assert_eq!(solver.endpoint_eval_time(1.0).unwrap(), 1.0);
        solver.set_discontinuity_stop_time(1.0).unwrap();
        let incoming = solver.endpoint_eval_time(1.0).unwrap();
        assert!(0.0 < incoming && incoming < 1.0);
    }

    // Prothero-Robinson model: y'=lambda*(y-sin t)+cos t, y(0)=0.
    // Exact solution is sin(t) for every negative lambda.
    #[test]
    fn paper_prothero_robinson_stiff_tracking() {
        const LAMBDA: f64 = -1000.0;
        let problem = OdeBuilder::<Mat>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .rhs_implicit(
                |x, _, t, f| f[0] = LAMBDA * (x[0] - t.sin()) + t.cos(),
                |_, _, _, v, jv| jv[0] = LAMBDA * v[0],
            )
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        advance_to(&mut solver, 1.0);
        assert!((solver.state().y[0] - 1.0f64.sin()).abs() < 5e-9);
    }

    fn fixed_step_error(h: f64, lambda: f64) -> f64 {
        let problem = OdeBuilder::<Mat>::new()
            .rtol(10.0)
            .atol([10.0])
            .rhs_implicit(
                move |x, _, t, f| f[0] = lambda * (x[0] - t.sin()) + t.cos(),
                move |_, _, _, v, jv| jv[0] = lambda * v[0],
            )
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        *solver.state_mut().h = h;
        solver.config_mut().maximum_timestep_growth = 1.0;
        solver.config_mut().minimum_timestep_growth = 1.0;
        advance_to(&mut solver, 1.0);
        (solver.state().y[0] - 1.0f64.sin()).abs()
    }

    #[test]
    fn fixed_step_fifth_order_and_stiff_order_reduction() {
        let coarse = fixed_step_error(0.2, -2.0);
        let fine = fixed_step_error(0.1, -2.0);
        assert!(coarse > 20.0 * fine, "coarse={coarse}, fine={fine}");
        let stiff_coarse = fixed_step_error(0.2, -1000.0);
        let stiff_fine = fixed_step_error(0.1, -1000.0);
        assert!(
            stiff_coarse > stiff_fine,
            "stiff coarse={stiff_coarse}, fine={stiff_fine}"
        );
    }

    #[test]
    fn dense_output_and_derivative_track_sine() {
        let problem = OdeBuilder::<Mat>::new()
            .rtol(10.0)
            .atol([10.0])
            .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        *solver.state_mut().h = 0.2;
        solver.step().unwrap();
        let mid = solver.state().t / 2.0;
        let value = solver.interpolate(mid).unwrap();
        let derivative = solver.interpolate_dy(mid).unwrap();
        assert!(
            (value[0] - mid.sin()).abs() < 5e-6,
            "value={}, exact={}",
            value[0],
            mid.sin()
        );
        assert!((derivative[0] - mid.cos()).abs() < 5e-5);
    }

    fn midpoint_error(h: f64) -> f64 {
        let problem = OdeBuilder::<Mat>::new()
            .rtol(10.0)
            .atol([10.0])
            .rhs_implicit(|_, _, t, f| f[0] = t.cos(), |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        *solver.state_mut().h = h;
        solver.step().unwrap();
        (solver.interpolate(h / 2.0).unwrap()[0] - (h / 2.0).sin()).abs()
    }

    #[test]
    fn paper_dense_extension_converges_at_midpoint() {
        let coarse = midpoint_error(0.4);
        let fine = midpoint_error(0.2);
        // A fourth-order continuous extension has O(h^5) one-step error.
        assert!(coarse > 12.0 * fine, "coarse={coarse}, fine={fine}");
    }

    #[test]
    fn declared_discontinuity_keeps_incoming_and_outgoing_forcing_separate() {
        let problem = OdeBuilder::<Mat>::new()
            .rtol(1e-7)
            .atol([1e-9])
            .rhs_implicit(
                |x, _, t, f| f[0] = -10.0 * x[0] + if t < 1.0 { 0.0 } else { 10.0 },
                |_, _, _, v, jv| jv[0] = -10.0 * v[0],
            )
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        solver.set_discontinuity_stop_time(1.0).unwrap();
        loop {
            if let OdeSolverStopReason::TstopReached = solver.step().unwrap() {
                break;
            }
        }
        assert!(solver.state().y[0].abs() < 1e-12);
        advance_to(&mut solver, 1.2);
        let exact = 1.0 - (-2.0f64).exp();
        assert!((solver.state().y[0] - exact).abs() < 2e-6);
    }

    #[test]
    fn rejected_attempt_preserves_start_state() {
        let problem = OdeBuilder::<Mat>::new()
            .rtol(1e-8)
            .atol([1e-10])
            .rhs_implicit(
                |x, _, _, f| f[0] = -100.0 * x[0],
                |_, _, _, v, jv| jv[0] = -100.0 * v[0],
            )
            .init(|_, _, y| y[0] = 1.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        *solver.state_mut().h = 0.1;
        solver.step().unwrap();
        assert!(solver.get_statistics().number_of_error_test_failures > 0);
        assert_eq!(
            solver
                .get_statistics()
                .number_of_linear_solver_setups_from_error_test_fail,
            solver.get_statistics().number_of_error_test_failures,
        );
        let t = solver.state().t;
        assert!(t > 0.0 && t < 0.1);
        assert!((solver.state().y[0] - (-100.0 * t).exp()).abs() < 2e-5);
        advance_to(&mut solver, 0.1);
        assert!((solver.state().y[0] - (-10.0f64).exp()).abs() < 2e-5);
    }

    #[test]
    fn time_derivative_probe_requires_interior_representable_time() {
        let problem = OdeBuilder::<Mat>::new()
            .rhs_implicit(|_, _, t, f| f[0] = t, |_, _, _, _, jv| jv[0] = 0.0)
            .init(|_, _, y| y[0] = 0.0, 1)
            .build()
            .unwrap();
        let mut solver = problem.rodas5p::<LS>().unwrap();
        let t_even = f64::from_bits(1e12f64.to_bits() & !1);
        for t in [t_even, f64::from_bits(t_even.to_bits() + 1)] {
            *solver.state_mut().t = t;
            let h = f64::from_bits(t.to_bits() + 1) - t;
            assert!(matches!(
                solver.attempt(h, false),
                Err(DiffsolError::OdeSolverError(OdeSolverError::Other(_)))
            ));
        }
    }

    #[test]
    fn unsupported_mass_matrix_and_integrated_output_are_rejected() {
        let mass_problem = OdeBuilder::<Mat>::new()
            .rhs_implicit(|x, _, _, f| f[0] = -x[0], |_, _, _, v, jv| jv[0] = -v[0])
            .init(|_, _, y| y[0] = 1.0, 1)
            .mass(|v, _, _, beta, y| y[0] = v[0] + beta * y[0])
            .build()
            .unwrap();
        assert!(matches!(
            mass_problem.rodas5p::<LS>(),
            Err(DiffsolError::OdeSolverError(
                OdeSolverError::MassMatrixNotSupported
            ))
        ));
        let output_problem = OdeBuilder::<Mat>::new()
            .rhs_implicit(|x, _, _, f| f[0] = -x[0], |_, _, _, v, jv| jv[0] = -v[0])
            .init(|_, _, y| y[0] = 1.0, 1)
            .out_implicit(
                |x, _, _, out| out[0] = x[0],
                |_, _, _, v, jv| jv[0] = v[0],
                1,
            )
            .integrate_out(true)
            .build()
            .unwrap();
        assert!(matches!(
            output_problem.rodas5p::<LS>(),
            Err(DiffsolError::OdeSolverError(
                OdeSolverError::IntegratedOutputNotSupported
            ))
        ));
    }
}
