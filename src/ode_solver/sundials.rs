use crate::sundials_sys::{
    realtype, IDACalcIC, IDACreate, IDAFree, IDAGetDky, IDAGetIntegratorStats,
    IDAGetNonlinSolvStats, IDAGetReturnFlagName, IDAInit, IDAReInit, IDASVtolerances, IDASetId,
    IDASetJacFn, IDASetLinearSolver, IDASetStopTime, IDASetUserData, IDASolve, N_Vector,
    SUNLinSolFree, SUNLinSolInitialize, SUNLinSol_Dense, SUNLinearSolver, SUNMatrix,
    IDA_CONSTR_FAIL, IDA_CONV_FAIL, IDA_ERR_FAIL, IDA_ILL_INPUT, IDA_LINIT_FAIL, IDA_LSETUP_FAIL,
    IDA_LSOLVE_FAIL, IDA_MEM_NULL, IDA_ONE_STEP, IDA_REP_RES_ERR, IDA_RES_FAIL, IDA_ROOT_RETURN,
    IDA_RTFUNC_FAIL, IDA_SUCCESS, IDA_TOO_MUCH_ACC, IDA_TOO_MUCH_WORK, IDA_TSTOP_RETURN,
    IDA_YA_YDP_INIT,
};
use num_traits::Zero;
use serde::Serialize;
use std::{
    ffi::{c_int, c_long, c_void, CStr},
    rc::Rc,
};

use crate::{
    matrix::sparsity::MatrixSparsityRef, scale, LinearOp, Matrix, NonLinearOp, OdeEquations,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason, Op, SundialsMatrix,
    SundialsVector, Vector,
};

#[cfg(not(sundials_version_major = "5"))]
use crate::vector::sundials::get_suncontext;

pub fn sundials_check(retval: c_int) -> Result<()> {
    if retval < 0 {
        let char_ptr = unsafe { IDAGetReturnFlagName(i64::from(retval)) };
        let c_str = unsafe { CStr::from_ptr(char_ptr) };
        Err(ode_solver_error!(SundialsError(
            c_str.to_str().unwrap().to_string()
        )))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SundialsStatistics {
    pub number_of_linear_solver_setups: usize,
    pub number_of_steps: usize,
    pub number_of_error_test_failures: usize,
    pub number_of_nonlinear_solver_iterations: usize,
    pub number_of_nonlinear_solver_fails: usize,
}

impl SundialsStatistics {
    fn new() -> Self {
        Self {
            number_of_linear_solver_setups: 0,
            number_of_steps: 0,
            number_of_error_test_failures: 0,
            number_of_nonlinear_solver_iterations: 0,
            number_of_nonlinear_solver_fails: 0,
        }
    }
    fn new_from_ida(ida_mem: *mut c_void) -> Result<Self> {
        let mut nsteps: c_long = 0;
        let mut nrevals: c_long = 0;
        let mut nlinsetups: c_long = 0;
        let mut netfails: c_long = 0;
        let mut klast: c_int = 0;
        let mut kcur: c_int = 0;
        let mut hinused: realtype = 0.;
        let mut hlast: realtype = 0.;
        let mut hcur: realtype = 0.;
        let mut tcur: realtype = 0.;

        sundials_check(unsafe {
            IDAGetIntegratorStats(
                ida_mem,
                &mut nsteps,
                &mut nrevals,
                &mut nlinsetups,
                &mut netfails,
                &mut klast,
                &mut kcur,
                &mut hinused,
                &mut hlast,
                &mut hcur,
                &mut tcur,
            )
        })?;

        let mut nniters: c_long = 0;
        let mut nncfails: c_long = 0;
        sundials_check(unsafe { IDAGetNonlinSolvStats(ida_mem, &mut nniters, &mut nncfails) })?;

        Ok(Self {
            number_of_linear_solver_setups: nlinsetups.try_into().unwrap(),
            number_of_steps: nsteps.try_into().unwrap(),
            number_of_error_test_failures: netfails.try_into().unwrap(),
            number_of_nonlinear_solver_iterations: nniters.try_into().unwrap(),
            number_of_nonlinear_solver_fails: nncfails.try_into().unwrap(),
        })
    }
}

struct SundialsData<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    eqn: Rc<Eqn>,
    rhs_jac: SundialsMatrix,
    mass: SundialsMatrix,
}

impl<Eqn> SundialsData<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn new(eqn: Rc<Eqn>) -> Self {
        let n = eqn.rhs().nstates();
        let rhs = eqn.rhs();
        let rhs_jac_sparsity = rhs.sparsity().map(|s| MatrixSparsityRef::to_owned(&s));
        let rhs_jac = SundialsMatrix::new_from_sparsity(n, n, rhs_jac_sparsity);
        let mass = if let Some(mass) = eqn.mass() {
            let mass_sparsity = mass.sparsity().map(|s| MatrixSparsityRef::to_owned(&s));
            SundialsMatrix::new_from_sparsity(n, n, mass_sparsity)
        } else {
            let ones = SundialsVector::from_element(n, 1.0);
            SundialsMatrix::from_diagonal(&ones)
        };
        Self { eqn, rhs_jac, mass }
    }
}

pub struct SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    ida_mem: *mut c_void,
    linear_solver: SUNLinearSolver,
    data: Option<SundialsData<Eqn>>,
    problem: Option<OdeSolverProblem<Eqn>>,
    yp: SundialsVector,
    jacobian: SundialsMatrix,
    statistics: SundialsStatistics,
    state: Option<OdeSolverState<Eqn::V>>,
    is_state_modified: bool,
}

impl<Eqn> SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    extern "C" fn residual(
        t: realtype,
        y: N_Vector,
        yp: N_Vector,
        rr: N_Vector,
        user_data: *mut c_void,
    ) -> i32 {
        let data = unsafe { &*(user_data as *const SundialsData<Eqn>) };
        let y = SundialsVector::new_not_owned(y);
        let yp = SundialsVector::new_not_owned(yp);
        let mut rr = SundialsVector::new_not_owned(rr);
        // F(t, y, y') =  M y' - f(t, y)
        // rr = f(t, y)
        data.eqn.rhs().call_inplace(&y, t, &mut rr);
        // rr = M y' - rr
        if let Some(mass) = data.eqn.mass() {
            mass.gemv_inplace(&yp, t, -1.0, &mut rr);
        } else {
            rr.axpy(1.0, &yp, -1.0);
        }
        0
    }

    extern "C" fn jacobian(
        t: realtype,
        c_j: realtype,
        y: N_Vector,
        _yp: N_Vector,
        _r: N_Vector,
        jac: SUNMatrix,
        user_data: *mut c_void,
        _tmp1: N_Vector,
        _tmp2: N_Vector,
        _tmp3: N_Vector,
    ) -> i32 {
        let data = unsafe { &mut *(user_data as *mut SundialsData<Eqn>) };
        let eqn = &data.eqn;

        // jac = c_j * M - rhs_jac
        let y = SundialsVector::new_not_owned(y);
        let mut jac = SundialsMatrix::new_not_owned(jac);
        if let Some(mass) = eqn.mass() {
            mass.matrix_inplace(t, &mut data.mass);
        }
        eqn.rhs().jacobian_inplace(&y, t, &mut data.rhs_jac);
        data.rhs_jac *= scale(-1.0);
        jac.scale_add_and_assign(&data.rhs_jac, c_j, &data.mass);
        0
    }

    fn check(retval: c_int) -> Result<()> {
        sundials_check(retval)
    }

    pub fn new() -> Self {
        #[cfg(not(sundials_version_major = "5"))]
        let ida_mem = unsafe { IDACreate(*get_suncontext()) };

        #[cfg(sundials_version_major = "5")]
        let ida_mem = unsafe { IDACreate() };

        let yp = SundialsVector::new_serial(0);
        let jacobian = SundialsMatrix::new_dense(0, 0);

        Self {
            ida_mem,
            data: None,
            problem: None,
            yp,
            linear_solver: std::ptr::null_mut(),
            statistics: SundialsStatistics::new(),
            jacobian,
            state: None,
            is_state_modified: false,
        }
    }

    pub fn get_statistics(&self) -> &SundialsStatistics {
        &self.statistics
    }

    pub fn calc_ic(&mut self, t: realtype) -> Result<(), DiffsolError> {
        if self.problem.is_none() {
            return Err(ode_solver_error!(ProblemNotSet));
        }
        if self.problem.as_ref().unwrap().eqn.mass().is_none() {
            return Ok(());
        }
        let diag = self
            .problem
            .as_ref()
            .unwrap()
            .eqn
            .mass()
            .unwrap()
            .matrix(t)
            .diagonal();
        let id = diag.filter_indices(|x| x == Eqn::T::zero());
        let number_of_states = self.problem.as_ref().unwrap().eqn.rhs().nstates();
        // need to convert to realtype sundials vector
        let mut id_realtype = SundialsVector::new_serial(number_of_states);
        for i in 0..number_of_states {
            match id[i] {
                1 => id_realtype[i] = 1.0,
                _ => id_realtype[i] = 0.0,
            }
        }
        Self::check(unsafe { IDASetId(self.ida_mem, id_realtype.sundials_vector()) })?;
        Self::check(unsafe { IDACalcIC(self.ida_mem, IDA_YA_YDP_INIT, t) })?;
        Ok(())
    }
}

impl<Eqn> Default for SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Eqn> Drop for SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn drop(&mut self) {
        if !self.linear_solver.is_null() {
            unsafe { SUNLinSolFree(self.linear_solver) };
        }
        unsafe { IDAFree(&mut self.ida_mem) };
    }
}

impl<Eqn> OdeSolverMethod<Eqn> for SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>> {
        self.problem.as_ref()
    }

    fn state(&self) -> Option<&OdeSolverState<Eqn::V>> {
        self.state.as_ref()
    }

    fn order(&self) -> usize {
        1
    }

    fn state_mut(&mut self) -> Option<&mut OdeSolverState<Eqn::V>> {
        self.is_state_modified = true;
        self.state.as_mut()
    }

    fn take_state(&mut self) -> Option<OdeSolverState<<Eqn>::V>> {
        Option::take(&mut self.state)
    }

    fn set_problem(&mut self, state: OdeSolverState<Eqn::V>, problem: &OdeSolverProblem<Eqn>) {
        self.state = Some(state);
        let state = self.state.as_ref().unwrap();
        self.problem = Some(problem.clone());
        let eqn = problem.eqn.as_ref();
        let number_of_states = eqn.rhs().nstates();
        let ida_mem = self.ida_mem;

        // set user data
        self.data = Some(SundialsData::new(problem.eqn.clone()));
        Self::check(unsafe { IDASetUserData(self.ida_mem, &self.data as *const _ as *mut c_void) })
            .unwrap();

        // initialize
        self.yp = <SundialsVector as Vector>::zeros(number_of_states);
        Self::check(unsafe {
            IDAInit(
                ida_mem,
                Some(Self::residual),
                state.t,
                state.y.sundials_vector(),
                self.yp.sundials_vector(),
            )
        })
        .unwrap();

        // tolerances
        let rtol = problem.rtol;
        let atol = problem.atol.as_ref();
        Self::check(unsafe { IDASVtolerances(ida_mem, rtol, atol.sundials_vector()) }).unwrap();

        // linear solver
        self.jacobian = SundialsMatrix::new_dense(number_of_states, number_of_states);

        self.linear_solver = unsafe {
            #[cfg(not(sundials_version_major = "5"))]
            {
                SUNLinSol_Dense(
                    state.y.sundials_vector(),
                    self.jacobian.sundials_matrix(),
                    *get_suncontext(),
                )
            }
            #[cfg(sundials_version_major = "5")]
            {
                SUNLinSol_Dense(state.y.sundials_vector(), self.jacobian.sundials_matrix())
            }
        };

        Self::check(unsafe { SUNLinSolInitialize(self.linear_solver) }).unwrap();
        Self::check(unsafe {
            IDASetLinearSolver(ida_mem, self.linear_solver, self.jacobian.sundials_matrix())
        })
        .unwrap();

        // set jacobian function
        Self::check(unsafe { IDASetJacFn(ida_mem, Some(Self::jacobian)) }).unwrap();

        // sensitivities
        if self.problem.as_ref().unwrap().eqn_sens.is_some() {
            panic!("Sensitivities not implemented for sundials solver");
        }
    }

    fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<()> {
        Self::check(unsafe { IDASetStopTime(self.ida_mem, tstop) })
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let state = self.state.as_mut().ok_or(ode_solver_error!(StateNotSet))?;
        if self.problem.is_none() {
            return Err(ode_solver_error!(ProblemNotSet));
        }
        if self.is_state_modified {
            // reinit as state has been modified
            Self::check(unsafe {
                IDAReInit(
                    self.ida_mem,
                    state.t,
                    state.y.sundials_vector(),
                    self.yp.sundials_vector(),
                )
            })?
        }
        let itask = IDA_ONE_STEP;
        let retval = unsafe {
            IDASolve(
                self.ida_mem,
                state.t + 1.0,
                &mut state.t as *mut realtype,
                state.y.sundials_vector(),
                self.yp.sundials_vector(),
                itask,
            )
        };

        // update stats
        self.statistics = SundialsStatistics::new_from_ida(self.ida_mem).unwrap();

        // check return value
        match retval {
            IDA_SUCCESS => Ok(OdeSolverStopReason::InternalTimestep),
            IDA_TSTOP_RETURN => Ok(OdeSolverStopReason::TstopReached),
            IDA_ROOT_RETURN => Ok(OdeSolverStopReason::RootFound(state.t)),
            IDA_MEM_NULL => Err(other_error!("The ida_mem argument was NULL.")),
            IDA_ILL_INPUT => Err(other_error!("One of the inputs to IDASolve() was illegal, or some other input to the solver was either illegal or missing.")),
            IDA_TOO_MUCH_WORK => Err(other_error!("The solver took mxstep internal steps but could not reach tout.")),
            IDA_TOO_MUCH_ACC => Err(other_error!("The solver could not satisfy the accuracy demanded by the user for some internal step.")),
            IDA_ERR_FAIL => Err(other_error!("Error test failures occurred too many times (MXNEF = 10) during one internal time step or occurred with.")),
            IDA_CONV_FAIL => Err(other_error!("Convergence test failures occurred too many times (MXNCF = 10) during one internal time step or occurred with.")),
            IDA_LINIT_FAIL => Err(other_error!("The linear solver’s initialization function failed.")),
            IDA_LSETUP_FAIL => Err(other_error!("The linear solver’s setup function failed in an unrecoverable manner.")),
            IDA_LSOLVE_FAIL => Err(other_error!("The linear solver’s solve function failed in an unrecoverable manner.")),
            IDA_CONSTR_FAIL => Err(other_error!("The inequality constraints were violated and the solver was unable to recover.")),
            IDA_REP_RES_ERR => Err(other_error!("The user’s residual function repeatedly returned a recoverable error flag, but the solver was unable to recover.")),
            IDA_RES_FAIL => Err(other_error!("The user’s residual function returned a nonrecoverable error flag.")),
            IDA_RTFUNC_FAIL => Err(other_error!("The rootfinding function failed.")),
            _ => Err(other_error!("Unknown error")),
        }
    }

    fn interpolate(&self, t: <Eqn>::T) -> Result<Eqn::V, DiffsolError> {
        if self.data.is_none() {
            return Err(ode_solver_error!(ProblemNotSet));
        }
        let state = self.state.as_ref().ok_or(ode_solver_error!(StateNotSet))?;
        if t > state.t {
            return Err(ode_solver_error!(InterpolationTimeGreaterThanCurrentTime));
        }
        let ret = SundialsVector::new_serial(self.data.as_ref().unwrap().eqn.rhs().nstates());
        Self::check(unsafe { IDAGetDky(self.ida_mem, t, 0, ret.sundials_vector()) }).unwrap();
        Ok(ret)
    }

    fn interpolate_sens(
        &self,
        _t: <Eqn as OdeEquations>::T,
    ) -> Result<Vec<<Eqn as OdeEquations>::V>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod test {

    use crate::{
        ode_solver::{
            test_models::{
                exponential_decay::exponential_decay_problem,
                foodweb::{foodweb_problem, FoodWebContext},
                heat2d::head2d_problem,
                robertson::robertson,
            },
            tests::{test_interpolate, test_no_set_problem, test_ode_solver, test_state_mut},
        },
        OdeEquations, Op, SundialsIda, SundialsMatrix,
    };

    type M = SundialsMatrix;
    #[test]
    fn sundials_no_set_problem() {
        test_no_set_problem::<M, _>(SundialsIda::default())
    }
    #[test]
    fn sundials_state_mut() {
        test_state_mut::<M, _>(SundialsIda::default())
    }
    #[test]
    fn sundials_interpolate() {
        test_interpolate::<M, _>(SundialsIda::default())
    }

    #[test]
    fn test_sundials_exponential_decay() {
        let mut s = crate::SundialsIda::default();
        let (problem, soln) = exponential_decay_problem::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 18
        number_of_steps: 43
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 63
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 65
        number_of_jac_muls: 36
        number_of_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_sundials_robertson() {
        let mut s = crate::SundialsIda::default();
        let (problem, soln) = robertson::<crate::SundialsMatrix>(false);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 59
        number_of_steps: 355
        number_of_error_test_failures: 15
        number_of_nonlinear_solver_iterations: 506
        number_of_nonlinear_solver_fails: 1
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 509
        number_of_jac_muls: 180
        number_of_matrix_evals: 60
        "###);
    }

    #[test]
    fn test_sundials_foodweb() {
        let foodweb_context = FoodWebContext::default();
        let mut s = crate::SundialsIda::default();
        let (problem, soln) = foodweb_problem::<crate::SundialsMatrix, 10>(&foodweb_context);
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 42
        number_of_steps: 256
        number_of_error_test_failures: 9
        number_of_nonlinear_solver_iterations: 458
        number_of_nonlinear_solver_fails: 1
        "###);
    }
    #[test]
    fn test_sundials_heat2d() {
        let mut s = crate::SundialsIda::default();
        let (problem, soln) = head2d_problem::<crate::SundialsMatrix, 10>();
        test_ode_solver(&mut s, &problem, soln, None, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 42
        number_of_steps: 165
        number_of_error_test_failures: 11
        number_of_nonlinear_solver_iterations: 214
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().rhs().statistics(), @r###"
        ---
        number_of_calls: 217
        number_of_jac_muls: 4300
        number_of_matrix_evals: 43
        "###);
    }
}
