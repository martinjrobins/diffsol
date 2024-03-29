use anyhow::{anyhow, Result};
use serde::Serialize;
use std::{
    ffi::{c_int, c_long, c_void, CStr},
    rc::Rc,
};
use sundials_sys::{
    realtype, IDACalcIC, IDACreate, IDAFree, IDAGetDky, IDAGetIntegratorStats,
    IDAGetNonlinSolvStats, IDAGetReturnFlagName, IDAInit, IDASVtolerances, IDASetId, IDASetJacFn,
    IDASetLinearSolver, IDASetUserData, IDASolve, N_Vector, SUNLinSolFree, SUNLinSolInitialize,
    SUNLinSol_Dense, SUNLinearSolver, SUNMatrix, IDA_CONSTR_FAIL, IDA_CONV_FAIL, IDA_ERR_FAIL,
    IDA_ILL_INPUT, IDA_LINIT_FAIL, IDA_LSETUP_FAIL, IDA_LSOLVE_FAIL, IDA_MEM_NULL, IDA_ONE_STEP,
    IDA_REP_RES_ERR, IDA_RES_FAIL, IDA_ROOT_RETURN, IDA_RTFUNC_FAIL, IDA_SUCCESS, IDA_TOO_MUCH_ACC,
    IDA_TOO_MUCH_WORK, IDA_TSTOP_RETURN, IDA_YA_YDP_INIT,
};

use crate::{
    vector::sundials::get_suncontext, Matrix, OdeEquations, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, SundialsMatrix, SundialsVector, Vector,
};

pub fn sundials_check(retval: c_int) -> Result<()> {
    if retval < 0 {
        let char_ptr = unsafe { IDAGetReturnFlagName(i64::from(retval)) };
        let c_str = unsafe { CStr::from_ptr(char_ptr) };
        Err(anyhow!("Sundials Error Name: {}", c_str.to_str()?))
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
    pub initial_step_size: realtype,
    pub final_step_size: realtype,
}

impl SundialsStatistics {
    fn new() -> Self {
        Self {
            number_of_linear_solver_setups: 0,
            number_of_steps: 0,
            number_of_error_test_failures: 0,
            number_of_nonlinear_solver_iterations: 0,
            number_of_nonlinear_solver_fails: 0,
            initial_step_size: 0.0,
            final_step_size: 0.0,
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
            initial_step_size: hinused,
            final_step_size: hcur,
        })
    }
}

struct SundialsData<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    eqn: Rc<Eqn>,
}

impl<Eqn> SundialsData<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn new(eqn: Rc<Eqn>) -> Self {
        Self { eqn }
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
        // F(t, y, y') = f(t, y) - M y'
        // rr = f(t, y)
        data.eqn.rhs_inplace(t, &y, &mut rr);
        // tmp = M y'
        let mut tmp = SundialsVector::new_clone(&y);
        data.eqn.mass_inplace(t, &yp, &mut tmp);
        // rr = -M y' + rr (gemv)
        rr.axpy(-1., &tmp, 1.);
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
        let data = unsafe { &*(user_data as *const SundialsData<Eqn>) };
        let eqn = &data.eqn;

        // jac = rhs_jac - c_j * M
        let y = SundialsVector::new_not_owned(y);
        let mut jac = SundialsMatrix::new_not_owned(jac);
        jac.copy_from(&eqn.mass_matrix(t));
        jac *= -c_j;
        jac += &eqn.jacobian_matrix(&y, t);
        0
    }

    fn check(retval: c_int) -> Result<()> {
        sundials_check(retval)
    }

    pub fn new() -> Self {
        let ctx = *get_suncontext();
        let ida_mem = unsafe { IDACreate(ctx) };
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
        }
    }

    pub fn get_statistics(&self) -> &SundialsStatistics {
        &self.statistics
    }

    pub fn calc_ic(&mut self, t: realtype) -> Result<()> {
        if self.problem.is_none() {
            return Err(anyhow!("Problem not set"));
        }
        let id = self.problem.as_ref().unwrap().eqn.algebraic_indices();
        let number_of_states = self.problem.as_ref().unwrap().eqn.nstates();
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

    fn set_problem(&mut self, state: &mut OdeSolverState<Eqn::M>, problem: &OdeSolverProblem<Eqn>) {
        self.problem = Some(problem.clone());
        let eqn = problem.eqn.as_ref();
        let number_of_states = eqn.nstates();
        let ctx = *get_suncontext();
        let ida_mem = self.ida_mem;

        // set user data
        self.data = Some(SundialsData::new(problem.eqn.clone()));
        Self::check(unsafe { IDASetUserData(self.ida_mem, &self.data as *const _ as *mut c_void) })
            .unwrap();

        // initialize
        self.yp = SundialsVector::zeros(number_of_states);
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
            SUNLinSol_Dense(
                state.y.sundials_vector(),
                self.jacobian.sundials_matrix(),
                ctx,
            )
        };
        Self::check(unsafe { SUNLinSolInitialize(self.linear_solver) }).unwrap();
        Self::check(unsafe {
            IDASetLinearSolver(ida_mem, self.linear_solver, self.jacobian.sundials_matrix())
        })
        .unwrap();

        // set jacobian function
        Self::check(unsafe { IDASetJacFn(ida_mem, Some(Self::jacobian)) }).unwrap();
    }

    fn step(&mut self, state: &mut OdeSolverState<<Eqn>::M>) -> Result<()> {
        if self.problem.is_none() {
            return Err(anyhow!("Problem not set"));
        }
        let retval = unsafe {
            IDASolve(
                self.ida_mem,
                state.t + 1.0,
                &mut state.t as *mut realtype,
                state.y.sundials_vector(),
                self.yp.sundials_vector(),
                IDA_ONE_STEP,
            )
        };

        // update stats
        self.statistics = SundialsStatistics::new_from_ida(self.ida_mem).unwrap();

        // check return value
        match retval {
            IDA_SUCCESS => Ok(()),
            IDA_TSTOP_RETURN => Ok(()),
            IDA_ROOT_RETURN => Ok(()),
            IDA_MEM_NULL => Err(anyhow!("The ida_mem argument was NULL.")),
            IDA_ILL_INPUT => Err(anyhow!("One of the inputs to IDASolve() was illegal, or some other input to the solver was either illegal or missing.")),
            IDA_TOO_MUCH_WORK => Err(anyhow!("The solver took mxstep internal steps but could not reach tout.")),
            IDA_TOO_MUCH_ACC => Err(anyhow!("The solver could not satisfy the accuracy demanded by the user for some internal step.")),
            IDA_ERR_FAIL => Err(anyhow!("Error test failures occurred too many times (MXNEF = 10) during one internal time step or occurred with.")),
            IDA_CONV_FAIL => Err(anyhow!("Convergence test failures occurred too many times (MXNCF = 10) during one internal time step or occurred with.")),
            IDA_LINIT_FAIL => Err(anyhow!("The linear solver’s initialization function failed.")),
            IDA_LSETUP_FAIL => Err(anyhow!("The linear solver’s setup function failed in an unrecoverable manner.")),
            IDA_LSOLVE_FAIL => Err(anyhow!("The linear solver’s solve function failed in an unrecoverable manner.")),
            IDA_CONSTR_FAIL => Err(anyhow!("The inequality constraints were violated and the solver was unable to recover.")),
            IDA_REP_RES_ERR => Err(anyhow!("The user’s residual function repeatedly returned a recoverable error flag, but the solver was unable to recover.")),
            IDA_RES_FAIL => Err(anyhow!("The user’s residual function returned a nonrecoverable error flag.")),
            IDA_RTFUNC_FAIL => Err(anyhow!("The rootfinding function failed.")),
            _ => Err(anyhow!("Unknown error")),
        }
    }

    fn interpolate(&self, _state: &OdeSolverState<<Eqn>::M>, t: <Eqn>::T) -> <Eqn>::V {
        if self.data.is_none() {
            panic!("Problem not set");
        }
        let ret = SundialsVector::new_serial(self.data.as_ref().unwrap().eqn.nstates());
        Self::check(unsafe { IDAGetDky(self.ida_mem, t, 0, ret.sundials_vector()) }).unwrap();
        ret
    }
}
