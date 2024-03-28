use anyhow::{anyhow, Result};
use std::{
    ffi::{c_int, c_void, CStr},
    rc::Rc,
};
use sundials_sys::{
    realtype, IDACalcIC, IDACreate, IDAFree, IDAGetDky, IDAGetReturnFlagName, IDAInit, IDAReInit,
    IDASVtolerances, IDASetId, IDASetJacFn, IDASetLinearSolver, IDASetUserData, IDASolve, N_Vector,
    SUNLinSolFree, SUNLinSolInitialize, SUNLinSol_Dense, SUNLinearSolver, SUNMatrix,
    IDA_CONSTR_FAIL, IDA_CONV_FAIL, IDA_ERR_FAIL, IDA_ILL_INPUT, IDA_LINIT_FAIL, IDA_LSETUP_FAIL,
    IDA_LSOLVE_FAIL, IDA_MEM_NULL, IDA_NORMAL, IDA_REP_RES_ERR, IDA_RES_FAIL, IDA_ROOT_RETURN,
    IDA_RTFUNC_FAIL, IDA_SUCCESS, IDA_TOO_MUCH_ACC, IDA_TOO_MUCH_WORK, IDA_TSTOP_RETURN,
    IDA_YA_YDP_INIT,
};

use crate::{
    matrix::sundials::SundialsMatrix,
    vector::{
        sundials::{get_suncontext, SundialsVector},
        Vector,
    },
    Matrix, OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverState,
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

struct SundialsData<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    eqn: Rc<Eqn>,
    number_of_states: usize,
}

impl<Eqn> SundialsData<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn new(eqn: Eqn) -> Self {
        let number_of_states = eqn.nstates();
        Self {
            eqn: Rc::new(eqn),
            number_of_states,
        }
    }
}

pub struct SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    ida_mem: *mut c_void,
    data: Box<SundialsData<Eqn>>,
    problem: Option<OdeSolverProblem<Eqn>>,
    yp: SundialsVector,
    linear_solver: SUNLinearSolver,
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
        // F(t, y, y') = M y' - f(t, y)
        // rr = f(t, y)
        data.eqn.rhs_inplace(t, &y, &mut rr);
        // tmp = M y'
        let mut tmp = SundialsVector::new_clone(&y);
        data.eqn.mass_inplace(t, &yp, &mut tmp);
        // rr = M y' - rr (gemv)
        rr.axpy(1., &tmp, 1.);
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

    pub fn new(eqn: Eqn) -> Result<Self> {
        let number_of_states = eqn.nstates();
        let ctx = *get_suncontext();

        let jacobian = SundialsMatrix::new_dense(number_of_states, number_of_states);
        let yy = SundialsVector::new_serial(number_of_states);
        let id = eqn.algebraic_indices();
        let data = Box::new(SundialsData::new(eqn));
        let ida_mem = unsafe { IDACreate(ctx) };
        Self::check(unsafe { IDASetUserData(ida_mem, data.as_ref() as *const _ as *mut c_void) })?;

        // need to convert to realtype sundials vector
        let mut id_realtype = SundialsVector::new_serial(number_of_states);
        for i in 0..number_of_states {
            match id[i] {
                1 => id_realtype[i] = 1.0,
                _ => id_realtype[i] = 0.0,
            }
        }
        let t0 = 0.0;
        let y = SundialsVector::zeros(number_of_states);
        let yp = SundialsVector::zeros(number_of_states);
        Self::check(unsafe {
            IDAInit(
                ida_mem,
                Some(Self::residual),
                t0,
                y.sundials_vector(),
                yp.sundials_vector(),
            )
        })?;
        Self::check(unsafe { IDASetJacFn(ida_mem, Some(Self::jacobian)) })?;
        Self::check(unsafe { IDASetId(ida_mem, id_realtype.sundials_vector()) })?;
        let linear_solver =
            unsafe { SUNLinSol_Dense(yy.sundials_vector(), jacobian.sundials_matrix(), ctx) };
        Self::check(unsafe {
            IDASetLinearSolver(ida_mem, linear_solver, jacobian.sundials_matrix())
        })?;
        Self::check(unsafe { SUNLinSolInitialize(linear_solver) })?;

        Ok(Self {
            ida_mem,
            data,
            problem: None,
            yp: yy,
            linear_solver,
        })
    }
}

impl<Eqn> Drop for SundialsIda<Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector, M = SundialsMatrix>,
{
    fn drop(&mut self) {
        unsafe { SUNLinSolFree(self.linear_solver) };
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
        let data = &self.data;
        let ida_mem = self.ida_mem;
        let rtol = problem.rtol;
        let atol = problem.atol.as_ref();
        let n = data.number_of_states;

        Self::check(unsafe { IDASVtolerances(ida_mem, rtol, atol.sundials_vector()) }).unwrap();

        let t0 = state.t;
        let y0 = state.y.clone();
        let yp0 = SundialsVector::zeros(n);
        Self::check(unsafe { IDAReInit(ida_mem, t0, y0.sundials_vector(), yp0.sundials_vector()) })
            .unwrap();
        Self::check(unsafe { IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t0 + 1.0) }).unwrap();
    }

    fn step(&mut self, state: &mut OdeSolverState<<Eqn>::M>) -> Result<()> {
        let mut t1: realtype = 0.0;
        let retval = unsafe {
            IDASolve(
                self.ida_mem,
                state.t + 1.0,
                &mut t1 as *mut realtype,
                state.y.sundials_vector(),
                self.yp.sundials_vector(),
                IDA_NORMAL,
            )
        };
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
        let ret = SundialsVector::new_serial(self.data.number_of_states);
        Self::check(unsafe { IDAGetDky(self.ida_mem, t, 0, ret.sundials_vector()) }).unwrap();
        ret
    }
}
