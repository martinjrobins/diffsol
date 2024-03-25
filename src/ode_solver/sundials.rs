use anyhow::{anyhow, Result};
use core::slice;
use std::{
    ffi::{c_int, c_void, CStr},
    rc::Rc,
};
use sundials_sys::{
    realtype, IDACalcIC, IDACreate, IDAFree, IDAGetConsistentIC, IDAGetIntegratorStats,
    IDAGetNonlinSolvStats, IDAGetReturnFlagName, IDAInit, IDAReInit, IDASVtolerances, IDASetId,
    IDASetLinearSolver, IDASetStopTime, IDASetUserData, IDASolve, N_VConst, N_VDestroy,
    N_VGetArrayPointer, N_VNew_Serial, N_Vector, SUNContext, SUNContext_Create, SUNDenseMatrix,
    SUNLinSolFree, SUNLinSolInitialize, SUNLinSol_Dense, SUNLinSol_SPBCGS, SUNLinSol_SPFGMR,
    SUNLinSol_SPGMR, SUNLinSol_SPTFQMR, SUNLinearSolver, SUNMatDestroy, SUNMatrix, IDA_NORMAL,
    IDA_ROOT_RETURN, IDA_SUCCESS, IDA_YA_YDP_INIT, PREC_LEFT, PREC_NONE,
};

use crate::{
    matrix::sundials::SundialsMatrix,
    vector::{sundials::SundialsVector, Vector, VectorView},
    OdeEquations, OdeSolverMethod,
};

struct SundialsData<'ctx, Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector<'ctx>, M = SundialsMatrix<'ctx>>,
{
    eqn: Rc<Eqn>,
    number_of_states: usize,
    ctx: &'ctx SUNContext,
}

pub struct SundialsIda<'ctx, Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector<'ctx>, M = SundialsMatrix<'ctx>>,
{
    eqn: Rc<Eqn>,
    ida_mem: *mut c_void,
    data: Box<SundialsData<'ctx, Eqn>>,
}

impl<'ctx, Eqn> SundialsIda<'ctx, Eqn>
where
    Eqn: OdeEquations<T = realtype, V = SundialsVector<'ctx>, M = SundialsMatrix<'ctx>>,
{
    extern "C" fn residual(
        t: realtype,
        y: N_Vector,
        yp: N_Vector,
        rr: N_Vector,
        user_data: *mut c_void,
    ) -> i32 {
        let data = unsafe { &*(user_data as *const SundialsData) };
        let n = data.eqn.nstates();
        let y = SundialsVector::new_not_owned(y, data.ctx);
        let yp = SundialsVector::new_not_owned(yp, data.ctx);
        let mut rr = SundialsVector::new_not_owned(rr, data.ctx);
        // F(t, y, y') = M y' - f(t, y)
        let tmp = SundialsVector::new_same_type(&y);
        // rr = f(t, y)
        // rr = M y' - rr (gemv)
        data.eqn.rhs_inplace(t, &y, &mut rr);
        0
    }

    fn check(retval: c_int) -> Result<()> {
        if retval < 0 {
            let char_ptr = unsafe { IDAGetReturnFlagName(i64::from(retval)) };
            let c_str = unsafe { CStr::from_ptr(char_ptr) };
            Err(anyhow!("Sundials Error Name: {}", c_str.to_str()?))
        } else {
            Ok(())
        }
    }

    fn create_nvector(n: usize, ctx: SUNContext) -> N_Vector {
        unsafe { N_VNew_Serial(n as i64, ctx) }
    }

    fn create_context() -> SUNContext {
        let sunctx: SUNContext;
        unsafe { SUNContext_Create(SUN_COMM_NULL, &sunctx) };
        sunctx
    }

    pub fn new(eqn: Eqn) -> Result<Self> {
        let number_of_states = eqn.nstates();
        let number_of_parameters = eqn.nparams();
        let ctx = Self::create_context();
        let yy = Self::create_nvector(number_of_states, ctx);
        let yp = Self::create_nvector(number_of_states, ctx);
        let avtol = Self::create_nvector(number_of_states, ctx);

        let jacobian = unsafe { SUNMatDense(number_of_states as i32, number_of_states as i32) };
        let linear_solver = unsafe { SUNLinSol_Dense(yy, jacobian) };
        let data = Box::new(SundialsData {
            number_of_states,
            number_of_parameters,
            yy,
            yp,
            avtol,
            yy_s,
            yp_s,
            id,
            jacobian,
            linear_solver,
        });
        let ida_mem = unsafe { IDACreate() };
        unsafe { IDASetUserData(ida_mem, data.as_ref() as *const _ as *mut c_void) };
        unsafe { IDAInit(ida_mem, Self::residual, 0.0, yy, yp) };
        unsafe { IDASVtolerances(ida_mem, 1e-6, avtol) };
        unsafe { IDASetId(ida_mem, id) };
        unsafe { IDASetLinearSolver(ida_mem, linear_solver, jacobian) };
        Ok(Self {
            eqn: Rc::new(eqn),
            ida_mem,
            data,
        })
    }
}

impl<Eqn: OdeEquations> OdeSolverMethod<Eqn> for SundialsIda<Eqn> {}
