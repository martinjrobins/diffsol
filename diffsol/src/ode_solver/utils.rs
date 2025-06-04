use crate::{ode_solver_error, DenseMatrix};
use crate::{Vector, StateRefMut, OdeSolverStopReason, error::{DiffsolError, OdeSolverError}, StateRef, Scalar};
use num_traits::{abs, One, Zero};
use std::ops::MulAssign;

pub(crate) fn handle_tstop<V: Vector>(
    state: StateRefMut<'_, V>,
    tstop: V::T,
) -> Result<Option<OdeSolverStopReason<V::T>>, DiffsolError> {

    // check if the we are at tstop
    let troundoff = V::T::from(100.0) * V::T::EPSILON * (abs(*state.t) + abs(*state.h));
    if abs(*state.t - tstop) <= troundoff {
        return Ok(Some(OdeSolverStopReason::TstopReached));
    } else if (*state.h > V::T::zero() && tstop < *state.t - troundoff)
        || (*state.h < V::T::zero() && tstop > *state.t + troundoff)
    {
        return Err(DiffsolError::from(
            OdeSolverError::StopTimeBeforeCurrentTime {
                stop_time: tstop.into(),
                state_time: (*state.t).into(),
            },
        ));
    }

    // check if the next step will be beyond tstop, if so adjust the step size
    if (*state.h > V::T::zero() && *state.t + *state.h > tstop + troundoff)
        || (*state.h < V::T::zero() && *state.t + *state.h < tstop - troundoff)
    {
        let factor = (tstop - *state.t) / *state.h;
        state.h.mul_assign(factor);
    }
    Ok(None)
}

pub(crate) fn interpolate_from_diff<M: DenseMatrix>(h: M::T, y0: &M::V, beta_f: &M::V, diff: &M) -> M::V {
    // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
    let mut ret = y0.clone();
    diff.gemv(h, beta_f, M::T::one(), &mut ret);
    ret
}

pub(crate) fn interpolate_beta_function<M: DenseMatrix>(theta: M::T, beta: &M) -> M::V {
    let poly_order = beta.ncols();
    let s_star = beta.nrows();
    let mut thetav = Vec::with_capacity(poly_order);
    thetav.push(theta);
    for i in 1..poly_order {
        thetav.push(theta * thetav[i - 1]);
    }
    // beta_poly = beta * thetav
    let thetav = M::V::from_vec(thetav, beta.context().clone());
    let mut beta_f = <M::V as Vector>::zeros(s_star, beta.context().clone());
    beta.gemv(M::T::one(), &thetav, M::T::zero(), &mut beta_f);
    beta_f
}

pub(crate) fn interpolate_hermite<M: DenseMatrix>(h: M::T, theta: M::T, u0: &M::V, u1: &M::V, diff: &M) -> M::V {
    let f0 = diff.column(0);
    let f1 = diff.column(diff.ncols() - 1);

    let mut y = u1.clone() - u0;
    y.axpy_v(
        h * (theta - M::T::from(1.0)),
        &f0,
        M::T::one() - M::T::from(2.0) * theta,
    );
    y.axpy_v(h * theta, &f1, M::T::one());
    y.axpy(
        M::T::from(1.0) - theta,
        u0,
        theta * (theta - M::T::from(1.0)),
    );
    y.axpy(theta, u1, M::T::one());
    y
}


pub(crate) fn interpolate<M: DenseMatrix>(is_state_mutated: bool, beta: Option<&M>, t: M::T, state: StateRef<M::V>, old_state: StateRef<M::V>, diff: &M) -> Result<M::V, DiffsolError> {
    if is_state_mutated {
        if t == state.t {
            return Ok(state.y.clone());
        } else {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }
    }

    // check that t is within the current step depending on the direction
    let is_forward = state.h > M::T::zero();
    if (is_forward && (t > state.t || t < old_state.t))
        || (!is_forward && (t < state.t || t > old_state.t))
    {
        return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
    }

    let dt = state.t - old_state.t;
    let theta = if dt == M::T::zero() {
        M::T::one()
    } else {
        (t - old_state.t) / dt
    };
    if let Some(beta) = beta {
        let beta_f = interpolate_beta_function(theta, beta);
        let ret = interpolate_from_diff(dt, old_state.y, &beta_f, diff);
        Ok(ret)
    } else {
        let ret = interpolate_hermite(dt, theta, old_state.y, state.y, diff);
        Ok(ret)
    }
}

pub(crate) fn interpolate_out<M: DenseMatrix>(is_state_mutated: bool, beta: Option<&M>, t: M::T, state: StateRef<M::V>, old_state: StateRef<M::V>, gdiff: &M) -> Result<M::V, DiffsolError> {
    if is_state_mutated {
        if t == state.t {
            return Ok(state.g.clone());
        } else {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }
    }

    // check that t is within the current step depending on the direction
    let is_forward = state.h > M::T::zero();
    if (is_forward && (t > state.t || t < old_state.t))
        || (!is_forward && (t < state.t || t > old_state.t))
    {
        return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
    }

    let dt = state.t - old_state.t;
    let theta = if dt == M::T::zero() {
        M::T::one()
    } else {
        (t - old_state.t) / dt
    };
    if let Some(beta) = beta {
        let beta_f = interpolate_beta_function(theta, beta);
        let ret = interpolate_from_diff(dt, old_state.g, &beta_f, gdiff);
        Ok(ret)
    } else {
        let ret = interpolate_hermite(dt, theta, old_state.g, state.g, gdiff);
        Ok(ret)
    }
}


pub(crate) fn interpolate_sens<M: DenseMatrix>(is_state_mutated: bool, beta: Option<&M>, t: M::T, state: StateRef<M::V>, old_state: StateRef<M::V>, sdiff: &[M]) -> Result<Vec<M::V>, DiffsolError> {
    if is_state_mutated {
        if t == state.t {
            return Ok(state.s.to_vec());
        } else {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }
    }

    // check that t is within the current step depending on the direction
    let is_forward = state.h > M::T::zero();
    if (is_forward && (t > state.t || t < old_state.t))
        || (!is_forward && (t < state.t || t > old_state.t))
    {
        return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
    }

    let dt = state.t - old_state.t;
    let theta = if dt == M::T::zero() {
        M::T::one()
    } else {
        (t - old_state.t) / dt
    };
    if let Some(beta) = beta {
        let beta_f = interpolate_beta_function(theta, beta);
        let ret = old_state.s
                .iter()
                .zip(sdiff.iter())
                .map(|(y, diff)| interpolate_from_diff(dt, y, &beta_f, diff))
                .collect();
        Ok(ret)
    } else {
        let ret = old_state.s
            .iter()
            .zip(state.s.iter())
            .zip(sdiff.iter())
            .map(|((s0, s1), diff)| interpolate_hermite(dt, theta, s0, s1, diff))
            .collect();
        Ok(ret)
    }
}