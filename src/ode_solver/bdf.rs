use std::rc::Rc;
use std::ops::AddAssign;

use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use num_traits::{One, Zero, Pow};

use crate::{callable::ode::BdfCallable, matrix::MatrixRef, ConstantJacobian, ConstantOp, IndexType, IterativeSolver, Jacobian, LinearOp, Matrix, MatrixViewMut, NewtonNonlinearSolver, NonLinearOp, Scalar, SolverProblem, Vector, VectorRef, VectorView, VectorViewMut, LU};

use super::{OdeSolverState, OdeSolverMethod, OdeSolverStatistics, OdeSolverProblem};

pub struct Bdf<M: Matrix, CRhs: NonLinearOp<V = M::V, T = M::T>, CMass: LinearOp<V = M::V, T = M::T>, CInit: ConstantOp<V = M::V, T = M::T>> {
    nonlinear_solver: Box<dyn IterativeSolver<BdfCallable<M, CRhs, CMass>>>,
    bdf_callable: Option<Rc<BdfCallable<M, CRhs, CMass>>>,
    ode_problem: Option<Rc<OdeSolverProblem<CRhs, CMass, CInit>>>,
    statistics: OdeSolverStatistics,
    order: usize,
    n_equal_steps: usize,
    diff: M,
    diff_tmp: M,
    u: M,
    r: M,
    alpha: Vec<CRhs::T>,
    gamma: Vec<CRhs::T>,
    error_const: Vec<CRhs::T>,
}

impl<T: Scalar, CRhs: Jacobian<M = DMatrix<T>, V = DVector<T>, T=T> + 'static, CMass: ConstantJacobian<M = DMatrix<T>, V = DVector<T>, T=T> + 'static, CInit: ConstantOp<V = DVector<T>, T=T> + 'static> Default for Bdf<DMatrix<T>, CRhs, CMass, CInit> 
{
    fn default() -> Self {
        let n = 1;
        let linear_solver = LU::<T>::default();
        let mut nonlinear_solver = Box::new(NewtonNonlinearSolver::<BdfCallable<DMatrix<T>, CRhs, CMass>>::new(linear_solver));
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        let statistics = OdeSolverStatistics { niter: 0, nmaxiter: 0 };
        Self { 
            ode_problem: None,
            statistics,
            nonlinear_solver,
            bdf_callable: None, 
            order: 1, 
            n_equal_steps: 0, 
            diff: DMatrix::<T>::zeros(n, Self::MAX_ORDER + 1), 
            diff_tmp: DMatrix::<T>::zeros(n, Self::MAX_ORDER + 1), 
            gamma: vec![T::from(1.0); Self::MAX_ORDER + 1], 
            alpha: vec![T::from(1.0); Self::MAX_ORDER + 1], 
            error_const: vec![T::from(1.0); Self::MAX_ORDER + 1], 
            u: DMatrix::<T>::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
            r: DMatrix::<T>::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
        }
    }
}

// implement clone for bdf
impl<T: Scalar, CRhs: Jacobian<M = DMatrix<T>, V = DVector<T>, T=T> + 'static, CMass: ConstantJacobian<M = DMatrix<T>, V = DVector<T>, T=T> + 'static, CInit: ConstantOp<V = DVector<T>, T=T> + 'static> Clone for Bdf<DMatrix<T>, CRhs, CMass, CInit> 
where
    for<'b> &'b DVector<T>: VectorRef<DVector<T>>,
{
    fn clone(&self) -> Self {
        let n = self.diff.nrows();
        let linear_solver = LU::<CRhs::T>::default();
        let mut nonlinear_solver = Box::new(NewtonNonlinearSolver::<BdfCallable<CRhs::M, CRhs, CMass>>::new(linear_solver));
        nonlinear_solver.set_max_iter(Self::NEWTON_MAXITER);
        let statistics = OdeSolverStatistics { niter: 0, nmaxiter: 0 };
        Self { 
            ode_problem: self.ode_problem.clone(),
            statistics,
            nonlinear_solver,
            bdf_callable: self.bdf_callable.clone(), 
            order: self.order, 
            n_equal_steps: self.n_equal_steps, 
            diff: CRhs::M::zeros(n, Self::MAX_ORDER + 1), 
            diff_tmp: CRhs::M::zeros(n, Self::MAX_ORDER + 1), 
            gamma: self.gamma.clone(), 
            alpha: self.alpha.clone(), 
            error_const: self.error_const.clone(), 
            u: CRhs::M::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
            r: CRhs::M::zeros(Self::MAX_ORDER + 1, Self::MAX_ORDER + 1),
        }
    }
}

impl<M: Matrix, CRhs: NonLinearOp<V = M::V, T = M::T>, CMass: LinearOp<V = M::V, T = M::T>, CInit: ConstantOp<V = M::V, T = M::T>> Bdf<M, CRhs, CMass, CInit> 
where
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    const MAX_ORDER: IndexType = 5;
    const NEWTON_MAXITER: IndexType = 4;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    
    fn _predict(&self, state: &mut OdeSolverState<M::V>) {
        // predict forward to new step (eq 2 in [1])
        for i in 1..=self.order {
            state.y += self.diff.column(i);
        }
    }
    
    fn _compute_r(order: usize, factor: M::T) -> M {
        //computes the R matrix with entries
        //given by the first equation on page 8 of [1]
        //
        //This is used to update the differences matrix when step size h is varied
        //according to factor = h_{n+1} / h_n
        //
        //Note that the U matrix also defined in the same section can be also be
        //found using factor = 1, which corresponds to R with a constant step size
        let mut r = M::zeros(order + 1, order + 1);
        
        // r[0, 0:order] = 1
        for j in 0..=order {
            r[(0, j)] = M::T::one();
        }
        // r[i, j] = r[i, j-1] * (j - 1 - factor * i) / j
        for i in 1..=order {
            for j in 1..=order {
                let i_t = M::T::from(i as f64);
                let j_t = M::T::from(j as f64);
                r[(i, j)] = r[(i - 1, j)] * (i_t - M::T::one() - factor * j_t) / i_t;
            }
        }
        r
    }

    fn _update_step_size(&mut self, factor: M::T, state: &mut OdeSolverState<M::V>) {
        //If step size h is changed then also need to update the terms in
        //the first equation of page 9 of [1]:
        //
        //- constant c = h / (1-kappa) gamma_k term
        //- lu factorisation of (M - c * J) used in newton iteration (same equation)
        //- psi term

        state.h *= factor;
        self.n_equal_steps = 0;

        // update D using equations in section 3.2 of [1]
        self.u = Self::_compute_r(self.order, M::T::one());
        self.r = Self::_compute_r(self.order, factor);
        let ru = self.r.mat_mul(&self.u);
        // D[0:order] = R * U * D[0:order]
        {
            let d_zero_order = self.diff.columns(0, self.order + 1);
            let mut d_zero_order_tmp = self.diff_tmp.columns_mut(0, self.order + 1);
            d_zero_order_tmp.gemm_vo(M::T::one(),  &d_zero_order, &ru, M::T::zero()); // diff_sub = diff * RU
        }
        std::mem::swap(&mut self.diff, &mut self.diff_tmp);

        // update y0 (D has changed)
        self._predict(state);

        // update psi and c (h, D, y0 has changed)
        let callable = self.bdf_callable.as_ref().unwrap();
        callable.set_psi_and_y0(&self.diff, &self.gamma, &self.alpha, self.order, &state.y);
        callable.set_c(state.h, &self.alpha, self.order);

        // clear nonlinear's linear solver problem as lu factorisation has changed
        self.nonlinear_solver.as_mut().set_problem(Rc::new(SolverProblem::new(self.bdf_callable.as_ref().unwrap().clone(), self.ode_problem.as_ref().unwrap().problem.p.clone())));

    }

    
    fn _update_differences(&mut self, d: &M::V) {
        //update of difference equations can be done efficiently
        //by reusing d and D.
        //
        //From first equation on page 4 of [1]:
        //d = y_n - y^0_n = D^{k + 1} y_n
        //
        //Standard backwards difference gives
        //D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}
        //
        //Combining these gives the following algorithm
        let order = self.order;
        let d_minus_order_plus_one = d - self.diff.column(order + 1);
        self.diff.column_mut(order + 2).copy_from(&d_minus_order_plus_one);
        self.diff.column_mut(order + 1).copy_from(d);
        for i in (0..=order).rev() {
            let tmp = self.diff.column(i + 1).into_owned();
            self.diff.column_mut(i).copy_from(&tmp);
        }
    }
}


impl<M: Matrix, CRhs: NonLinearOp<V = M::V, T = M::T>, CMass: LinearOp<V = M::V, T = M::T>, CInit: ConstantOp<V = M::V, T = M::T>> OdeSolverMethod<CRhs, CMass, CInit> for Bdf<M, CRhs, CMass, CInit> 
where
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    fn interpolate(&self, state: &OdeSolverState<M::V>, t: CRhs::T) -> M::V {
        //interpolate solution at time values t* where t-h < t* < t
        //
        //definition of the interpolating polynomial can be found on page 7 of [1]
        let mut time_factor = M::T::from(1.0);
        let mut order_summation = self.diff.column(0).into_owned();
        for i in 0..self.order {
            let i_t = M::T::from(i as f64);
            time_factor *= (t - (state.t - state.h * i_t)) / (state.h * (M::T::one() + i_t));
            order_summation += self.diff.column(i + 1) * time_factor;
        }
        order_summation
    }
    
    fn problem(&self) -> Option<&Rc<OdeSolverProblem<CRhs, CMass, CInit>>> {
        self.ode_problem.as_ref()
    }
    
    fn get_statistics(&self) -> &OdeSolverStatistics {
        &self.statistics
    }
    
    fn set_problem(&mut self, state: &mut OdeSolverState<M::V>, problem: Rc<OdeSolverProblem<CRhs, CMass, CInit>>) {
        self.ode_problem = Some(problem);
        let ode_problem = self.ode_problem.as_ref().unwrap();
        let problem = &ode_problem.problem;
        let nstates = problem.f.nstates();
        self.order = 1usize; 
        self.n_equal_steps = 0;
        self.diff = M::zeros(nstates, Self::MAX_ORDER + 1);
        self.diff_tmp = M::zeros(nstates, Self::MAX_ORDER + 1);
        self.diff.column_mut(0).copy_from(&state.y);
        
        // kappa values for difference orders, taken from Table 1 of [1]
        let kappa = [M::T::from(0.0), M::T::from(-0.1850), M::T::from(-1.0) / M::T::from(9.0), M::T::from(-0.0823), M::T::from(-0.0415), M::T::from(0.0)];
        self.alpha = vec![M::T::zero()];
        self.gamma = vec![M::T::zero()];
        self.error_const = vec![M::T::one()];

        #[allow(clippy::needless_range_loop)]
        for i in 1..=Self::MAX_ORDER {
            let i_t = M::T::from(i as f64);
            let one_over_i = M::T::one() / i_t;
            let one_over_i_plus_one = M::T::one() / (i_t + M::T::one());
            self.gamma.push(self.gamma[i-1] + one_over_i);
            self.alpha.push(M::T::one() / ((M::T::one() - kappa[i]) * self.gamma[i]));
            self.error_const.push(kappa[i] * self.gamma[i] + one_over_i_plus_one);
        }

        // update initial step size based on function
        let mut scale = state.y.abs();
        scale *= problem.rtol;
        scale += &problem.atol;

        let f0 = problem.f.call(&state.y, &problem.p);

        // y1 = y0 + h * f0
        let mut y1 = state.y.clone();
        y1.axpy(state.h, &f0, M::T::one());

        // df = f1 here
        let mut df = problem.f.call(&y1, &problem.p);
        
        // store f1 in diff[1] for use in step size control
        let h_times_f1 = &df * state.h;
        self.diff.column_mut(1).copy_from(&h_times_f1);

        // now df = f1 - f0
        df.axpy(M::T::from(-1.0), &f0, M::T::one());
        df.component_div_assign(&scale);
        let d2 = df.norm();
        let one_over_order_plus_one = M::T::one() / (M::T::from(self.order as f64) + M::T::one());
        let mut new_h = state.h * d2.pow(-one_over_order_plus_one);
        if new_h > M::T::from(100.0) * state.h {
            new_h = M::T::from(100.0) * state.h;
        }
        state.h = new_h;

        // setup linear solver for first step
        self.bdf_callable = Some(Rc::new(BdfCallable::new(ode_problem.clone())));
        self.nonlinear_solver.as_mut().set_problem(Rc::new(SolverProblem::new(self.bdf_callable.as_ref().unwrap().clone(), ode_problem.problem.p.clone())));
        
        // setup U
        self.u = Self::_compute_r(self.order, M::T::one());
    }

    fn step(&mut self, state: &mut OdeSolverState<M::V>) -> Result<()> {
        // we will try and use the old jacobian unless convergence of newton iteration
        // fails
        // tells callable to update rhs jacobian if the jacobian is requested (by nonlinear solver)
        self.bdf_callable.as_ref().unwrap().set_rhs_jacobian_is_stale();
        // initialise step size and try to make the step,
        // iterate, reducing step size until error is in bounds
        let mut step_accepted = false;
        let nstates = self.diff.nrows();
        let mut d = <M::V as Vector>::zeros(nstates);
        let mut safety = M::T::from(0.0);
        let mut error_norm = M::T::from(0.0);
        let mut scale_y = <M::V as Vector>::zeros(0);

        // loop until step is accepted
        while !step_accepted {
            // solve BDF equation using y0 as starting point
            match self.nonlinear_solver.solve(&state.y) {
                Ok(y) => {
                    // test error is within tolerance
                    {
                        let problem = &self.ode_problem.as_ref().unwrap().problem;
                        scale_y = y.abs() * problem.rtol;
                        scale_y += &problem.atol;
                    }

                    // combine eq 3, 4 and 6 from [1] to obtain error
                    // Note that error = C_k * h^{k+1} y^{k+1}
                    // and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
                    //d.add_assign(&y);
                    d = y - &state.y;

                    let mut error =  &d * self.error_const[self.order];
                    error.component_div_assign(&scale_y);
                    error_norm = error.norm();
                    let maxiter = self.nonlinear_solver.max_iter() as f64;
                    let niter = self.nonlinear_solver.niter() as f64;
                    safety = M::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));
                    
                    if error_norm <= M::T::from(1.0) {
                        // step is accepted
                        step_accepted = true;
                    } else {
                        // step is rejected
                        // calculate optimal step size factor as per eq 2.46 of [2]
                        // and reduce step size and try again
                        let order = self.order as f64;
                        let mut factor = safety * error_norm.pow(M::T::from(-1.0 / (order + 1.0)));
                        if factor < M::T::from(Self::MIN_FACTOR) {
                            factor = M::T::from(Self::MIN_FACTOR);
                        }
                        self._update_step_size(factor, state);
                        step_accepted = false; 
                        continue
                    }
                }
                Err(_e) => {
                    // newton iteration did not converge, but jacobian has already been
                    // evaluated so reduce step size by 0.3 (as per [1]) and try again
                    self._update_step_size(M::T::from(0.3), state);
                    step_accepted = false;
                    continue
                }
            };
        }

        // take the accepted step
        state.t += state.h;
        
        self._update_differences(&d);

        // a change in order is only done after running at order k for k + 1 steps
        // (see page 83 of [2])
        self.n_equal_steps += 1;
        
        if self.n_equal_steps < self.order + 1 {
            self._predict(state);
            self.bdf_callable.as_mut().unwrap().set_psi_and_y0(&self.diff, &self.gamma, &self.alpha, self.order, &state.y);
        } else {
            let order = self.order;
            // similar to the optimal step size factor we calculated above for the current
            // order k, we need to calculate the optimal step size factors for orders
            // k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
            let error_m_norm = if order > 1 {
                let mut error_m = self.diff.column(order) * self.error_const[order - 1];
                error_m.component_div_assign(&scale_y);
                error_m.norm()
            } else {
                M::T::INFINITY
            };
            let error_p_norm = if order < Self::MAX_ORDER {
                let mut error_p = self.diff.column(order + 2) * self.error_const[order + 1];
                error_p.component_div_assign(&scale_y);
                error_p.norm()
            } else {
                M::T::INFINITY
            };

            let error_norms = [error_m_norm, error_norm, error_p_norm];
            let factors = error_norms.into_iter().enumerate().map(|(i, error_norm)| {
                error_norm.pow(M::T::from(-1.0 / (i as f64 + order as f64)))
            }).collect::<Vec<_>>();

            // now we have the three factors for orders k-1, k and k+1, pick the maximum in
            // order to maximise the resultant step size
            let max_index = factors.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            self.order += max_index - 1;

            let mut factor = safety * factors[max_index];
            if factor > M::T::from(Self::MAX_FACTOR) {
                factor = M::T::from(Self::MAX_FACTOR);
            }
            self._update_step_size(factor, state);
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use crate::{callable::{closure::Closure, constant_closure::ConstantClosure, filter::FilterCallable, linear_closure::LinearClosure}, ode_solver::tests::test_ode_solver};

    use super::*;

    #[test]
    fn test_bdf_nalgebra() {
        type T = f64;
        type M = nalgebra::DMatrix<T>;
        type CRhs = Closure<M, M>;
        type CMass = LinearClosure<M, M>;
        type CInit = ConstantClosure<M, M>;
        type S = Bdf<M, CRhs, CMass, CInit>;
        let s = S::default();
        let rs = NewtonNonlinearSolver::<FilterCallable<CRhs>>::default();
        test_ode_solver(s, rs);
    }
}