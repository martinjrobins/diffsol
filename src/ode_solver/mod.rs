use anyhow::Context;
use std::rc::Rc;

use crate::{
    op::{filter::FilterCallable, ode_rhs::OdeRhs, Op},
    Matrix, NonLinearSolver, OdeEquations, SolverProblem, Vector, VectorIndex,
};

use anyhow::Result;

use self::equations::{OdeSolverEquations, OdeSolverEquationsMassI};

#[cfg(feature = "diffsl")]
use self::diffsl::DiffSl;

pub mod bdf;
pub mod equations;
pub mod test_models;

#[cfg(feature = "diffsl")]
pub mod diffsl;

#[cfg(feature = "sundials")]
pub mod sundials;

pub trait OdeSolverMethod<Eqn: OdeEquations> {
    fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;
    fn set_problem(&mut self, state: &mut OdeSolverState<Eqn::M>, problem: &OdeSolverProblem<Eqn>);
    fn step(&mut self, state: &mut OdeSolverState<Eqn::M>) -> Result<()>;
    fn interpolate(&self, state: &OdeSolverState<Eqn::M>, t: Eqn::T) -> Eqn::V;
    fn solve(&mut self, problem: &OdeSolverProblem<Eqn>, t: Eqn::T) -> Result<Eqn::V> {
        let mut state = OdeSolverState::new(&problem);
        self.set_problem(&mut state, &problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
    fn make_consistent_and_solve<RS: NonLinearSolver<FilterCallable<OdeRhs<Eqn>>>>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        t: Eqn::T,
        root_solver: &mut RS,
    ) -> Result<Eqn::V> {
        let mut state = OdeSolverState::new_consistent(&problem, root_solver)?;
        self.set_problem(&mut state, &problem);
        while state.t <= t {
            self.step(&mut state)?;
        }
        Ok(self.interpolate(&state, t))
    }
}

pub struct OdeSolverState<M: Matrix> {
    pub y: M::V,
    pub t: M::T,
    pub h: M::T,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix> OdeSolverState<M> {
    pub fn new<Eqn>(ode_problem: &OdeSolverProblem<Eqn>) -> Self
    where
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let y = ode_problem.eqn.init(t);
        Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        }
    }
    fn new_consistent<Eqn, S>(
        ode_problem: &OdeSolverProblem<Eqn>,
        root_solver: &mut S,
    ) -> Result<Self>
    where
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
        S: NonLinearSolver<FilterCallable<OdeRhs<Eqn>>> + ?Sized,
    {
        let t = ode_problem.t0;
        let h = ode_problem.h0;
        let indices = ode_problem.eqn.algebraic_indices();
        let mut y = ode_problem.eqn.init(t);
        if indices.len() == 0 {
            return Ok(Self {
                y,
                t,
                h,
                _phantom: std::marker::PhantomData,
            });
        }
        let mut y_filtered = y.filter(&indices);
        let atol = Rc::new(ode_problem.atol.as_ref().filter(&indices));
        let rhs = Rc::new(OdeRhs::new(ode_problem.eqn.clone()));
        let f = Rc::new(FilterCallable::new(rhs, &y, indices));
        let rtol = ode_problem.rtol;
        let init_problem = SolverProblem::new(f, t, atol, rtol);
        root_solver.set_problem(init_problem);
        root_solver.solve_in_place(&mut y_filtered)?;
        let init_problem = root_solver.problem().unwrap();
        let indices = init_problem.f.indices();
        y.scatter_from(&y_filtered, indices);
        Ok(Self {
            y,
            t,
            h,
            _phantom: std::marker::PhantomData,
        })
    }
}

pub struct OdeSolverProblem<Eqn: OdeEquations> {
    pub eqn: Rc<Eqn>,
    pub rtol: Eqn::T,
    pub atol: Rc<Eqn::V>,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
}

// impl clone
impl<Eqn: OdeEquations> Clone for OdeSolverProblem<Eqn> {
    fn clone(&self) -> Self {
        Self {
            eqn: self.eqn.clone(),
            rtol: self.rtol,
            atol: self.atol.clone(),
            t0: self.t0,
            h0: self.h0,
        }
    }
}

impl<Eqn: OdeEquations> OdeSolverProblem<Eqn> {
    pub fn default_rtol() -> Eqn::T {
        Eqn::T::from(1e-6)
    }
    pub fn default_atol(nstates: usize) -> Eqn::V {
        Eqn::V::from_element(nstates, Eqn::T::from(1e-6))
    }
    pub fn new(eqn: Eqn, rtol: Eqn::T, atol: Eqn::V, t0: Eqn::T, h0: Eqn::T) -> Self {
        let eqn = Rc::new(eqn);
        let atol = Rc::new(atol);
        Self {
            eqn,
            rtol,
            atol,
            t0,
            h0,
        }
    }

    pub fn set_params(&mut self, p: Eqn::V) -> Result<()> {
        let eqn = Rc::get_mut(&mut self.eqn).context("Failed to get mutable reference to equations, is there a solver created with this problem?")?;
        eqn.set_params(p);
        Ok(())
    }
}

pub struct OdeBuilder {
    t0: f64,
    h0: f64,
    rtol: f64,
    atol: Vec<f64>,
    p: Vec<f64>,
    use_coloring: bool,
}

impl Default for OdeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OdeBuilder {
    pub fn new() -> Self {
        Self {
            t0: 0.0,
            h0: 1.0,
            rtol: 1e-6,
            atol: vec![1e-6],
            p: vec![],
            use_coloring: false,
        }
    }
    pub fn t0(mut self, t0: f64) -> Self {
        self.t0 = t0;
        self
    }
    pub fn h0(mut self, h0: f64) -> Self {
        self.h0 = h0;
        self
    }
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }
    pub fn atol<V, T>(mut self, atol: V) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.atol = atol.into_iter().map(|x| f64::from(x)).collect();
        self
    }
    pub fn p<V, T>(mut self, p: V) -> Self
    where
        V: IntoIterator<Item = T>,
        f64: From<T>,
    {
        self.p = p.into_iter().map(|x| f64::from(x)).collect();
        self
    }
    pub fn use_coloring(mut self, use_coloring: bool) -> Self {
        self.use_coloring = use_coloring;
        self
    }

    fn build_atol<V: Vector>(atol: Vec<f64>, nstates: usize) -> Result<V> {
        if atol.len() == 1 {
            Ok(V::from_element(nstates, V::T::from(atol[0])))
        } else if atol.len() != nstates {
            Err(anyhow::anyhow!(
                "atol must have length 1 or equal to the number of states"
            ))
        } else {
            let mut v = V::zeros(nstates);
            for (i, &a) in atol.iter().enumerate() {
                v[i] = V::T::from(a);
            }
            Ok(v)
        }
    }

    fn build_p<V: Vector>(p: Vec<f64>) -> V {
        let mut v = V::zeros(p.len());
        for (i, &p) in p.iter().enumerate() {
            v[i] = V::T::from(p);
        }
        v
    }

    #[allow(clippy::type_complexity)]
    pub fn build_ode_with_mass<M, F, G, H, I>(
        self,
        rhs: F,
        rhs_jac: G,
        mass: H,
        init: I,
    ) -> Result<OdeSolverProblem<OdeSolverEquations<M, F, G, H, I>>>
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        H: Fn(&M::V, &M::V, M::T, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Self::build_p(self.p);
        let eqn = OdeSolverEquations::new_ode_with_mass(
            rhs,
            rhs_jac,
            mass,
            init,
            p,
            M::T::from(self.t0),
            self.use_coloring,
        );
        let atol = Self::build_atol(self.atol, eqn.nstates())?;
        Ok(OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
        ))
    }

    pub fn build_ode<M, F, G, I>(
        self,
        rhs: F,
        rhs_jac: G,
        init: I,
    ) -> Result<OdeSolverProblem<OdeSolverEquationsMassI<M, F, G, I>>>
    where
        M: Matrix,
        F: Fn(&M::V, &M::V, M::T, &mut M::V),
        G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        I: Fn(&M::V, M::T) -> M::V,
    {
        let p = Self::build_p(self.p);
        let eqn = OdeSolverEquationsMassI::new_ode(
            rhs,
            rhs_jac,
            init,
            p,
            M::T::from(self.t0),
            self.use_coloring,
        );
        let atol = Self::build_atol(self.atol, eqn.nstates())?;
        Ok(OdeSolverProblem::new(
            eqn,
            M::T::from(self.rtol),
            atol,
            M::T::from(self.t0),
            M::T::from(self.h0),
        ))
    }

    #[cfg(feature = "diffsl")]
    pub fn build_diffsl(self, source: &str) -> Result<OdeSolverProblem<DiffSl>> {
        type V = diffsl::V;
        type T = diffsl::T;
        let p = Self::build_p::<V>(self.p);
        let eqn = DiffSl::new(source, p, self.use_coloring)?;
        let atol = Self::build_atol::<V>(self.atol, eqn.nstates())?;
        Ok(OdeSolverProblem::new(
            eqn,
            T::from(self.rtol),
            atol,
            T::from(self.t0),
            T::from(self.h0),
        ))
    }
}

pub struct OdeSolverSolutionPoint<V: Vector> {
    pub state: V,
    pub t: V::T,
}

pub struct OdeSolverSolution<V: Vector> {
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
}

impl<V: Vector> OdeSolverSolution<V> {
    pub fn push(&mut self, state: V, t: V::T) {
        // find the index to insert the new point keeping the times sorted
        let index = self
            .solution_points
            .iter()
            .position(|x| x.t > t)
            .unwrap_or(self.solution_points.len());
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
    }
}

impl<V: Vector> Default for OdeSolverSolution<V> {
    fn default() -> Self {
        Self {
            solution_points: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_models::{
        exponential_decay::exponential_decay_problem,
        exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem,
        robertson::robertson, robertson_ode::robertson_ode,
    };
    use super::*;
    use crate::nonlinear_solver::newton::NewtonNonlinearSolver;
    use crate::scalar::scale;
    use tests::bdf::Bdf;
    use tests::test_models::dydt_y2::dydt_y2_problem;
    use tests::test_models::gaussian_decay::gaussian_decay_problem;

    fn test_ode_solver<M, Eqn>(
        method: &mut impl OdeSolverMethod<Eqn>,
        mut root_solver: impl NonLinearSolver<FilterCallable<OdeRhs<Eqn>>>,
        problem: OdeSolverProblem<Eqn>,
        solution: OdeSolverSolution<M::V>,
        override_tol: Option<M::T>,
    ) where
        M: Matrix + 'static,
        Eqn: OdeEquations<M = M, T = M::T, V = M::V>,
    {
        let mut state = OdeSolverState::new_consistent(&problem, &mut root_solver).unwrap();
        method.set_problem(&mut state, &problem);
        for point in solution.solution_points.iter() {
            while state.t < point.t {
                method.step(&mut state).unwrap();
            }

            let soln = method.interpolate(&state, point.t);

            if let Some(override_tol) = override_tol {
                soln.assert_eq(&point.state, override_tol);
            } else {
                let tol = {
                    let problem = method.problem().unwrap();
                    soln.abs() * scale(problem.rtol) + problem.atol.as_ref()
                };
                soln.assert_eq(&point.state, tol[0]);
            }
        }
    }

    type Mcpu = nalgebra::DMatrix<f64>;

    #[test]
    fn test_bdf_nalgebra_exponential_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = exponential_decay_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 16
        number_of_steps: 19
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 54
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.011892071150027213
        final_step_size: 0.23215911532645564
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 56
        number_of_jac_mul_evals: 2
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, None);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 18
        number_of_steps: 21
        number_of_error_test_failures: 7
        number_of_nonlinear_solver_iterations: 58
        number_of_nonlinear_solver_fails: 2
        initial_step_size: 0.004450050658086208
        final_step_size: 0.20974041151932246
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 62
        number_of_jac_mul_evals: 7
        number_of_mass_evals: 64
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 2
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = robertson::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 129
        number_of_steps: 374
        number_of_error_test_failures: 17
        number_of_nonlinear_solver_iterations: 1046
        number_of_nonlinear_solver_fails: 25
        initial_step_size: 0.0000045643545698038086
        final_step_size: 7622676567.923919
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1049
        number_of_jac_mul_evals: 55
        number_of_mass_evals: 1052
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = robertson::<Mcpu>(true);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 129
        number_of_steps: 374
        number_of_error_test_failures: 17
        number_of_nonlinear_solver_iterations: 1046
        number_of_nonlinear_solver_fails: 25
        initial_step_size: 0.0000045643545698038086
        final_step_size: 7622676567.923919
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1049
        number_of_jac_mul_evals: 58
        number_of_mass_evals: 1051
        number_of_mass_matrix_evals: 2
        number_of_jacobian_matrix_evals: 18
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = robertson_ode::<Mcpu>(false);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 105
        number_of_steps: 346
        number_of_error_test_failures: 8
        number_of_nonlinear_solver_iterations: 941
        number_of_nonlinear_solver_fails: 18
        initial_step_size: 0.0000038381494276795106
        final_step_size: 7310380599.023874
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 943
        number_of_jac_mul_evals: 51
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 17
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = dydt_y2_problem::<Mcpu>(false, 10);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 109
        number_of_steps: 469
        number_of_error_test_failures: 16
        number_of_nonlinear_solver_iterations: 1440
        number_of_nonlinear_solver_fails: 4
        initial_step_size: 0.000000024472764934039197
        final_step_size: 0.6825461945378934
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1442
        number_of_jac_mul_evals: 50
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 5
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = dydt_y2_problem::<Mcpu>(true, 10);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 109
        number_of_steps: 469
        number_of_error_test_failures: 16
        number_of_nonlinear_solver_iterations: 1440
        number_of_nonlinear_solver_fails: 4
        initial_step_size: 0.000000024472764934039197
        final_step_size: 0.6825461945378934
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 1442
        number_of_jac_mul_evals: 15
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 5
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let mut s = Bdf::default();
        let rs = NewtonNonlinearSolver::default();
        let (problem, soln) = gaussian_decay_problem::<Mcpu>(false, 10);
        test_ode_solver(&mut s, rs, problem.clone(), soln, Some(1.0e-4));
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        ---
        number_of_linear_solver_setups: 15
        number_of_steps: 56
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 144
        number_of_nonlinear_solver_fails: 0
        initial_step_size: 0.0025148668593658707
        final_step_size: 0.19513473994542221
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.as_ref().get_statistics(), @r###"
        ---
        number_of_rhs_evals: 146
        number_of_jac_mul_evals: 10
        number_of_mass_evals: 0
        number_of_mass_matrix_evals: 0
        number_of_jacobian_matrix_evals: 1
        "###);
    }
}
