use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{ode_solver::test_models::{
    exponential_decay::exponential_decay_problem, heat2d::head2d_problem, robertson::robertson
}, SparseColMat};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("nalgebra_bdf_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<nalgebra::DMatrix<f64>>(false);
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("nalgebra_esdirk34_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<nalgebra::DMatrix<f64>>(false);
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("nalgebra_tr_bdf2_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<nalgebra::DMatrix<f64>>(false);
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });

    #[cfg(feature = "sundials")]
    c.bench_function("sundials_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<diffsol::SundialsMatrix>(false);
        b.iter(|| benchmarks::sundials(&problem, soln.solution_points.last().unwrap().t))
    });

    c.bench_function("nalgebra_bdf_robertson", |b| {
        let (problem, soln) = robertson::<nalgebra::DMatrix<f64>>(false);
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("nalgebra_esdirk34_robertson", |b| {
        let (problem, soln) = robertson::<nalgebra::DMatrix<f64>>(false);
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("nalgebra_tr_bdf2_robertson", |b| {
        let (problem, soln) = robertson::<nalgebra::DMatrix<f64>>(false);
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });

    #[cfg(feature = "sundials")]
    c.bench_function("sundials_robertson", |b| {
        let (problem, soln) = robertson::<diffsol::SundialsMatrix>(false);
        b.iter(|| benchmarks::sundials(&problem, soln.solution_points.last().unwrap().t))
    });

    c.bench_function("faer_bdf_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<faer::Mat<f64>>(false);
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_esdirk34_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<faer::Mat<f64>>(false);
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_tr_bdf2_exponential_decay", |b| {
        let (problem, soln) = exponential_decay_problem::<faer::Mat<f64>>(false);
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });

    c.bench_function("faer_bdf_robertson", |b| {
        let (problem, soln) = robertson::<faer::Mat<f64>>(false);
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_esdirk34_robertson", |b| {
        let (problem, soln) = robertson::<faer::Mat<f64>>(false);
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_tr_bdf2_robertson", |b| {
        let (problem, soln) = robertson::<faer::Mat<f64>>(false);
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_bdf_heat2d_5", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 5>();
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_tr_bdf2_heat2d_5", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 5>();
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_esdirk34_heat2d_5", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 5>();
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_bdf_heat2d_10", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_tr_bdf2_heat2d_10", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_esdirk34_heat2d_10", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_bdf_heat2d_20", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 20>();
        b.iter(|| benchmarks::bdf(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_tr_bdf2_heat2d_20", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 20>();
        b.iter(|| benchmarks::tr_bdf2(&problem, soln.solution_points.last().unwrap().t))
    });
    c.bench_function("faer_sparse_esdirk34_heat2d_20", |b| {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 20>();
        b.iter(|| benchmarks::esdirk34(&problem, soln.solution_points.last().unwrap().t))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

mod benchmarks {
    use diffsol::matrix::MatrixRef;
    use diffsol::vector::VectorRef;
    use diffsol::{
        Bdf, DefaultDenseMatrix, DefaultSolver, Matrix, OdeEquations, OdeSolverMethod,
        OdeSolverProblem, Sdirk, Tableau,
    };

    // bdf
    pub fn bdf<Eqn>(problem: &OdeSolverProblem<Eqn>, t: Eqn::T)
    where
        Eqn: OdeEquations,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let mut s = Bdf::default();
        let _y = s.solve(problem, t);
    }

    pub fn esdirk34<Eqn>(problem: &OdeSolverProblem<Eqn>, t: Eqn::T)
    where
        Eqn: OdeEquations,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let tableau = Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::esdirk34();
        let linear_solver = <Eqn::M as DefaultSolver>::default_solver();
        let mut s = Sdirk::new(tableau, linear_solver);
        let _y = s.solve(problem, t);
    }

    pub fn tr_bdf2<Eqn>(problem: &OdeSolverProblem<Eqn>, t: Eqn::T)
    where
        Eqn: OdeEquations,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let tableau = Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::tr_bdf2();
        let linear_solver = <Eqn::M as DefaultSolver>::default_solver();
        let mut s = Sdirk::new(tableau, linear_solver);
        let _y = s.solve(problem, t);
    }

    #[cfg(feature = "sundials")]
    pub fn sundials<Eqn>(problem: &OdeSolverProblem<Eqn>, t: Eqn::T)
    where
        Eqn: OdeEquations<M = diffsol::SundialsMatrix, V = diffsol::SundialsVector, T = f64>,
    {
        use diffsol::SundialsIda;

        let mut s = SundialsIda::default();
        let _y = s.solve(problem, t);
    }
}
