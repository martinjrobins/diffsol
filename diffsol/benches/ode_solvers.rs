use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{
    ode_equations::test_models::{
        exponential_decay::exponential_decay_problem, foodweb::foodweb_problem,
        heat2d::head2d_problem, robertson::robertson, robertson_ode::robertson_ode,
    },
    FaerLU, FaerMat, FaerSparseLU, FaerSparseMat, NalgebraLU, NalgebraMat,
};

mod sundials_benches;

fn criterion_benchmark(c: &mut Criterion) {
    macro_rules! bench {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty) => {
            c.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let (problem, soln) = $model_problem::<$matrix>(false);
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals);
                })
            });
        };
    }

    macro_rules! bench_explicit {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix:ty) => {
            c.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let (problem, soln) = $model_problem::<$matrix>(false);
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    benchmarks::$solver::<_>(&problem, &t_evals);
                })
            });
        };
    }

    bench!(
        nalgebra_bdf_exponential_decay,
        bdf,
        NalgebraLU,
        exponential_decay,
        exponential_decay_problem,
        NalgebraMat<f64>
    );
    bench!(
        nalgebra_esdirk34_exponential_decay,
        esdirk34,
        NalgebraLU,
        exponential_decay,
        exponential_decay_problem,
        NalgebraMat<f64>
    );
    bench!(
        nalgebra_tr_bdf2_exponential_decay,
        tr_bdf2,
        NalgebraLU,
        exponential_decay,
        exponential_decay_problem,
        NalgebraMat<f64>
    );
    bench_explicit!(
        nalgebra_tsit45_exponential_decay,
        tsit45,
        exponential_decay,
        exponential_decay_problem,
        NalgebraMat<f64>
    );
    bench!(
        nalgebra_bdf_robertson,
        bdf,
        NalgebraLU,
        robertson,
        robertson,
        NalgebraMat<f64>
    );
    bench!(
        nalgebra_esdirk34_robertson,
        esdirk34,
        NalgebraLU,
        robertson,
        robertson,
        NalgebraMat<f64>
    );
    bench!(
        nalgebra_tr_bdf2_robertson,
        tr_bdf2,
        NalgebraLU,
        robertson,
        robertson,
        NalgebraMat<f64>
    );
    bench!(
        faer_bdf_exponential_decay,
        bdf,
        FaerLU,
        exponential_decay,
        exponential_decay_problem,
        FaerMat<f64>
    );
    bench!(
        faer_esdirk34_exponential_decay,
        esdirk34,
        FaerLU,
        exponential_decay,
        exponential_decay_problem,
        FaerMat<f64>
    );
    bench!(
        faer_tr_bdf2_exponential_decay,
        tr_bdf2,
        FaerLU,
        exponential_decay,
        exponential_decay_problem,
        FaerMat<f64>
    );
    bench!(
        faer_bdf_robertson,
        bdf,
        FaerLU,
        robertson,
        robertson,
        FaerMat<f64>
    );
    bench!(
        faer_esdirk34_robertson,
        esdirk34,
        FaerLU,
        robertson,
        robertson,
        FaerMat<f64>
    );
    bench!(
        faer_tr_bdf2_robertson,
        tr_bdf2,
        FaerLU,
        robertson,
        robertson,
        FaerMat<f64>
    );

    macro_rules! bench_robertson_ode {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty,  $($N:expr),+) => {
            $(c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $model_problem::<$matrix>(false, $N);
                    let t_evals = soln.solution_points.iter().map(|sp| sp.t).collect::<Vec<_>>();
                    benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals);
                })
            });)+
        };
    }

    bench_robertson_ode!(
        faer_sparse_bdf_robertson_ode,
        bdf,
        FaerSparseLU,
        robertson_ode,
        robertson_ode,
        FaerSparseMat<f64>,
        25,
        100,
        400,
        900
    );

    bench_robertson_ode!(
        faer_sparse_tr_bdf2_robertson_ode,
        tr_bdf2,
        FaerSparseLU,
        robertson_ode,
        robertson_ode,
        FaerSparseMat<f64>,
        25,
        100,
        400,
        900
    );

    bench_robertson_ode!(
        faer_sparse_esdirk_robertson_ode,
        esdirk34,
        FaerSparseLU,
        robertson_ode,
        robertson_ode,
        FaerSparseMat<f64>,
        25,
        100,
        400,
        900
    );

    macro_rules! bench_diffsl_robertson {
        ($name:ident, $solver:ident, $linear_solver:ident, $matrix:ty) => {
            #[cfg(feature = "diffsl-llvm")]
            c.bench_function(stringify!($name), |b| {
                use diffsol::ode_equations::test_models::robertson::*;
                use diffsol::LlvmModule;
                let (problem, soln) = robertson_diffsl_problem::<$matrix, LlvmModule>();
                let t_evals = soln
                    .solution_points
                    .iter()
                    .map(|sp| sp.t)
                    .collect::<Vec<_>>();
                b.iter(|| benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals))
            });
        };
    }

    bench_diffsl_robertson!(
        nalgebra_bdf_diffsl_robertson,
        bdf,
        NalgebraLU,
        NalgebraMat<f64>
    );

    macro_rules! bench_wsize {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty, $($N:expr),+) => {
            $(c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $model_problem::<$matrix, $N>();
                    let t_evals = soln.solution_points.iter().map(|sp| sp.t).collect::<Vec<_>>();
                    benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals)
                })
            });)+
        };
    }

    bench_wsize!(
        faer_sparse_bdf_heat2d,
        bdf,
        FaerSparseLU,
        heat2d,
        head2d_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    bench_wsize!(
        faer_sparse_tr_bdf2_heat2d,
        tr_bdf2,
        FaerSparseLU,
        heat2d,
        head2d_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    bench_wsize!(
        faer_sparse_esdirk_heat2d,
        esdirk34,
        FaerSparseLU,
        heat2d,
        head2d_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    macro_rules! bench_foodweb {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty, $($N:expr),+) => {
            $(c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $model_problem::<$matrix, $N>();
                    let t_evals = soln.solution_points.iter().map(|sp| sp.t).collect::<Vec<_>>();
                    benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals)
                })
            });)+
        };
    }

    bench_foodweb!(
        faer_sparse_bdf_foodweb,
        bdf,
        FaerSparseLU,
        foodweb,
        foodweb_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    bench_foodweb!(
        faer_sparse_tr_bdf2_foodweb,
        tr_bdf2,
        FaerSparseLU,
        foodweb,
        foodweb_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    bench_foodweb!(
        faer_sparse_esdirk_foodweb,
        esdirk34,
        FaerSparseLU,
        foodweb,
        foodweb_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    macro_rules! bench_diffsl_heat2d {
        ($name:ident, $solver:ident, $linear_solver:ident, $model_problem:ident, $matrix:ty, $($N:expr),+) => {
            $(#[cfg(feature = "diffsl-llvm")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                use diffsol::ode_equations::test_models::heat2d::*;
                use diffsol::LlvmModule;
                let (problem, soln) = $model_problem::<$matrix, LlvmModule, $N>();
                let t_evals = soln.solution_points.iter().map(|sp| sp.t).collect::<Vec<_>>();
                b.iter(|| {
                    benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals)
                })
            });)+
        };
    }

    macro_rules! bench_diffsl_heat1d {
        ($name:ident, $solver:ident, $model_problem:ident, $matrix:ty, $($N:expr),+) => {
            $(#[cfg(feature = "diffsl-llvm")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                use diffsol::ode_equations::test_models::heat1d::*;
                use diffsol::LlvmModule;
                let (problem, soln) = $model_problem::<$matrix, LlvmModule, $N>();
                let t_evals = soln.solution_points.iter().map(|sp| sp.t).collect::<Vec<_>>();
                b.iter(|| {
                    benchmarks::$solver::<_>(&problem, &t_evals)
                })
            });)+
        };
    }

    bench_diffsl_heat2d!(
        faer_sparse_bdf_diffsl_heat2d,
        bdf,
        FaerSparseLU,
        heat2d_diffsl_problem,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );

    bench_diffsl_heat1d!(
        faer_tsit45_diffsl_heat1d,
        tsit45,
        heat1d_diffsl_problem,
        FaerMat<f64>,
        10,
        20,
        40,
        80
    );

    bench_diffsl_heat1d!(
        nalgebra_tsit45_diffsl_heat1d,
        tsit45,
        heat1d_diffsl_problem,
        NalgebraMat<f64>,
        10,
        20,
        40,
        80
    );

    macro_rules! bench_sundials {
        ($name:ident, $solver:ident) => {
            #[cfg(feature = "sundials")]
            c.bench_function(stringify!($name), |b| {
                b.iter(|| unsafe { sundials_benches::$solver() })
            });
        };
        () => {};
    }

    bench_sundials!(sundials_heat2d_klu_5, idaHeat2d_klu_5);
    bench_sundials!(sundials_heat2d_klu_10, idaHeat2d_klu_10);
    bench_sundials!(sundials_heat2d_klu_20, idaHeat2d_klu_20);
    bench_sundials!(sundials_heat2d_klu_30, idaHeat2d_klu_30);
    bench_sundials!(sundials_foodweb_bnd_5, idaFoodWeb_bnd_5);
    bench_sundials!(sundials_foodweb_bnd_10, idaFoodWeb_bnd_10);
    bench_sundials!(sundials_foodweb_bnd_20, idaFoodWeb_bnd_20);
    bench_sundials!(sundials_foodweb_bnd_30, idaFoodWeb_bnd_30);
    bench_sundials!(sundials_roberts_dns, idaRoberts_dns);

    macro_rules! bench_sundials_ngroups {
        ($name:ident, $solver:ident, $($N:expr),+) => {
            $(#[cfg(feature = "sundials")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| unsafe { sundials_benches::$solver($N) })
            });)+
        };
    }

    bench_sundials_ngroups!(
        sundials_robertson_ode_klu,
        cvRoberts_block_klu,
        25,
        100,
        400,
        900
    );

    macro_rules! bench_diffsl_foodweb {
        ($name:ident, $solver:ident, $linear_solver:ident, $matrix:ty, $($N:expr),+) => {
            $(#[cfg(feature = "diffsl-llvm")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                use diffsol::ode_equations::test_models::foodweb::*;
                use diffsol::LlvmModule;
                let (problem, soln) = foodweb_diffsl_problem::<$matrix, LlvmModule, $N>();
                let t_evals = soln.solution_points.iter().map(|sp| sp.t).collect::<Vec<_>>();
                b.iter(|| {
                    benchmarks::$solver::<_, $linear_solver<_>>(&problem, &t_evals)
                })
            });)+

        };
    }

    bench_diffsl_foodweb!(
        faer_sparse_bdf_diffsl_foodweb,
        bdf,
        FaerSparseLU,
        FaerSparseMat<f64>,
        5,
        10,
        20,
        30
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

mod benchmarks {
    use diffsol::matrix::MatrixRef;
    use diffsol::vector::VectorRef;
    use diffsol::LinearSolver;
    use diffsol::{
        DefaultDenseMatrix, DefaultSolver, Matrix, OdeEquationsImplicit, OdeSolverMethod,
        OdeSolverProblem,
    };

    // bdf
    pub fn bdf<Eqn, LS>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
    where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        LS: LinearSolver<Eqn::M>,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let mut s = problem.bdf::<LS>().unwrap();
        let _y = s.solve_dense(t_evals);
    }

    pub fn esdirk34<Eqn, LS>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
    where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        LS: LinearSolver<Eqn::M>,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let mut s = problem.esdirk34::<LS>().unwrap();
        let _y = s.solve_dense(t_evals);
    }

    pub fn tr_bdf2<Eqn, LS>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
    where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        LS: LinearSolver<Eqn::M>,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        let _y = s.solve_dense(t_evals);
    }

    pub fn tsit45<Eqn>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
    where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let mut s = problem.tsit45().unwrap();
        let _y = s.solve_dense(t_evals);
    }
}
