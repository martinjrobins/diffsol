use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{
    ode_solver::test_models::{
        exponential_decay::exponential_decay_problem,
        foodweb::{foodweb_problem, FoodWebContext},
        heat2d::head2d_problem,
        robertson::robertson,
    },
    SparseColMat,
};
use faer::Mat;
mod sundials_benches;

fn criterion_benchmark(c: &mut Criterion) {
    macro_rules! bench {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix:ty) => {
            c.bench_function(stringify!($name), |b| {
                let (problem, soln) = $model_problem::<$matrix>(false);
                b.iter(|| benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t))
            });
        };
    }

    bench!(
        nalgebra_bdf_exponential_decay,
        bdf,
        exponential_decay,
        exponential_decay_problem,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_esdirk34_exponential_decay,
        esdirk34,
        exponential_decay,
        exponential_decay_problem,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_tr_bdf2_exponential_decay,
        tr_bdf2,
        exponential_decay,
        exponential_decay_problem,
        nalgebra::DMatrix<f64>
    );
    #[cfg(feature = "sundials")]
    bench!(
        sundials_exponential_decay,
        sundials,
        exponential_decay,
        exponential_decay_problem,
        diffsol::SundialsMatrix
    );
    bench!(
        nalgebra_bdf_robertson,
        bdf,
        robertson,
        robertson,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_esdirk34_robertson,
        esdirk34,
        robertson,
        robertson,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_tr_bdf2_robertson,
        tr_bdf2,
        robertson,
        robertson,
        nalgebra::DMatrix<f64>
    );
    #[cfg(feature = "sundials")]
    bench!(
        sundials_robertson,
        sundials,
        robertson,
        robertson,
        diffsol::SundialsMatrix
    );
    bench!(
        faer_bdf_exponential_decay,
        bdf,
        exponential_decay,
        exponential_decay_problem,
        faer::Mat<f64>
    );
    bench!(
        faer_esdirk34_exponential_decay,
        esdirk34,
        exponential_decay,
        exponential_decay_problem,
        faer::Mat<f64>
    );
    bench!(
        faer_tr_bdf2_exponential_decay,
        tr_bdf2,
        exponential_decay,
        exponential_decay_problem,
        faer::Mat<f64>
    );
    bench!(
        faer_bdf_robertson,
        bdf,
        robertson,
        robertson,
        faer::Mat<f64>
    );
    bench!(
        faer_esdirk34_robertson,
        esdirk34,
        robertson,
        robertson,
        faer::Mat<f64>
    );
    bench!(
        faer_tr_bdf2_robertson,
        tr_bdf2,
        robertson,
        robertson,
        faer::Mat<f64>
    );

    macro_rules! bench_diffsl {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix:ty) => {
            #[cfg(feature = "diffsl")]
            c.bench_function(stringify!($name), |b| {
                let mut context = diffsol::DiffSlContext::default();
                let (problem, soln) = diffsol::ode_solver::test_models::$model::$model_problem::<
                    $matrix,
                >(&mut context, false);
                b.iter(|| benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t))
            });
        };
    }

    bench_diffsl!(
        nalgebra_bdf_diffsl_exponential_decay,
        bdf,
        robertson,
        robertson_diffsl,
        nalgebra::DMatrix<f64>
    );

    macro_rules! bench_wsize {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix:ty, $size:expr) => {
            c.bench_function(stringify!($name), |b| {
                let (problem, soln) = $model_problem::<$matrix, $size>();
                b.iter(|| benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t))
            });
        };
    }

    bench_wsize!(
        faer_sparse_bdf_heat2d_5,
        bdf,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5
    );
    bench_wsize!(
        faer_sparse_bdf_heat2d_10,
        bdf,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        10
    );
    bench_wsize!(
        faer_sparse_bdf_heat2d_20,
        bdf,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        20
    );
    bench_wsize!(
        faer_sparse_tr_bdf2_heat2d_5,
        tr_bdf2,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5
    );
    bench_wsize!(
        faer_sparse_tr_bdf2_heat2d_10,
        tr_bdf2,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        10
    );
    bench_wsize!(
        faer_sparse_tr_bdf2_heat2d_20,
        tr_bdf2,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        20
    );
    bench_wsize!(
        faer_sparse_esdirk_heat2d_5,
        esdirk34,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5
    );
    bench_wsize!(
        faer_sparse_esdirk_heat2d_10,
        esdirk34,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        10
    );
    bench_wsize!(
        faer_sparse_esdirk_heat2d_20,
        esdirk34,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        20
    );

    macro_rules! bench_foodweb {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix_dense:ty, $matrix:ty, $size:expr) => {
            c.bench_function(stringify!($name), |b| {
                let context = FoodWebContext::default();
                let (problem, soln) = $model_problem::<$matrix_dense, $matrix, $size>(&context);
                b.iter(|| benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t))
            });
        };
    }

    bench_foodweb!(
        faer_sparse_bdf_foodweb_5,
        bdf,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        5
    );
    bench_foodweb!(
        faer_sparse_bdf_foodweb_10,
        bdf,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        10
    );
    bench_foodweb!(
        faer_sparse_bdf_foodweb_20,
        bdf,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        20
    );
    bench_foodweb!(
        faer_sparse_tr_bdf2_foodweb_5,
        tr_bdf2,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        5
    );
    bench_foodweb!(
        faer_sparse_tr_bdf2_foodweb_10,
        tr_bdf2,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        10
    );
    bench_foodweb!(
        faer_sparse_tr_bdf2_foodweb_20,
        tr_bdf2,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        20
    );
    bench_foodweb!(
        faer_sparse_esdirk_foodweb_5,
        esdirk34,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        5
    );
    bench_foodweb!(
        faer_sparse_esdirk_foodweb_10,
        esdirk34,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        10
    );
    bench_foodweb!(
        faer_sparse_esdirk_foodweb_20,
        esdirk34,
        foodweb,
        foodweb_problem,
        Mat<f64>,
        SparseColMat<f64>,
        20
    );

    macro_rules! bench_diffsl_wsize {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix:ty, $size:expr) => {
            #[cfg(feature = "diffsl")]
            c.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut context = diffsol::DiffSlContext::default();
                    let (problem, soln) = diffsol::ode_solver::test_models::$model::$model_problem::<
                        $matrix,
                        $size,
                    >(&mut context);
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t);
                })
            });
        };
    }
    bench_diffsl_wsize!(
        faer_sparse_bdf_diffsl_heat2d_5,
        bdf,
        heat2d,
        heat2d_diffsl,
        SparseColMat<f64>,
        5
    );
    bench_diffsl_wsize!(
        faer_sparse_bdf_diffsl_heat2d_10,
        bdf,
        heat2d,
        heat2d_diffsl,
        SparseColMat<f64>,
        10
    );
    bench_diffsl_wsize!(
        faer_sparse_bdf_diffsl_heat2d_20,
        bdf,
        heat2d,
        heat2d_diffsl,
        SparseColMat<f64>,
        20
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
    bench_sundials!(sundials_foodweb_bnd_5, idaFoodWeb_bnd_5);
    bench_sundials!(sundials_foodweb_bnd_10, idaFoodWeb_bnd_10);
    bench_sundials!(sundials_foodweb_bnd_20, idaFoodWeb_bnd_20);
    bench_sundials!(sundials_roberts_dns, idaRoberts_dns);

    macro_rules! bench_diffsl_foodweb {
        ($name:ident, $solver:ident, $model:ident, $model_problem:ident, $matrix_dense:ty, $matrix:ty, $size:expr) => {
            #[cfg(feature = "diffsl")]
            c.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut context = diffsol::DiffSlContext::default();
                    let (problem, soln) = diffsol::ode_solver::test_models::$model::$model_problem::<
                        $matrix_dense,
                        $matrix,
                        $size,
                    >(&mut context);
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t);
                })
            });
        };
    }

    bench_diffsl_foodweb!(
        faer_sparse_bdf_diffsl_foodweb_5,
        bdf,
        foodweb,
        foodweb_diffsl,
        Mat<f64>,
        SparseColMat<f64>,
        5
    );
    bench_diffsl_foodweb!(
        faer_sparse_bdf_diffsl_foodweb_10,
        bdf,
        foodweb,
        foodweb_diffsl,
        Mat<f64>,
        SparseColMat<f64>,
        10
    );
    bench_diffsl_foodweb!(
        faer_sparse_bdf_diffsl_foodweb_20,
        bdf,
        foodweb,
        foodweb_diffsl,
        Mat<f64>,
        SparseColMat<f64>,
        20
    );
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
