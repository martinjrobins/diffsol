use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{
    ode_equations::test_models::{
        exponential_decay::exponential_decay_problem, foodweb::foodweb_problem,
        heat2d::head2d_problem, robertson::robertson, robertson_ode::robertson_ode,
    },
    FaerLU, FaerMat, FaerSparseLU, FaerSparseMat, NalgebraLU, NalgebraMat,
};

#[cfg(feature = "diffsl-llvm")]
use diffsol::{
    ode_equations::test_models::{
        foodweb::foodweb_diffsl_problem,
        heat1d::heat1d_diffsl_problem,
        heat2d::heat2d_diffsl_problem,
        robertson::robertson_diffsl_problem,
    },
    LlvmModule,
};

mod sundials_benches;

macro_rules! bench_implicit {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty) => {
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                let (problem, soln) = $problem::<$m>(false);
                let t_evals = soln
                    .solution_points
                    .iter()
                    .map(|sp| sp.t)
                    .collect::<Vec<_>>();
                benchmarks::$solver::<_, $ls<_>>(&problem, &t_evals);
            })
        });
    };
}

macro_rules! bench_implicit_cg {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $problem::<$m, $N>();
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    benchmarks::$solver::<_, $ls<_>>(&problem, &t_evals);
                })
            });
        )+
    };
}

macro_rules! bench_implicit_rt {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $problem::<$m>(false, $N);
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    benchmarks::$solver::<_, $ls<_>>(&problem, &t_evals);
                })
            });
        )+
    };
}

macro_rules! bench_explicit {
    ($g:ident, $name:ident, $solver:ident, $problem:ident, $m:ty) => {
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                let (problem, soln) = $problem::<$m>(false);
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

macro_rules! bench_diffsl {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty) => {
        #[cfg(feature = "diffsl-llvm")]
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                let (problem, soln) = $problem::<$m, LlvmModule>();
                let t_evals = soln
                    .solution_points
                    .iter()
                    .map(|sp| sp.t)
                    .collect::<Vec<_>>();
                benchmarks::$solver::<_, $ls<_>>(&problem, &t_evals);
            })
        });
    };
}

macro_rules! bench_diffsl_cg {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            #[cfg(feature = "diffsl-llvm")]
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $problem::<$m, LlvmModule, $N>();
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    benchmarks::$solver::<_, $ls<_>>(&problem, &t_evals);
                })
            });
        )+
    };
}

macro_rules! bench_diffsl_explicit_cg {
    ($g:ident, $name:ident, $solver:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            #[cfg(feature = "diffsl-llvm")]
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $problem::<$m, LlvmModule, $N>();
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    benchmarks::$solver::<_>(&problem, &t_evals);
                })
            });
        )+
    };
}

macro_rules! bench_sundials {
    ($g:ident, $name:ident, $solver:ident) => {
        #[cfg(feature = "sundials")]
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| unsafe { sundials_benches::$solver() })
        });
    };
}

macro_rules! bench_sundials_rt {
    ($g:ident, $name:ident, $solver:ident, $($N:expr),+ $(,)?) => {
        $(
            #[cfg(feature = "sundials")]
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| unsafe { sundials_benches::$solver($N) })
            });
        )+
    };
}

fn criterion_benchmark(c: &mut Criterion) {
    // -------------------------------------------------------------------------
    // Exponential Decay
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("exponential_decay");
        bench_implicit!(
            g, nalgebra_bdf, bdf, NalgebraLU, exponential_decay_problem, NalgebraMat<f64>
        );
        bench_implicit!(
            g, nalgebra_esdirk34, esdirk34, NalgebraLU, exponential_decay_problem, NalgebraMat<f64>
        );
        bench_implicit!(
            g, nalgebra_tr_bdf2, tr_bdf2, NalgebraLU, exponential_decay_problem, NalgebraMat<f64>
        );
        bench_explicit!(
            g, nalgebra_tsit45, tsit45, exponential_decay_problem, NalgebraMat<f64>
        );
        bench_implicit!(
            g, faer_bdf, bdf, FaerLU, exponential_decay_problem, FaerMat<f64>
        );
        bench_implicit!(
            g, faer_esdirk34, esdirk34, FaerLU, exponential_decay_problem, FaerMat<f64>
        );
        bench_implicit!(
            g, faer_tr_bdf2, tr_bdf2, FaerLU, exponential_decay_problem, FaerMat<f64>
        );
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Robertson
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("robertson");
        bench_implicit!(g, nalgebra_bdf, bdf, NalgebraLU, robertson, NalgebraMat<f64>);
        bench_implicit!(
            g, nalgebra_esdirk34, esdirk34, NalgebraLU, robertson, NalgebraMat<f64>
        );
        bench_implicit!(
            g, nalgebra_tr_bdf2, tr_bdf2, NalgebraLU, robertson, NalgebraMat<f64>
        );
        bench_implicit!(g, faer_bdf, bdf, FaerLU, robertson, FaerMat<f64>);
        bench_implicit!(g, faer_esdirk34, esdirk34, FaerLU, robertson, FaerMat<f64>);
        bench_implicit!(g, faer_tr_bdf2, tr_bdf2, FaerLU, robertson, FaerMat<f64>);

        bench_diffsl!(
            g,
            nalgebra_bdf_diffsl,
            bdf,
            NalgebraLU,
            robertson_diffsl_problem,
            NalgebraMat<f64>
        );
        bench_sundials!(g, sundials_dns, idaRoberts_dns);
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Robertson ODE (sparse, multi-size)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("robertson_ode");
        bench_implicit_rt!(
            g,
            faer_sparse_bdf,
            bdf,
            FaerSparseLU,
            robertson_ode,
            FaerSparseMat<f64>,
            25,
            100,
            400,
            900
        );
        bench_implicit_rt!(
            g,
            faer_sparse_tr_bdf2,
            tr_bdf2,
            FaerSparseLU,
            robertson_ode,
            FaerSparseMat<f64>,
            25,
            100,
            400,
            900
        );
        bench_implicit_rt!(
            g,
            faer_sparse_esdirk,
            esdirk34,
            FaerSparseLU,
            robertson_ode,
            FaerSparseMat<f64>,
            25,
            100,
            400,
            900
        );
        bench_sundials_rt!(g, sundials_klu, cvRoberts_block_klu, 25, 100, 400, 900);
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Heat2D (sparse, multi-size)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("heat2d");
        bench_implicit_cg!(
            g,
            faer_sparse_bdf,
            bdf,
            FaerSparseLU,
            head2d_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );
        bench_implicit_cg!(
            g,
            faer_sparse_tr_bdf2,
            tr_bdf2,
            FaerSparseLU,
            head2d_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );
        bench_implicit_cg!(
            g,
            faer_sparse_esdirk,
            esdirk34,
            FaerSparseLU,
            head2d_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );

        bench_diffsl_cg!(
            g,
            faer_sparse_bdf_diffsl,
            bdf,
            FaerSparseLU,
            heat2d_diffsl_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );

        bench_sundials!(g, sundials_klu_5, idaHeat2d_klu_5);
        bench_sundials!(g, sundials_klu_10, idaHeat2d_klu_10);
        bench_sundials!(g, sundials_klu_20, idaHeat2d_klu_20);
        bench_sundials!(g, sundials_klu_30, idaHeat2d_klu_30);
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Foodweb (sparse, multi-size)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("foodweb");
        bench_implicit_cg!(
            g,
            faer_sparse_bdf,
            bdf,
            FaerSparseLU,
            foodweb_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );
        bench_implicit_cg!(
            g,
            faer_sparse_tr_bdf2,
            tr_bdf2,
            FaerSparseLU,
            foodweb_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );
        bench_implicit_cg!(
            g,
            faer_sparse_esdirk,
            esdirk34,
            FaerSparseLU,
            foodweb_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );

        bench_diffsl_cg!(
            g,
            faer_sparse_bdf_diffsl,
            bdf,
            FaerSparseLU,
            foodweb_diffsl_problem,
            FaerSparseMat<f64>,
            5,
            10,
            20,
            30
        );

        bench_sundials!(g, sundials_bnd_5, idaFoodWeb_bnd_5);
        bench_sundials!(g, sundials_bnd_10, idaFoodWeb_bnd_10);
        bench_sundials!(g, sundials_bnd_20, idaFoodWeb_bnd_20);
        bench_sundials!(g, sundials_bnd_30, idaFoodWeb_bnd_30);
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Heat1D (explicit, diffsl only)
    // -------------------------------------------------------------------------
    {
        #[allow(unused_mut)]
        let mut g = c.benchmark_group("heat1d");
        bench_diffsl_explicit_cg!(
            g,
            faer_tsit45_diffsl,
            tsit45,
            heat1d_diffsl_problem,
            FaerMat<f64>,
            10,
            20,
            40,
            80
        );
        bench_diffsl_explicit_cg!(
            g,
            nalgebra_tsit45_diffsl,
            tsit45,
            heat1d_diffsl_problem,
            NalgebraMat<f64>,
            10,
            20,
            40,
            80
        );
        g.finish();
    }
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
