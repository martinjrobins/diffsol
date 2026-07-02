use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{
    ode_equations::test_models::{
        exponential_decay::exponential_decay_problem, foodweb::foodweb_problem,
        heat2d::head2d_problem, robertson::robertson, robertson_ode::robertson_ode,
    },
    FaerLU, FaerMat, FaerSparseLU, FaerSparseMat, NalgebraLU, NalgebraMat,
};

mod common;
use common::{bench_explicit, bench_implicit, bench_implicit_cg, bench_implicit_rt};

fn criterion_benchmark(c: &mut Criterion) {
    *c = std::mem::take(c)
        .measurement_time(Duration::from_secs(1))
        .sample_size(5)
        .warm_up_time(Duration::from_secs(1));

    // -------------------------------------------------------------------------
    // Exponential Decay
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("exponential_decay_ci");
        bench_implicit!(
            g,
            nalgebra_bdf,
            bdf,
            NalgebraLU,
            exponential_decay_problem,
            NalgebraMat<f64>
        );
        bench_implicit!(
            g,
            nalgebra_esdirk34,
            esdirk34,
            NalgebraLU,
            exponential_decay_problem,
            NalgebraMat<f64>
        );
        bench_implicit!(
            g,
            nalgebra_tr_bdf2,
            tr_bdf2,
            NalgebraLU,
            exponential_decay_problem,
            NalgebraMat<f64>
        );
        bench_explicit!(
            g,
            nalgebra_tsit45,
            tsit45,
            exponential_decay_problem,
            NalgebraMat<f64>
        );
        bench_implicit!(
            g,
            faer_bdf,
            bdf,
            FaerLU,
            exponential_decay_problem,
            FaerMat<f64>
        );
        bench_implicit!(
            g,
            faer_esdirk34,
            esdirk34,
            FaerLU,
            exponential_decay_problem,
            FaerMat<f64>
        );
        bench_implicit!(
            g,
            faer_tr_bdf2,
            tr_bdf2,
            FaerLU,
            exponential_decay_problem,
            FaerMat<f64>
        );
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Robertson
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("robertson_ci");
        bench_implicit!(
            g,
            nalgebra_bdf,
            bdf,
            NalgebraLU,
            robertson,
            NalgebraMat<f64>
        );
        bench_implicit!(
            g,
            nalgebra_esdirk34,
            esdirk34,
            NalgebraLU,
            robertson,
            NalgebraMat<f64>
        );
        bench_implicit!(
            g,
            nalgebra_tr_bdf2,
            tr_bdf2,
            NalgebraLU,
            robertson,
            NalgebraMat<f64>
        );
        bench_implicit!(g, faer_bdf, bdf, FaerLU, robertson, FaerMat<f64>);
        bench_implicit!(g, faer_esdirk34, esdirk34, FaerLU, robertson, FaerMat<f64>);
        bench_implicit!(g, faer_tr_bdf2, tr_bdf2, FaerLU, robertson, FaerMat<f64>);
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Robertson ODE (sparse, reduced sizes for CI)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("robertson_ode_ci");
        bench_implicit_rt!(
            g,
            faer_sparse_bdf,
            bdf,
            FaerSparseLU,
            robertson_ode,
            FaerSparseMat<f64>,
            25,
            100
        );
        bench_implicit_rt!(
            g,
            faer_sparse_tr_bdf2,
            tr_bdf2,
            FaerSparseLU,
            robertson_ode,
            FaerSparseMat<f64>,
            25,
            100
        );
        bench_implicit_rt!(
            g,
            faer_sparse_esdirk,
            esdirk34,
            FaerSparseLU,
            robertson_ode,
            FaerSparseMat<f64>,
            25,
            100
        );
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Heat2D (sparse, reduced sizes for CI)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("heat2d_ci");
        bench_implicit_cg!(
            g,
            faer_sparse_bdf,
            bdf,
            FaerSparseLU,
            head2d_problem,
            FaerSparseMat<f64>,
            5,
            10
        );
        bench_implicit_cg!(
            g,
            faer_sparse_tr_bdf2,
            tr_bdf2,
            FaerSparseLU,
            head2d_problem,
            FaerSparseMat<f64>,
            5,
            10
        );
        bench_implicit_cg!(
            g,
            faer_sparse_esdirk,
            esdirk34,
            FaerSparseLU,
            head2d_problem,
            FaerSparseMat<f64>,
            5,
            10
        );
        g.finish();
    }

    // -------------------------------------------------------------------------
    // Foodweb (sparse, reduced sizes for CI)
    // -------------------------------------------------------------------------
    {
        let mut g = c.benchmark_group("foodweb_ci");
        bench_implicit_cg!(
            g,
            faer_sparse_bdf,
            bdf,
            FaerSparseLU,
            foodweb_problem,
            FaerSparseMat<f64>,
            5,
            10
        );
        bench_implicit_cg!(
            g,
            faer_sparse_tr_bdf2,
            tr_bdf2,
            FaerSparseLU,
            foodweb_problem,
            FaerSparseMat<f64>,
            5,
            10
        );
        bench_implicit_cg!(
            g,
            faer_sparse_esdirk,
            esdirk34,
            FaerSparseLU,
            foodweb_problem,
            FaerSparseMat<f64>,
            5,
            10
        );
        g.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
