//! Full BDF solves of the element-parallel test models on the `cuda-oxide` backend, swept over
//! lane count and state count.

use criterion::{criterion_group, criterion_main, Criterion};

// the shared helpers carry the whole bench-suite toolkit; this bench uses only two of them
#[allow(dead_code, unused_macros, unused_imports)]
mod common;

#[cfg(feature = "cuda-oxide")]
mod oxide {
    use super::*;
    use diffsol::{
        ode_equations::test_models::{
            foodweb::foodweb_elem_problem, heat2d::heat2d_elem_problem,
            robertson_ode::robertson_ode_elem_problem,
        },
        OxideLU, OxideMat,
    };

    /// Lane counts every model is swept over.
    const NBATCH: &[usize] = &[1, 10, 100];

    /// `heat2d` and `foodweb` take their grid size as a const generic, so the size sweep has to
    /// be a macro over literals rather than a loop.
    macro_rules! bench_const_generic {
        ($g:ident, $problem:ident, $($n:expr),+ $(,)?) => {
            $(
                for &nbatch in NBATCH {
                    let (problem, t_evals) =
                        common::setup_problem!($problem::<OxideMat, $n>(nbatch));
                    let id = format!("oxide_bdf/n{}_nbatch{}", $n, nbatch);
                    $g.bench_function(id, |b| {
                        b.iter(|| {
                            common::bdf::<_, OxideLU>(&problem, &t_evals);
                        })
                    });
                }
            )+
        };
    }

    pub fn benches(c: &mut Criterion) {
        {
            let mut g = c.benchmark_group("heat2d");
            g.sample_size(20);
            bench_const_generic!(g, heat2d_elem_problem, 5, 10, 20, 30);
            g.finish();
        }
        {
            let mut g = c.benchmark_group("foodweb");
            g.sample_size(20);
            bench_const_generic!(g, foodweb_elem_problem, 5, 10, 20, 30);
            g.finish();
        }
        {
            // 3 states per lane, so the lane sweep can go further than the PDE models'
            let mut g = c.benchmark_group("robertson_ode");
            g.sample_size(20);
            for &nbatch in &[1usize, 10, 100, 1000] {
                let (problem, t_evals) =
                    common::setup_problem!(robertson_ode_elem_problem::<OxideMat>(nbatch));
                g.bench_function(format!("oxide_bdf/nbatch{nbatch}"), |b| {
                    b.iter(|| {
                        common::bdf::<_, OxideLU>(&problem, &t_evals);
                    })
                });
            }
            g.finish();
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(feature = "cuda-oxide")]
    oxide::benches(c);
    #[cfg(not(feature = "cuda-oxide"))]
    let _ = c;
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
