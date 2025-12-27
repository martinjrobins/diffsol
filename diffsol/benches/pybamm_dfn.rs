use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{OdeBuilder, OdeSolverMethod};

type M = diffsol::FaerSparseMat<f64>;
type LS = diffsol::FaerSparseLU<f64>;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pybamm_dfn");
    group.sample_size(10);

    #[cfg(feature = "diffsl-llvm")]
    group.bench_function("pybamm_dfn_solve", |b| {
        type CG = diffsol::LlvmModule;
        let full_text = std::fs::read_to_string("benches/pybamm_dfn.diffsl").unwrap();
        let mut problem = OdeBuilder::<M>::new()
            .build_from_diffsl::<CG>(full_text.as_str())
            .unwrap();
        problem.ic_options.armijo_constant = 1e-1;
        let t0 = 0.0;
        let tf = 3600.0;
        let t_interp = (0..100)
            .map(|i| t0 + (tf - t0) * (i as f64) / 99.0)
            .collect::<Vec<_>>();
        b.iter(|| {
            let mut solver = problem.bdf::<LS>().unwrap();
            solver.solve_dense(t_interp.as_slice()).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
