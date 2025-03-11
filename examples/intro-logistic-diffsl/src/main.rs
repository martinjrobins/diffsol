use diffsol::{CraneliftModule, DiffSl};
use diffsol::{OdeBuilder, OdeSolverMethod};
type M = nalgebra::DMatrix<f64>;
type CG = CraneliftModule;
type LS = diffsol::NalgebraLU<f64>;

fn main() {
    let eqn = DiffSl::<M, CG>::compile(
        "
     in = [r, k]
     r { 1.0 }
     k { 1.0 }
     u { 0.1 }
     F { r * u * (1.0 - u / k) }
 ",
    )
    .unwrap();
    let problem = OdeBuilder::<M>::new()
        .rtol(1e-6)
        .p([1.0, 10.0])
        .build_from_eqn(eqn)
        .unwrap();
    let mut solver = problem.bdf::<LS>().unwrap();
    let t = 0.4;
    let _soln = solver.solve(t).unwrap();
}
