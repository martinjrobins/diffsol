use diffsol::{NalgebraContext, NalgebraVec};
use diffsol::{NalgebraLU, NalgebraMat};
mod problem_implicit;
use problem_implicit::problem_implicit;
mod problem_explicit;
use problem_explicit::problem_explicit;
mod problem_mass;
use problem_mass::problem_mass;
mod problem_root;
use problem_root::problem_root;
mod problem_fwd_sens;
use problem_fwd_sens::problem_fwd_sens;
mod problem_sparse;
use problem_sparse::problem_sparse;
mod solve;
use solve::solve;
mod solve_dense;
use solve_dense::solve_dense;
mod solve_interpolate;
use solve_interpolate::solve_interpolate;
mod solve_step;
use solve_step::solve_step;
mod solve_match_step;
use solve_match_step::solve_match_step;
mod solve_fwd_sens;
use solve_fwd_sens::solve_fwd_sens;
mod solve_fwd_sens_step;
use solve_fwd_sens_step::solve_fwd_sens_step;
mod print_jacobian;
mod solve_adjoint_sens;
mod solve_adjoint_sens_sum_squares;
use print_jacobian::print_jacobian;
mod create_solvers;
use create_solvers::create_solvers;
mod create_solvers_uninit;
use create_solvers_uninit::create_solvers_uninit;
mod create_solvers_tableau;
use create_solvers_tableau::create_solvers_tableau;
type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;
type T = f64;
type LS = NalgebraLU<f64>;
type C = NalgebraContext;

fn main() {
    //
    // SPECIFYING THE PROBLEM
    //
    let problem = problem_fwd_sens();
    let mut solver = problem.bdf_sens::<LS>().unwrap();
    solve_fwd_sens(&mut solver);
    let mut solver = problem.bdf_sens::<LS>().unwrap();
    solve_fwd_sens_step(&mut solver);
    let _problem = problem_root();
    let _problem = problem_mass();
    let _problem = problem_explicit();
    let problem = problem_sparse();
    print_jacobian(&problem);
    let problem = problem_implicit();
    let mut solver = problem.bdf::<LS>().unwrap();
    solve_match_step(&mut solver);

    //
    // CHOOSING A SOLVER
    //
    create_solvers();
    create_solvers_uninit();
    create_solvers_tableau();

    //
    // SOLVING THE PROBLEM
    //
    let mut solver = problem.bdf::<LS>().unwrap();
    solve(&mut solver);
    let mut solver = problem.bdf::<LS>().unwrap();
    solve_dense(&mut solver);
    let mut solver = problem.bdf::<LS>().unwrap();
    solve_step(&mut solver);
    let mut solver = problem.bdf::<LS>().unwrap();
    solve_interpolate(&mut solver);
}
