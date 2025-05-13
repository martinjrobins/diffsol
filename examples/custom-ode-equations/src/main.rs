mod common;
use common::{T, V, M, C};
mod my_rhs;
use my_rhs::MyRhs;
mod my_rhs_impl_op;
mod my_rhs_impl_nonlinear;
mod my_mass;
use my_mass::MyMass;
mod my_init;
use my_init::MyInit;
mod my_root;
use my_root::MyRoot;
mod my_out;
use my_out::MyOut;
mod my_equations;
use my_equations::MyEquations;
mod my_equations_impl_op;
mod my_equations_impl_ode_equations;
mod build;
use build::build;

fn main() {
    build();
}
    
