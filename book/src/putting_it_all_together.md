# Putting it all together

Once you have structs implementing the functions for your system of equations, you can use the [`OdeSolverEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/struct.OdeSolverEquations.html) struct
to put it all together. This struct implements the [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) trait, and can be used to specify the problem to the solver.

Note that it is optional to use the `OdeSolverEquations` struct, you can implement the `OdeEquations` trait directly if you prefer, but the `OdeSolverEquations` struct can be useful to reduce boilerplate code 
and make it easier to specify the problem.

## Getting all your traits in order

The `OdeSolverEquations` struct requires arguments corresponding to the right-hand side function, mass matrix, root function, initial condition, and output functions.
For those that you want to provide, you can implement `NonLinearOp`, `LinearOp`, and `ConstantOp` traits for your structs, as described in the previous sections.
However, some of these arguments are optional and can be set to `None` if not needed. To do this, you still need to provide a placeholder type for these arguments, so you can use the 
included [`UnitCallable`](https://docs.rs/diffsol/latest/diffsol/op/unit/struct.UnitCallable.html) type for this purpose. For example lets assume that we already have objects implementing 
the `NonLinearOp` trait for the right-hand side function, and the `ConstantOp` trait for the initial condition, but we don't have a mass matrix, root function, or output function.
We can specify the missing arguments like so:

```rust
# fn main() {
# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
use std::rc::Rc;
use diffsol::UnitCallable;

let mass: Option<Rc<UnitCallable<M>>> = None;
let root: Option<Rc<UnitCallable<M>>> = None;
let out: Option<Rc<UnitCallable<M>>> = None;
# }
```

## Creating the equations

Now we have variables `rhs` and `init` that are structs implementing the required traits, and `mass`, `root`, and `out` set to `None`. Using these, we can create the `OdeSolverEquations` struct,
and then provide it to the `OdeBuilder` struct to create the problem. 

```rust
# fn main() {
# use std::rc::Rc;
# use diffsol::{NonLinearOp, NonLinearOpJacobian, OdeSolverProblem, Op, UnitCallable, ConstantClosure};
use diffsol::{OdeSolverEquations, OdeBuilder};

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
# 
# struct MyProblem {
#     p: Rc<V>,
# }
# 
# impl MyProblem {
#     fn new(p: Rc<V>) -> Self {
#         MyProblem { p }
#     }
# }
# 
# impl Op for MyProblem {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
# }
# 
# impl NonLinearOp for MyProblem {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = self.p[0] * x[0] * (1.0 - x[0] / self.p[1]);
#     }
# }
# impl NonLinearOpJacobian for MyProblem {
#     fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) {
#         y[0] = self.p[0] * v[0] * (1.0 - 2.0 * x[0] / self.p[1]);
#     }
# }
# 
# 
# let p = Rc::new(V::from_vec(vec![1.0, 10.0]));
# let rhs = Rc::new(MyProblem::new(p.clone()));
# 
# // use the provided constant closure to define the initial condition
# let init_fn = |_p: &V, _t: T| V::from_element(1, 0.1);
# let init = Rc::new(ConstantClosure::new(init_fn, p.clone()));
# 
# // we don't have a mass matrix, root or output functions, so we can set to None
# // we still need to give a placeholder type for these, so we use the diffsol::UnitCallable type
# let mass: Option<Rc<UnitCallable<M>>> = None;
# let root: Option<Rc<UnitCallable<M>>> = None;
# let out: Option<Rc<UnitCallable<M>>> = None;
# 
# let p = Rc::new(V::zeros(0));
let eqn = OdeSolverEquations::new(rhs, mass, root, init, out, p.clone());
let _problem = OdeBuilder::new().build_from_eqn(eqn).unwrap();
# }
```

Note the last two arguments to `OdeSolverProblem::new` are for sensitivity analysis which we will turn off for now.
