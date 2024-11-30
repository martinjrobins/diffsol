# ODE systems

So far we've focused on using custom structs to specify individual equations, now we need to put these together to specify the entire system of equations. 

## Implementing the OdeEquations trait

To specify the entire system of equations, you need to implement the `Op`, [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) 
and [`OdeEquationsRef`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquationsRef.html) traits for your struct.

## Getting all your traits in order

The `OdeEquations` trait requires methods that return objects corresponding to the right-hand side function, mass matrix, root function, initial condition, and output functions.
Therefore, you need to already have structs that implement the `NonLinearOp`, `LinearOp`, and `ConstantOp` traits for these functions. For the purposes of this example, we will assume that
you have already implemented these traits for your structs. 

Often, the structs that implement these traits will have to use data defined in the struct that implements the `OdeEquations` trait. For example, they might wish to have a reference to the same parameter vector `p`. Therefore, you will often need to define lifetimes for these structs to ensure that they can access the data they need. 

Note that these struct will need to be lightweight and should not contain a significant amount of data. The data should be stored in the struct that implements the `OdeEquations` trait. This is because these structs will be created and destroyed many times during the course of the simulation (e.g. every time the right-hand side function is called).


```rust
# fn main() {
type T = f64;
type V = nalgebra::DVector<f64>;
type M = nalgebra::DMatrix<f64>;
struct MyRhs<'a> { p: &'a V } // implements NonLinearOp
struct MyMass<'a> { p: &'a V } // implements LinearOp
struct MyInit<'a> { p: &'a V } // implements ConstantOp
struct MyRoot<'a> { p: &'a V } // implements NonLinearOp
struct MyOut<'a> { p: &'a V } // implements NonLinearOp
# }
```

## Implementing the OdeEquations traits

Lets imagine we have a struct `MyProblem` that we want to use to specify the entire system of equations. We can implement the `Op`, `OdeEquations`, and `OdeEquationsRef` traits for this struct like so:

```rust
use diffsol::{Op, NonLinearOp, LinearOp, ConstantOp, OdeEquations, OdeEquationsRef};
# fn main() {
# type T = f64;
# type V = nalgebra::DVector<f64>;
# type M = nalgebra::DMatrix<f64>;
# struct MyRhs<'a> { p: &'a V } // implements NonLinearOp
# struct MyMass<'a> { p: &'a V } // implements LinearOp
# struct MyInit<'a> { p: &'a V } // implements ConstantOp
# struct MyRoot<'a> { p: &'a V } // implements NonLinearOp
# struct MyOut<'a> { p: &'a V } // implements NonLinearOp
# impl Op for MyRhs<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         2
#     }
# }
# impl NonLinearOp for MyRhs<'_> {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = x[0] * x[0];
#     }
# }
# impl Op for MyMass<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl LinearOp for MyMass<'_> {
#     fn gemv_inplace(&self, x: &V, _t: T, beta: T, y: &mut V) {
#         y[0] = x[0] * beta;
#     }
# }
# impl Op for MyInit<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl ConstantOp for MyInit<'_> {
#     fn call_inplace(&self, _t: T, y: &mut V) {
#         y[0] = 0.1;
#     }
# }
# impl Op for MyRoot<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl NonLinearOp for MyRoot<'_> {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = x[0] - 1.0;
#     }
# }
# impl Op for MyOut<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl NonLinearOp for MyOut<'_> {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = x[0];
#     }
# }

struct MyProblem {
    p: V,
}

impl MyProblem {
    fn new() -> Self {
        MyProblem { p: V::zeros(2) }
    }
}

impl Op for MyProblem {
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        2
    }
}

impl<'a> OdeEquationsRef<'a> for MyProblem {
    type Rhs = MyRhs<'a>;
    type Mass = MyMass<'a>;
    type Init = MyInit<'a>;
    type Root = MyRoot<'a>;
    type Out = MyOut<'a>;
}

impl OdeEquations for MyProblem {
    fn rhs(&self) -> <MyProblem as OdeEquationsRef<'_>>::Rhs {
        MyRhs { p: &self.p }
    }
    fn mass(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Mass> {
        Some(MyMass { p: &self.p })
    }
    fn init(&self) -> <MyProblem as OdeEquationsRef<'_>>::Init {
        MyInit { p: &self.p }
    }
    fn root(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Root> {
        Some(MyRoot { p: &self.p })
    }
    fn out(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Out> {
        Some(MyOut { p: &self.p })
    }
    fn set_params(&mut self, p: &V) {
        self.p.copy_from(p);
    }
}
# }
```

## Creating the problem

Now that we have our custom `OdeEquations` struct, we can use it in an `OdeBuilder` to create the problem. Hint: click the button below to see the full code, which includes the implementation of the `Op`, `NonLinearOp`, `LinearOp`, and `ConstantOp` traits for the `MyRhs`, `MyMass`, `MyInit`, `MyRoot`, and `MyOut` structs.

```rust
use diffsol::{Op, NonLinearOp, LinearOp, ConstantOp, OdeEquations, OdeEquationsRef};
# fn main() {
# type T = f64;
# type V = nalgebra::DVector<f64>;
# type M = nalgebra::DMatrix<f64>;
# struct MyRhs<'a> { p: &'a V } // implements NonLinearOp
# struct MyMass<'a> { p: &'a V } // implements LinearOp
# struct MyInit<'a> { p: &'a V } // implements ConstantOp
# struct MyRoot<'a> { p: &'a V } // implements NonLinearOp
# struct MyOut<'a> { p: &'a V } // implements NonLinearOp
# impl Op for MyRhs<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         2
#     }
# }
# impl NonLinearOp for MyRhs<'_> {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = x[0] * x[0];
#     }
# }
# impl Op for MyMass<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl LinearOp for MyMass<'_> {
#     fn gemv_inplace(&self, x: &V, _t: T, beta: T, y: &mut V) {
#         y[0] = x[0] * beta;
#     }
# }
# impl Op for MyInit<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl ConstantOp for MyInit<'_> {
#     fn call_inplace(&self, _t: T, y: &mut V) {
#         y[0] = 0.1;
#     }
# }
# impl Op for MyRoot<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl NonLinearOp for MyRoot<'_> {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = x[0] - 1.0;
#     }
# }
# impl Op for MyOut<'_> {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }
# impl NonLinearOp for MyOut<'_> {
#     fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
#         y[0] = x[0];
#     }
# }
# 
# struct MyProblem {
#     p: V,
# }
# 
# impl MyProblem {
#     fn new() -> Self {
#         MyProblem { p: V::zeros(2) }
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
#     fn nparams(&self) -> usize {
#         2
#     }
# }
# 
# impl<'a> OdeEquationsRef<'a> for MyProblem {
#     type Rhs = MyRhs<'a>;
#     type Mass = MyMass<'a>;
#     type Init = MyInit<'a>;
#     type Root = MyRoot<'a>;
#     type Out = MyOut<'a>;
# }
# 
# impl OdeEquations for MyProblem {
#     fn rhs(&self) -> <MyProblem as OdeEquationsRef<'_>>::Rhs {
#         MyRhs { p: &self.p }
#     }
#     fn mass(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Mass> {
#         Some(MyMass { p: &self.p })
#     }
#     fn init(&self) -> <MyProblem as OdeEquationsRef<'_>>::Init {
#         MyInit { p: &self.p }
#     }
#     fn root(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Root> {
#         Some(MyRoot { p: &self.p })
#     }
#     fn out(&self) -> Option<<MyProblem as OdeEquationsRef<'_>>::Out> {
#         Some(MyOut { p: &self.p })
#     }
#     fn set_params(&mut self, p: &V) {
#         self.p.copy_from(p);
#     }
# }
use diffsol::OdeBuilder;
let problem = OdeBuilder::<M>::new()
    .p(vec![1.0, 10.0])
    .build_from_eqn(MyProblem::new())
    .unwrap();
# }
```