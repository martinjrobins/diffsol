# ODE equations

The simplest way to create a new ode problem is to use the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct. 
You can use methods to set the equations to be solve, initial time, initial step size, relative & absolute tolerances, and parameters, or leave them at their default values.
Then, call the `build` method to create a new problem.

Below is an example of how to create a new ODE problem using the `OdeBuilder` struct. 
The specific problem we will solve is the logistic equation 

\\[\frac{dy}{dt} = r y (1 - y/K),\\] 

where \\(r\\) is the growth rate and \\(K\\) is the carrying capacity. 
To specify the problem, we need to provide the \\(dy/dt\\) function \\(f(y, p, t)\\), 
and the jacobian of \\(f\\) multiplied by a vector \\(v\\) function, which we will call \\(f'(y, p, t, v)\\). That is

\\[f(y, p, t) = r y (1 - y/K),\\]
\\[f'(y, p, t, v) = rv (1 - 2y/K),\\]

and the initial state 

\\[y_0(p, t) = 0.1\\]

This can be done using the following code:

```rust
# fn main() {
use diffsol::OdeBuilder;
use nalgebra::DVector;
type M = nalgebra::DMatrix<f64>;

let problem = OdeBuilder::<M>::new()
    .t0(0.0)
    .rtol(1e-6)
    .atol([1e-6])
    .p(vec![1.0, 10.0])
    .rhs_implicit(
        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
    )
    .init(
        |_p, _t| DVector::from_element(1, 0.1),
    )
    .build()
    .unwrap();
# }
```

The `rhs_implicit` method is used to specify the \\(f(y, p, t)\\) and \\(f'(y, p, t, v)\\) functions, whereas the `init` method is used to specify the initial state vector \\(y_0(p, t)\\).
We also use the `t0`, `rtol`, `atol`, and `p` methods to set the initial time, relative tolerance, absolute tolerance, and parameters, respectively. 

We have also specified the matrix type `M` to be `nalgebra::DMatrix<f64>`, using a generic parameter of the `OdeBuilder` struct.
The `nalgebra::DMatrix<f64>` type is a dense matrix type from the [nalgebra](https://nalgebra.org) crate. Other options are:
- `faer::Mat<T>` from [faer](https://github.com/sarah-ek/faer-rs), which is a dense matrix type.
- `diffsol::SparseColMat<T>`, which is a thin wrapper around `faer::sparse::SparseColMat<T>`, a sparse compressed sparse column matrix type.
    
Each of these matrix types have an associated vector type that is used to represent the vectors in the problem (i.e. the state vector \\(y\\), the parameter vector \\(p\\), and the gradient vector \\(v\\)).
You can see in the example above that the `DVector` type is explicitly used to create the initial state vector in the third closure.
For these matrix types the associated vector type is:
- `nalgebra::DVector<T>` for `nalgebra::DMatrix<T>`.
- `faer::Col<T>` for `faer::Mat<T>`.
- `faer::Coll<T>` for `diffsol::SparseColMat<T>`.