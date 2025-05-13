# An explicit example

Below is an example of how to create an explicit ODE problem using the `OdeBuilder` struct. This type of problem is suitable only for explicit solver methods.

The specific problem we will solve is the logistic equation 

\\[\frac{dy}{dt} = r y (1 - y/K),\\] 

where \\(r\\) is the growth rate and \\(K\\) is the carrying capacity. 
To specify the problem, we need to provide the \\(dy/dt\\) function \\(f(y, p, t)\\), 

\\[f(y, p, t) = r y (1 - y/K),\\]

and the initial state 

\\[y_0(p, t) = 0.1\\]

This can be done using the following code:

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_explicit.rs}}
```

The return type of this function, `OdeSolverProblem<impl OdeEquations<M=M, V=V, T=T, C=C>>` details the type of problem that we are returning. The `OdeSolverProblem` struct contains our equations, which are of type `impl OdeEquations<M=M, V=V, T=T, C=C>`, where `M` is the matrix type, `V` is the vector type, `T` is the time type, and `C` is the context type. We need to specify the matrix, vector, scalar and context types as these are used for the underlying linear algebra containers and operations. The `OdeEquations` trait specifies that this system of equations is only suitable for explicit solver methods.

The `rhs` method is used to specify the \\(f(y, p, t)\\) function, whereas the `init` method is used to specify the initial state vector \\(y_0(p, t)\\).