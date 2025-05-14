# An implicit example

We will now create an implicit ODE problem, again using the logistic equation as an example. This problem can be used in both the explicit and implicit diffsol solvers.

\\[\frac{dy}{dt} = r y (1 - y/K),\\] 

where \\(r\\) is the growth rate and \\(K\\) is the carrying capacity. 
To specify the problem, we need to provide the \\(dy/dt\\) function \\(f(y, p, t)\\), 
and the jacobian of \\(f\\) multiplied by a vector \\(v\\) function, which we will call \\(f'(y, p, t, v)\\). That is

\\[f(y, p, t) = r y (1 - y/K),\\]
\\[f'(y, p, t, v) = rv (1 - 2y/K),\\]

and the initial state 

\\[y_0(p, t) = 0.1\\]

This can be done using the following code:

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_implicit.rs}}
```

The `rhs_implicit` method is used to specify the \\(f(y, p, t)\\) and \\(f'(y, p, t, v)\\) functions, whereas the `init` method is used to specify the initial state vector \\(y_0(p, t)\\).
