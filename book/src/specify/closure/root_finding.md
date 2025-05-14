# Root finding

Root finding is the process of finding the values of the variables that make a set of equations equal to zero. This is a common problem where you want to
stop the solver or perform some action when a certain condition is met. 

## Specifying the root finding function

Using the logistic example, we can add a root finding function \\(r(y, p, t)\\) that will stop the solver when the value of \\(y\\) is such that \\(r(y, p, t) = 0\\).
For this example we'll use the root finding function \\(r(y, p, t) = y - 0.5\\), which will stop the solver when the value of \\(y\\) is 0.5.


\\[\frac{dy}{dt} = r y (1 - y/K),\\] 
\\[r(y, p, t) = y - 0.5,\\]

This can be done using the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) via the following code:

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_root.rs}}
```

here we have added the root finding function \\(r(y, p, t) = y - 0.5\\), and also let Diffsol know that we have one root function by passing `1` as the last argument to the `root` method.
If we had specified more than one root function, the solver would stop when any of the root functions are zero.

## Detecting roots during the solve

To detect the root during the solve, we can use the return type on the [`step`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#tymethod.step) method of the solver. 
If successful the `step` method returns an [`OdeSolverStopReason`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/enum.OdeSolverStopReason.html) enum that contains the reason the solver stopped.


```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/solve_match_step.rs}}
```