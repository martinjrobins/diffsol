# Diffsol

<img src="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml/badge.svg" alt="CI build status badge">

Diffsol is a library for solving ordinary differential equations (ODEs) or
semi-explicit differential algebraic equations (DAEs) in Rust. You can use it
out-of-the-box with vectors and matrices from the
[nalgebra](https://nalgebra.org) crate, or you can implement your own types by
implementing the various vector and matrix traits in diffsol.

**Note**: This library is still in the early stages of development and is not
ready for production use. The API is likely to change in the future.

## Features

Currently only one solver is implemented, the Backward Differentiation Formula
(BDF) method. This is a variable step-size implicit method that is suitable for
stiff ODEs and semi-explicit DAEs and is similar to the BDF method in MATLAB's
`ode15s` solver or the `bdf` solver in SciPy's `solve_ivp` function.

Users can specify the equations to solve in the following ODE form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where `M` is a mass matrix, `y` is the state vector, `t` is the time, `p` is a
vector of parameters, and `f` is the right-hand side function. The mass matrix
`M` is optional (assumed to be the identity matrix if not provided).

The RHS function `f`  can be specified as a function
that takes the state vector `y`, the parameter vector `p`, and the time `t` as
arguments and returns `f(t, y, p)`. The action of the jacobian `J` of `f` must also be specified as a
function that takes the state vector `y`, the parameter vector `p`, the time `t`
and a vector `v` and returns the product `Jv`.

The action of the mass matrix `M` can also be specified as a function that takes an input vector `v`,
the parameter vector `p` and the time `t` and returns the product mass matrix `Mv`. 
Note that the only requirement for this mass matrix operator is that it must be linear. 
It can be, for example, a singular matrix with zeros on the diagonal (i.e. defining a semi-explicit DAE).

## Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
diffeq = "0.1"
```

or add it on the command line:

```sh
cargo add diffeq
```

## Usage

This example solves the Robertson chemical kinetics problem, which is a stiff ODE with the equations:

```math
\begin{align*}
\frac{dy_1}{dt} &= -0.04 y_1 + 10^4 y_2 y_3 \\
\frac{dy_2}{dt} &= 0.04 y_1 - 10^4 y_2 y_3 - 3 \times 10^7 y_2^2 \\
\frac{dy_3}{dt} &= 3 \times 10^7 y_2^2
\end{align*}
```

with the initial conditions `y_1(0) = 1`, `y_2(0) = 0`, and `y_3(0) = 0`. We set
the tolerances to `1.0e-4` for the relative tolerance and `1.0e-8`, `1.0e-6`,
and `1.0e-6` for the absolute tolerances of `y_1`, `y_2`, and `y_3`,
respectively. We set the problem up with the following code:

```rust
type T = f64;
type V = nalgebra::DVector<T>;
let problem = OdeBuilder::new()
    .p([0.04, 1.0e4, 3.0e7])
    .rtol(1e-4)
    .atol([1.0e-8, 1.0e-6, 1.0e-6])
    .build_ode(
        |x: &V, p: &V, _t: T, y: &mut V| {
            y[0] = -p[0] * x[0] + p[1] * x[1] * x[2];
            y[1] = p[0] * x[0] - p[1] * x[1] * x[2] - p[2] * x[1] * x[1];
            y[2] = p[2] * x[1] * x[1];
        },
        |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
            y[0] = -p[0] * v[0] + p[1] * v[1] * x[2] + p[1] * x[1] * v[2];
            y[1] = p[0] * v[0]
                - p[1] * v[1] * x[2]
                - p[1] * x[1] * v[2]
                - 2.0 * p[2] * x[1] * v[1];
            y[2] = 2.0 * p[2] * x[1] * v[1];
        },
        |_p: &V, _t: T| V::from_vec(vec![1.0, 0.0, 0.0]),
    )
    .unwrap();

let mut solver = Bdf::default();
```

We can then solve the problem up to time `t = 0.4` with the following code:

```rust
let t = 0.4;
let _y = solver.solve(&problem, t).unwrap();
```

Or if we want to explicitly step through time and have access to the solution
state, we can use the following code:

```rust
let mut state = OdeSolverState::new(&problem);
solver.set_problem(&mut state, problem);
while state.t <= t {
    solver.step(&mut state).unwrap();
}
let _y = solver.interpolate(&state, t);
```

Note that `step` will advance the state to the next time step as chosen by the
solver, and `interpolate` will interpolate the solution back to the exact time
`t` that is requested.



