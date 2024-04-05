<div align="center">
<a href="https://docs.rs/diffsol">
    <img src="https://img.shields.io/crates/v/diffsol.svg?label=docs&color=blue&logo=rust" alt="docs.rs badge">
</a>
<a href="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml">
    <img src="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml/badge.svg" alt="CI build status badge">
</a>
</div>

# DiffSol

Diffsol is a library for solving ordinary differential equations (ODEs) or
semi-explicit differential algebraic equations (DAEs) in Rust. You can use it
out-of-the-box with vectors and matrices from the
[nalgebra](https://nalgebra.org) crate, or you can implement your own types by
implementing the various vector and matrix traits in diffsol.

## Features

DiffSol has two implementations of the Backward Differentiation Formula
(BDF) method, one in pure rust, the other wrapping the [Sundials](https://github.com/LLNL/sundials) IDA solver.
This method is a variable step-size implicit method that is suitable for
stiff ODEs and semi-explicit DAEs and is similar to the BDF method in MATLAB's
`ode15s` solver or the `bdf` solver in SciPy's `solve_ivp` function.

Users can specify the equations to solve in the following ODE form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where $M$ is a mass matrix, $y$ is the state vector, $t$ is the time, $p$ is a
vector of parameters, and $f$ is the right-hand side function. The mass matrix
$M$ is optional (assumed to be the identity matrix if not provided).

## Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
diffsol = "0.1.4"
```

or add it on the command line:

```sh
cargo add diffsol
```

## Usage

For more documentation and examples, see the [API documentation](https://docs.rs/diffsol/latest/diffsol/).
