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
[nalgebra](https://nalgebra.org) or [faer](https://github.com/sarah-ek/faer-rs) crates, or you can implement your own types by
implementing the various vector and matrix traits in diffsol.

## Features

DiffSol implements the following solvers:
- A variable order Backwards Difference Formulae (BDF) solver, suitable for stiff problems and singular mass matrices.
- A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver, suitable for moderately stiff problems and singular mass matrices. You can use your own butcher tableau or use one of the provided (`tr_bdf2` or `esdirk34`).
- A BDF solver that wraps the IDA solver solver from the [Sundials library](https://github.com/LLNL/sundials) (requires the `sundials` feature). This is similar to the BDF solver above and is include for comparison purposes.

All solvers feature adaptive step-size control to given tolerances, dense output, event handling and stepping to specific times. 
For comparison, the BDF solvers are similar to MATLAB's `ode15s` solver or the `bdf` solver in SciPy's `solve_ivp` function. 
The ESDIRK solver using the provided `tr_bdf2` tableau is similar to MATLAB's `ode23t` solver.

Users can specify the equations to solve in the following ODE form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where $M$ is a (possibly singular) mass matrix, $y$ is the state vector, $t$ is the time, $p$ is a
vector of parameters, and $f$ is the right-hand side function. The mass matrix
$M$ is optional (assumed to be the identity matrix if not provided).

## Installation

See instructions on the [crates.io page](https://crates.io/crates/diffsol).

## Usage

For more documentation and examples, see the [API documentation](https://docs.rs/diffsol/latest/diffsol/).
