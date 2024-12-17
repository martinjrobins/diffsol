<div align="center">
<a href="https://docs.rs/diffsol">
    <img src="https://img.shields.io/crates/v/diffsol.svg?label=docs&color=blue&logo=rust" alt="docs.rs badge">
</a>
<a href="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml">
    <img src="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml/badge.svg" alt="CI build status badge">
</a>
</div>

# DiffSol

Diffsol is a library for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs) in Rust. It can solve equations in the following form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where $M$ is a (possibly singular and optional) mass matrix, $y$ is the state vector, $t$ is the time and $p$ is a vector of parameters. 

The equations can be given by either rust closures or the [DiffSL](https://martinjrobins.github.io/diffsl/) Domain Specific Language (DSL). The DSL uses automatic differentiation using [Enzyme](https://enzyme.mit.edu/) to calculate the necessary jacobians, and JIT compilation (using either [LLVM](https://llvm.org/) or [Cranelift](https://cranelift.dev/)) to generate efficient native code at runtime. The DSL is ideal for using DiffSol from a higher-level language like Python or R while still maintaining similar performance to pure rust.

You can use DiffSol out-of-the-box with vectors, matrices and linear solvers from the [nalgebra](https://nalgebra.org) or [faer](https://github.com/sarah-ek/faer-rs) crates, or you can implement your own types or solvers by implementing the required traits.

## Installation and Usage

See installation instructions on the [crates.io page](https://crates.io/crates/diffsol).

The [DiffSol book](https://martinjrobins.github.io/diffsol/) describes how to use DiffSol using examples taken from several application areas (e.g. population dynamics, electrical circuits and pharmacological modelling), as well as more detailed information on the various APIs used to specify the ODE equations. For a more complete description of the API, please see the [docs.rs API documentation](https://docs.rs/diffsol). 

## Solvers 

DiffSol implements the following solvers:
- A variable order Backwards Difference Formulae (BDF) solver, suitable for stiff problems and singular mass matrices.
- A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver, suitable for moderately stiff problems and singular mass matrices. You can use your own butcher tableau or use one of the provided (`tr_bdf2` or `esdirk34`).

All solvers feature:
- adaptive step-size control to given tolerances, 
- dense output, 
- event handling, 
- stepping to specific times,
- numerical quadrature of an optional output function over time
- forward sensitivity analysis,
- backwards or adjoint sensitivity analysis,

For comparison, the BDF solvers are similar to MATLAB's `ode15s` solver, the `bdf` solver in SciPy's `solve_ivp` function, or the BDF solver in SUNDIALS. The ESDIRK solver using the provided `tr_bdf2` tableau is similar to MATLAB's `ode23t` solver.