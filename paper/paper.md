---
title: 'DiffSol: A Rust crate for solving differential equations'
tags:
  - rust
  - scientific computing
  - solver
  - ordinary differential equations
  - differential Algebraic equations
  - runge-kutta
  - backward differentiation formula
authors:
  - name: Martin Robinson
    orcid: 0000-0002-1572-6782
    corresponding: true
    affiliation: 1
  - name: Alex Allmont
    affiliation: 1


affiliations:
 - name: Oxford Research Software Engineering Group, Doctoral Training Centre, University of Oxford, Oxford, UK
   index: 1
   ror: 052gg0110
date: 12 March 2025
bibliography: paper.bib
---

# Summary

Ordinary Differential Equations (ODEs) are powerful tools for modelling a wide range of physical systems. Unlike purely data-driven models, ODEs can be based on the underlying physics, biology, or chemistry of the system being modelled, making them particularly useful for predicting the behaviour of a system under conditions that have not been observed. ODEs can be used to model everything from the motion of planets to the spread of infectious diseases.

`Diffsol` is a Rust crate for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs). It can solve equations in the following form:

$$
M \frac{dy}{dt} = f(t, y, p)
$$

where $y$ is the state of the system, $p$ are a set of parameters,  $t$ is time, $f(t, y, p)$ is a function that describes how the state of the system changes over time and $M$ is an optional and possibly singular mass matrix. The solution to an ODE is a function $y(t)$ that satisfies the ODE and any initial conditions.

The equations (e.g. $f(t, y, p)$) can be provided by the user either using Rust code, or a custom Domain Specific Language (DSL) called `DiffSL`. `DiffSL` uses automatic differentiation using Enzyme [@moses22enzyme] to calculate the necessary gradients, and JIT compilation (using either `LLVM` [@lattner2004llvm] or `Cranelift` [@cranelift]) to generate efficient native code at runtime. The DSL is ideal for using `DiffSol` from a higher-level language like Python or R while still maintaining similar performance to pure rust.

ODE solvers require linear algebra containers (e.g. vectors, matrices), operators and linear solvers. `DiffSol` allows users to choose both dense and sparse matrices and solvers from the `nalgebra` [@nalgebra] or `faer` [@faer] crates, and uses a trait-based approach to allow other linear algebra libraries to be added at a later date.

# Statement of need

ODE solvers have a long history in scientific computing, and many libraries currently exist. Some notable examples include `scipy.integrate.odeint` [@virtanen2020scipy] in Python, `ode45` [@shampine1997matlab] in MATLAB, and the `Sundials` suite of solvers [@gardner2022sundials] in C. Rust is a systems programming language that is gaining popularity in the scientific computing community due to its performance, safety, and ease of use. There is currently no ODE solver library written in Rust that provides the same level of functionality as these other libraries, and this is the gap that `DiffSol` aims to fill.

ODE solvers written in lower-level languages like C, Fortran or Rust offer significant performance benifits. However, these solvers are more difficult to wrap and use in higher-level languages like Python or MATLAB, primarily because users must supply their equations in the language of the solver. `DiffSol` solves this issue by providing its own custom `DiffSL` DSL which is JIT compiled to efficient native code at run-time, meaning that users from from a higher-level language like Python or R can specify their equations using a simple string-based format while still maintaining similar performance to pure Rust. Two other popular ODE solvers that take advantage of JIT compilation are `DifferentialEquations.jl` [@DifferentialEquations.jl-2017] in Julia, and `diffrax` [@kidger2021on] in Python. However, both these packages compile the entire solver as well as the equations, which is a significant amount of code. `DiffSol` only compiles the equations, meaning that it has a significantly smaller "time-to-first-plot" for users.

# Features

The following solvers are available in `DiffSol`:

1. A variable order Backwards Difference Formulae (BDF) solver, suitable for stiff problems and singular mass matrices.  The basic algorithm is derived in @byrne1975polyalgorithm, however this particular implementation follows that implemented in the Matlab routine ode15s [@shampine1997matlab] and the SciPy implementation [@virtanen2020scipy], which features the NDF formulas for improved stability
2. A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver, suitable for moderately stiff problems and singular mass matrices. Two different butcher tableau are provided, TR-BDF2 [@bank1985transient, @hosea1996analysis] and ESDIRK34 [@jorgensen2018family], or users can supply their own.

All solvers feature:

- Linear algebra containers and linear solvers from the `nalgebra` or `faer` crates, including both dense and sparse matrix support.
- Adaptive step-size control to given relative and absolute tolerances. Tolerances can be set separately for the main equations, quadrature of the output function, and sensitivity analysis.
- Dense output, interpolating to times provided by the user.
- Event handling, stopping when a given condition $g_e(t, y, p)$ is met or at a specific time.
- Numerical quadrature of an optional output $g_o(t, y, p)$ function over time.
- Forward sensitivity analysis, calculating the gradient of an output function or the solver states $y$ with respect to the parameters $p$.
- Adjoint sensitivity analysis, calculating the gradient of cost function $G(p)$ with respect to the parameters $p$. The cost function can be the integral of a continuous output function $g(t, y, p)$ or a sum of a set of discrete functions $g_i(t_i, y, p)$ at time points $t_i$.

# Acknowledgements

We greatfully acknowledge the support of all the past and future contributors to the `DiffSol` project, for their advice, enthusiasm, bug reports and code. In particular, we would like to thank the authors of the `pharmsol` crate, Julian Otalvaro and Markus Hovd.

# References
