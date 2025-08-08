<div align="center">
<a href="https://docs.rs/diffsol">
    <img src="https://img.shields.io/crates/v/diffsol.svg?label=docs&color=blue&logo=rust" alt="docs.rs badge">
</a>
<a href="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml">
    <img src="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml/badge.svg" alt="CI build status badge">
</a>
<a href="https://codecov.io/gh/martinjrobins/diffsol">
    <img src="https://codecov.io/gh/martinjrobins/diffsol/branch/main/graph/badge.svg" alt="code coverage">
</a>
</div>

# Diffsol

Diffsol is a library for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs) in Rust. It can solve equations in the following form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where $M$ is a (possibly singular and optional) mass matrix, $y$ is the state vector, $t$ is the time and $p$ is a vector of parameters.

The equations can be given by either rust code or the [DiffSL](https://martinjrobins.github.io/diffsl/) Domain Specific Language (DSL). The DSL uses automatic differentiation using [Enzyme](https://enzyme.mit.edu/) to calculate the necessary jacobians, and JIT compilation (using either [LLVM](https://llvm.org/) or [Cranelift](https://cranelift.dev/)) to generate efficient native code at runtime. The DSL is ideal for using Diffsol from a higher-level language like Python or R while still maintaining similar performance to pure rust.

## Installation and Usage

See installation instructions on the [crates.io page](https://crates.io/crates/diffsol).

The [Diffsol book](https://martinjrobins.github.io/diffsol/) describes how to use Diffsol using examples taken from several application areas (e.g. population dynamics, electrical circuits and pharmacological modelling), as well as more detailed information on the various APIs used to specify the ODE equations. For a more complete description of the API, please see the [docs.rs API documentation](https://docs.rs/diffsol).

## Features

The following solvers are available in Diffsol

1. A variable order Backwards Difference Formulae (BDF) solver, suitable for stiff problems and singular mass matrices. The basic algorithm is derived in [(Byrne & Hindmarsh, 1975)](#1), however this particular implementation follows that implemented in the Matlab routine ode15s [(Shampine & Reichelt, 1997)](#4) and the SciPy implementation [(Virtanen et al., 2020)](#5), which features the NDF formulas for improved stability
2. A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver, suitable for moderately stiff problems and singular mass matrices. Two different butcher tableau are provided, TR-BDF2 [(Hosea & Shampine, 1996)](#2) and ESDIRK34 [(Jørgensen et al., 2018)](#3), or users can supply their own.
3. A variable order Explict Runge-Kutta (ERK) solver, suitable for non-stiff problems. One butcher tableau is provided, the 4th order TSIT45 [(Tsitouras, 2011)](#5), or users can supply their own.

All solvers feature:

- Linear algebra containers and linear solvers from the nalgebra or faer crates, including both dense and sparse matrix support.
- Adaptive step-size control to given relative and absolute tolerances. Tolerances can be set separately for the main equations, quadrature of the output function, and sensitivity analysis.
- Dense output, interpolating to times provided by the user.
- Event handling, stopping when a given condition $g_e(t, y , p)$ is met or at a specific time.
- Numerical quadrature of an optional output $g_o(t, y, p)$ function over time.
- Forward sensitivity analysis, calculating the gradient of an output function or the solver states $y$ with respect to the parameters $p$.
- Adjoint sensitivity analysis, calculating the gradient of cost function $G(p)$ with respect to the parameters $p$. The cost function can be the integral of a continuous output function $g(t, y, p)$ or a sum of a set of discrete functions $h_i(t_i, y_i, p)$ at time points $t_i$.

## Wanted - Developers for higher-level language wrappers

Diffsol is designed to be easy to use from higher-level languages like Python or R. I'd prefer not to split my focus away from the core library, so I'm looking for developers who would like to lead the development of these wrappers. If you're interested, please get in touch.

- [x] Python (e.g. using [PyO3](https://pyo3.rs/v0.24.0/)). <https://github.com/alexallmont/pydiffsol>.
- [ ] Python ML frameworks (e.g. [JAX](https://docs.jax.dev/en/latest/ffi.html), [PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html))
- [ ] R (e.g. using [extendr](https://extendr.github.io/)).
- [ ] Julia
- [ ] Matlab
- [ ] Javascript in backend (e.g using [Neon](https://neon-rs.dev/))
- [ ] Javascript in browser (e.g. using [wasm-pack](https://rustwasm.github.io/wasm-pack/))
- [ ] Others, feel free to suggest your favourite language.

## References

- <a id="1"></a> Byrne, G. D., & Hindmarsh, A. C. (1975). A polyalgorithm for the numerical solution of ordinary differential equations. ACM Transactions on Mathematical Software (TOMS), 1(1), 71–96.81
- <a id="2"></a> Hosea, M., & Shampine, L. (1996). Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20(1-2), 21–37.
- <a id="3"></a> Jørgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv Preprint arXiv:1803.01613.
- <a id="4"></a> Shampine, L. F., & Reichelt, M. W. (1997). The matlab ode suite. SIAM Journal on Scientific Computing, 18(1), 1–22.
- <a id="5"></a> Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., & others. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods, 17(3), 261–272.
 - <a id="5"></a> Tsitouras, C. (2011). Runge–Kutta pairs of order 5 (4) satisfying only the first column simplifying assumption. Computers & Mathematics with Applications, 62(2), 770-775.
