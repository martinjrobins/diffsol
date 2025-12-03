<div align="center">
<p></p><img src="https://raw.githubusercontent.com/martinjrobins/diffsol/refs/heads/main/assets/diffsol_rectangle.svg" alt="diffsol logo" width="300"/></p>
<a href="https://martinjrobins.github.io/diffsol/">
    <img src="https://img.shields.io/crates/v/diffsol.svg?label=mdbook&color=green&logo=mdbook" alt="mdbook badge">
</a>
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
---

Diffsol is a library for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs) in Rust. It can solve equations in the following form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where $M$ is a (possibly singular and optional) mass matrix, $y$ is the state vector, $t$ is the time and $p$ is a vector of parameters.

The equations can be given by either rust code or the [DiffSL](https://martinjrobins.github.io/diffsl/) Domain Specific Language (DSL). The DSL uses automatic differentiation using [Enzyme](https://enzyme.mit.edu/) to calculate the necessary jacobians, and JIT compilation (using either [LLVM](https://llvm.org/) or [Cranelift](https://cranelift.dev/)) to generate efficient native code at runtime. The DSL is ideal for using diffsol from a higher-level language like Python or R while still maintaining similar performance to pure rust.

## Installation

You can add diffsol using `cargo add diffsol` or directly in your `Cargo.toml`:

```toml
[dependencies]
diffsol = "0.8"
```

Diffsol has the following features that can be enabled or disabled:

- `nalgebra`: Use nalgebra for linear algebra containers and solvers (enabled by default).
- `faer`: Use faer for linear algebra containers and solvers (enabled by default).
- `cuda`: Use in-built CUDA linear algebra containers and solvers (disabled by default, experimental).
- `diffsl-llvm15`, `diffsl-llvm16`, `diffsl-llvm17`, `diffsl-llvm18`, `diffsl-llvm19`, `diffsl-llvm20`, `diffsl-cranelift`: Enable DiffSL with the specified JIT backend (disabled by default). You will need to set the `LLVM_SYS_XXX_PREFIX` (see [`llvm-sys`](https://gitlab.com/taricorp/llvm-sys.rs)) and `LLVM_DIR` environment variables to point to your LLVM installation, where `XXX` is the version number (`150`, `160`, `170`, `181`, `191`, `201`, `211`).
- `suitesparse`: Enable SuiteSparse KLU sparse linear solver (disabled by default, requires `faer`).

You can add any of the above features by specifying them in your `Cargo.toml`. For example, to enable the `diffsl-cranelift` JIT backend, you would add:

```toml
[dependencies]
diffsol = { version = "0.8", features = "diffsl-cranelift" }
```

See the [Cargo.toml documentation](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html) for more information on specifying features.

# Usage

The [diffsol book](https://martinjrobins.github.io/diffsol/) describes how to use diffsol using examples taken from several application areas (e.g. population dynamics, electrical circuits and pharmacological modelling), as well as more detailed information on the various APIs used to specify the ODE equations. For a more complete description of the API, please see the [docs.rs API documentation](https://docs.rs/diffsol).

For a quick start, see the following example of solving the Lorenz system of equations using the BDF solver and the DiffSL DSL with the LLVM JIT backend:

```rust
use diffsol::{LlvmModule, NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod};

pub fn lorenz() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeBuilder::<NalgebraMat<f64>>::new().build_from_diffsl::<LlvmModule>(
        "
            a { 14.0 } b { 10.0 } c { 8.0 / 3.0 }
            u_i {
                x = 1.0,
                y = 0.0,
                z = 0.0,
            }
            F_i {
                b * (y - x);
                x * (a - z) - y;
                x * y - c * z;
            }
        ",
    )?;
    let mut solver = problem.bdf::<NalgebraLU<f64>>()?;
    let (_ys, _ts) = solver.solve(0.0)?;
    Ok(())
}
```

## ODE solvers

The following ODE solvers are available in diffsol

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

## Contributing

Contributions are very welcome, as are bug reports! Please see the [contributing guidelines](CONTRIBUTING.md) for more information, but in summary:

- Please open an [issue](https://github.com/martinjrobins/diffsol/issues) or [discussion](https://github.com/martinjrobins/diffsol/discussions) to report any issues or problems using diffsol
- There are a number of repositories in the diffsol ecosystem, please route your issue/request to the appropriate repository:
  - [diffsol](https://github.com/martinjrobins/diffsol) - the core ODE solver library
  - [diffsl](https://github.com/martinjrobins/diffsl) - the DiffSL DSL compiler and JIT backends
  - [pydiffsol](https://github.com/alexallmont/pydiffsol) - Python bindings
- Feel free to submit a pull request with your changes or improvements, but please open an issue first if the change is significant. The [contributing guidelines](CONTRIBUTING.md) describe how to set up a development environment, run tests, and format code.
  
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
