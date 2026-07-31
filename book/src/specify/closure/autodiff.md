# Automatic differentiation

With the `autodiff` feature, Diffsol can generate the Jacobian-vector and
negative-transpose products needed by adjoint sensitivity analysis from ordinary
Rust closures. This avoids manually writing `rhs_jac`, `rhs_adjoint`,
`rhs_sens_adjoint`, and the initial-condition parameter-adjoint closure.

As in the previous closure examples, consider the logistic equation

$$\frac{dy}{dt} = r y (1 - y/K),$$

with the parameterized initial state

$$y(0) = y_0.$$

The parameter vector is \\(p = [r, K, y_0]\\), so the right-hand side is

$$f(y, p, t) = p_0 y (1 - y/p_1),$$

and the initial-condition function is

$$y_0(p, t) = p_2.$$

Using the `autodiff` feature, we can simply supply these equations, and `std::autodiff` will
automatically generate the required gradients for any type of solver in diffsol.

```rust,ignore
{{#include ../../../../examples/logistic-autodiff/src/problem_autodiff.rs}}
```

## Compiling standalone programs

Autodiff requires nightly Rust, the Enzyme backend, and fat LTO. Install the
backend once for the nightly toolchain:

```bash
rustup +nightly component add enzyme
```

You can then enable autodiff for a single crate with `RUSTFLAGS`:

```bash
CARGO_PROFILE_RELEASE_LTO=fat \
RUSTFLAGS="-Zautodiff=Enable" \
cargo +nightly run --release
```

For more details on installing and using `std::autodiff` please consult the
[internal](https://rustc-dev-guide.rust-lang.org/autodiff/internals.html) and
[user docs](https://doc.rust-lang.org/std/autodiff/index.html)

## Compiling workspaces

`RUSTFLAGS` applies to every crate that Cargo compiles, including dependencies.
When compiling the `diffsol` tests, we found that some of our dependencies
can fail to compile with `-Zautodiff=Enable`. If you run into a similar problem, a rustc wrapper can
append the flag only for crates that use autodiff. This repository's wrapper is
an example:

```sh
{{#include ../../../../.github/scripts/autodiff-rustc-wrapper.sh}}
```

Use it when building the `logistic-autodiff` workspace example:

```bash
CARGO_PROFILE_RELEASE_LTO=fat \
RUSTC_WRAPPER="$PWD/.github/scripts/autodiff-rustc-wrapper.sh" \
cargo +nightly run -p logistic-autodiff --features autodiff --release
```
