# Python

The `PyO3` and `maturin` crates allow you to easily create Python bindings for Rust libraries, and Diffsol is no exception. For a full example of how to use Diffsol from Python, see the [Diffsol Python example](https://github.com/martinjrobins/diffsol/tree/main/examples/python-diffsol). Below is a brief overview of some of the key features from this example.

## Getting started

You can install `maturin` using `pip`:

```bash
pip install maturin
```

Then, you can create a template `maturin` project using `pyo3`:

```bash
maturin new -b pyo3 python-diffsol
```

```bash
cd python-diffsol
cargo add diffsol
```


## Wrapping a Diffsol problem

First, lets define some types that we'll use, including a matrix `M`, vector `V` and context type `C` for the linear algebra operations, as well as a linear solver type `LS`. For the JIT compilation backend, we'll swap between the LLVM and Cranelift backends, depending on whether the `diffsol-llvm` feature is enabled.

```rust,ignore
{{#include ../../../examples/python-diffsol/src/lib.rs::21}}
```

Now lets create a simple struct that we're going to wrap to use from Python. This will just store a Diffsol problem that we can solve later. Since we're using the `DiffSL` equations type (see `Eqn` above), this isn't threadsafe, so we'll use an `Arc<Mutex<_>>` to allow us to share the problem between threads safely.

```rust,ignore
{{#include ../../../examples/python-diffsol/src/lib.rs:23:26}}
```

## Solving the problem

For our implementation for `PyDiffsol`, we'll create two methods: one for creating the problem from a DiffSL string (`new`), and another for solving the problem (`solve`). The `solve` method will take a set of parameters and a set of times to solve the problem at. It then creates an `Array2` to store the solution (we are using the `numpy` crate to allow us to return a NumPy array), and then iterates over the times, stepping the solver and interpolating the solution at each time. If the problem has an output function, it will call that function to get the output, otherwise it will just return the state vector.

```rust,ignore
{{#include ../../../examples/python-diffsol/src/lib.rs:28:76}}
```

## Error handling

In our implementation, we need to handle errors that may occur when working with the Diffsol library. We'll create a custom error type `PyDiffsolError` that wraps the `DiffsolError` type, and implement the `From` trait to convert between the two types and the Python `PyErr` type. This will allow us to easily convert `DiffsolError` errors into Python exceptions.

```rust,ignore
{{#include ../../../examples/python-diffsol/src/lib.rs:78:90}}
```

## The Python module

Finally, we need to create the Python module that will be imported by Python.

```rust,ignore
{{#include ../../../examples/python-diffsol/src/lib.rs:92:}}
```

## Testing from Python

To build the Python module, we can use the `maturin` tool. This will compile the Rust code and create a Python wheel.

```bash
maturin develop
```

Then we can write and run a simple Python test to ensure everything is working correctly:

```python,ignore
{{#include ../../../examples/python-diffsol/test/test_solve.py}}
```
